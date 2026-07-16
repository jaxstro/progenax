r"""Simulation-Based Calibration (SBC) driver for the gravoturb inference layer (Task 6).

Simulation-Based Calibration (Talts et al. 2018, *Validating Bayesian Inference Algorithms
with Simulation-Based Calibration*) checks that the inference machinery (prior + likelihood +
NUTS) is *self-consistent*: if for each trial we draw theta* from the prior, simulate a mock
from theta*, and run the posterior, then the rank of theta* among (thinned, ~independent)
posterior draws is **DiscreteUniform** over ``{0, ..., L}`` when the sampler is calibrated.
Deviations from uniformity diagnose miscalibration (a U/inverted-U = over/under-dispersion, a
slope = bias). The heavy uniformity test lives in AC18 / Task 7; this module is the *driver*
that produces the rank table.

Two pieces:

- :func:`build_logdensity` -- the shared, **prior-aware** unconstrained log-density factory used
  by BOTH the SBC driver and AC16. It composes (over a constrained ``z`` reparametrization)::

        logdensity(z) = tail_exceedance_loglike(...)               # POT tail -> alpha
                      + sum_c log_count_variance_loglike(meas_v_c, # tail-robust CIC sigma_s^2
                            ..., var_v_c)                          #   -> mach (scalar; weak beta)
                      + bandpower_gaussian_loglike(P_pred-bp, ...) # 2-pt band powers -> beta
                      + prior.logpdf([M, alpha, beta])             # proper BM19 prior
                      + log_jacobian(z)                            # reparam Jacobian

  The **2-pt band-power block** is the beta carrier. The single-scalar log-count variance (even
  at multiple cell scales) barely constrains the GRF slope beta -- it integrates the spectrum to
  one number, so SBC found beta UNDER-constrained (rank-uniformity fails). The design-intended
  fix is the field-level log-density power-spectrum band-powers (à la AC15): predicted analytic
  band-powers ``P_pred(theta)`` (:func:`power_spectrum_bandpowers`, differentiable, the beta
  carrier) fit to the measured periodogram ``band_powers`` of the latent log-density field via a
  Gaussian ``-1/2 r^T Cinv r`` with a FIXED-FIDUCIAL Hartlap precision ``bp_precision`` (no
  log|C| term, exactly like :func:`gaussian_loglike`). This is a FIELD-LEVEL UPPER BOUND on beta
  information: the band-powers are measured from the continuous log-density field, with NO star
  shot noise on the 2-pt (cf. the AC15 scoping caveat). ``bp_precision`` is computed ONCE at a
  FIXED fiducial theta (NOT the trial truth), so -- like ``var_v`` -- it is a truth-independent
  constant and does not break SBC.

  The stellar CIC block is the **tail-robust log-count-variance** statistic
  (:func:`log_count_variance_loglike`, Task 7) -- a Gaussian fit of the measured
  ``Var_cells[log_plus(N_cell)]`` to its analytic prediction with a FIXED-FIDUCIAL estimator
  variance ``var_v``. It REPLACES the per-cell ``count_loglike`` count-histogram block, which
  was tail-sensitive and biased mach high (the AC18 ℳ-bias). Crucially, ``var_v`` is computed
  ONCE per inference at a fixed fiducial theta (NOT at the trial truth ``theta*``) so it is a
  truth-independent constant -- a truth-keyed var_v would be exactly the kind of SBC artifact
  the old POT validity barrier was.

  AC16 originally carried a *flat-in-theta* (improper) prior; SBC REQUIRES a proper prior, so
  the prior term is part of this shared factory. AC16 is refactored to call this factory with
  its own :class:`BM19Prior` so there is a single source of truth for the composition.

  **No POT-validity barrier.** Earlier versions added ``pot_validity_barrier(theta, s_thr)`` to
  keep ``s_t(theta) <= s_thr``. With the per-trial threshold ``s_thr = s_t(theta*) + margin``
  that barrier is a *trial-dependent prior keyed to the truth* (it is absent from the SBC
  sampling prior and from mock generation), which chops the upper-alpha posterior asymmetrically
  around the truth and skews SBC ranks high at high Mach -- a driver artifact, not an engine
  fault. Because ``tail_exceedance_loglike`` is **shift-immune in s_thr** (the lognormal norm
  cancels, so the truth-keyed threshold does NOT bias alpha through the likelihood) and the
  proper :class:`BM19Prior` already bounds ``(M, alpha, beta)``, the barrier is dropped. SBC
  validity is then verified empirically by AC18's rank-uniformity test.

- :func:`sbc_ranks` -- the calibration loop. A python loop over trials (the validation/oracle
  side -- each trial is a full NUTS run), with per-trial fold-in keys for determinism /
  resumability. Per trial it draws theta*, builds the BM19 mock (gas tail exceedances + the
  tail-robust stellar-CIC log-count variance + the field-level 2-pt band powers) exactly as AC16
  does, builds the prior-aware log-density, runs NUTS, thins the draws to ~independence (Talts
  2018 sec. 5.1), and ranks theta* among them. BOTH fixed-fiducial precisions -- the count
  block's estimator variance ``var_vs`` and the 2-pt band-power precision ``bp_precision`` -- are
  computed ONCE (before the trial loop) at a fixed fiducial theta -- never at the trial truth --
  and threaded into every trial's log-density (SBC-valid as truth-independent constants).

The mock construction (n_bar per cell, the rank-copula gas/stellar fields) mirrors
:func:`gravoturb.validation.acceptance.ac16_hmc_recovery` so the calibrated object is the
SAME inference machinery AC16 exercises. The numpy paths (``measure_exceedances``,
``measure_log_count_variance``, the per-trial scalar threshold) are the non-differentiable
oracle side -- the differentiable interface is the log-density itself (Phase 5 design).
"""

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from gravoturb.realization.gaussian_field import gaussian_random_field
from gravoturb.realization.copula import rank_copula_field
from gravoturb.realization.placement import sample_cic_counts
from gravoturb.inference.covariance import (
    measured_bandpowers,
    mock_precision,
    power_spectrum_bandpowers,
)
from gravoturb.inference.hmc import (
    log_jacobian,
    run_nuts,
    to_constrained,
    to_unconstrained,
)
from gravoturb.inference.likelihood import (
    log_count_variance_loglike,
    tail_exceedance_loglike,
)
from gravoturb.inference.priors import BM19Prior
from gravoturb.theory.density_pdf import sigma_s_squared, transition_density
from gravoturb.validation.measure import (
    estimate_log_count_variance_var,
    measure_exceedances,
    measure_log_count_variance,
)

# Fixed fiducial theta for the truth-INDEPENDENT log-count-variance estimator variance
# (var_v). Used ONCE per inference in sbc_ranks to compute var_v at a fixed point -- NEVER at
# the trial truth theta* (a truth-keyed var_v would break SBC, like the old POT barrier did).
# The SAME fixed fiducial theta also keys the band-power precision bp_precision (below).
_MACH_FID = 8.0
_ALPHA_FID = 2.5
_BETA_FID = 3.0
_N_REAL_VAR_V = 12

# Fixed |k| band-power edges for the field-level 2-pt block (the beta carrier). 5 bins on the
# 24^3 stellar grid the count model FFTs on. NB: AC15/data_vector use jnp.linspace(2.0, 11.0, 5)
# (4 bins); we use one more bin (6 edges -> 5 bins) to give beta a touch more 2-pt leverage on
# the small grid -- harmless, since bp_precision is the matched mock covariance of the SAME edges.
_K_EDGES = jnp.linspace(2.0, 11.0, 6)
_N_REAL_BP = 64  # ensemble size for the fixed-fiducial band-power mock precision

# Fixed turbulence driving parameter the fiducial band-power ensemble uses (matches the fixed b
# threaded through inference; b is not a free param here).
_B_FID = 0.4


def build_logdensity(
    prior: BM19Prior,
    data: dict,
    b: float,
    s_thr: float,
    s_max: float,
    shape: tuple[int, int, int],
    cell_sizes: tuple[int, ...],
    bp_precision,
    n_max: int = 10,
    n_s: int = 400,
) -> Callable:
    r"""Build the shared prior-aware unconstrained log-density closure (see module docstring).

    Parameters
    ----------
    prior : BM19Prior
        Proper prior over the 3 free params ``[M, alpha, beta]`` (``b`` is fixed and outside it).
    data : dict
        Per-trial mock data bundle with keys:
        ``"exc_counts"`` (nb,), ``"exc_edges"`` (nb+1,) -- the gas-tail POT histogram from
        :func:`measure_exceedances`; ``"log_count_vars"`` -- tuple of per-cell measured
        ``Var_cells[log_plus(N)]`` scalars (:func:`measure_log_count_variance`, Task 7);
        ``"var_vs"`` -- tuple of per-cell FIXED-FIDUCIAL estimator variances (threaded in from
        :func:`sbc_ranks`, computed ONCE at a fixed fiducial theta, NOT at the trial truth ->
        SBC-valid); ``"n_bars"`` -- tuple of per-cell ``n_bar`` (matched to ``cell_sizes``);
        ``"band_powers"`` -- (k,) measured periodogram band-powers of the latent log-density
        field on ``_K_EDGES`` (the 2-pt beta channel; :func:`measured_bandpowers`).
    b : float
        Fixed turbulence driving parameter (the likelihood constant in ``theta4``).
    s_thr, s_max : float
        POT threshold and realized field maximum (data-derived constants; same field as
        ``exc_counts``, which makes the POT block shift-immune).
    shape : (n, n, n)
        Stellar CIC grid the count model FFTs on (forward-bias-matched). The band-power block
        FFTs on this SAME grid (``_K_EDGES``).
    cell_sizes : tuple[int, ...]
        CIC cell sizes; ``data["log_count_vars"]`` / ``data["var_vs"]`` / ``data["n_bars"]``
        align with this.
    bp_precision : (k, k) array
        FIXED-FIDUCIAL Hartlap-corrected band-power precision (:func:`mock_precision` over an
        ensemble at the fixed fiducial theta; threaded in from :func:`sbc_ranks`). Truth-
        independent constant (like ``var_v``) -> SBC-valid. numpy/jnp both ok (fixed data).
    n_max, n_s : int
        ``log_count_variance_loglike`` quadrature controls (mirror AC16 defaults). ``n_max`` also
        sets the Gaussianization order of the predicted band-powers.

    Returns
    -------
    logdensity_fn : z (3,) -> scalar log-density (differentiable in z).
    """
    exc_counts = jnp.asarray(data["exc_counts"])
    exc_edges = jnp.asarray(data["exc_edges"])
    log_count_vars = tuple(jnp.asarray(mv) for mv in data["log_count_vars"])
    var_vs = tuple(jnp.asarray(vv) for vv in data["var_vs"])
    n_bars = tuple(float(nb) for nb in data["n_bars"])
    band_powers = jnp.asarray(data["band_powers"])  # measured 2-pt band powers (beta channel)
    bp_precision = jnp.asarray(bp_precision)         # fixed-fiducial Hartlap precision (k, k)

    def logdensity(z):
        m_, a_, be_ = to_constrained(z)
        theta3 = jnp.array([m_, a_, be_])
        theta4 = jnp.array([m_, b, a_, be_])
        # POT tail block -> alpha. SHIFT-IMMUNE in s_thr (the lognormal norm
        # cancels), so a per-trial s_thr keyed to the truth does NOT bias alpha.
        ll = tail_exceedance_loglike(exc_counts, exc_edges, theta4, s_thr, s_max)
        # tail-robust stellar CIC blocks (log-count variance) -> mach (scalar sigma_s^2; only a
        # weak beta dependence). var_v is the FIXED-FIDUCIAL estimator variance (truth-
        # independent; see build_logdensity docstring).
        for c, mv, vv, nb in zip(cell_sizes, log_count_vars, var_vs, n_bars):
            ll = ll + log_count_variance_loglike(
                mv, theta4, shape, c, nb, vv, n_max=n_max, n_s=n_s
            )
        # field-level 2-pt band-power block -> beta. Predicted analytic band-powers (the beta
        # carrier; differentiable) vs the measured periodogram, Gaussian with the fixed-fiducial
        # precision -- same -1/2 r^T Cinv r pattern as gaussian_loglike (no log|C| term). This is
        # the channel the single-scalar log-count variance lacked; it is a FIELD-LEVEL UPPER
        # BOUND (no star shot noise on the 2-pt; see build_logdensity docstring).
        _, P_pred, _ = power_spectrum_bandpowers(
            shape, be_, m_, b, a_, _K_EDGES, n_max=n_max
        )
        r = P_pred - band_powers
        ll = ll + (-0.5 * r @ (bp_precision @ r))
        # proper prior (replaces AC16's flat-in-theta improper prior)
        ll = ll + prior.logpdf(theta3)
        # NB: NO pot_validity_barrier here. The barrier penalizes draws with
        # s_t(theta) > s_thr, and with the per-trial s_thr = s_t(theta*)+margin it
        # is a *trial-dependent prior* (keyed to the truth) absent from the SBC
        # sampling prior -- it would chop the upper-alpha posterior asymmetrically
        # around the truth and skew SBC ranks high at high Mach (a driver artifact,
        # not an engine fault). The proper BM19Prior already bounds (M, alpha, beta),
        # and the POT likelihood is well-defined without it, so it is dropped. See
        # the SBC-validity note in the per-paper Talts/Sailynoja docs.
        # reparametrization Jacobian (sampling in unconstrained z)
        return ll + log_jacobian(z)

    return logdensity


def _build_mock(
    theta_star,
    b: float,
    s_thr_margin: float,
    shape: tuple[int, int, int],
    density_shape: tuple[int, int, int],
    cell_sizes: tuple[int, ...],
    n_stars: float,
    key,
    n_exc_bins: int = 12,
):
    r"""Build the BM19 mock for one SBC trial (mirrors ac16_hmc_recovery's construction).

    Returns ``(data, s_thr, s_max)`` where ``data`` is the bundle :func:`build_logdensity`
    consumes. Option B per-trial POT threshold: ``s_thr = s_t(theta*) + margin`` (the
    transition density at the *injected* theta, not a global constant).
    """
    M_star, alpha_star, beta_star = (
        float(theta_star[0]),
        float(theta_star[1]),
        float(theta_star[2]),
    )

    # Option B per-trial threshold from the injected theta.
    s_t = float(transition_density(alpha_star, sigma_s_squared(M_star, b)))
    s_thr = s_t + s_thr_margin

    # Gas-density map (faithful rank copula) -> threshold exceedances -> alpha (POT).
    g_hi = gaussian_random_field(density_shape, beta_star, jax.random.fold_in(key, 7))
    s_hi = np.asarray(rank_copula_field(g_hi, M_star, b, alpha_star))
    field_max = float(s_hi.max())

    if field_max > s_thr:
        exc_counts, exc_edges, s_max, _n_tail = measure_exceedances(
            s_hi, s_thr, n_bins=n_exc_bins
        )
    else:
        # Finite-field truncation: at high (Mach, alpha) the per-trial transition density
        # s_t(theta*) (hence s_thr = s_t + margin) can exceed the densest cell this finite
        # grid realizes, so NO cells lie above threshold. This is honest, not a failure: we
        # hand the likelihood a VALID empty POT histogram -- zero counts on a monotone
        # [s_thr, s_thr + L] edge array with L > 0 so tail_exceedance_loglike's normalizer
        # 1 - e^{-alpha L} stays finite -- which contributes EXACTLY 0 (sum of count*log_p with
        # all counts 0). The empty-tail factor is thus identically 1 under BOTH the generative
        # and the inference model, so the posterior remains the correct Bayesian posterior under
        # the same joint model and theta* still ranks uniformly -> SBC validity is preserved.
        # (alpha is NOT "the prior alone" here -- it is still informed by the stellar CIC count
        # block below, which depends on alpha; the empty tail merely adds no alpha information.)
        s_max = s_thr + max(s_thr_margin, 1.0)  # any L > 0; counts are zero so L is immaterial
        exc_edges = np.linspace(s_thr, s_max, n_exc_bins + 1)
        exc_counts = np.zeros(n_exc_bins, dtype=float)

    # Stellar CIC counts on the SAME grid the model FFTs on (forward-bias-matched) -> M, beta.
    s_lo = rank_copula_field(
        gaussian_random_field(shape, beta_star, jax.random.fold_in(key, 1)),
        M_star,
        b,
        alpha_star,
    )
    log_count_vars, n_bars = [], []
    for c in cell_sizes:
        nb = n_stars / (shape[0] // c) ** 3
        cnt = np.asarray(
            sample_cic_counts(s_lo, nb, c, jax.random.fold_in(key, 100 + c))
        )
        # Tail-robust statistic (Task 7): measured Var_cells[log_plus(N)] per cell scale,
        # replacing the count histogram. var_v (fixed-fiducial estimator variance) is threaded
        # in by sbc_ranks, NOT computed here -- it must be truth-independent for SBC validity.
        log_count_vars.append(measure_log_count_variance(cnt, nb))
        n_bars.append(nb)

    # Field-level 2-pt band powers (the beta channel): measured periodogram of the SAME latent
    # stellar log-density field s_lo, on the SAME grid/_K_EDGES the predicted band-powers FFT on.
    band_powers = measured_bandpowers(np.asarray(s_lo), shape, _K_EDGES)

    data = {
        "exc_counts": exc_counts,
        "exc_edges": exc_edges,
        "log_count_vars": tuple(log_count_vars),
        "n_bars": tuple(n_bars),
        "band_powers": band_powers,
    }
    return data, float(s_thr), float(s_max)


def sbc_ranks(
    prior: BM19Prior,
    key,
    n_trials: int,
    b: float,
    s_thr_margin: float,
    shape: tuple[int, int, int],
    density_shape: tuple[int, int, int],
    n_warmup: int,
    n_samples: int,
    n_thin: int,
    cell_sizes: tuple[int, ...],
    n_stars: float,
    n_max: int = 10,
    n_s: int = 400,
    n_exc_bins: int = 12,
) -> dict:
    r"""Run the SBC calibration loop and return the rank table (Talts 2018).

    For each of ``n_trials`` trials (python loop, per-trial fold-in keys for determinism /
    resumability): draw ``theta* = (M*, alpha*, beta*) ~ prior``; build a BM19 mock from
    theta* (Option B per-trial POT threshold ``s_thr = s_t(theta*) + s_thr_margin``); build the
    prior-aware log-density (:func:`build_logdensity`); run NUTS initialized at the unconstrained
    truth; thin the draws by ``n_thin`` to ~independence; rank theta* among the thinned draws.

    The rank of a calibrated posterior is DiscreteUniform over ``{0, ..., L}`` (``L`` = number of
    thinned draws). Per free parameter (M, alpha, beta) the rank is
    ``sum_l 1[theta_l < theta*]`` in ``{0, ..., L}``.

    Returns
    -------
    dict with keys:
        ``"ranks"``       (n_trials, 3) int   -- per-param rank in {0..L}
        ``"n_draws"``     int                 -- L = number of thinned draws per trial
        ``"n_trials"``    int
        ``"param_names"`` ["M", "alpha", "beta"]
        ``"thetas_true"`` (n_trials, 3) float -- the drawn theta* per trial
    """
    ranks = np.zeros((n_trials, 3), dtype=int)
    thetas_true = np.zeros((n_trials, 3), dtype=float)
    n_draws = None

    # --- Fixed-fiducial estimator variance var_v for the log-count-variance block. ---
    # CRITICAL FOR SBC VALIDITY: var_v MUST be truth-independent. It is computed ONCE here,
    # before the trial loop, at a FIXED fiducial theta (M=_MACH_FID, alpha=_ALPHA_FID,
    # beta=_BETA_FID; the same fixed b as inference) -- NEVER at the trial truth theta*. A
    # truth-keyed var_v is exactly the kind of artifact (like the old POT validity barrier)
    # that breaks SBC by tying the likelihood's precision to the injected truth. Per cell,
    # with the SAME shape/cell_size/n_bar the inference uses, its own fold-in key.
    # Tag 2**31 keeps this stream disjoint from the per-trial fold-ins (i in 0..n_trials-1).
    k_var = jax.random.fold_in(key, 2**31)
    var_vs = tuple(
        estimate_log_count_variance_var(
            mach=_MACH_FID,
            b=b,
            alpha=_ALPHA_FID,
            beta=_BETA_FID,
            shape=shape,
            cell_size=c,
            n_bar=n_stars / (shape[0] // c) ** 3,
            n_real=_N_REAL_VAR_V,
            key=jax.random.fold_in(k_var, c),
        )
        for c in cell_sizes
    )

    # --- Fixed-fiducial band-power precision bp_precision for the 2-pt beta block. ---
    # SAME SBC-validity contract as var_v: it MUST be truth-independent. Computed ONCE here, at the
    # FIXED fiducial theta (_MACH_FID, _ALPHA_FID, _BETA_FID; fixed b), as the Hartlap-corrected
    # mock precision of an ensemble of measured band-powers -- NEVER at the trial truth theta*.
    # Tag 2**30 keeps this stream disjoint from the var_v (2**31) and per-trial (0..n_trials-1)
    # fold-ins. n_real (=_N_REAL_BP) > k (band-power bins) is required for an invertible Hartlap C.
    k_bp = jax.random.fold_in(key, 2**30)
    bp_rows = [
        measured_bandpowers(
            np.asarray(
                rank_copula_field(
                    gaussian_random_field(shape, _BETA_FID, jax.random.fold_in(k_bp, i)),
                    _MACH_FID,
                    b,
                    _ALPHA_FID,
                )
            ),
            shape,
            _K_EDGES,
        )
        for i in range(_N_REAL_BP)
    ]
    bp_precision = mock_precision(bp_rows)

    for i in range(n_trials):
        # Per-trial fold-in keys: prior draw / mock realization / NUTS, each its own stream.
        k_trial = jax.random.fold_in(key, i)
        k_prior, k_mock, k_nuts = jax.random.split(k_trial, 3)

        theta_star = prior.sample(k_prior)  # (M*, alpha*, beta*)
        thetas_true[i] = np.asarray(theta_star)

        data, s_thr, s_max = _build_mock(
            theta_star,
            b=b,
            s_thr_margin=s_thr_margin,
            shape=shape,
            density_shape=density_shape,
            cell_sizes=cell_sizes,
            n_stars=n_stars,
            key=k_mock,
            n_exc_bins=n_exc_bins,
        )
        # Thread the fixed-fiducial (truth-independent) var_v per cell into the data bundle.
        data["var_vs"] = var_vs

        logdensity = build_logdensity(
            prior,
            data,
            b=b,
            s_thr=s_thr,
            s_max=s_max,
            shape=shape,
            cell_sizes=cell_sizes,
            bp_precision=bp_precision,
            n_max=n_max,
            n_s=n_s,
        )

        # Disperse the init OFF the truth (Talts 2018 sec. 5.1): initializing exactly at
        # theta* can leave residual init-correlation that under-disperses ranks if warmup is
        # short. A modest jitter in unconstrained space breaks the truth-pinning while staying
        # well-conditioned; window adaptation then forgets the start.
        k_nuts, k_init = jax.random.split(k_nuts)
        z0 = to_unconstrained(theta_star) + 0.3 * jax.random.normal(k_init, (3,))
        draws = run_nuts(logdensity, z0, k_nuts, n_warmup, n_samples)  # (n_samples, 3) in z
        constrained = np.asarray(jax.vmap(to_constrained)(draws))  # (n_samples, 3) in theta

        # Thin to ~independent draws (Talts 2018 sec. 5.1).
        thinned = constrained[::n_thin]
        L = thinned.shape[0]
        n_draws = L

        # Rank theta* among the thinned draws, per free param (M, alpha, beta) in {0..L}.
        ranks[i] = np.sum(thinned < np.asarray(theta_star)[None, :], axis=0)

    return {
        "ranks": ranks,
        "n_draws": int(n_draws),
        "n_trials": int(n_trials),
        "param_names": ["M", "alpha", "beta"],
        "thetas_true": thetas_true,
    }

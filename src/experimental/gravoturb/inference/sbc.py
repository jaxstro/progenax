r"""Simulation-Based Calibration (SBC) driver for the gravoturb inference layer (Task 6).

Simulation-Based Calibration (Talts et al. 2018, *Validating Bayesian Inference Algorithms
with Simulation-Based Calibration*) checks that the inference machinery (prior + likelihood +
NUTS) is *self-consistent*: if for each trial we draw theta* from the prior, simulate a mock
from theta*, and run the posterior, then the rank of theta* among (thinned, ~independent)
posterior draws is **DiscreteUniform** over ``{0, ..., L}`` when the sampler is calibrated.
Deviations from uniformity diagnose miscalibration (a U/inverted-U = over/under-dispersion, a
slope = bias). The heavy uniformity test lives in AC18 / Task 7; this module is the *driver*
that produces the rank table.

This module holds the mock construction + the SBC loop. The shared, prior-aware unconstrained
log-density factory :func:`gravoturb.inference.model.build_logdensity` -- the single source of
truth for the model composition, used by BOTH this driver and AC16 -- lives in
:mod:`gravoturb.inference.model` (see that module's docstring for the composition, the 2-pt
band-power beta carrier, the tail-robust CIC block, and the no-POT-barrier rationale).

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

import jax
import numpy as np

from gravoturb.diagnostics.measure import (
    estimate_log_count_variance_var,
    measure_exceedances,
    measure_log_count_variance,
)
from gravoturb.inference.covariance import (
    measured_bandpowers,
    mock_precision,
)
from gravoturb.inference.hmc import (
    run_nuts,
    to_constrained,
    to_unconstrained,
)
from gravoturb.inference.model import K_EDGES, build_logdensity
from gravoturb.inference.priors import BM19Prior
from gravoturb.realization.copula import rank_copula_field
from gravoturb.realization.gaussian_field import gaussian_random_field
from gravoturb.realization.placement import sample_cic_counts
from gravoturb.theory.density_pdf import sigma_s_squared, transition_density

# Fixed fiducial theta for the truth-INDEPENDENT log-count-variance estimator variance
# (var_v). Used ONCE per inference in sbc_ranks to compute var_v at a fixed point -- NEVER at
# the trial truth theta* (a truth-keyed var_v would break SBC, like the old POT barrier did).
# The SAME fixed fiducial theta also keys the band-power precision bp_precision (below).
_MACH_FID = 8.0
_ALPHA_FID = 2.5
_BETA_FID = 3.0
_N_REAL_VAR_V = 12

_N_REAL_BP = 64  # ensemble size for the fixed-fiducial band-power mock precision

# Fixed turbulence driving parameter the fiducial band-power ensemble uses (matches the fixed b
# threaded through inference; b is not a free param here).
_B_FID = 0.4


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
    # stellar log-density field s_lo, on the SAME grid/K_EDGES the predicted band-powers FFT on.
    band_powers = measured_bandpowers(np.asarray(s_lo), shape, K_EDGES)

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
            K_EDGES,
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

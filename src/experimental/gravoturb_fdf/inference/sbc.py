r"""Simulation-Based Calibration (SBC) driver for the gravoturb_fdf inference layer (Task 6).

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

        logdensity(z) = tail_exceedance_loglike(...)          # POT tail -> alpha
                      + sum_c count_loglike(count_hist_c, ...) # stellar CIC -> mach, beta
                      + prior.logpdf([M, alpha, beta])         # proper BM19 prior
                      + log_jacobian(z)                        # reparam Jacobian

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
  resumability. Per trial it draws theta*, builds the BM19 mock (gas tail exceedances + stellar
  CIC count histograms) exactly as AC16 does, builds the prior-aware log-density, runs NUTS,
  thins the draws to ~independence (Talts 2018 sec. 5.1), and ranks theta* among them.

The mock construction (n_bar per cell, count-hist length, the rank-copula gas/stellar fields)
mirrors :func:`gravoturb_fdf.validation.acceptance.ac16_hmc_recovery` so the calibrated object
is the SAME inference machinery AC16 exercises. The numpy paths (``measure_exceedances``,
``np.bincount``, the per-trial scalar threshold) are the non-differentiable oracle side --
the differentiable interface is the log-density itself (Phase 5 design).
"""

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from gravoturb_fdf.field.field import gaussian_random_field, rank_copula_field
from gravoturb_fdf.field.sampling import sample_cic_counts
from gravoturb_fdf.inference.hmc import (
    log_jacobian,
    run_nuts,
    to_constrained,
    to_unconstrained,
)
from gravoturb_fdf.inference.likelihood import (
    count_loglike,
    tail_exceedance_loglike,
)
from gravoturb_fdf.inference.priors import BM19Prior
from gravoturb_fdf.theory.bm19 import sigma_s_squared, transition_density
from gravoturb_fdf.validation.measure import measure_exceedances


def build_logdensity(
    prior: BM19Prior,
    data: dict,
    b: float,
    s_thr: float,
    s_max: float,
    shape: tuple[int, int, int],
    cell_sizes: tuple[int, ...],
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
        :func:`measure_exceedances`; ``"count_hists"`` -- tuple of per-cell count histograms
        (each ``(nmax_c,)`` from ``np.bincount``); ``"n_bars"`` -- tuple of per-cell ``n_bar``
        (matched to ``cell_sizes``).
    b : float
        Fixed turbulence driving parameter (the likelihood constant in ``theta4``).
    s_thr, s_max : float
        POT threshold and realized field maximum (data-derived constants; same field as
        ``exc_counts``, which makes the POT block shift-immune).
    shape : (n, n, n)
        Stellar CIC grid the count model FFTs on (forward-bias-matched).
    cell_sizes : tuple[int, ...]
        CIC cell sizes; ``data["count_hists"]`` / ``data["n_bars"]`` align with this.
    n_max, n_s : int
        ``count_loglike`` quadrature controls (mirror AC16 defaults).

    Returns
    -------
    logdensity_fn : z (3,) -> scalar log-density (differentiable in z).
    """
    exc_counts = jnp.asarray(data["exc_counts"])
    exc_edges = jnp.asarray(data["exc_edges"])
    count_hists = tuple(jnp.asarray(h) for h in data["count_hists"])
    n_bars = tuple(float(nb) for nb in data["n_bars"])

    def logdensity(z):
        m_, a_, be_ = to_constrained(z)
        theta3 = jnp.array([m_, a_, be_])
        theta4 = jnp.array([m_, b, a_, be_])
        # POT tail block -> alpha. SHIFT-IMMUNE in s_thr (the lognormal norm
        # cancels), so a per-trial s_thr keyed to the truth does NOT bias alpha.
        ll = tail_exceedance_loglike(exc_counts, exc_edges, theta4, s_thr, s_max)
        # stellar CIC blocks -> mach, beta
        for c, h, nb in zip(cell_sizes, count_hists, n_bars):
            ll = ll + count_loglike(h, theta4, shape, c, nb, n_max=n_max, n_s=n_s)
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
    count_hists, n_bars = [], []
    for c in cell_sizes:
        nb = n_stars / (shape[0] // c) ** 3
        cnt = np.asarray(
            sample_cic_counts(s_lo, nb, c, jax.random.fold_in(key, 100 + c))
        ).ravel()
        nmaxN = int(nb * 8) + 30
        count_hists.append(np.bincount(cnt, minlength=nmaxN)[:nmaxN].astype(float))
        n_bars.append(nb)

    data = {
        "exc_counts": exc_counts,
        "exc_edges": exc_edges,
        "count_hists": tuple(count_hists),
        "n_bars": tuple(n_bars),
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

        logdensity = build_logdensity(
            prior,
            data,
            b=b,
            s_thr=s_thr,
            s_max=s_max,
            shape=shape,
            cell_sizes=cell_sizes,
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

r"""HMC convergence diagnostics for gravoturb_fdf inference (AC19, Task 3).

A thin **arviz** wrapper that turns the multi-chain output of
:func:`gravoturb_fdf.inference.hmc.run_nuts_diagnostic` into the standard per-fit
NUTS convergence diagnostics: split-R-hat, bulk-ESS, tail-ESS, divergence rate,
BFMI, and max-tree-depth saturation, plus an overall ``passed`` boolean.

This is the **numpy/arviz validation side**, NOT a JAX hot path -- numpy + arviz are
appropriate here (the released-core JAX-native rules govern the differentiable sampler in
``hmc.py``/``likelihood.py``, not this post-hoc check).

Thresholds (grounded in Vehtari, Gelman, Simpson, Carpenter & Burkner 2021,
"Rank-normalization, folding, and localization: An improved R-hat for assessing
convergence of MCMC", Bayesian Analysis 16(2):667-718, doi:10.1214/20-BA1221, and the
arviz convergence-diagnostics docs):

- **R-hat < 1.01** -- the recommended modern threshold; the classic 1.1 is too loose.
  arviz/Vehtari use the rank-normalized split-R-hat.
- **ESS > 400** -- the rule of thumb (>= ~100 effective draws per chain, x4 chains) below
  which the split-R-hat / quantile estimates are themselves unreliable. We require BOTH
  bulk-ESS (mixing in the bulk) and tail-ESS (mixing in the 5%/95% tails).
- **Divergence rate < 1%** and **max-tree-depth saturation < 1%** -- standard NUTS
  pathology gates (divergences => biased posterior near sharp curvature; tree-depth
  saturation => the sampler is hitting ``max_num_doublings`` and under-exploring).
- **BFMI** -- E-BFMI (Betancourt 2016) flags poor momentum resampling; values << 0.3 are
  problematic. We report the per-chain minimum (conservative) as a scalar summary.

This is the *per-fit* convergence check, complementary to (and orthogonal from) the
SBC rank-uniformity calibration check (AC18): convergence says "this one chain ensemble
mixed", calibration says "the posteriors are well-calibrated under the BM19 model".
"""

import warnings

import arviz as az
import numpy as np

# Tail-ESS evaluates effective sample size at the 5% and 95% quantiles, matching the
# Vehtari et al. (2021) tail-ESS convention. arviz 1.1.0 requires this probability pair
# to be passed explicitly when method="tail" (no default).
_TAIL_PROB = (0.05, 0.95)


def compute_hmc_diagnostics(
    positions,
    *,
    divergences=None,
    energy=None,
    tree_depth=None,
    max_tree_depth=None,
    param_names=None,
    r_hat_max=1.01,
    ess_min=400.0,
):
    r"""Compute HMC convergence diagnostics from multi-chain NUTS output.

    Parameters
    ----------
    positions : array_like
        Sampled positions, shape ``(n_chains, n_draws, n_params)`` or
        ``(n_chains, n_draws)`` for a single (1-D) parameter.
    divergences : array_like of bool, optional
        Per-step divergence flags, shape ``(n_chains, n_draws)``. If ``None`` the
        divergence gate is skipped and ``divergence_rate`` reported as 0.0.
    energy : array_like, optional
        Per-step Hamiltonian energy, shape ``(n_chains, n_draws)``, for E-BFMI. If
        ``None`` the BFMI gate is skipped and ``bfmi`` reported as ``nan``.
    tree_depth : array_like, optional
        Per-step NUTS tree depth (number of trajectory doublings), shape
        ``(n_chains, n_draws)``. Used with ``max_tree_depth`` for saturation.
    max_tree_depth : int, optional
        The configured NUTS ``max_num_doublings`` ceiling. Required (with
        ``tree_depth``) to compute saturation; otherwise saturation is 0.0.
    param_names : list of str, optional
        Names indexed by parameter; defaults to ``["theta_0", ...]``.
    r_hat_max : float, default 1.01
        R-hat pass threshold (Vehtari et al. 2021).
    ess_min : float, default 400.0
        Minimum bulk- AND tail-ESS pass threshold.

    Returns
    -------
    dict with keys:
        ``r_hat``                    (n_params,) float -- rank-normalized split-R-hat
        ``ess_bulk``                 (n_params,) float -- bulk effective sample size
        ``ess_tail``                 (n_params,) float -- tail (5%/95%) effective sample size
        ``divergence_rate``          float             -- mean of ``divergences`` (0.0 if None)
        ``bfmi``                     float             -- min per-chain E-BFMI (nan if no energy)
        ``max_tree_depth_saturation``float             -- frac of steps at the depth ceiling
        ``passed``                   bool              -- all active gates satisfied
        ``param_names``              list of str
    """
    positions = np.asarray(positions)
    if positions.ndim == 2:
        # (n_chains, n_draws) -> single parameter
        positions = positions[:, :, None]
    if positions.ndim != 3:
        raise ValueError(
            f"positions must have shape (n_chains, n_draws) or "
            f"(n_chains, n_draws, n_params); got {positions.shape}"
        )
    n_chains, n_draws, n_params = positions.shape

    if param_names is None:
        param_names = [f"theta_{p}" for p in range(n_params)]

    # Per-parameter R-hat and bulk/tail ESS. arviz accepts a (chain, draw) ndarray slice
    # directly (chain_axis=0, draw_axis=1 by default) -- no need for the deprecated
    # az.InferenceData constructor (migrated to xarray DataTree in arviz 1.x).
    r_hat = np.empty(n_params)
    ess_bulk = np.empty(n_params)
    ess_tail = np.empty(n_params)
    # Convergence stats can warn on degenerate (e.g. exactly-constant) chains; those are
    # exactly the cases we WANT to flag via R-hat, so silence the cosmetic warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in range(n_params):
            chain_draw = positions[:, :, p]
            r_hat[p] = float(az.rhat(chain_draw))
            ess_bulk[p] = float(az.ess(chain_draw, method="bulk"))
            ess_tail[p] = float(az.ess(chain_draw, method="tail", prob=_TAIL_PROB))

    # Divergence rate (skip gate gracefully if not provided).
    if divergences is not None:
        divergence_rate = float(np.mean(np.asarray(divergences)))
        has_div = True
    else:
        divergence_rate = 0.0
        has_div = False

    # E-BFMI: az.bfmi returns one value per chain; report the minimum (most conservative
    # -- one bad chain is enough to signal poor momentum resampling).
    if energy is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bfmi_per_chain = np.asarray(az.bfmi(np.asarray(energy)))
        bfmi = float(np.min(bfmi_per_chain))
        has_bfmi = True
    else:
        bfmi = float("nan")
        has_bfmi = False

    # Max-tree-depth saturation: fraction of steps that hit the doubling ceiling.
    if tree_depth is not None and max_tree_depth is not None:
        saturation = float(np.mean(np.asarray(tree_depth) >= max_tree_depth))
        has_depth = True
    else:
        saturation = 0.0
        has_depth = False

    # Note: a *provided* but degenerate `energy` (e.g. constant) makes az.bfmi -> nan, and
    # `nan > 0.3` is False, so the run intentionally FAILS the gate (degenerate energy is a
    # broken-sampler signal). An *absent* energy (has_bfmi False) skips the gate entirely.
    passed = bool(
        np.all(r_hat < r_hat_max)
        and np.all(ess_bulk > ess_min)
        and np.all(ess_tail > ess_min)
        and (not has_div or divergence_rate < 0.01)
        and (not has_depth or saturation < 0.01)
        and (not has_bfmi or bfmi > 0.3)
    )

    return {
        "r_hat": r_hat,
        "ess_bulk": ess_bulk,
        "ess_tail": ess_tail,
        "divergence_rate": divergence_rate,
        "bfmi": bfmi,
        "max_tree_depth_saturation": saturation,
        "passed": passed,
        "param_names": list(param_names),
    }

"""f_sub → Q calibration driver (spec §3.8 / §8 steps 4-6).

Realize FDF fields across cloud parameters and substructure fractions, sample N⋆ stars,
and measure the CW04 Q with the AC5-validated estimator (2D-projected). The headline is
the monotone-decreasing Q(f_sub) relation with realization-scatter bands — a *forward*
calibration of how dense-tail star fraction maps to projected substructure, reported with
its scatter (not an inversion of Q to recover FBM parameters, which Lomax+2018 show is
ill-posed).

numpy/scipy are permitted here (validation/analysis side); the CW04 Q estimator is
non-differentiable. The differentiable interface is the fitted surrogate (see P3.3).
"""

import jax
import numpy as np

from gravoturb_fdf.diagnostics.q import compute_q_parameter
from gravoturb_fdf.field.pipeline import build_fdf_field, cloud_to_stars
from gravoturb_fdf.surrogate import surrogate_features
from gravoturb_fdf.theory.bm19 import sigma_s_squared


def measure_q_ensemble(
    mach: float,
    b: float,
    alpha: float,
    beta: float,
    f_sub: float,
    n_stars: int,
    n_real: int,
    shape: tuple[int, int, int],
    key: jax.Array,
) -> np.ndarray:
    """Q over ``n_real`` independent FDF realizations at one (cloud params, f_sub).

    Each realization gets a fresh field key and star-sampling key; Q is measured on the
    2D-projected positions by the CW04 estimator. Returns a length-``n_real`` array.
    """
    q = np.empty(n_real)
    for i in range(n_real):
        k_field, k_stars = jax.random.split(jax.random.fold_in(key, i))
        fld = build_fdf_field(mach, b, alpha, beta, shape, k_field)
        pos = cloud_to_stars(fld, f_sub, n_stars, k_stars)
        q[i] = compute_q_parameter(np.asarray(pos))
    return q


def q_vs_fsub(
    mach: float,
    b: float,
    alpha: float,
    beta: float,
    f_sub_values: tuple[float, ...],
    n_stars: int,
    n_real: int,
    shape: tuple[int, int, int],
    key: jax.Array,
) -> dict:
    """Sweep f_sub on a PAIRED set of field realizations; return mean ± std of Q.

    Each realization builds **one** FDF field and samples stars at every f_sub from that
    same field, so field-to-field variance is differenced out and the f_sub effect is
    isolated (the cross-f_sub trend is paired). Returns
    ``{"f_sub", "q_mean", "q_std", "q_all"}`` with ``q_all`` shaped ``(n_real, n_fsub)``.
    """
    f_sub_values = np.asarray(f_sub_values, dtype=float)
    q_all = np.empty((n_real, f_sub_values.size))
    for i in range(n_real):
        k_field, k_stars = jax.random.split(jax.random.fold_in(key, i))
        fld = build_fdf_field(mach, b, alpha, beta, shape, k_field)
        for j, f_sub in enumerate(f_sub_values):
            pos = cloud_to_stars(fld, float(f_sub), n_stars, jax.random.fold_in(k_stars, j))
            q_all[i, j] = compute_q_parameter(np.asarray(pos))
    return {
        "f_sub": f_sub_values,
        "q_mean": q_all.mean(axis=0),
        "q_std": q_all.std(axis=0),
        "q_all": q_all,
    }


def fit_q_surrogate(
    f_sub: np.ndarray,
    sigma_s: np.ndarray,
    beta: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """Least-squares fit of the Q(f_sub; σ_s, β) surrogate coefficients.

    Builds the shared feature design matrix (numpy) and solves ``M c ≈ q``. Returns the
    length-7 coefficient vector consumed by ``gravoturb_fdf.surrogate.q_surrogate``.
    """
    M = np.asarray(surrogate_features(np.asarray(f_sub), np.asarray(sigma_s), np.asarray(beta), np))
    coeffs, *_ = np.linalg.lstsq(M, np.asarray(q), rcond=None)
    return coeffs


def q_calibration_grid(
    param_sets,
    f_sub_values: tuple[float, ...],
    n_stars: int,
    n_real: int,
    shape: tuple[int, int, int],
    key: jax.Array,
) -> dict:
    """Q(f_sub) over a grid of cloud-parameter sets (the production calibration).

    ``param_sets`` is a sequence of ``(mach, b, alpha, beta)``. For each set the paired
    ``q_vs_fsub`` sweep is run; results are stacked into Q surfaces over (param, f_sub),
    with the matching σ_s = √(σ_s²(ℳ,b)) and β recorded for surrogate fitting.

    Returns ``{"sigma_s", "beta", "f_sub", "q_mean", "q_std"}`` with q surfaces shaped
    ``(n_param, n_fsub)``.
    """
    f_arr = np.asarray(f_sub_values, dtype=float)
    n_p = len(param_sets)
    q_mean = np.empty((n_p, f_arr.size))
    q_std = np.empty((n_p, f_arr.size))
    sigma_s = np.empty(n_p)
    beta_arr = np.empty(n_p)
    for p, (mach, b, alpha, beta) in enumerate(param_sets):
        res = q_vs_fsub(
            mach, b, alpha, beta, f_sub_values, n_stars, n_real, shape,
            jax.random.fold_in(key, p),
        )
        q_mean[p] = res["q_mean"]
        q_std[p] = res["q_std"]
        sigma_s[p] = float(np.sqrt(sigma_s_squared(mach, b)))
        beta_arr[p] = beta
    return {
        "sigma_s": sigma_s,
        "beta": beta_arr,
        "f_sub": f_arr,
        "q_mean": q_mean,
        "q_std": q_std,
    }

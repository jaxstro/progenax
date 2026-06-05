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

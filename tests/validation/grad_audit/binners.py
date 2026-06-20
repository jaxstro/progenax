r"""Frozen-edge binned summary statistics for the binned-kinematic Fisher audit.

Provenance
----------
Ported VERBATIM from ``scripts/_demo_inference.py`` @ commit 2090dcd (the
scripts-local physics-direct inference helper used by the Batch-B science demos).
The five binners — ``_bin_index``, ``_grouped_bin_sums``, ``binned_sigma1d``,
``binned_sigma_beta`` (+ ``SigmaBetaResult``), and ``binned_number_density`` — are
copied here, unchanged, so the grad-audit has a STABLE frozen-edge binner that is
independent of the demo script (``scripts/`` is not a package; importing from it is
not supported). Keep this file in sync with the demo by re-vendoring if the demo's
estimators change.

These are the data-side summary statistics whose Jacobian J = d(summary)/d(theta)
the Fisher matrix F = JᵀJ is built from. The radial bin edges ``r_edges`` are STATIC
(the data-side bin geometry is FROZEN — this is correct and in-scope-as-frozen: the
observer fixes the bins, then gradients flow through the model that fills them). For
the kinematic channels σ_1d(r) and β(r) the BIN VALUES move smoothly as the params
scale the velocity magnitudes, so d(σ_k)/dθ and d(β_k)/dθ are genuinely live; for
``binned_number_density`` the COUNTS are the frozen data of a Poisson likelihood and
"gradients flow through the model only" (see that function's docstring).

JAX-native throughout (``jax.numpy``, ``jax.lax``); every function is jit/grad-safe.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


# --------------------------------------------------------------------------- #
# Binned summary statistics
# --------------------------------------------------------------------------- #
def _bin_index(radii, r_edges):
    """Radial bin index in [0, K) for each radius, or -1 if outside [r0, rK].

    ``jnp.searchsorted`` with ``side='right'`` gives 1..K for in-range radii; we
    shift to 0..K-1 and mark out-of-range radii (index 0 or K+1 after shift) -1.
    """
    k = jnp.searchsorted(r_edges, radii, side="right") - 1
    n_bins = r_edges.shape[0] - 1
    in_range = (radii >= r_edges[0]) & (radii <= r_edges[-1])
    return jnp.where(in_range, jnp.clip(k, 0, n_bins - 1), -1)


def _grouped_bin_sums(values, group_ids, bin_ids, n_groups, n_bins):
    """Sum ``values`` and count members into a (n_groups, n_bins) grid.

    Valid samples have ``bin_ids >= 0``; invalid ones are routed to a throwaway
    overflow cell so no Python control flow is needed (fully vmappable)."""
    valid = bin_ids >= 0
    g = jnp.where(valid, group_ids, n_groups)  # overflow row
    b = jnp.where(valid, bin_ids, 0)
    flat = g * n_bins + b
    size = (n_groups + 1) * n_bins
    sums = jnp.zeros(size).at[flat].add(jnp.where(valid, values, 0.0))
    counts = jnp.zeros(size).at[flat].add(jnp.where(valid, 1.0, 0.0))
    sums = sums[: n_groups * n_bins].reshape(n_groups, n_bins)
    counts = counts[: n_groups * n_bins].reshape(n_groups, n_bins)
    return sums, counts


def binned_sigma1d(pos, vel, group_ids, n_groups, r_edges, n_min=30):
    r"""Per-group binned 1-D velocity dispersion sigma_1d(r).

    Isotropic estimator ``sigma_1d^2 = <|v|^2> / 3`` within each (group j,
    radial bin k). Standard error of the dispersion estimate is
    ``se = sig_hat / sqrt(6 n)``. The estimator pools all 3 velocity
    components, so the inner sum runs over ``3n`` iid one-D squared normals:
    ``sum |v|^2 / sigma^2 ~ chi^2(3n)``, giving ``Var(sigma_hat^2) = 2 sigma^4
    / (3n)`` and (delta method) ``Var(sigma_hat) ~ sigma^2 / (6n)``. The
    effective sample size is ``3n``, not ``n`` -- hence ``sqrt(6 n)`` rather
    than the single-component ``sqrt(2 n)``.

    Parameters
    ----------
    pos, vel : (N, 3) arrays
    group_ids : (N,) int in [0, n_groups)
    r_edges : (K+1,) monotone radial bin edges
    n_min : bins with fewer than ``n_min`` members get weight 0 (not NaN).

    Returns
    -------
    sig_hat, se, weight, n : each (n_groups, K). ``weight`` is 1.0 for populated
    bins (n >= n_min) and 0.0 otherwise; outputs are NaN-free everywhere.
    """
    n_bins = r_edges.shape[0] - 1
    radii = jnp.linalg.norm(pos, axis=1)
    speed_sq = jnp.sum(vel * vel, axis=1)
    bin_ids = _bin_index(radii, r_edges)

    sum_v2, counts = _grouped_bin_sums(speed_sq, group_ids, bin_ids, n_groups, n_bins)
    populated = counts >= n_min
    safe_counts = jnp.where(counts > 0, counts, 1.0)  # avoid /0 in empty bins

    sigma_sq = jnp.where(populated, sum_v2 / safe_counts / 3.0, 0.0)
    sig_hat = jnp.sqrt(sigma_sq)
    se = jnp.where(populated, sig_hat / jnp.sqrt(6.0 * safe_counts), 0.0)
    weight = jnp.where(populated, 1.0, 0.0)
    return sig_hat, se, weight, counts


class SigmaBetaResult(NamedTuple):
    """Return of :func:`binned_sigma_beta`; all arrays shaped (C, K)."""

    sig_hat: jax.Array  # 1-D dispersion sqrt((sigma_r^2 + sigma_t^2)/3)
    se: jax.Array  # SE of sig_hat = sig_hat / sqrt(6 n) (3 pooled components)
    beta_hat: jax.Array  # Binney anisotropy 1 - sigma_t^2 / (2 sigma_r^2)
    weight: jax.Array  # 1.0 if populated (n >= n_min) else 0.0
    n: jax.Array  # member counts


def binned_sigma_beta(pos, vel, r_edges, component_id=None, n_min=50):
    r"""sigma_1d(r) and Binney anisotropy beta(r) per component.

    Radial/tangential split: ``v_r = v . r_hat``; ``v_t^2 = |v|^2 - v_r^2``
    (two tangential dof). With ``sigma_r^2 = <v_r^2>`` and ``sigma_t^2 = <v_t^2>``
    (summing BOTH tangential components),
    ``beta = 1 - sigma_t^2 / (2 sigma_r^2)`` and
    ``sig_hat = sqrt((sigma_r^2 + sigma_t^2) / 3)``.

    The standard error of ``sig_hat`` is ``se = sig_hat / sqrt(6 n)``: all 3
    velocity components are pooled into ``sig_hat^2``, so the inner sum is over
    ``3n`` iid one-D squared normals (``chi^2(3n)``), giving effective sample
    size ``3n`` (delta method ``Var(sig_hat) ~ sigma^2 / (6n)``), not ``n``.

    ``component_id=None`` treats all stars as a single component (C = 1).
    Otherwise ``component_id`` must be a CONCRETE (static) array: ``n_comp`` is
    read host-side via ``int(component_id.max()) + 1``, so this function is NOT
    traceable over a traced ``component_id`` (unlike :func:`binned_sigma1d`,
    which takes ``n_groups`` explicitly and stays fully traceable).

    Returns a :class:`SigmaBetaResult` namedtuple (``sig_hat, se, beta_hat,
    weight, n``), each (C, K), NaN-free.
    """
    n = pos.shape[0]
    if component_id is None:
        component_id = jnp.zeros((n,), dtype=jnp.int32)
        n_comp = 1
    else:
        n_comp = int(component_id.max()) + 1

    n_bins = r_edges.shape[0] - 1
    radii = jnp.linalg.norm(pos, axis=1)
    r_hat = pos / jnp.maximum(radii, 1e-30)[:, None]
    v_r = jnp.sum(vel * r_hat, axis=1)
    v_r_sq = v_r * v_r
    v_t_sq = jnp.sum(vel * vel, axis=1) - v_r_sq

    bin_ids = _bin_index(radii, r_edges)
    sum_vr2, counts = _grouped_bin_sums(v_r_sq, component_id, bin_ids, n_comp, n_bins)
    sum_vt2, _ = _grouped_bin_sums(v_t_sq, component_id, bin_ids, n_comp, n_bins)

    populated = counts >= n_min
    safe_counts = jnp.where(counts > 0, counts, 1.0)
    sigma_r_sq = jnp.where(populated, sum_vr2 / safe_counts, 0.0)
    sigma_t_sq = jnp.where(populated, sum_vt2 / safe_counts, 0.0)

    sig_hat = jnp.sqrt(jnp.where(populated, (sigma_r_sq + sigma_t_sq) / 3.0, 0.0))
    safe_sigma_r_sq = jnp.where(sigma_r_sq > 0, sigma_r_sq, 1.0)
    beta_hat = jnp.where(populated, 1.0 - sigma_t_sq / (2.0 * safe_sigma_r_sq), 0.0)
    se = jnp.where(populated, sig_hat / jnp.sqrt(6.0 * safe_counts), 0.0)
    weight = jnp.where(populated, 1.0, 0.0)
    return SigmaBetaResult(sig_hat, se, beta_hat, weight, counts)


def binned_number_density(pos, r_edges):
    r"""Frozen per-shell star counts N_k for a radial number-density profile fit.

    Returns the observed count in each of the ``K`` radial bins defined by
    ``r_edges`` (K+1 monotone edges). Radii outside ``[r_edges[0], r_edges[-1]]``
    are excluded. The counts are the FROZEN data of a Poisson profile likelihood
    (:func:`poisson_loglike`): the model supplies the expected counts
    ``mu_k(theta) = N_total * p_k(theta)`` (``p_k`` = the profile's enclosed-number
    fraction in shell ``k``), and gradients flow through the model only.

    A Poisson model (not a Gaussian on the counts or their logs) is used so that
    low-occupancy OUTER bins -- the regime that constrains a tidal / truncation
    radius -- carry honest counting errors (``Var = mu``) rather than an
    ill-defined log-count Gaussian that diverges as the count approaches zero.

    Returns
    -------
    counts : (K,) float array, NaN-free and traceable.
    """
    radii = jnp.linalg.norm(pos, axis=1)
    bin_ids = _bin_index(radii, r_edges)  # -1 if outside [r0, rK]
    n_bins = r_edges.shape[0] - 1
    valid = bin_ids >= 0
    b = jnp.where(valid, bin_ids, 0)
    return jnp.zeros(n_bins).at[b].add(jnp.where(valid, 1.0, 0.0))

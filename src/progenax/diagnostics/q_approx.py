"""
JAX-native approximate Q parameter computation.

Provides differentiable, JIT-compatible approximations of the Cartwright & Whitworth
(2004) Q parameter using k-nearest neighbor distances instead of exact MST.

Two implementations:
    q_approx_naive: O(N^2) brute-force for small N (< 2000)
    q_approx_fast: O(N log N) with Morton-based spatial indexing for large N

The approximation:
    Q_exact = m_bar / s_bar
    Q_approx ~ m_bar_knn / s_bar
    where m_bar_knn = (N-1) * mean(d_1NN) / sqrt(N * A)

References:
    Cartwright & Whitworth (2004), MNRAS 348, 589
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = [
    "q_approx_naive",
    "q_approx_fast",
    "q_approx",
    "calibrate_q_approx",
    "DEFAULT_CALIBRATION",
]

# Default calibration factor (determined empirically in Task 5)
DEFAULT_CALIBRATION: float = 1.0


def q_approx_naive(
    positions: Float[Array, "N 3"],
    project_to_2d: bool = True,
    calibration: float = DEFAULT_CALIBRATION,
) -> Float[Array, ""]:
    """
    Compute approximate Q parameter using O(N^2) brute-force kNN.

    Suitable for N < 2000 where O(N^2) is acceptable.

    Args:
        positions: Particle positions [N, 3] or [N, 2]
        project_to_2d: Project to xy plane (CW04 methodology)
        calibration: Multiplicative calibration factor

    Returns:
        Q_approx: Approximate Q parameter (scalar)
    """
    # Handle 2D input
    if positions.shape[1] == 2:
        xy = positions
    elif positions.shape[1] == 3:
        xy = positions[:, :2] if project_to_2d else positions
    else:
        raise ValueError(f"positions must be (N, 2) or (N, 3), got {positions.shape}")

    N = xy.shape[0]

    # Degenerate case
    def degenerate_case(_):
        return jnp.array(0.79)

    def normal_case(_):
        # 1. Compute all pairwise distances [N, N]
        diff = xy[:, None, :] - xy[None, :, :]  # [N, N, D]
        dist_sq = jnp.sum(diff ** 2, axis=-1)  # [N, N]
        dist = jnp.sqrt(dist_sq + 1e-12)

        # 2. Find 1-NN distance (exclude self with large value on diagonal)
        dist_no_self = dist + jnp.eye(N) * 1e10
        nn_dist = jnp.min(dist_no_self, axis=1)  # [N]

        # 3. Approximate MST length: L_MST ~ (N-1) * mean(d_1NN)
        L_approx = (N - 1) * jnp.mean(nn_dist)

        # 4. Cluster geometry
        center = jnp.mean(xy, axis=0)
        radii = jnp.sqrt(jnp.sum((xy - center) ** 2, axis=1))
        R_cluster = jnp.maximum(jnp.max(radii), 1e-10)

        # Approximate area (bounding circle)
        A_approx = jnp.pi * R_cluster ** 2

        # 5. m_bar approximation
        m_bar = L_approx / jnp.sqrt(N * A_approx)

        # 6. s_bar: mean pairwise separation / R_cluster
        triu_mask = jnp.triu(jnp.ones((N, N), dtype=bool), k=1)
        n_pairs = N * (N - 1) // 2
        s_raw = jnp.sum(dist * triu_mask) / n_pairs
        s_bar = s_raw / R_cluster

        # 7. Q = calibration * m_bar / s_bar
        return calibration * m_bar / (s_bar + 1e-10)

    return jax.lax.cond(N < 3, degenerate_case, normal_case, operand=None)


def q_approx_fast(
    positions: Float[Array, "N 3"],
    project_to_2d: bool = True,
    nbins_per_dim: int = 16,
    calibration: float = DEFAULT_CALIBRATION,
) -> Float[Array, ""]:
    """
    Compute approximate Q parameter using O(N log N) spatial indexing.

    Uses Morton-based spatial binning from jaxstro.spatial for efficient
    k-nearest neighbor computation.

    Args:
        positions: Particle positions [N, 3] or [N, 2]
        project_to_2d: Project to xy plane (CW04 methodology)
        nbins_per_dim: Spatial bins per dimension (16-32 recommended)
        calibration: Multiplicative calibration factor

    Returns:
        Q_approx: Approximate Q parameter (scalar)
    """
    from jaxstro.spatial import assign_particles_to_bins, fill_bins, approx_knn_candidates

    # Handle 2D/3D input
    if positions.shape[1] == 2:
        xy = positions
        pos_3d = jnp.concatenate([positions, jnp.zeros((positions.shape[0], 1))], axis=1)
    elif positions.shape[1] == 3:
        if project_to_2d:
            xy = positions[:, :2]
            pos_3d = jnp.concatenate([xy, jnp.zeros((positions.shape[0], 1))], axis=1)
        else:
            xy = positions
            pos_3d = positions
    else:
        raise ValueError(f"positions must be (N, 2) or (N, 3)")

    N = xy.shape[0]

    # Pre-compute static parameters outside of cond
    Nbins = nbins_per_dim ** 3
    # Use fixed Bcap (maximum expected particles per bin)
    # For N=5000, nbins=32: 5000/(32^3/8) = ~1.2, so 128 is safe
    Bcap = max(128, int(N // (Nbins // 8) + 1))

    def degenerate_case(_):
        return jnp.array(0.79)

    def normal_case(_):
        # 1. Build spatial grid
        pos_min = pos_3d.min(axis=0)
        pos_max = pos_3d.max(axis=0)
        L_box = (pos_max - pos_min).max() * 1.01
        center = (pos_min + pos_max) / 2

        bin_of = assign_particles_to_bins(
            pos_3d, L_box=L_box, Nbins_per_dim=nbins_per_dim,
            box_center=center
        )

        particle_ids = jnp.arange(N, dtype=jnp.int32)
        bin_members, bin_mask = fill_bins(particle_ids, bin_of, Nbins=Nbins, Bcap=Bcap)

        # 2. Add sentinel position
        pos_with_sentinel = jnp.concatenate([pos_3d, jnp.zeros((1, 3))], axis=0)

        # 3. Get kNN candidates
        cand_idx, cand_mask = approx_knn_candidates(
            pos=pos_with_sentinel,
            bin_members=bin_members,
            bin_mask=bin_mask,
            bin_of=bin_of,
            Nbins_per_dim=nbins_per_dim,
            K_target=6,
        )

        # 4. Compute distances to candidates (2D)
        xy_sentinel = jnp.concatenate([xy, jnp.zeros((1, 2))], axis=0)
        cand_pos = xy_sentinel[cand_idx]  # [N, Cand_max, 2]
        diff = xy[:, None, :] - cand_pos
        cand_dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-12)
        cand_dist = jnp.where(cand_mask, cand_dist, jnp.inf)

        # 5. Find 1-NN distance
        nn_dist = jnp.min(cand_dist, axis=1)

        # 6. Approximate MST length
        L_approx = (N - 1) * jnp.mean(nn_dist)

        # 7. Cluster geometry
        center_2d = jnp.mean(xy, axis=0)
        radii = jnp.sqrt(jnp.sum((xy - center_2d) ** 2, axis=1))
        R_cluster = jnp.maximum(jnp.max(radii), 1e-10)
        A_approx = jnp.pi * R_cluster ** 2

        # 8. m_bar
        m_bar = L_approx / jnp.sqrt(N * A_approx)

        # 9. s_bar (subsampled for large N)
        s_bar = _compute_s_bar_subsampled(xy, R_cluster, max_pairs=10000)

        return calibration * m_bar / (s_bar + 1e-10)

    return jax.lax.cond(N < 3, degenerate_case, normal_case, operand=None)


def _compute_s_bar_subsampled(
    xy: Float[Array, "N 2"],
    R_cluster: float,
    max_pairs: int = 10000,
) -> Float[Array, ""]:
    """Compute s_bar with deterministic subsampling for large N."""
    N = xy.shape[0]
    n_pairs = N * (N - 1) // 2

    def exact():
        diff = xy[:, None, :] - xy[None, :, :]
        dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-12)
        triu_mask = jnp.triu(jnp.ones((N, N), dtype=bool), k=1)
        return jnp.sum(dist * triu_mask) / n_pairs / R_cluster

    def subsampled():
        n_sample = max_pairs
        idx_i = jnp.arange(n_sample) % N
        idx_j = (jnp.arange(n_sample) * 7919 + 1) % N
        idx_j = jnp.where(idx_i == idx_j, (idx_j + 1) % N, idx_j)
        diffs = xy[idx_i] - xy[idx_j]
        return jnp.mean(jnp.sqrt(jnp.sum(diffs ** 2, axis=-1) + 1e-12)) / R_cluster

    return jax.lax.cond(n_pairs <= max_pairs, lambda _: exact(), lambda _: subsampled(), None)


def q_approx(
    positions: Float[Array, "N 3"],
    project_to_2d: bool = True,
    method: Literal["auto", "naive", "fast"] = "auto",
    calibration: float = DEFAULT_CALIBRATION,
    **kwargs,
) -> Float[Array, ""]:
    """Unified interface. Implementation in Task 4."""
    raise NotImplementedError("Task 4")


def calibrate_q_approx(
    n_samples: int = 100,
    N_stars: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    """Calibration function. Implementation in Task 5."""
    raise NotImplementedError("Task 5")

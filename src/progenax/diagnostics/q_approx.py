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
    """O(N log N) Q approximation. Implementation in Task 3."""
    raise NotImplementedError("Task 3")


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

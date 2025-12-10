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
    """O(N^2) Q approximation. Implementation in Task 2."""
    raise NotImplementedError("Task 2")


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

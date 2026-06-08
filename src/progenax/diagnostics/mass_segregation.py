# progenax/src/progenax/diagnostics/mass_segregation.py
"""
Mass segregation diagnostics using Minimum Spanning Tree.

Implements the Λ_MSR metric from Allison et al. (2009) for quantifying
mass segregation in star clusters.

This module uses NumPy and SciPy (not JAX) and is intended for validation
and visualization, not gradient-based inference.

References:
    Allison et al. (2009), ApJ 700, L99
    Allison et al. (2009), MNRAS 395, 1449
"""

from typing import Tuple

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform


def compute_lambda_msr(
    positions: np.ndarray,
    masses: np.ndarray,
    N_massive: int = 10,
    N_random_samples: int = 50,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute Λ_MSR mass segregation ratio (Allison et al. 2009).

    Uses Minimum Spanning Tree lengths to compare the spatial distribution
    of massive stars vs. random subsets:

        Λ_MSR = <L_random> / L_massive ± σ_random / L_massive

    Interpretation:
        - Λ_MSR ≈ 1: No mass segregation
        - Λ_MSR > 1: Massive stars more concentrated (segregated)
        - Λ_MSR >> 1 (e.g., 3-5): Strong segregation
        - Λ_MSR < 1: Inverse segregation (rare)

    Args:
        positions: Stellar positions (N, 3) as NumPy array
        masses: Stellar masses (N,) as NumPy array
        N_massive: Number of most massive stars to use for comparison.
                   Typical values: 10-20 for clusters with N~1000.
        N_random_samples: Number of random subsets for comparison.
                          Default 50 is for quick validation; use >= 200
                          for science-quality results.
        seed: Random seed for reproducibility

    Returns:
        lambda_msr: Mass segregation ratio
        error: Standard error estimate (σ_random / L_massive)

    Raises:
        ValueError: If N_massive < 2 or N_massive >= N

    Example:
        >>> import numpy as np
        >>> positions = np.random.randn(1000, 3)
        >>> masses = np.random.power(2.3, 1000)
        >>> lam, err = compute_lambda_msr(positions, masses, N_massive=10)
        >>> print(f"Λ_MSR = {lam:.2f} ± {err:.2f}")

    Notes:
        Uses scipy.sparse.csgraph.minimum_spanning_tree for MST computation.
        Not differentiable; for validation/calibration only.

        Caution: Strongly affected by binaries (massive binaries have very
        short MST edges). For systems with binaries, consider using only
        binary center-of-mass positions.

    References:
        Allison et al. (2009), MNRAS 395, 1449 — formal Λ_MSR definition.
        Allison et al. (2009), ApJ 700, L99 — application (note: L99 Eq. 1 is the
            Spitzer t_seg relation, NOT Λ_MSR; verified against the held PDF 2026-06-08).
    """
    rng = np.random.default_rng(seed)
    N = len(masses)

    # Validate inputs
    if N_massive < 2:
        raise ValueError(f"N_massive must be >= 2, got {N_massive}")
    if N_massive >= N:
        raise ValueError(f"N_massive ({N_massive}) must be < N ({N})")

    # MST of N_massive most massive stars
    massive_indices = np.argsort(-masses)[:N_massive]
    massive_positions = positions[massive_indices]
    l_massive = _compute_mst_length(massive_positions)

    # Handle edge case: zero length (shouldn't happen with real data)
    if l_massive < 1e-10:
        return 1.0, 0.0

    # MSTs of random subsets
    l_random = []
    for _ in range(N_random_samples):
        random_indices = rng.choice(N, size=N_massive, replace=False)
        random_positions = positions[random_indices]
        l_random.append(_compute_mst_length(random_positions))

    l_random = np.array(l_random)
    lambda_msr = np.mean(l_random) / l_massive
    error = np.std(l_random) / l_massive

    return float(lambda_msr), float(error)


def _compute_mst_length(positions: np.ndarray) -> float:
    """
    Compute Minimum Spanning Tree length using SciPy.

    Args:
        positions: Particle positions (N, 3)

    Returns:
        Total MST edge length
    """
    if len(positions) < 2:
        return 0.0

    # Compute pairwise distance matrix
    dist_matrix = squareform(pdist(positions))

    # Compute MST
    mst = minimum_spanning_tree(dist_matrix)

    # Sum of MST edge weights
    return float(mst.sum())


__all__ = [
    "compute_lambda_msr",
]

# progenax/src/progenax/diagnostics/substructure.py
"""
Spatial substructure diagnostics for star clusters.

Implements metrics for quantifying fractal/clumpy structure:
- Q parameter (Cartwright & Whitworth 2004)
- Azimuthal density variation (Küpper et al. 2011)

These are NOT the virial ratio Q_vir. The Cartwright-Whitworth Q measures
spatial substructure (clumpiness), distinct from the energy balance Q_vir.

This module uses NumPy and SciPy (not JAX) and is intended for validation
and visualization, not gradient-based inference.

References:
    Cartwright & Whitworth (2004), MNRAS 348, 589
    Küpper et al. (2011), MNRAS 417, 2300
"""

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform


def compute_q_parameter(positions: np.ndarray) -> float:
    """
    Compute Cartwright & Whitworth Q parameter for spatial substructure.

    Q = mean_mst_edge / mean_separation

    This is the substructure metric, NOT the virial ratio Q_vir.

    Interpretation:
        - Q < 0.8: Substructured (fractal, clumpy)
        - Q ≈ 0.8: Homogeneous sphere
        - Q > 0.8: Centrally concentrated (radial profile dominates)

    Args:
        positions: Stellar positions (N, 3) as NumPy array

    Returns:
        Q: Cartwright-Whitworth Q parameter

    Example:
        >>> import numpy as np
        >>> # Random uniform sphere
        >>> r = np.random.uniform(0, 1, 1000)**(1/3)
        >>> theta = np.arccos(2*np.random.uniform(0, 1, 1000) - 1)
        >>> phi = np.random.uniform(0, 2*np.pi, 1000)
        >>> positions = np.column_stack([
        ...     r * np.sin(theta) * np.cos(phi),
        ...     r * np.sin(theta) * np.sin(phi),
        ...     r * np.cos(theta),
        ... ])
        >>> Q = compute_q_parameter(positions)
        >>> print(f"Q = {Q:.3f}")  # Should be ~0.8 for uniform sphere

    Notes:
        O(N²) complexity for full cluster due to pairwise distance computation.
        For large N (> 5000), consider using a random subsample.

        Not differentiable; for validation/calibration only.

    References:
        Cartwright & Whitworth (2004), MNRAS 348, 589
    """
    N = len(positions)

    if N < 3:
        return 0.8  # Default for degenerate case

    # Compute pairwise distances
    dist_matrix = squareform(pdist(positions))

    # Mean MST edge length
    mst = minimum_spanning_tree(dist_matrix)
    mst_length = mst.sum()
    mean_m = mst_length / (N - 1)

    # Mean separation (all pairs)
    # Upper triangle of distance matrix (excluding diagonal)
    upper_indices = np.triu_indices(N, k=1)
    all_distances = dist_matrix[upper_indices]
    mean_s = np.mean(all_distances)

    # Q = mean_mst_edge / mean_separation
    Q = mean_m / mean_s if mean_s > 0 else 0.8

    return float(Q)


def compute_azimuthal_variation(
    positions: np.ndarray,
    n_bins: int = 12,
) -> float:
    """
    Compute azimuthal density variation σ_Σ / <Σ>.

    Divides the projected cluster into azimuthal bins and computes the
    relative standard deviation of star counts. This metric correlates
    linearly with fractal dimension:

        σ_Σ/<Σ> ≈ -0.46 * D + 1.45

    Where D is the fractal dimension (1.5-3.0).

    Args:
        positions: Stellar positions (N, 3) as NumPy array
        n_bins: Number of azimuthal bins (default 12, i.e., 30° sectors)

    Returns:
        sigma_over_mean: Relative variation σ_Σ / <Σ>

    Example:
        >>> import numpy as np
        >>> positions = np.random.randn(1000, 3)  # Gaussian cluster
        >>> var = compute_azimuthal_variation(positions)
        >>> print(f"σ_Σ/<Σ> = {var:.3f}")

    Notes:
        Practical alternative to Q parameter for large clusters.

        The linear relation to D allows estimation of fractal dimension:
            D ≈ (1.45 - σ_Σ/<Σ>) / 0.46

    References:
        Küpper et al. (2011), MNRAS 417, 2300
    """
    # Project to x-y plane and compute azimuthal angle
    phi = np.arctan2(positions[:, 1], positions[:, 0])

    # Bin by azimuthal angle
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    counts, _ = np.histogram(phi, bins=bin_edges)

    # Relative variation
    mean_count = np.mean(counts)
    if mean_count < 1e-10:
        return 0.0

    std_count = np.std(counts)
    sigma_over_mean = std_count / mean_count

    return float(sigma_over_mean)


__all__ = [
    "compute_q_parameter",
    "compute_azimuthal_variation",
]

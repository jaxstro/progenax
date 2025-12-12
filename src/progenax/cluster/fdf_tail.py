"""FDF tail selection: BM19-consistent and legacy methods.

This module provides tail PMF construction for gravoturbulent sampling.

Primary Methods
---------------
- compute_tail_pmfs_bm19(): BM19-consistent direct s-threshold selection (DEFAULT)
- compute_tail_pmfs_pn11_legacy(): Legacy local overdensity selection (for comparison)

The BM19 method implements the physics-consistent approach:
- Uses s = ln(ρ/ρ_mean) as the threshold variable
- Applies soft sigmoid w(x) = sigmoid(κ(s - s_t)) for differentiability
- Produces f_tail_actual that matches f_dense from theory

The legacy method uses local overdensity ranking (ρ/ρ_smoothed)
which is phenomenological but NOT BM19-consistent.

All functions are:
- @jax.jit compatible
- Differentiable via jax.grad (BM19 method)
- Vectorizable via jax.vmap

References
----------
Burkhart, B. & Mocz, P. 2019, ApJ, 879, 129
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, PRNGKeyArray


# =============================================================================
# Result Container
# =============================================================================


class TailPMFResult(NamedTuple):
    """Result of tail PMF computation.

    Attributes
    ----------
    p_tail : Array, shape (N_cells,)
        Normalized tail PMF (flattened). Sums to 1.
        For BM19: mass-weighted where w(x) > 0.
        For legacy: mass of voxels in dense tail.
    p_smooth : Array, shape (N_cells,)
        Normalized smooth PMF (flattened). Sums to 1.
        Covers the full domain, weighted by density.
    f_tail_actual : float
        Measured mass fraction in the tail.
        For BM19: should match f_dense from theory within ~5%.
        For legacy: equals dense_tail_mass_frac by construction.
    tail_weights : Array, shape (Nx, Ny, Nz)
        3D soft weights for each cell.
        For BM19: sigmoid(κ(s - s_t)) in [0, 1].
        For legacy: binary mask (0 or 1).
        Used by zeta_fdf_direct() for magnification factor.
    """

    p_tail: Float[Array, "N"]
    p_smooth: Float[Array, "N"]
    f_tail_actual: Float[Array, ""]  # Scalar JAX array for JIT compatibility
    tail_weights: Float[Array, "Nx Ny Nz"]


# =============================================================================
# BM19-Consistent Tail Selection (PRIMARY METHOD)
# =============================================================================


@jax.jit
def compute_tail_pmfs_bm19(
    rho_grid: Float[Array, "Nx Ny Nz"],
    s_t: float,
    kappa: float = 10.0,
) -> TailPMFResult:
    """BM19-consistent tail selection via direct s-threshold.

    This is the PRIMARY method for tail identification in gravoturbulent
    star formation. It implements the physics-consistent approach:

    1. Compute log-density contrast: s(x) = ln(ρ(x) / ⟨ρ⟩)
    2. Apply soft sigmoid: w(x) = sigmoid(κ × (s(x) - s_t))
    3. Construct tail PMF: p_tail ∝ w × ρ
    4. Construct smooth PMF: p_smooth ∝ ρ (full domain)
    5. Compute f_tail_actual = Σ(w × ρ) / Σρ

    The soft sigmoid ensures differentiability while approximating the
    hard threshold s > s_t. Higher κ = sharper transition.

    Parameters
    ----------
    rho_grid : Array, shape (Nx, Ny, Nz)
        3D density field from init_turbulent_density_field().
    s_t : float
        BM19 transition density from bm19_pipeline().
        Cells with s > s_t are preferentially selected as tail.
    kappa : float, default 10.0
        Sigmoid sharpness parameter.
        - κ = 1: Very soft transition (gradual)
        - κ = 10: Reasonably sharp (default, good balance)
        - κ = 100: Nearly hard threshold (approaches step function)

    Returns
    -------
    TailPMFResult
        PMFs and validation quantities.

    Notes
    -----
    **Validation check**: f_tail_actual should match f_dense from BM19
    to within ~5-10% for a single realization at 64³ resolution.
    Larger deviations suggest:
    - Non-lognormal field (e.g., strong shocks)
    - Resolution effects (increase grid_size)
    - Bug in either computation

    **Differentiability**: All computations use smooth operations.
    Gradients flow through s_t, kappa, and (stop_gradiented) rho_grid.

    Examples
    --------
    >>> from progenax.gravoturb import bm19_pipeline
    >>> from progenax.cluster.fdf_density import init_turbulent_density_field
    >>>
    >>> # Get BM19 quantities
    >>> result = bm19_pipeline(mach=10.0, alpha=2.0)
    >>>
    >>> # Generate density field and compute tail PMFs
    >>> field = init_turbulent_density_field(key, R_half=1.0, layer=layer)
    >>> pmf_result = compute_tail_pmfs_bm19(field.rho_grid, float(result.s_t))
    >>>
    >>> # Validate: f_tail_actual ≈ f_dense
    >>> print(f"Theory f_dense: {result.f_dense:.3f}")
    >>> print(f"Measured f_tail: {pmf_result.f_tail_actual:.3f}")
    """
    # Flatten for 1D operations
    rho_flat = rho_grid.ravel()
    n_cells = rho_flat.size

    # Mean density
    rho_mean = jnp.mean(rho_flat)

    # Log-density contrast: s = ln(ρ / ⟨ρ⟩)
    # Add small epsilon to avoid log(0)
    s_flat = jnp.log(rho_flat / (rho_mean + 1e-30) + 1e-30)

    # Soft sigmoid weights: w = sigmoid(κ(s - s_t))
    # w → 1 for s >> s_t (in tail)
    # w → 0 for s << s_t (not in tail)
    w_flat = jax.nn.sigmoid(kappa * (s_flat - s_t))

    # Total mass (for normalization)
    total_mass = jnp.sum(rho_flat)

    # Tail mass fraction: f_tail = Σ(w × ρ) / Σρ
    weighted_mass = jnp.sum(w_flat * rho_flat)
    f_tail_actual = weighted_mass / (total_mass + 1e-30)

    # Tail PMF: p_tail ∝ w × ρ
    # Cells with higher w and higher ρ are more likely to be selected
    p_tail_unnorm = w_flat * rho_flat
    p_tail = p_tail_unnorm / (jnp.sum(p_tail_unnorm) + 1e-30)

    # Smooth PMF: p_smooth ∝ ρ (full domain, no threshold)
    p_smooth = rho_flat / (total_mass + 1e-30)

    # Reshape weights back to 3D for zeta_fdf_direct
    tail_weights = w_flat.reshape(rho_grid.shape)

    return TailPMFResult(
        p_tail=p_tail,
        p_smooth=p_smooth,
        f_tail_actual=f_tail_actual,  # Keep as JAX scalar for JIT compatibility
        tail_weights=tail_weights,
    )


# =============================================================================
# Legacy Local Overdensity Selection (PN11 Comparison)
# =============================================================================


def _gaussian_blur_3d_fft(
    field: Float[Array, "Nx Ny Nz"],
    sigma_cells: float = 5.0,
) -> Float[Array, "Nx Ny Nz"]:
    """Apply Gaussian blur via FFT convolution.

    This is used by the legacy local overdensity method.
    Duplicated here to keep fdf_tail.py self-contained.

    Parameters
    ----------
    field : Array, shape (Nx, Ny, Nz)
        Input 3D field.
    sigma_cells : float
        Gaussian width in grid cells (default 5.0).

    Returns
    -------
    blurred : Array, shape (Nx, Ny, Nz)
        Smoothed field.
    """
    Nx, Ny, Nz = field.shape

    # Frequency grids
    kx = jnp.fft.fftfreq(Nx)
    ky = jnp.fft.fftfreq(Ny)
    kz = jnp.fft.fftfreq(Nz)
    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing="ij")
    k_sq = KX**2 + KY**2 + KZ**2

    # Gaussian kernel in Fourier space
    # G(k) = exp(-2 π² σ² k²)
    kernel_fft = jnp.exp(-2.0 * jnp.pi**2 * sigma_cells**2 * k_sq)

    # Convolve via FFT
    field_fft = jnp.fft.fftn(field)
    blurred_fft = field_fft * kernel_fft
    blurred = jnp.real(jnp.fft.ifftn(blurred_fft))

    return blurred


def compute_tail_pmfs_pn11_legacy(
    rho_grid: Float[Array, "Nx Ny Nz"],
    dense_tail_mass_frac: float = 0.10,
    smoothing_sigma: float = 5.0,
) -> TailPMFResult:
    """Legacy local overdensity tail selection.

    ⚠️ LEGACY: Use compute_tail_pmfs_bm19() for physics-consistent work.

    This method is retained for:
    - Reproducing older-style setups
    - Validation comparisons with BM19 approach

    Algorithm:
    1. Gaussian blur: ρ_smoothed = blur(ρ, σ=smoothing_sigma)
    2. Local overdensity: ρ_local = ρ / ρ_smoothed
    3. Rank voxels by ρ_local (descending)
    4. Dense tail = voxels containing top dense_tail_mass_frac of mass
    5. Construct PMFs from dense/smooth masks

    This identifies cells denser than their local environment, which is
    a phenomenological proxy for gravitational collapse sites. However,
    it does NOT match the BM19 theoretical framework (absolute s > s_t).

    Parameters
    ----------
    rho_grid : Array, shape (Nx, Ny, Nz)
        3D density field.
    dense_tail_mass_frac : float, default 0.10
        Fixed mass fraction for dense tail (top 10% of mass by default).
        Note: This is typically ~1-2% of cells for lognormal fields.
    smoothing_sigma : float, default 5.0
        Gaussian blur σ in grid cells.

    Returns
    -------
    TailPMFResult
        PMFs and validation quantities.
        Note: f_tail_actual will equal dense_tail_mass_frac by construction.

    Notes
    -----
    **NOT differentiable** with respect to dense_tail_mass_frac because
    the algorithm uses hard ranking and thresholding.

    **Physical interpretation**: Local overdensity identifies regions
    where ρ/ρ_smoothed >> 1, i.e., cells denser than their surroundings.
    This is related to local gravitational instability but does not
    directly correspond to BM19's s > s_t criterion.
    """
    import warnings

    warnings.warn(
        "compute_tail_pmfs_pn11_legacy() is deprecated. "
        "Use compute_tail_pmfs_bm19() for physics-consistent tail selection.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Flatten
    rho_flat = rho_grid.ravel()
    n_cells = rho_flat.size

    # Normalize to mass probabilities
    total_mass = jnp.sum(rho_flat)
    mass = rho_flat / (total_mass + 1e-12)

    # Local overdensity via Gaussian smoothing
    rho_smoothed = _gaussian_blur_3d_fft(rho_grid, sigma_cells=smoothing_sigma)
    rho_local = rho_grid / (rho_smoothed + 1e-12)
    rho_local_flat = rho_local.ravel()

    # Rank by local overdensity (high → low)
    sort_idx = jnp.argsort(-rho_local_flat)
    mass_sorted = mass[sort_idx]
    cum_mass = jnp.cumsum(mass_sorted)

    # Dense tail = voxels containing top dense_tail_mass_frac of mass
    dense_cut = jnp.searchsorted(cum_mass, dense_tail_mass_frac, side="right")
    dense_cut = jnp.clip(dense_cut, 1, n_cells - 1)

    # Create masks
    idx_range = jnp.arange(n_cells)
    is_dense = idx_range < dense_cut
    is_smooth = ~is_dense

    # Dense PMF (in sorted order)
    mass_dense_raw = jnp.where(is_dense, mass_sorted, 0.0)
    p_dense_sorted = mass_dense_raw / (jnp.sum(mass_dense_raw) + 1e-12)

    # Smooth PMF (in sorted order)
    mass_smooth_raw = jnp.where(is_smooth, mass_sorted, 0.0)
    p_smooth_sorted = mass_smooth_raw / (jnp.sum(mass_smooth_raw) + 1e-12)

    # Convert back to original order for consistency with BM19 method
    # Create inverse permutation
    inv_sort_idx = jnp.argsort(sort_idx)
    p_tail = p_dense_sorted[inv_sort_idx]
    p_smooth = p_smooth_sorted[inv_sort_idx]

    # Binary tail weights (1 if in dense tail, 0 otherwise)
    # In original order: cell i is dense if its sorted rank < dense_cut
    ranks = jnp.argsort(jnp.argsort(-rho_local_flat))  # Rank of each cell
    cum_mass_at_rank = cum_mass[ranks]
    is_dense_original = cum_mass_at_rank <= dense_tail_mass_frac
    tail_weights = is_dense_original.reshape(rho_grid.shape).astype(jnp.float32)

    # f_tail_actual = dense_tail_mass_frac by construction
    f_tail_actual = dense_tail_mass_frac

    return TailPMFResult(
        p_tail=p_tail,
        p_smooth=p_smooth,
        f_tail_actual=f_tail_actual,
        tail_weights=tail_weights,
    )


# =============================================================================
# Shared Sampling Functions
# =============================================================================


def sample_from_pmf(
    key: PRNGKeyArray,
    pmf: Float[Array, "N"],
    n_samples: int,
) -> Float[Array, "n_samples"]:
    """Sample indices from a discrete PMF.

    Parameters
    ----------
    key : PRNGKey
        JAX random key.
    pmf : Array, shape (N,)
        Probability mass function (must sum to 1).
    n_samples : int
        Number of samples to draw (must be concrete Python int).

    Returns
    -------
    indices : Array, shape (n_samples,)
        Sampled cell indices.

    Notes
    -----
    This function is NOT JIT-compiled because n_samples must be a
    concrete integer for JAX shape requirements.
    """
    log_pmf = jnp.log(pmf + 1e-30)
    return random.categorical(key, log_pmf, shape=(n_samples,))


def sample_positions_from_pmfs(
    key: PRNGKeyArray,
    rho_grid: Float[Array, "Nx Ny Nz"],
    x_grid: Float[Array, "Nx"],
    y_grid: Float[Array, "Ny"],
    z_grid: Float[Array, "Nz"],
    pmf_result: TailPMFResult,
    N_stars: int,
    f_sub: float,
) -> Float[Array, "N 3"]:
    """Sample star positions from tail + smooth PMFs.

    Allocates N_dense = round(f_sub × N_stars) to the tail,
    and N_smooth = N_stars - N_dense to the smooth component.

    Parameters
    ----------
    key : PRNGKey
        JAX random key.
    rho_grid : Array, shape (Nx, Ny, Nz)
        3D density field (for shape information).
    x_grid, y_grid, z_grid : Array
        1D coordinate arrays for each axis.
    pmf_result : TailPMFResult
        Output from compute_tail_pmfs_*().
    N_stars : int
        Total number of stars.
    f_sub : float
        Fraction allocated to tail (0 to 1).

    Returns
    -------
    positions : Array, shape (N, 3)
        Sampled positions in physical units.

    Notes
    -----
    Stars are sampled from both PMFs independently, then the first
    N_dense are taken from tail samples and the rest from smooth samples.
    This ensures exact N_stars output with the desired split.
    """
    Nx, Ny, Nz = rho_grid.shape

    # Split keys
    key_tail, key_smooth, key_jitter = random.split(key, 3)

    # Compute allocation (JAX-native for JIT compatibility)
    f_sub_clipped = jnp.clip(f_sub, 0.0, 1.0)
    N_dense_float = jnp.round(f_sub_clipped * N_stars)
    N_dense = jnp.clip(N_dense_float, 0, N_stars).astype(jnp.int32)

    # Sample from both PMFs (always sample N_stars from each for simplicity)
    tail_idx = sample_from_pmf(key_tail, pmf_result.p_tail, N_stars)
    smooth_idx = sample_from_pmf(key_smooth, pmf_result.p_smooth, N_stars)

    # Select: first N_dense from tail, rest from smooth
    star_idx = jnp.arange(N_stars)
    use_tail = star_idx < N_dense
    cell_idx = jnp.where(use_tail, tail_idx, smooth_idx)

    # Convert flat index to 3D indices
    x_idx = cell_idx // (Ny * Nz)
    y_idx = (cell_idx % (Ny * Nz)) // Nz
    z_idx = cell_idx % Nz

    # Cell centers
    x_c = x_grid[x_idx]
    y_c = y_grid[y_idx]
    z_c = z_grid[z_idx]

    # Sub-cell jitter for smoothness
    dx = x_grid[1] - x_grid[0]
    jitter = (random.uniform(key_jitter, (N_stars, 3)) - 0.5) * dx

    positions = jnp.stack([x_c, y_c, z_c], axis=1) + jitter

    return positions


# =============================================================================
# High-Level Sampling Interface
# =============================================================================


def sample_positions_tail_bm19(
    key: PRNGKeyArray,
    rho_grid: Float[Array, "Nx Ny Nz"],
    x_grid: Float[Array, "Nx"],
    y_grid: Float[Array, "Ny"],
    z_grid: Float[Array, "Nz"],
    N_stars: int,
    f_sub: float,
    s_t: float,
    kappa: float = 10.0,
) -> tuple[Float[Array, "N 3"], TailPMFResult]:
    """Sample star positions using BM19-consistent tail selection.

    This is the recommended interface for gravoturbulent IC generation.

    Parameters
    ----------
    key : PRNGKey
        JAX random key.
    rho_grid : Array, shape (Nx, Ny, Nz)
        3D density field.
    x_grid, y_grid, z_grid : Array
        1D coordinate arrays.
    N_stars : int
        Total number of stars.
    f_sub : float
        Fraction allocated to tail (0 to 1).
        f_sub ≈ f_dense × eta_survive from BM19 pipeline.
    s_t : float
        BM19 transition density.
    kappa : float, default 10.0
        Sigmoid sharpness.

    Returns
    -------
    positions : Array, shape (N, 3)
        Sampled star positions.
    pmf_result : TailPMFResult
        PMF details for diagnostics/validation.

    Examples
    --------
    >>> from progenax.gravoturb import bm19_pipeline
    >>>
    >>> # Get BM19 quantities
    >>> result = bm19_pipeline(mach=10.0, alpha=2.0, eta_survive=0.6)
    >>>
    >>> # Sample positions
    >>> positions, pmf = sample_positions_tail_bm19(
    ...     key, field.rho_grid, field.x_grid, field.y_grid, field.z_grid,
    ...     N_stars=1000, f_sub=float(result.f_sub), s_t=float(result.s_t)
    ... )
    >>>
    >>> # Validate consistency
    >>> print(f"Theory f_dense: {result.f_dense:.3f}")
    >>> print(f"Measured f_tail: {pmf.f_tail_actual:.3f}")
    """
    if N_stars < 100:
        import warnings

        warnings.warn(
            f"Sampling with N_stars={N_stars} < 100: Q and clustering "
            "diagnostics will be dominated by Poisson noise.",
            UserWarning,
            stacklevel=2,
        )

    # Compute tail PMFs
    pmf_result = compute_tail_pmfs_bm19(rho_grid, s_t, kappa)

    # Sample positions
    positions = sample_positions_from_pmfs(
        key, rho_grid, x_grid, y_grid, z_grid, pmf_result, N_stars, f_sub
    )

    return positions, pmf_result


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Result container
    "TailPMFResult",
    # Primary method (BM19)
    "compute_tail_pmfs_bm19",
    "sample_positions_tail_bm19",
    # Legacy method (PN11)
    "compute_tail_pmfs_pn11_legacy",
    # Shared utilities
    "sample_from_pmf",
    "sample_positions_from_pmfs",
]

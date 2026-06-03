"""Position sampling from a density field (split from fdf_density.py)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import random
from jax import Array
from jaxtyping import Float, PRNGKeyArray

from progenax import defaults
from progenax.cluster.fdf_config import FDF_HEURISTICS, CHI_MIN, CHI_MAX

if TYPE_CHECKING:
    from progenax.cluster.fdf_config import GravoturbulentEnv, GravoturbulentResult
    from progenax.gravoturb.bm19_model import BM19Result

from .density_field import DensityField3D, _gaussian_blur_3d_fft


def sample_positions_from_density(
    key: PRNGKeyArray,
    field: DensityField3D,
    N_stars: int,
) -> Float[Array, "N 3"]:
    """Sample N_stars positions from the density field.

    Uses inverse CDF sampling on the flattened density grid,
    with uniform sub-cell offsets for smoothness.

    Parameters
    ----------
    key : PRNGKey
        JAX random key.
    field : DensityField3D
        The density field to sample from.
    N_stars : int
        Number of positions to sample.

    Returns
    -------
    positions : Array, shape (N, 3)
        Sampled positions in pc.

    Notes
    -----
    The algorithm:
    1. Flatten the 3D density grid to 1D
    2. Compute cumulative distribution function (CDF)
    3. Draw uniform samples u ~ U[0, 1]
    4. Find cell indices via searchsorted(CDF, u)
    5. Add uniform sub-cell offsets for smoothness

    This is O(N_stars * log(N_cells)) due to searchsorted.
    """
    Nx = field.x_grid.shape[0]
    Ny = field.y_grid.shape[0]
    Nz = field.z_grid.shape[0]

    # Flatten and normalize to probability
    rho_flat = field.rho_grid.ravel()
    rho_flat = rho_flat / (jnp.sum(rho_flat) + 1e-12)

    # Cumulative distribution function
    cdf = jnp.cumsum(rho_flat)

    # Draw uniform samples
    key_u, key_offset = random.split(key)
    u = random.uniform(key_u, (N_stars,))

    # Find cell indices via searchsorted
    idx = jnp.searchsorted(cdf, u)
    idx = jnp.clip(idx, 0, rho_flat.size - 1)

    # Convert 1D index to 3D indices
    i = idx // (Ny * Nz)
    j = (idx % (Ny * Nz)) // Nz
    k = idx % Nz

    # Cell centers
    x_c = field.x_grid[i]
    y_c = field.y_grid[j]
    z_c = field.z_grid[k]

    # Sub-cell uniform offsets for smoothness
    dx = field.x_grid[1] - field.x_grid[0]
    offsets = (random.uniform(key_offset, (N_stars, 3)) - 0.5) * dx

    positions = jnp.stack([x_c, y_c, z_c], axis=1) + offsets

    return positions


def sample_positions_tail(
    key: PRNGKeyArray,
    field: DensityField3D,
    N_stars: int,
    f_sub: float,
    *,
    mode: str = "bm19",
    s_t: float | None = None,
    kappa: float = 10.0,
    dense_tail_mass_frac: float = 0.10,
) -> Float[Array, "N 3"]:
    """Sample star positions from gravoturbulent dense tail + smooth component.

    This implements two-component sampling that separates:
    - Dense tail: cells with s > s_t (BM19 mode) or top 10% mass (legacy)
    - Smooth component: remaining cells

    Stars are allocated: N_dense ≈ f_sub × N_stars go to dense tail,
    N_smooth = N - N_dense to smooth. Higher f_sub = more stars concentrated
    in the dense tail = more substructure = LOWER Q.

    Parameters
    ----------
    key : PRNGKey
        JAX random key.
    field : DensityField3D
        The density field to sample from (turbulent gas density).
    N_stars : int
        Total number of star positions to sample. MUST be a concrete integer
        (not a JAX tracer) because jax.random.categorical requires static shape.
    f_sub : float
        Fraction of stars to sample from dense tail (0..1).
        - f_sub=0: all stars from smooth component (spread out → high Q)
        - f_sub=1: all stars from dense tail (concentrated → low Q)
    mode : str, default "bm19"
        Tail selection method:
        - "bm19": BM19-consistent direct s > s_t threshold (RECOMMENDED)
        - "pn11_legacy": Local overdensity ranking (legacy, for comparison)
    s_t : float, optional
        BM19 transition density. REQUIRED when mode="bm19".
        Get from bm19.bm19_pipeline(mach, alpha).s_t.
    kappa : float, default 10.0
        Sigmoid sharpness for BM19 mode. Higher = sharper threshold.
    dense_tail_mass_frac : float, default 0.10
        Fixed mass fraction for dense tail in legacy mode.
        Ignored when mode="bm19".

    Returns
    -------
    positions : Array, shape (N, 3)
        Sampled positions in pc.

    Notes
    -----
    **BM19 Mode (Recommended)**

    Uses direct s = ln(ρ/ρ_mean) > s_t threshold with soft sigmoid:
    - w(x) = sigmoid(κ × (s(x) - s_t))
    - Cells with s > s_t have w ≈ 1 (in tail)
    - f_tail_actual ≈ f_dense from BM19 theory

    **Legacy Mode (PN11)**

    Uses local overdensity ranking (ρ/ρ_smoothed):
    - Identifies cells denser than local environment
    - Selects top dense_tail_mass_frac of mass
    - NOT physics-consistent with BM19

    From CW04: f_sub ↑ → more stars in spatially-correlated clumps → Q ↓

    Examples
    --------
    >>> # BM19 mode (recommended)
    >>> from progenax.gravoturb import bm19_pipeline
    >>> result = bm19_pipeline(mach=10.0, alpha=2.0, eta_survive=0.6)
    >>> positions = sample_positions_tail(
    ...     key, field, N_stars=1000, f_sub=float(result.f_sub),
    ...     mode="bm19", s_t=float(result.s_t)
    ... )

    >>> # Legacy mode
    >>> positions = sample_positions_tail(
    ...     key, field, N_stars=1000, f_sub=0.3,
    ...     mode="pn11_legacy", dense_tail_mass_frac=0.10
    ... )
    """
    if mode == "bm19":
        if s_t is None:
            raise ValueError(
                "s_t is required for mode='bm19'. "
                "Get it from bm19.bm19_pipeline(mach, alpha).s_t"
            )
        # Use new BM19-consistent tail selection
        from progenax.cluster.fdf_tail import sample_positions_tail_bm19

        positions, _ = sample_positions_tail_bm19(
            key,
            field.rho_grid,
            field.x_grid,
            field.y_grid,
            field.z_grid,
            N_stars,
            f_sub,
            s_t,
            kappa,
        )
        return positions

    elif mode == "pn11_legacy":
        import warnings

        warnings.warn(
            "mode='pn11_legacy' uses deprecated local overdensity ranking. "
            "Use mode='bm19' with s_t from bm19_pipeline() for "
            "physics-consistent results.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Fall through to legacy implementation below
    else:
        raise ValueError(f"Invalid mode '{mode}'. Use 'bm19' or 'pn11_legacy'.")

    # =========================================================================
    # Legacy local overdensity implementation (mode='pn11_legacy')
    # =========================================================================
    Nx = field.x_grid.shape[0]
    Ny = field.y_grid.shape[0]
    Nz = field.z_grid.shape[0]
    n_cells = Nx * Ny * Nz

    # Step 1: Flatten and normalize cell masses as probabilities
    rho_flat = field.rho_grid.ravel()
    mass = rho_flat / (jnp.sum(rho_flat) + 1e-12)  # Dimensionless probabilities

    # Step 2: Rank voxels by LOCAL overdensity (ρ / ρ_smoothed)
    #
    # CRITICAL: Global density ranking fails because the turbulent field has
    # spatially-correlated structure - the highest-density cells cluster together.
    # Selecting the top 10% by global density picks a single clump, not distributed
    # clumps. This creates CENTRAL CONCENTRATION (Q > 0.79), not SUBSTRUCTURE (Q < 0.79).
    #
    # Fix: Use local overdensity = ρ / ρ_smoothed to identify cells that are
    # denser than their LOCAL environment. This ensures the "dense" cells are
    # DISTRIBUTED across the volume as multiple independent clumps.
    #
    # Physics: Local overdensity = local α_vir << 1 = gravitational collapse site
    rho_smoothed = _gaussian_blur_3d_fft(field.rho_grid, sigma_cells=5.0)
    rho_local = field.rho_grid / (rho_smoothed + 1e-12)  # Local overdensity ratio
    rho_local_flat = rho_local.ravel()

    sort_idx = jnp.argsort(-rho_local_flat)  # Rank by LOCAL overdensity (high → low)
    mass_sorted = mass[sort_idx]
    cum_mass = jnp.cumsum(mass_sorted)

    # Step 3: Define dense vs smooth by FIXED mass fraction (not f_sub!)
    # Dense tail = voxels containing top dense_tail_mass_frac of the total mass
    # This is typically ~1-2% of cells for lognormal density fields
    dense_cut = jnp.searchsorted(cum_mass, dense_tail_mass_frac, side="right")
    dense_cut = jnp.clip(dense_cut, 1, n_cells - 1)  # At least 1 cell in each

    # Create masks for dense and smooth indices
    idx_range = jnp.arange(n_cells)
    is_dense = idx_range < dense_cut
    is_smooth = ~is_dense

    # Step 4: Build two normalized PMFs
    # For dense component: use mass of dense voxels
    mass_dense_raw = jnp.where(is_dense, mass_sorted, 0.0)
    sum_dense = jnp.sum(mass_dense_raw) + 1e-12
    p_dense = mass_dense_raw / sum_dense

    # For smooth component: use mass of smooth voxels
    mass_smooth_raw = jnp.where(is_smooth, mass_sorted, 0.0)
    sum_smooth = jnp.sum(mass_smooth_raw) + 1e-12
    p_smooth = mass_smooth_raw / sum_smooth

    # Step 5: Allocate stars to each component
    # f_sub controls how many stars go to the FIXED dense tail
    # Use JAX-clean computation: jnp.round returns float, then clip and cast
    f_sub_clipped = jnp.clip(f_sub, 0.0, 1.0)
    N_dense_float = jnp.round(f_sub_clipped * N_stars)
    N_dense = jnp.clip(N_dense_float, 0, N_stars).astype(jnp.int32)
    N_smooth = N_stars - N_dense

    # Step 6: Sample from each component via categorical
    key_dense, key_smooth, key_jitter = random.split(key, 3)

    # Sample sorted indices from dense PMF
    # categorical samples from logits, so we use log(p)
    log_p_dense = jnp.log(p_dense + 1e-30)
    dense_sorted_idx = random.categorical(key_dense, log_p_dense, shape=(N_stars,))

    # Sample sorted indices from smooth PMF
    log_p_smooth = jnp.log(p_smooth + 1e-30)
    smooth_sorted_idx = random.categorical(key_smooth, log_p_smooth, shape=(N_stars,))

    # Convert sorted indices back to original flat indices
    dense_flat_idx = sort_idx[dense_sorted_idx]
    smooth_flat_idx = sort_idx[smooth_sorted_idx]

    # Create selection mask: first N_dense from dense, rest from smooth
    star_idx = jnp.arange(N_stars)
    use_dense = star_idx < N_dense
    cell_idx_all = jnp.where(use_dense, dense_flat_idx, smooth_flat_idx)

    # Step 7: Convert flat indices → 3D → physical coordinates
    x_idx = cell_idx_all // (Ny * Nz)
    y_idx = (cell_idx_all % (Ny * Nz)) // Nz
    z_idx = cell_idx_all % Nz

    # Cell centers
    x_c = field.x_grid[x_idx]
    y_c = field.y_grid[y_idx]
    z_c = field.z_grid[z_idx]

    # Sub-voxel uniform jitter for smoothness
    dx = field.x_grid[1] - field.x_grid[0]
    jitter = (random.uniform(key_jitter, (N_stars, 3)) - 0.5) * dx

    positions = jnp.stack([x_c, y_c, z_c], axis=1) + jitter

    return positions


# =============================================================================
# Complete IC Generator
# =============================================================================



"""Turbulent / BM19 density-field initialization (split from fdf_density.py)."""

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

from .density_field import (
    _legacy_chi_to_beta,
    FractalDensityLayer,
    DensityField3D,
    _make_hermitian,
    _plummer_density_grid,
    _uniform_sphere_mask,
)


def init_turbulent_density_field(
    key: PRNGKeyArray,
    R_half: float,
    layer: FractalDensityLayer,
) -> DensityField3D:
    """Initialize a 3D turbulent density field.

    Parameters
    ----------
    key : PRNGKey
        JAX random key.
    R_half : float
        Half-mass radius in pc.
    layer : FractalDensityLayer
        Parameters controlling the density field.

    Returns
    -------
    DensityField3D
        The frozen density field ready for position sampling.

    Notes
    -----
    The field is constructed as:
    1. Generate Gaussian random field δ(x) with P(k) ∝ k^{-β}
    2. Convert to lognormal: ρ_turb = exp(δ - σ²/2) (mean = 1)
    3. Multiply by base profile: ρ_combined = ρ_base × ρ_turb^{λ_frac}
    4. Normalize to integrate to 1

    The spectral slope β depends on chi:
        β = BETA_0_DENSITY + BETA_1_DENSITY * (chi - 1.5)

    This gives:
        - chi = 1.6: β = 2.15 (steeper, more small-scale power)
        - chi = 3.0: β = 4.25 (shallower for small scales, large-scale dominated)
    """
    Nx = Ny = Nz = layer.grid_size
    L_box = layer.box_size_factor * R_half

    # Grid setup
    x_grid = jnp.linspace(-L_box, L_box, Nx)
    y_grid = jnp.linspace(-L_box, L_box, Ny)
    z_grid = jnp.linspace(-L_box, L_box, Nz)
    dx = 2 * L_box / Nx  # periodic spacing for the FFT k-grid (fftfreq below)
    # ∫ρ dV uses the ACTUAL grid spacing (linspace endpoints inclusive -> 2L/(N-1));
    # using dx here left the absolute normalization off by (N/(N-1))^3 ~ 1.048 at
    # N=64 (audit minor). Decoupled so the k-grid / turbulent spectrum is unchanged.
    dV = (x_grid[1] - x_grid[0]) ** 3

    # k-space grid
    kx = 2 * jnp.pi * jnp.fft.fftfreq(Nx, d=dx)
    ky = 2 * jnp.pi * jnp.fft.fftfreq(Ny, d=dx)
    kz = 2 * jnp.pi * jnp.fft.fftfreq(Nz, d=dx)
    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing="ij")
    k_mag = jnp.sqrt(KX**2 + KY**2 + KZ**2)

    # Avoid division by zero at DC
    k_mag_safe = jnp.where(k_mag == 0, 1.0, k_mag)

    # Spectral slope from chi (UNCALIBRATED legacy mapping)
    # Higher chi → higher beta → less small-scale power → smoother
    beta = _legacy_chi_to_beta(layer.chi)

    # Power spectrum P(k) ∝ k^{-β}
    # The amplitude spectrum is sqrt(P(k)) = k^{-β/2}
    P_k = k_mag_safe ** (-beta)
    P_k = jnp.where(k_mag == 0, 0.0, P_k)  # Zero DC mode

    # Draw complex Gaussian modes
    key_real, key_imag = random.split(key)
    sigma_k = jnp.sqrt(P_k / 2.0)
    real_part = sigma_k * random.normal(key_real, (Nx, Ny, Nz))
    imag_part = sigma_k * random.normal(key_imag, (Nx, Ny, Nz))
    delta_k = real_part + 1j * imag_part

    # Enforce Hermitian symmetry for real IFFT output
    delta_k = _make_hermitian(delta_k)
    delta_k = delta_k.at[0, 0, 0].set(0.0)  # Ensure DC = 0

    # Inverse FFT to real space
    delta_x = jnp.real(jnp.fft.ifftn(delta_k))

    # Rescale to target variance
    var_delta = jnp.var(delta_x)
    delta_scaled = delta_x * (layer.sigma_ln_rho / jnp.sqrt(var_delta + 1e-12))

    # Convert to density
    if layer.use_log_normal:
        # Lognormal: ρ = exp(δ - σ²/2) has mean = 1
        rho_turb = jnp.exp(delta_scaled - 0.5 * layer.sigma_ln_rho**2)
    else:
        # Gaussian perturbation around 1
        rho_turb = 1.0 + delta_scaled
        rho_turb = jnp.clip(rho_turb, 1e-6, None)

    # Base profile density
    if layer.base_profile == "uniform":
        # Uniform density within a spherical region
        R_sphere = layer.sphere_radius_factor * R_half
        rho_base = _uniform_sphere_mask(x_grid, y_grid, z_grid, R_sphere)
    elif layer.base_profile == "plummer":
        # Plummer density profile
        rho_base = _plummer_density_grid(x_grid, y_grid, z_grid, R_half)
    else:
        raise ValueError(f"Unknown base_profile: {layer.base_profile}")

    # Normalize base profile so integral = 1
    mass_base = jnp.sum(rho_base) * dV
    rho_base_norm = rho_base / (mass_base + 1e-12)

    # Blend: ρ_combined = (1-λ) * ρ_base + λ * (ρ_base * ρ_turb)
    # At λ=0: pure smooth profile (Plummer or uniform sphere)
    # At λ=1: full turbulent modulation
    #
    # For "uniform" base: at λ=1, this becomes pure turbulent field in sphere
    # For "plummer" base: at λ=1, turbulent field modulates Plummer radial profile
    rho_combined = (1.0 - layer.lambda_frac) * rho_base_norm + layer.lambda_frac * (
        rho_base_norm * rho_turb
    )

    # Normalize to total mass = 1
    mass_combined = jnp.sum(rho_combined) * dV
    rho_final = rho_combined / (mass_combined + 1e-12)

    return DensityField3D(
        rho_grid=rho_final,
        x_grid=x_grid,
        y_grid=y_grid,
        z_grid=z_grid,
        box_half_size=L_box,
    )


def init_bm19_density_field(
    key: PRNGKeyArray,
    sigma_s_sq: float,
    s_t: float,
    alpha: float,
    grid_size: int = 64,
    box_half_size: float = 1.0,
    beta: float = 4.0,
    copula: str = "rank",
) -> DensityField3D:
    """Initialize a 3D turbulent density field with BM19 LN+PL PDF.

    Uses Gaussian copula (CDF remap) to generate a density field where the
    one-point PDF exactly matches the BM19 piecewise lognormal+powerlaw
    distribution while preserving turbulent spatial correlations.

    This is the PHYSICS-CORRECT method for generating FDF fields consistent
    with BM19 gravoturbulent theory. Use this for validation and production.

    Parameters
    ----------
    key : PRNGKey
        JAX random key.
    sigma_s_sq : float
        BM19 PDF variance σ_s² = ln(1 + b²M²).
    s_t : float
        BM19 transition density s_t = (α - 0.5)σ_s².
    alpha : float
        BM19 powerlaw slope (1.5-3.0).
    grid_size : int
        Number of grid cells per dimension (default 64).
    box_half_size : float
        Half-size of cubic box (default 1.0, normalized units).
    beta : float
        Power spectrum slope for turbulent structure (default 4.0 = Burgers).

    Returns
    -------
    DensityField3D
        Density field with BM19 LN+PL PDF and turbulent geometry.

    Notes
    -----
    The algorithm:
    1. Generate Gaussian GRF g(x) with power spectrum P(k) ∝ k^{-β}
    2. Transform: u(x) = Φ(g(x)) where Φ is standard normal CDF
    3. Apply inverse BM19 CDF: s(x) = F_V^{-1}(u(x))
    4. Convert to density: ρ(x) = exp(s(x))

    This preserves spatial correlations from the Gaussian GRF while enforcing
    the exact BM19 one-point PDF. The resulting field has:
    - Lognormal PDF for s < s_t
    - Powerlaw tail (∝ exp(-αs)) for s ≥ s_t
    - f_tail_actual ≈ f_dense from BM19 theory

    References
    ----------
    Burkhart, B. & Mocz, P. 2019, ApJ, 879, 129

    Examples
    --------
    >>> from progenax.gravoturb import bm19_pipeline
    >>> result = bm19_pipeline(mach=10.0, alpha=2.0)
    >>> field = init_bm19_density_field(
    ...     key, sigma_s_sq=result.sigma_s_sq, s_t=result.s_t, alpha=2.0
    ... )
    """
    from progenax.gravoturb import gaussian_to_bm19, build_bm19_cdf_table

    Nx = Ny = Nz = grid_size
    L_box = box_half_size

    # Grid setup
    x_grid = jnp.linspace(-L_box, L_box, Nx)
    y_grid = jnp.linspace(-L_box, L_box, Ny)
    z_grid = jnp.linspace(-L_box, L_box, Nz)
    dx = 2 * L_box / Nx  # periodic spacing for the FFT k-grid (fftfreq below)
    # ∫ρ dV uses the ACTUAL grid spacing (linspace endpoints inclusive -> 2L/(N-1));
    # using dx here left the absolute normalization off by (N/(N-1))^3 ~ 1.048 at
    # N=64 (audit minor). Decoupled so the k-grid / turbulent spectrum is unchanged.
    dV = (x_grid[1] - x_grid[0]) ** 3

    # k-space grid for power spectrum
    kx = 2 * jnp.pi * jnp.fft.fftfreq(Nx, d=dx)
    ky = 2 * jnp.pi * jnp.fft.fftfreq(Ny, d=dx)
    kz = 2 * jnp.pi * jnp.fft.fftfreq(Nz, d=dx)
    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing="ij")
    k_mag = jnp.sqrt(KX**2 + KY**2 + KZ**2)
    k_mag_safe = jnp.where(k_mag == 0, 1.0, k_mag)

    # Power spectrum P(k) ∝ k^{-β} (turbulent structure)
    P_k = k_mag_safe ** (-beta)
    P_k = jnp.where(k_mag == 0, 0.0, P_k)

    # Draw complex Gaussian modes
    key_real, key_imag = random.split(key)
    sigma_k = jnp.sqrt(P_k / 2.0)
    real_part = sigma_k * random.normal(key_real, (Nx, Ny, Nz))
    imag_part = sigma_k * random.normal(key_imag, (Nx, Ny, Nz))
    g_k = real_part + 1j * imag_part

    # Enforce Hermitian symmetry for real IFFT output
    g_k = _make_hermitian(g_k)
    g_k = g_k.at[0, 0, 0].set(0.0)

    # Inverse FFT to get Gaussian random field g(x)
    g_x = jnp.real(jnp.fft.ifftn(g_k))

    # Standardize to N(0, 1) for CDF remap
    g_mean = jnp.mean(g_x)
    g_std = jnp.std(g_x)
    g_standardized = (g_x - g_mean) / (g_std + 1e-12)

    # Build BM19 CDF table
    s_grid, F_grid = build_bm19_cdf_table(sigma_s_sq, s_t, alpha)

    # Resolution guard (audit M3) -- host-side check on CONCRETE inputs only; under
    # jax.grad/jit the BM19 param gradients flow through the CDF table (rank copula
    # is grad-safe), so skip the float()/warn when tracing.
    try:
        tail_prob = float(jnp.clip(1.0 - jnp.interp(s_t, s_grid, F_grid), 0.0, 1.0))
    except jax.errors.ConcretizationTypeError:
        tail_prob = None
    if tail_prob is not None:
        expected_tail_cells = tail_prob * (Nx * Ny * Nz)
        if expected_tail_cells < 5.0:
            import warnings

            need = 5.0 / max(tail_prob, 1e-30)
            warnings.warn(
                f"BM19 dense tail under-resolved at grid_size={grid_size}: only "
                f"~{expected_tail_cells:.1f} cells expected above s_t={s_t:.2f} "
                f"(tail probability {tail_prob:.2e}). Realized f_tail will read low even "
                f"with the rank copula; increase grid_size (need N^3 >~ {need:.0e}).",
                UserWarning,
                stacklevel=2,
            )

    # Apply CDF remap: g -> u -> s = F_V^{-1}(u).
    # Default copula="rank" (empirical CDF) forces the exact BM19 marginal at any
    # beta; the legacy "phi" (normal CDF) collapses the dense tail at steep beta
    # because the realized GRF is non-Gaussian (audit M3).
    s_field = gaussian_to_bm19(
        g_standardized, sigma_s_sq, s_t, alpha, s_grid, F_grid, copula=copula
    )

    # Convert to density
    rho = jnp.exp(s_field)

    # Normalize to total mass = 1
    mass_total = jnp.sum(rho) * dV
    rho_normalized = rho / (mass_total + 1e-12)

    return DensityField3D(
        rho_grid=rho_normalized,
        x_grid=x_grid,
        y_grid=y_grid,
        z_grid=z_grid,
        box_half_size=L_box,
    )


# =============================================================================
# Position Sampling
# =============================================================================



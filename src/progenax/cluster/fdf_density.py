# progenax/src/progenax/cluster/fdf_density.py
"""
Fractal Density Field (FDF-D) for star cluster initial conditions.

This module implements a density-field-based approach to generating
fractal substructure. Unlike the displacement-field FDF, this method:

1. Constructs a 3D turbulent density field ρ_turb(x) with power spectrum P(k) ∝ k^{-β}
2. Multiplies by the base profile to get ρ_final(x) = ρ_base(x) × ρ_turb(x)
3. Samples star positions proportionally to ρ_final(x)

The chi parameter controls the spectral slope:
    - chi ≈ 1.6 (clumpy): more small-scale power → more small clumps
    - chi ≈ 3.0 (smooth): more large-scale power → smoother distribution

Base Profile Choice
-------------------
The ``base_profile`` parameter critically affects Q interpretation:

**"uniform" (RECOMMENDED for CW04 comparison)**
    Turbulent field in a spherical region. Q calibration matches CW04 Table 1.

**"plummer" (for realistic star clusters)**
    Turbulent field modulates Plummer radial profile. Q will be HIGHER than
    CW04 values because the base profile is already centrally concentrated.
    Use with caveat in papers.

WARNING - Uncalibrated Parameters
---------------------------------
The spectral slopes (BETA_0_DENSITY, BETA_1_DENSITY) and σ_ln_ρ defaults
are preliminary heuristics. They are NOT calibrated against:

- MHD simulations of turbulent molecular clouds
- Observed cloud power spectra
- Cartwright & Whitworth (2004) Q(D) measurements

For physically motivated parameters, use ``env_to_fdf_layer()`` from
``progenax.cluster.fdf_config``, which derives σ_ln_ρ from Federrath+2010
density-Mach scaling.

References
----------
- Federrath et al. (2010) A&A 512, A81 - Turbulence spectra, density-Mach relation
- Goodwin & Whitworth (2004) A&A 413, 929 - Fractal dimension D
- Cartwright & Whitworth (2004) MNRAS 348, 589 - Q parameter definition
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import random
from jax import Array
from jaxtyping import Float, PRNGKeyArray


# =============================================================================
# Heuristics Import (quarantined - not physics-derived)
# =============================================================================

from progenax.cluster.fdf_config import FDF_HEURISTICS, CHI_MIN, CHI_MAX


def _legacy_chi_to_beta(chi: float) -> float:
    """INTERNAL: Convert chi to beta for legacy API.

    WARNING: This mapping is UNCALIBRATED. For physics-based β,
    use BirthEnvironment.spectral_slope() instead.

    The formula β = β₀ + β₁ × (χ - 1.5) is arbitrary and NOT derived from:
    - Kolmogorov/Burgers turbulence theory
    - MHD simulations
    - Observed cloud power spectra

    Parameters
    ----------
    chi : float
        Clumpiness parameter in [1.6, 3.0].

    Returns
    -------
    beta : float
        Spectral slope for power spectrum P(k) ∝ k^{-β}.
    """
    return FDF_HEURISTICS.beta_0 + FDF_HEURISTICS.beta_1 * (chi - 1.5)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True)
class FractalDensityLayer:
    """Parameters for fractal density field layer.

    Attributes
    ----------
    chi : float
        Clumpiness parameter in [1.6, 3.0]. Controls spectral slope.
        Lower chi → steeper spectrum → more small-scale power → clumpier.
    sigma_ln_rho : float
        Standard deviation of log-density field. Controls amplitude of
        density fluctuations. Typical values: 1.5-2.5 for visible substructure.
    lambda_frac : float
        Blend fraction [0, 1]. 0 = pure smooth profile, 1 = full turbulent.
    grid_size : int
        Number of grid cells per dimension. 64 is usually sufficient.
    box_size_factor : float
        Box extends to ±box_size_factor * R_half. Should be ≥ 4 to capture
        the full cluster with margin.
    use_log_normal : bool
        If True, density field is lognormal (positive-definite, realistic).
        If False, uses Gaussian perturbation (can go negative, less physical).
    virial_ratio : float
        Target Q_vir = K/|U| for velocity assignment.
    base_profile : str
        Base density profile: "uniform" or "plummer".
        "uniform": Turbulent field in a spherical region (RECOMMENDED for
            producing visible clumpy → smooth Q(D) progression).
        "plummer": Turbulent field modulated by Plummer profile (produces
            centrally concentrated structures regardless of chi).
    sphere_radius_factor : float
        For uniform base: radius = sphere_radius_factor * R_half.
        Stars are sampled within this spherical region.
    """

    chi: float = 2.0
    sigma_ln_rho: float = 2.0  # Higher default for visible substructure
    lambda_frac: float = 1.0
    grid_size: int = 64
    box_size_factor: float = 4.0  # Smaller default for uniform base
    use_log_normal: bool = True
    virial_ratio: float = 0.5
    base_profile: str = "uniform"  # Default to uniform for proper Q(D) trend
    sphere_radius_factor: float = 2.5


@dataclass(frozen=True)
class DensityField3D:
    """Frozen 3D density field for position sampling.

    The density field is normalized so that ∫ρ dV = 1 over the box.

    Attributes
    ----------
    rho_grid : Array, shape (Nx, Ny, Nz)
        Normalized density values on the grid.
    x_grid : Array, shape (Nx,)
        x-coordinates of grid cell centers.
    y_grid : Array, shape (Ny,)
        y-coordinates of grid cell centers.
    z_grid : Array, shape (Nz,)
        z-coordinates of grid cell centers.
    box_half_size : float
        Half-size of the cubic box in pc.
    """

    rho_grid: Float[Array, "Nx Ny Nz"]
    x_grid: Float[Array, "Nx"]
    y_grid: Float[Array, "Ny"]
    z_grid: Float[Array, "Nz"]
    box_half_size: float


# Register as JAX pytrees
jax.tree_util.register_dataclass(
    FractalDensityLayer,
    data_fields=[
        "chi",
        "sigma_ln_rho",
        "lambda_frac",
        "grid_size",
        "box_size_factor",
        "use_log_normal",
        "virial_ratio",
        "base_profile",
        "sphere_radius_factor",
    ],
    meta_fields=[],
)

jax.tree_util.register_dataclass(
    DensityField3D,
    data_fields=["rho_grid", "x_grid", "y_grid", "z_grid", "box_half_size"],
    meta_fields=[],
)


# =============================================================================
# Helper Functions
# =============================================================================


def _make_hermitian(delta_k: jnp.ndarray) -> jnp.ndarray:
    """Enforce Hermitian symmetry so IFFT gives real output.

    For a real field δ(x), its Fourier transform satisfies:
        δ_k[i,j,k] = conj(δ_k[-i,-j,-k])

    This function symmetrizes an arbitrary complex array.

    Parameters
    ----------
    delta_k : Array, shape (Nx, Ny, Nz)
        Complex Fourier coefficients.

    Returns
    -------
    delta_k_sym : Array, shape (Nx, Ny, Nz)
        Hermitian-symmetric coefficients.
    """
    Nx, Ny, Nz = delta_k.shape

    # Create index arrays for the conjugate positions
    # Using wrap-around: -i = (Nx - i) % Nx
    ix = jnp.arange(Nx)
    iy = jnp.arange(Ny)
    iz = jnp.arange(Nz)

    # Conjugate indices
    ix_conj = (Nx - ix) % Nx
    iy_conj = (Ny - iy) % Ny
    iz_conj = (Nz - iz) % Nz

    # Get the conjugate values using meshgrid indexing
    IX, IY, IZ = jnp.meshgrid(ix_conj, iy_conj, iz_conj, indexing="ij")
    delta_k_conj = jnp.conj(delta_k[IX, IY, IZ])

    # Average to enforce symmetry: (δ_k + conj(δ_k[-i,-j,-k]))/2
    delta_k_sym = 0.5 * (delta_k + delta_k_conj)

    return delta_k_sym


def _plummer_density_grid(
    x_grid: Float[Array, "Nx"],
    y_grid: Float[Array, "Ny"],
    z_grid: Float[Array, "Nz"],
    R_half: float,
) -> Float[Array, "Nx Ny Nz"]:
    """Compute Plummer density on a 3D grid.

    ρ(r) ∝ (1 + r²/a²)^{-5/2}

    where a = R_half * sqrt((1 - 0.5^{2/3}) / 0.5^{2/3}) ≈ 0.7664 * R_half

    Parameters
    ----------
    x_grid, y_grid, z_grid : Array
        Grid coordinates in each dimension.
    R_half : float
        Half-mass radius in pc.

    Returns
    -------
    rho : Array, shape (Nx, Ny, Nz)
        Unnormalized Plummer density on grid.
    """
    # Scale radius from half-mass radius
    # r_h / a = (1 / sqrt(2^(2/3) - 1)) => a = r_h * sqrt(2^(2/3) - 1)
    a = R_half * jnp.sqrt(2 ** (2 / 3) - 1)

    # Build 3D radius grid
    X, Y, Z = jnp.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
    R2 = X**2 + Y**2 + Z**2

    # Plummer density (unnormalized)
    rho = (1.0 + R2 / a**2) ** (-2.5)

    return rho


def _uniform_sphere_mask(
    x_grid: Float[Array, "Nx"],
    y_grid: Float[Array, "Ny"],
    z_grid: Float[Array, "Nz"],
    R_sphere: float,
) -> Float[Array, "Nx Ny Nz"]:
    """Create a uniform density mask within a spherical region.

    ρ(r) = 1 if r < R_sphere, else 0

    Parameters
    ----------
    x_grid, y_grid, z_grid : Array
        Grid coordinates in each dimension.
    R_sphere : float
        Radius of the spherical region.

    Returns
    -------
    mask : Array, shape (Nx, Ny, Nz)
        Binary mask (1 inside sphere, 0 outside).
    """
    X, Y, Z = jnp.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
    R2 = X**2 + Y**2 + Z**2

    # Smooth boundary with tanh for differentiability (optional)
    # Using sharp boundary for now
    mask = (R2 < R_sphere**2).astype(jnp.float64)

    return mask


# =============================================================================
# Density Field Construction
# =============================================================================


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
    dx = 2 * L_box / Nx
    dV = dx**3

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


# =============================================================================
# Position Sampling
# =============================================================================


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


# =============================================================================
# Complete IC Generator
# =============================================================================


def generate_fractal_ic_density(
    key: PRNGKeyArray,
    N_stars: int,
    M_total: float,
    R_half: float,
    imf_params,
    layer: FractalDensityLayer,
    G: float = None,
):
    """Generate cluster IC using density-field fractal method.

    Parameters
    ----------
    key : PRNGKey
        JAX random key.
    N_stars : int
        Number of stars.
    M_total : float
        Total cluster mass in M_sun.
    R_half : float
        Half-mass radius in pc.
    imf_params
        IMF instance with .sample(key, n) method.
    layer : FractalDensityLayer
        Density field parameters.
    G : float, optional
        Gravitational constant. Uses jaxstro.units.STELLAR.G if None.

    Returns
    -------
    ClusterState
        Cluster with masses, positions, velocities.

    Notes
    -----
    The density field is constructed and frozen once via stop_gradient.
    Gradients flow through:
        - sigma_ln_rho (amplitude of density fluctuations)
        - lambda_frac (blend fraction) via blending
        - virial_ratio (velocity scaling)

    Gradients do NOT flow through:
        - Stochastic realization of the field (frozen structure)
        - Cell selection in position sampling (discrete)
    """
    from progenax.cluster.core import ClusterState
    from progenax.dynamics.virial import compute_potential_energy

    if G is None:
        from jaxstro.units import STELLAR

        G = STELLAR.G

    # Split keys
    key_imf, key_field, key_pos, key_vel = random.split(key, 4)

    # Step 1: Sample masses from IMF
    masses = imf_params.sample(key_imf, N_stars)
    masses = masses * (M_total / jnp.sum(masses))

    # Step 2: Initialize and freeze density field
    field = init_turbulent_density_field(key_field, R_half, layer)
    field = jax.tree_util.tree_map(jax.lax.stop_gradient, field)

    # Step 3: Sample positions from density field
    positions = sample_positions_from_density(key_pos, field, N_stars)

    # Step 4: Recenter to center of mass
    M_total_actual = jnp.sum(masses)
    x_com = jnp.sum(masses[:, None] * positions, axis=0) / M_total_actual
    positions = positions - x_com

    # Step 5: Sample velocities from equilibrium DF and rescale to virial ratio
    # Use Plummer velocities as baseline (similar to displacement FDF)
    from progenax.kinematics import PlummerVelocityDF

    df = PlummerVelocityDF(r_h=R_half)
    velocities = df.sample_velocities(positions, masses, key_vel, G=G)

    # Remove COM velocity
    v_com = jnp.sum(masses[:, None] * velocities, axis=0) / M_total_actual
    velocities = velocities - v_com

    # Rescale to target virial ratio
    U = compute_potential_energy(positions, masses, G)
    K_actual = 0.5 * jnp.sum(masses[:, None] * velocities**2)
    K_target = layer.virial_ratio * jnp.abs(U)

    scale = jnp.sqrt(K_target / jnp.maximum(K_actual, 1e-12))
    velocities = velocities * scale

    return ClusterState(
        masses=masses,
        positions=positions,
        velocities=velocities,
    )


# =============================================================================
# Calibration Helper
# =============================================================================


def density_layer_from_D(
    D: float,
    sigma_ln_rho: float,
    lambda_frac: float = 1.0,
    virial_ratio: float = 0.5,
    grid_size: int = 64,
    base_profile: str = "uniform",
) -> FractalDensityLayer:
    """Create FractalDensityLayer from GW-style D parameter.

    UNCALIBRATED: Use env_to_fdf_layer() for physics-based parameters.

    Parameters
    ----------
    D : float
        Target fractal dimension in [1.6, 3.0].
    sigma_ln_rho : float
        Amplitude of log-density fluctuations. REQUIRED - no default.
        Use env_to_fdf_layer() for physics-derived values (~1.1-1.5).
    lambda_frac : float, default 1.0
        Blend fraction [0, 1].
    virial_ratio : float, default 0.5
        Target Q_vir.
    grid_size : int, default 64
        Grid resolution per dimension.
    base_profile : str, default "uniform"
        Base density profile: "uniform" or "plummer".

    Returns
    -------
    FractalDensityLayer
        Configured layer (chi ≈ D, uncalibrated).
    """

    chi = jnp.clip(D, CHI_MIN, CHI_MAX)

    return FractalDensityLayer(
        chi=float(chi),  # Convert to float for dataclass
        sigma_ln_rho=sigma_ln_rho,
        lambda_frac=lambda_frac,
        base_profile=base_profile,
        virial_ratio=virial_ratio,
        grid_size=grid_size,
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Data structures
    "FractalDensityLayer",
    "DensityField3D",
    # Field operations
    "init_turbulent_density_field",
    "sample_positions_from_density",
    # IC Generator
    "generate_fractal_ic_density",
    # Calibration (DEPRECATED - use env_to_fdf_layer() instead)
    "density_layer_from_D",
]

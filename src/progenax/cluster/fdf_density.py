# progenax/src/progenax/cluster/fdf_density.py
"""
Fractal Density Field (FDF-D) for star cluster initial conditions.

This module implements a two-layer approach to generating substructured
star cluster initial conditions:

**Layer 1: Turbulent Gas Density Field (FractalDensityLayer)**

    Constructs a 3D gas density field from ISM turbulence physics:
    - σ_ln_ρ and β are derived from Larson + Federrath theory
    - χ is an INTERNAL turbulence shaping parameter, NOT a "fractal dimension"
    - This layer describes WHERE gas is dense, not where stars form

**Layer 2: Gravoturbulent Collapse Selection (TailSubstructureLayer)**

    Controls STELLAR substructure via f_sub:
    - f_sub = fraction of stars from dense tail of gas PDF
    - Higher f_sub → more substructured (clumpy) stellar distribution
    - This is the PRIMARY stellar substructure knob

Key Physics Separation
----------------------
- **Turbulence (σ_ln_ρ, β)**: Set by ISM physics (Larson + Federrath+2010)
- **Substructure (f_sub)**: Controlled by gravoturbulent collapse selection

χ and β are turbulence-shaping parameters, NOT clumpiness knobs.
Stellar substructure is controlled by f_sub in TailSubstructureLayer.

Position Sampling Methods
-------------------------
**Standard sampling** (sample_positions_from_density):
    Samples stars proportionally to the full density field.

**Tail sampling** (sample_positions_tail):
    Two-component sampling with dense tail + smooth component split.
    Stars allocated: N_dense ≈ f_sub × N_stars from dense regions.

Base Profile Choice
-------------------
The ``base_profile`` parameter affects Q interpretation:

- **"uniform"**: RECOMMENDED for CW04-comparable Q values
- **"plummer"**: Realistic clusters, but Q >> 1 due to central concentration

WARNING - Calibration Status
----------------------------
The spectral slopes and χ→β mapping are UNCALIBRATED heuristics.
For physics-based turbulence parameters, use ``env_to_fdf_layer()``.
The D→f_sub mapping is phenomenological, NOT calibrated to CW04 Q(D).

References
----------
- Federrath et al. (2010) A&A 512, A81 - Density-Mach relation, turbulence
- Goodwin & Whitworth (2004) A&A 413, 929 - Fractal dimension D
- Cartwright & Whitworth (2004) MNRAS 348, 589 - Q parameter definition
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import random
from jax import Array
from jaxtyping import Float, PRNGKeyArray

from progenax import defaults


# =============================================================================
# Heuristics Import (quarantined - not physics-derived)
# =============================================================================

from progenax.cluster.fdf_config import FDF_HEURISTICS, CHI_MIN, CHI_MAX

if TYPE_CHECKING:
    from progenax.cluster.fdf_config import GravoturbulentEnv, GravoturbulentResult
    from progenax.gravoturb.bm19_model import BM19Result


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
    """Parameters for fractal density field layer (TURBULENCE LAYER).

    This controls the GAS density field structure from ISM turbulence physics.
    It does NOT directly control stellar substructure - that is controlled by
    TailSubstructureLayer via f_sub.

    The parameters σ_ln_ρ and β (derived from chi) describe turbulence in the
    parent molecular cloud, NOT the "fractal dimension" of stellar positions.

    Attributes
    ----------
    chi : float
        INTERNAL turbulence shaping parameter in [1.6, 3.0]. Controls spectral
        slope of the gas density field. This is NOT a "fractal dimension" proxy.
        For stellar substructure control, use TailSubstructureLayer with f_sub.
    sigma_ln_rho : float
        Standard deviation of log-density field from Federrath+2010.
        σ²_ln_ρ = ln(1 + b²M²) where b is driving parameter, M is Mach number.
        Derived from environment via env_to_fdf_layer() for physical values.
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

    See Also
    --------
    TailSubstructureLayer : Controls stellar substructure via f_sub.
    env_to_fdf_layer : Derives physical σ_ln_ρ from environment.
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
class TailSubstructureLayer:
    """Gravoturbulent dense-tail substructure parameters.

    This controls STELLAR substructure by specifying what fraction of stars
    form in the densest, locally-collapsing regions of the turbulent gas field.

    This is separate from the turbulence layer (FractalDensityLayer):
    - Turbulence (σ_ln_ρ, β) sets the gas density PDF from ISM physics
    - Substructure (f_sub) determines which parts of that PDF form stars

    The physical motivation is gravoturbulent fragmentation: local regions
    with α_vir,loc << 1 undergo runaway collapse and form stars, while
    lower-density regions may be disrupted by feedback before star formation.

    Attributes
    ----------
    f_sub : float
        Fraction of stars sampled from the dense tail (0..1).
        This is the PRIMARY stellar substructure knob.

        Phenomenological defaults by cluster type:
        - f_sub ~ 0.15: loose stellar associations (low Σ, weak confinement)
        - f_sub ~ 0.30: typical open cluster birth conditions
        - f_sub ~ 0.55: young massive cluster (YMC) formation
        - f_sub ~ 0.70: extreme, globular cluster-like proto-clusters

        Higher f_sub → more stars from dense clumps → more substructured.

    mode : str
        How f_sub was determined (for provenance tracking):
        - "direct": User-specified value (default)
        - "bm19": Derived via BM19 pipeline (RECOMMENDED)
        - "gravoturbulent": Derived via legacy PN11 path (DEPRECATED)
        - "cluster_type": From phenomenological cluster type defaults
        - "D_mapping": From legacy fractal dimension D mapping

    env : GravoturbulentEnv | None
        If mode="bm19" or "gravoturbulent", the environment used for derivation.
        None for other modes.

    result : BM19Result | GravoturbulentResult | None
        If mode="bm19", the BM19Result with fields:
        (sigma_s, sigma_s_sq, s_t, f_dense, f_sub, beta, p, zeta).
        If mode="gravoturbulent", the legacy GravoturbulentResult.
        Useful for diagnostics and sensitivity analysis.
        None for other modes.

    Notes
    -----
    The dense tail is defined by MASS FRACTION in the gas density field,
    not by volume or cell count. This means f_sub=0.3 samples stars from
    the voxels containing the densest 30% of the gas mass.

    f_sub encodes unresolved physics: feedback efficiency, collision geometry,
    external pressure, magnetic support.

    For physics-based f_sub values, use tail_layer_from_env() which derives
    f_sub from the gravoturbulent framework (Burkhart 2018).

    See Also
    --------
    FractalDensityLayer : Controls turbulent gas density field.
    default_f_sub_for_cluster_type : Phenomenological defaults.
    f_sub_from_D : Maps GW-style D to f_sub (phenomenological).
    tail_layer_from_env : Physics-based f_sub from cloud environment.
    GravoturbulentEnv : Cloud environment parameters.
    """

    f_sub: float = 0.3  # Default "OC-like"
    mode: str = "direct"
    env: "GravoturbulentEnv | None" = None
    result: "BM19Result | GravoturbulentResult | None" = None


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
    TailSubstructureLayer,
    data_fields=["f_sub"],
    meta_fields=["mode", "env", "result"],  # Non-array fields as metadata
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


def _gaussian_blur_3d_fft(
    field: Float[Array, "Nx Ny Nz"],
    sigma_cells: float = 5.0,
) -> Float[Array, "Nx Ny Nz"]:
    """Apply 3D Gaussian blur using FFT convolution.

    This is used to compute local overdensity: ρ_local = ρ / ρ_smoothed.
    By smoothing the density field, we remove large-scale structure and
    keep only local fluctuations for tail selection.

    Parameters
    ----------
    field : Array, shape (Nx, Ny, Nz)
        3D density field to blur.
    sigma_cells : float
        Standard deviation of Gaussian in grid cells.
        Larger values → more smoothing → only large-scale structure remains.
        Default 5.0 cells is ~10% of typical 64-cell grid.

    Returns
    -------
    blurred : Array, shape (Nx, Ny, Nz)
        Smoothed density field.

    Notes
    -----
    Uses FFT-based convolution which is:
    1. Fully JAX-native (differentiable)
    2. Fast O(N log N) complexity
    3. Handles periodic boundaries naturally
    """
    Nx, Ny, Nz = field.shape

    # Create Gaussian kernel in Fourier space
    # For Gaussian g(x) ∝ exp(-x²/2σ²), the FT is also Gaussian:
    # G(k) ∝ exp(-σ²k²/2)
    kx = jnp.fft.fftfreq(Nx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(Ny) * 2 * jnp.pi
    kz = jnp.fft.fftfreq(Nz) * 2 * jnp.pi
    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing="ij")
    k2 = KX**2 + KY**2 + KZ**2

    # Gaussian filter in Fourier space
    gaussian_filter_k = jnp.exp(-0.5 * sigma_cells**2 * k2)

    # Apply filter via FFT convolution
    field_k = jnp.fft.fftn(field)
    blurred_k = field_k * gaussian_filter_k
    blurred = jnp.real(jnp.fft.ifftn(blurred_k))

    # Ensure positive (numerical noise can create tiny negatives)
    blurred = jnp.maximum(blurred, 1e-10)

    return blurred


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

    # Apply CDF remap: g -> u = Φ(g) -> s = F_V^{-1}(u)
    s_field = gaussian_to_bm19(g_standardized, sigma_s_sq, s_t, alpha, s_grid, F_grid)

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


def generate_fractal_ic_density(
    key: PRNGKeyArray,
    N_stars: int,
    M_total: float,
    R_half: float,
    imf_params,
    layer: FractalDensityLayer,
    tail: TailSubstructureLayer | None = None,
    env: GravoturbulentEnv | None = None,
    G: float = None,
):
    """Generate cluster IC using density-field fractal method.

    This function creates initial conditions for a star cluster by:
    1. Generating a turbulent gas density field (controlled by FractalDensityLayer)
    2. Sampling star positions from that field

    The position sampling can use either:
    - Pure density sampling (default, when tail=None and env=None)
    - Gravoturbulent tail sampling (when tail or env is provided)

    The tail sampling preferentially places stars in the densest regions of
    the gas field, controlled by f_sub in TailSubstructureLayer.

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
        Turbulence parameters for gas density field (σ_ln_ρ, β).
    tail : TailSubstructureLayer, optional
        Gravoturbulent substructure parameters. If provided, uses two-component
        dense tail + smooth sampling with f_sub controlling the split.
        If None (default), uses standard density-weighted sampling.
    env : GravoturbulentEnv, optional
        If provided, derives tail layer from gravoturbulent theory (Burkhart 2018).
        This OVERRIDES the `tail` parameter with a physics-derived f_sub.
        This is the RECOMMENDED interface when birth cloud properties are known.
    G : float, optional
        Gravitational constant. Uses progenax.DEFAULT_UNITS.G if None.

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

    Examples
    --------
    >>> # Standard density sampling (legacy behavior)
    >>> cluster = generate_fractal_ic_density(key, N, M, R, imf, layer)
    >>>
    >>> # With gravoturbulent tail sampling (direct f_sub)
    >>> tail = TailSubstructureLayer(f_sub=0.5)  # YMC-like
    >>> cluster = generate_fractal_ic_density(key, N, M, R, imf, layer, tail=tail)
    >>>
    >>> # With physics-derived f_sub from environment (RECOMMENDED)
    >>> from progenax.cluster.fdf_config import GravoturbulentEnv
    >>> env = GravoturbulentEnv(Sigma=1000, Mach=20, eta_survive=0.85)
    >>> cluster = generate_fractal_ic_density(key, N, M, R, imf, layer, env=env)
    """
    # If env provided, derive tail layer from gravoturbulent theory
    if env is not None:
        from progenax.cluster.fdf_config import tail_layer_from_env

        tail = tail_layer_from_env(env)
    from progenax.cluster.core import ClusterState
    from progenax.dynamics.virial import compute_potential_energy

    if G is None:
        G = defaults.DEFAULT_UNITS.G

    # Split keys
    key_imf, key_field, key_pos, key_vel = random.split(key, 4)

    # Step 1: Sample masses from IMF
    masses = imf_params.sample(key_imf, N_stars)
    masses = masses * (M_total / jnp.sum(masses))

    # Step 2: Initialize and freeze density field
    field = init_turbulent_density_field(key_field, R_half, layer)
    field = jax.tree_util.tree_map(jax.lax.stop_gradient, field)

    # Step 3: Sample positions from density field
    # Use tail sampling if TailSubstructureLayer provided, else standard sampling
    if tail is not None:
        # Get mode and s_t from tail layer
        tail_mode = tail.mode
        tail_s_t = None

        # For BM19/gravoturbulent modes, extract s_t from result
        if tail_mode in ("bm19", "gravoturbulent") and tail.result is not None:
            tail_s_t = float(tail.result.s_t)
            # Map "gravoturbulent" to "bm19" for sample_positions_tail
            if tail_mode == "gravoturbulent":
                tail_mode = "bm19"
        elif tail_mode == "direct" or tail_mode in ("cluster_type", "D_mapping"):
            # Direct/cluster_type/D_mapping modes have no BM19 result
            # Fall back to legacy mode
            tail_mode = "pn11_legacy"

        positions = sample_positions_tail(
            key_pos, field, N_stars, tail.f_sub,
            mode=tail_mode, s_t=tail_s_t
        )
    else:
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
    "TailSubstructureLayer",
    "DensityField3D",
    # Field operations
    "init_turbulent_density_field",
    "sample_positions_from_density",
    "sample_positions_tail",
    # IC Generator
    "generate_fractal_ic_density",
    # Calibration (DEPRECATED - use env_to_fdf_layer() instead)
    "density_layer_from_D",
]

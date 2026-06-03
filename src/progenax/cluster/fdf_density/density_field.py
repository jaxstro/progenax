"""Density-field data structures + FFT/grid helpers (split from fdf_density.py)."""

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



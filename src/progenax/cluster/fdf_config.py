# progenax/src/progenax/cluster/fdf_config.py
"""FDF hyperparameter configuration and turbulence-based parameter derivation.

This module provides:
1. Versioned hyperparameter dataclasses for FDF modules
2. Physically motivated helpers for deriving parameters from turbulence theory
3. Functions to convert BirthEnvironment → FDF parameters

All tunable constants are collected here for:
- Transparency about what's arbitrary vs calibrated
- Easy swapping between calibration versions
- Reproducible parameter tracking in papers

Turbulence Physics
------------------
Parameters are derived from ISM turbulence theory:

- **σ_ln_ρ** (density contrast): Federrath+2010 Eq. 14
    σ²_ln_ρ = ln(1 + b² M²)
    where M = Mach number, b = driving parameter (0.33 solenoidal, 1.0 compressive)

- **β** (spectral slope): Interpolates between regimes
    - Subsonic (M << 1): Kolmogorov β = 11/3 ≈ 3.67
    - Supersonic (M >> 1): Burgers β ≈ 4.0

- **Mach number**: Derived from Larson velocity-size relation
    M = σ_v / c_s where σ_v = σ_v0 × (R_cloud / 1 pc)^α
    R_cloud = (3 M_gas / 4π ρ_cl)^(1/3), Larson α ≈ 0.5
    c_s ~ 0.2 km/s for cold GMC (T ~ 10 K)

    NOTE: We use the PARENT CLOUD size, not stellar r_h, because the fractal
    density structure is imprinted by gas turbulence before star formation.

References
----------
- Federrath et al. (2010) A&A 512, A81 - Density-Mach relation
- Padoan & Nordlund (2002) ApJ 576, 870 - Turbulent fragmentation
- Larson (1981) MNRAS 194, 809 - Turbulent velocity scaling
- Burgers (1948) - Shock-dominated turbulence

WARNING
-------
The default hyperparameters (v0_uncalibrated) are heuristic placeholders.
They are NOT calibrated against:
- MHD simulations of turbulent clouds
- Observed molecular cloud spectra
- Cartwright & Whitworth (2004) Q(D) measurements

A calibration sweep is required before claiming "physically motivated" in papers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jaxtyping import Array, Float

if TYPE_CHECKING:
    from progenax.cluster.fdf_density import FractalDensityLayer


# =============================================================================
# Physical Constants
# =============================================================================

# Gravitational constant in km²/s²/pc/M☉ (for Mach calculation)
G_KMS = 4.3e-3

# Sound speed in cold GMC (T ~ 10 K) [km/s]
C_S_DEFAULT = 0.2

# Turbulence driving parameter (Federrath+2010)
# b ≈ 1/3 for solenoidal (incompressible) driving
# b ≈ 1.0 for compressive driving
# b ≈ 0.4 for natural mixture
B_DEFAULT = 0.4

# Spectral slopes from turbulence theory
BETA_KOLMOGOROV = 11.0 / 3.0  # ≈ 3.67, incompressible limit
BETA_BURGERS = 4.0  # Shock-dominated limit

# Larson velocity-size relation (Larson 1981, Solomon+1987)
# σ_v = σ_v0 × (R / 1 pc)^α
SIGMA_V0_DEFAULT = 1.0  # km/s at 1 pc
ALPHA_LARSON = 0.5  # Velocity-size exponent


# =============================================================================
# Versioned Hyperparameter Dataclasses
# =============================================================================


@dataclass(frozen=True)
class FDFDensityHyperparams:
    """Hyperparameters for density-field FDF (fdf_density.py).

    These control the spectral slope and amplitude of the lognormal
    density field. Currently UNCALIBRATED heuristics.

    Attributes
    ----------
    beta_0 : float
        Baseline spectral slope for δ field. Default 2.0.
        Physical reference:
            Kolmogorov: β ≈ 11/3 ≈ 3.67 (velocity)
            Burgers: β ≈ 4 (shock-dominated)
        Current value is arbitrary.
    beta_1 : float
        Sensitivity of β to chi parameter. Default 1.5.
        Maps: β = beta_0 + beta_1 × (χ - 1.5)
        This mapping is NOT calibrated.
    sigma_ln_rho_default : float
        Default amplitude of log-density fluctuations.
        Physical reference (Federrath+2010):
            σ²_ln_ρ = ln(1 + b² M²) where b ~ 0.3-0.5, M ~ 5-50
            Gives σ_ln_ρ ~ 1.0-2.5 for typical clusters
        Current default 2.0 is in reasonable range but arbitrary.
    version : str
        Version identifier for tracking calibration state.

    Notes
    -----
    Version history:
        - v0_uncalibrated (2024-12): Initial heuristics, NOT calibrated
    """

    beta_0: float = 2.0
    beta_1: float = 1.5
    sigma_ln_rho_default: float = 2.0
    version: str = "v0_uncalibrated"


@dataclass(frozen=True)
class FDFDisplacementHyperparams:
    """Hyperparameters for displacement-field FDF (fdf.py).

    These control the spectral shape and amplitude of the Fourier-mode
    displacement field. Currently UNCALIBRATED heuristics.

    Attributes
    ----------
    beta_base : float
        Mild baseline power-law slope. Default 1.5.
        Combined with lognormal envelope in k-space.
    sigma_logk : float
        Width of lognormal envelope in log-k space. Default 0.8.
        Controls the k-range where modes have significant power.
    sigma_u_default : float
        Default displacement amplitude scale (dimensionless).
        Actual RMS displacement ≈ sigma_u × R_half / √2.
        Typical values: 0.1-0.5.
    version : str
        Version identifier for tracking calibration state.

    Notes
    -----
    Version history:
        - v0_uncalibrated (2024-12): Initial heuristics, NOT calibrated
    """

    beta_base: float = 1.5
    sigma_logk: float = 0.8
    sigma_u_default: float = 0.3
    version: str = "v0_uncalibrated"


# Default instances
FDF_DENSITY_DEFAULTS = FDFDensityHyperparams()
FDF_DISPLACEMENT_DEFAULTS = FDFDisplacementHyperparams()


# =============================================================================
# Uncalibrated Heuristics (QUARANTINED)
# =============================================================================

# Chi parameter bounds (Goodwin & Whitworth 2004)
# 1.6 = most clumpy (small-scale dominated)
# 3.0 = smoothest (large-scale dominated)
CHI_MIN = 1.6
CHI_MAX = 3.0


@dataclass(frozen=True)
class FDFUncalibratedHeuristics:
    """HEURISTIC CONSTANTS - NOT PHYSICS-DERIVED.

    WARNING: These values are placeholders awaiting calibration against:
    - MHD turbulence simulations
    - Cartwright & Whitworth (2004) Q(D) measurements

    DO NOT use these in env_to_fdf_layer(). They exist only for:
    - Legacy API compatibility (density_layer_from_D)
    - Manual experimentation
    - Calibration sweeps

    For physics-based parameters, use env_to_fdf_layer() which derives
    σ_ln_ρ and β from turbulence theory (Federrath+2010, Kolmogorov/Burgers).

    Attributes
    ----------
    beta_0 : float
        Baseline spectral slope for legacy χ→β mapping.
        Formula: β = beta_0 + beta_1 × (χ - 1.5)
        NOT from physics - arbitrary heuristic.
    beta_1 : float
        Sensitivity of β to chi. NOT calibrated.
    sigma_ln_rho_manual : float
        Default σ_ln_ρ for manual FractalDensityLayer construction.
        Should use env_to_fdf_layer() instead which derives from Federrath+2010.
    beta_base_displacement : float
        Spectral slope for displacement FDF. NOT physics-based.
    sigma_logk_displacement : float
        Envelope width for displacement FDF. NOT physics-based.
    sigma_u_default : float
        Displacement amplitude scale. NOT physics-based.

    Notes
    -----
    CW04 calibration targets (NOT currently achieved):
        D=1.5 → Q ≈ 0.47
        D=2.0 → Q ≈ 0.58
        D=2.5 → Q ≈ 0.70
        D=3.0 → Q ≈ 0.79-0.82

    Current D→χ is identity mapping, produces UNKNOWN Q values.
    """

    # Legacy χ→β mapping (used when chi is set directly)
    beta_0: float = 2.0  # Baseline: arbitrary, NOT Kolmogorov
    beta_1: float = 1.5  # Sensitivity: arbitrary, NOT from physics

    # Default σ_ln_ρ for manual construction (bypasses Federrath+2010)
    sigma_ln_rho_manual: float = 2.0  # Arbitrary "visible substructure"

    # Displacement field (fdf.py) parameters - LEGACY/EXPERIMENTAL
    beta_base_displacement: float = 1.5
    sigma_logk_displacement: float = 0.8
    sigma_u_default: float = 0.3

    version: str = "v0_uncalibrated_2024-12"


# Singleton instance for heuristics
FDF_HEURISTICS = FDFUncalibratedHeuristics()


# =============================================================================
# Turbulence Physics Helpers
# =============================================================================


def sigma_ln_rho_from_mach(
    mach: Float[Array, "..."],
    b: float = B_DEFAULT,
) -> Float[Array, "..."]:
    """Density contrast from Mach number (Federrath+2010 Eq. 14).

    The variance of the log-density field in supersonic turbulence:
        σ²_ln_ρ = ln(1 + b² M²)

    This is a fundamental result from turbulent fragmentation theory,
    relating the density PDF width to the turbulent Mach number.

    Parameters
    ----------
    mach : array
        Turbulent Mach number M = σ_v / c_s.
    b : float, optional
        Turbulence driving parameter (default 0.4).
        - b ≈ 1/3 (0.33): Solenoidal (incompressible, rotational) driving
        - b ≈ 1.0: Compressive (irrotational) driving
        - b ≈ 0.4: Natural mixture (default for star-forming clouds)

    Returns
    -------
    sigma_ln_rho : array
        Standard deviation of log-density field.

    Notes
    -----
    Physical ranges:
        - Small OC (M ~ 5): σ_ln_ρ ~ 0.9
        - Large OC (M ~ 10): σ_ln_ρ ~ 1.4
        - YMC (M ~ 25): σ_ln_ρ ~ 1.9
        - GC (M ~ 50): σ_ln_ρ ~ 2.4

    References
    ----------
    .. [1] Federrath et al. (2010) A&A 512, A81, Eq. 14
    .. [2] Padoan & Nordlund (2002) ApJ 576, 870

    Examples
    --------
    >>> mach = jnp.array(10.0)  # Typical OC Mach number
    >>> sigma = sigma_ln_rho_from_mach(mach)
    >>> print(f"σ_ln_ρ = {float(sigma):.2f}")  # ~1.4
    """
    return jnp.sqrt(jnp.log(1.0 + b**2 * mach**2))


def spectral_slope_from_mach(mach: Float[Array, "..."]) -> Float[Array, "..."]:
    """Power spectrum slope from Mach number.

    Interpolates between two limiting regimes:
        - Subsonic (M << 1): Kolmogorov β = 11/3 ≈ 3.67
        - Supersonic (M >> 1): Burgers β ≈ 4.0

    Uses smooth tanh interpolation centered at M = 1.

    Parameters
    ----------
    mach : array
        Turbulent Mach number M = σ_v / c_s.

    Returns
    -------
    beta : array
        Power spectrum slope P(k) ∝ k^{-β}.

    Notes
    -----
    Physical interpretation:
        - Kolmogorov (incompressible): Energy cascade E(k) ∝ k^{-5/3}
          → velocity power spectrum P(k) ∝ k^{-11/3}
        - Burgers (shock-dominated): Velocity discontinuities → steeper spectrum
          P(k) ∝ k^{-4}

    For star-forming clouds with M >> 1, expect β ≈ 4.

    References
    ----------
    .. [1] Kolmogorov (1941) - Incompressible turbulence theory
    .. [2] Burgers (1948) - Shock-dominated turbulence
    .. [3] Federrath et al. (2010) A&A 512, A81 - ISM turbulence spectra

    Examples
    --------
    >>> mach_subsonic = jnp.array(0.1)
    >>> mach_supersonic = jnp.array(10.0)
    >>> print(f"β(M=0.1) = {float(spectral_slope_from_mach(mach_subsonic)):.2f}")  # ~3.67
    >>> print(f"β(M=10) = {float(spectral_slope_from_mach(mach_supersonic)):.2f}")  # ~4.0
    """
    # Smooth interpolation: t ∈ [0, 1] as M goes 0 → ∞
    # Transition width ~ 0.5 in Mach number (adjustable)
    t = 0.5 * (1.0 + jnp.tanh((mach - 1.0) / 0.5))
    return BETA_KOLMOGOROV * (1.0 - t) + BETA_BURGERS * t


# =============================================================================
# Larson Velocity-Size Relation (RECOMMENDED)
# =============================================================================


def cloud_radius_from_density(
    M_ecl: Float[Array, "..."],
    sfe: Float[Array, "..."],
    rho_cl: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Parent cloud radius [pc] from cluster mass, SFE, and cloud density.

    Derives the size of the parent molecular cloud that formed the cluster,
    assuming a spherical geometry:

        R_cloud = (3 M_gas / (4π ρ_cl))^(1/3)

    where M_gas = M_ecl / SFE is the gas mass.

    Parameters
    ----------
    M_ecl : array
        Stellar mass of embedded cluster [M☉].
    sfe : array
        Star formation efficiency ε = M_ecl / M_gas.
    rho_cl : array
        Cloud density [M☉ pc⁻³].

    Returns
    -------
    R_cloud : array
        Parent cloud radius [pc].

    Notes
    -----
    This is used to derive the turbulent velocity dispersion via the
    Larson velocity-size relation, which then gives the Mach number.

    The cloud radius is NOT the stellar half-mass radius r_h (which is
    much smaller). The fractal density structure is imprinted at the
    cloud scale, not the stellar scale.

    Expected ranges:
        | M_ecl   | SFE  | ρ_cl       | R_cloud |
        |---------|------|------------|---------|
        | 10³ M☉  | 0.33 | ~10³ M☉/pc³| ~2 pc   |
        | 10⁴ M☉  | 0.33 | ~10⁴ M☉/pc³| ~3 pc   |
        | 10⁵ M☉  | 0.33 | ~10⁵ M☉/pc³| ~4 pc   |
        | 10⁶ M☉  | 0.33 | ~10⁶ M☉/pc³| ~5 pc   |

    References
    ----------
    .. [1] Larson (1981) MNRAS 194, 809
    """
    M_gas = M_ecl / sfe
    volume = M_gas / rho_cl
    return jnp.power(3.0 * volume / (4.0 * jnp.pi), 1.0 / 3.0)


def larson_sigma_v(
    R_cloud: Float[Array, "..."],
    sigma_v0: float = SIGMA_V0_DEFAULT,
    alpha: float = ALPHA_LARSON,
) -> Float[Array, "..."]:
    """Velocity dispersion [km/s] from Larson velocity-size relation.

    The classic scaling of turbulent velocity dispersion with cloud size:

        σ_v(R) = σ_v0 × (R / 1 pc)^α

    Parameters
    ----------
    R_cloud : array
        Cloud radius [pc].
    sigma_v0 : float, optional
        Normalization velocity [km/s] at 1 pc. Default 1.0.
    alpha : float, optional
        Power-law exponent. Default 0.5 (Larson 1981).

    Returns
    -------
    sigma_v : array
        Turbulent velocity dispersion [km/s].

    Notes
    -----
    Typical values from literature:
        - Larson (1981): σ_v0 ≈ 1.1 km/s, α ≈ 0.38
        - Solomon et al. (1987): σ_v0 ≈ 0.72 km/s, α ≈ 0.5
        - Heyer & Brunt (2004): α ≈ 0.5 (with scatter)

    We use σ_v0 = 1.0 km/s, α = 0.5 as reasonable defaults.

    For R ~ 1-10 pc clouds:
        - R = 2 pc → σ_v ≈ 1.4 km/s
        - R = 5 pc → σ_v ≈ 2.2 km/s
        - R = 10 pc → σ_v ≈ 3.2 km/s

    References
    ----------
    .. [1] Larson (1981) MNRAS 194, 809
    .. [2] Solomon et al. (1987) ApJ 319, 730
    .. [3] Heyer & Brunt (2004) ApJ 615, L45
    """
    return sigma_v0 * jnp.power(jnp.maximum(R_cloud, 1e-3), alpha)


def turbulent_mach_from_cloud(
    R_cloud: Float[Array, "..."],
    c_s: float = C_S_DEFAULT,
    sigma_v0: float = SIGMA_V0_DEFAULT,
    alpha: float = ALPHA_LARSON,
) -> Float[Array, "..."]:
    """Gas turbulent Mach number from Larson velocity-size relation.

    Combines the Larson relation with the sound speed to get Mach number:

        M = σ_v(R_cloud) / c_s

    where σ_v = σ_v0 × (R_cloud)^α.

    This is the RECOMMENDED method for deriving Mach numbers because it uses
    the parent cloud properties (where turbulence imprints density structure),
    not the stellar half-mass radius.

    Parameters
    ----------
    R_cloud : array
        Parent cloud radius [pc].
    c_s : float, optional
        Sound speed [km/s]. Default 0.2 (cold GMC at T ~ 10 K).
    sigma_v0 : float, optional
        Larson normalization [km/s]. Default 1.0.
    alpha : float, optional
        Larson exponent. Default 0.5.

    Returns
    -------
    mach : array
        Turbulent Mach number M = σ_v / c_s.

    Notes
    -----
    Expected ranges for typical star-forming clouds:

        | R_cloud | σ_v    | Mach |
        |---------|--------|------|
        | 2 pc    | 1.4    | ~7   |
        | 3 pc    | 1.7    | ~9   |
        | 5 pc    | 2.2    | ~11  |
        | 10 pc   | 3.2    | ~16  |

    These are MUCH more realistic than virial-based estimates which give
    M ~ 20-400 for the same cluster masses.

    References
    ----------
    .. [1] Larson (1981) MNRAS 194, 809 - Velocity-size relation
    .. [2] Solomon et al. (1987) ApJ 319, 730 - GMC properties
    .. [3] Federrath et al. (2010) A&A 512, A81 - Turbulence-density relation

    Examples
    --------
    >>> R_cloud = jnp.array(3.0)  # 3 pc cloud
    >>> M = turbulent_mach_from_cloud(R_cloud)
    >>> print(f"Mach = {float(M):.1f}")  # ~8.7
    """
    sigma_v = larson_sigma_v(R_cloud, sigma_v0, alpha)
    return sigma_v / c_s


# =============================================================================
# Virial-Based Mach (DEPRECATED - gives unrealistic values)
# =============================================================================


def turbulent_mach_from_virial(
    M_ecl: Float[Array, "..."],
    r_h: Float[Array, "..."],
    c_s: float = C_S_DEFAULT,
) -> Float[Array, "..."]:
    """DEPRECATED: Estimate turbulent Mach number from virial equilibrium.

    .. deprecated::
        This function uses stellar r_h which gives unrealistically high Mach
        numbers (M ~ 20-400). Use `turbulent_mach_from_cloud()` instead, which
        uses the parent cloud radius via Larson's velocity-size relation and
        gives physically realistic Mach numbers (M ~ 5-15).

    Assumes the cluster's velocity dispersion follows virial scaling:
        σ_v = √(G M_ecl / r_h)

    Then Mach number:
        M = σ_v / c_s

    Parameters
    ----------
    M_ecl : array
        Stellar mass of embedded cluster [M☉].
    r_h : array
        Half-mass radius [pc].
    c_s : float, optional
        Sound speed [km/s]. Default 0.2 (cold GMC at T ~ 10 K).

    Returns
    -------
    mach : array
        Turbulent Mach number.

    Warning
    -------
    THIS GIVES UNREALISTIC MACH NUMBERS!

    The stellar half-mass radius r_h is NOT the relevant scale for gas turbulence.
    Using r_h gives M ~ 21-422 for 10³-10⁶ M☉ clusters, which is way too high.

    Use `turbulent_mach_from_cloud()` instead, which:
    1. Derives the parent cloud radius from (M_ecl, SFE, ρ_cl)
    2. Uses Larson's velocity-size relation: σ_v = σ_v0 × R^α
    3. Gives realistic M ~ 5-15

    See Also
    --------
    turbulent_mach_from_cloud : Recommended replacement using Larson relation.
    cloud_radius_from_density : Helper to derive cloud radius.
    """
    sigma_v = jnp.sqrt(G_KMS * M_ecl / r_h)
    return sigma_v / c_s


# =============================================================================
# Environment-Dependent Turbulence Driving Parameter
# =============================================================================


def b_from_environment(
    log_rho_cl: Float[Array, "..."],
    log_rho_transition: float = 4.0,
    b_low: float = 0.33,
    b_high: float = 0.7,
    width: float = 1.0,
) -> Float[Array, "..."]:
    """Turbulence driving parameter b from cloud density.

    Physical motivation (Federrath+2010, Federrath+2013):

    - Low-density clouds: Turbulence is primarily driven by large-scale flows
      (galactic shear, supernova feedback) which are more solenoidal. b ≈ 0.33.

    - High-density star-forming cores: Self-gravity and accretion flows create
      more compressive motions. Observed b ≈ 0.5-1.0 in dense regions.

    This function interpolates smoothly using tanh for JAX compatibility.

    Parameters
    ----------
    log_rho_cl : array
        Log₁₀ of cloud density [M☉/pc³].
    log_rho_transition : float, optional
        Density at which transition from solenoidal to compressive occurs.
        Default 10⁴ M☉/pc³ (typical dense core threshold).
    b_low : float, optional
        Driving parameter for low-density regions (solenoidal limit).
        Default 0.33 ≈ 1/3 from pure solenoidal driving.
    b_high : float, optional
        Driving parameter for high-density regions (compressive).
        Default 0.7 (moderate compressive, not pure compressive b=1).
    width : float, optional
        Width of transition in log₁₀(ρ) units. Default 1.0 dex.

    Returns
    -------
    b : array
        Turbulence driving parameter for Federrath+2010 formula.
        σ²_ln_ρ = ln(1 + b²M²)

    Notes
    -----
    The b parameter in the density-Mach relation (Federrath+2010 Eq. 14):

    - b ≈ 1/3 (0.33): Pure solenoidal (incompressible) driving
    - b ≈ 1.0: Pure compressive driving
    - b ≈ 0.4: Natural mixture (often used as default)

    This environment-dependent mapping is TENTATIVE. The relationship between
    cloud density and driving mechanism is not firmly established. Use with
    scientific caveat.

    References
    ----------
    .. [1] Federrath et al. (2010) A&A 512, A81 - Eq. 14
    .. [2] Federrath (2013) MNRAS 436, 1245 - Driving modes
    .. [3] Padoan & Nordlund (2011) ApJ 730, 40 - Turbulence in cores

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> # Low density cloud → solenoidal
    >>> b_low_cloud = b_from_environment(jnp.array(2.0))  # 100 M☉/pc³
    >>> print(f"b = {float(b_low_cloud):.2f}")  # ~0.33
    >>>
    >>> # High density core → more compressive
    >>> b_high_cloud = b_from_environment(jnp.array(6.0))  # 10⁶ M☉/pc³
    >>> print(f"b = {float(b_high_cloud):.2f}")  # ~0.70
    """
    # Smooth tanh interpolation for JAX compatibility
    t = 0.5 * (1.0 + jnp.tanh((log_rho_cl - log_rho_transition) / width))
    return b_low * (1.0 - t) + b_high * t


# =============================================================================
# BirthEnvironment → FDF Parameter Mapping
# =============================================================================


def env_to_fdf_layer(
    log_mecl: Float[Array, ""],
    sfe: Float[Array, ""] | None = None,
    log_rho_cl: Float[Array, ""] | None = None,
    b: float | None = None,
    c_s: float = C_S_DEFAULT,
    sigma_v0: float = SIGMA_V0_DEFAULT,
    alpha: float = ALPHA_LARSON,
    base_profile: str = "uniform",
    lambda_frac: float = 1.0,
    virial_ratio: float = 0.5,
) -> "FractalDensityLayer":
    """CANONICAL entry point: Environment → FDF parameters.

    This is the RECOMMENDED way to create a FractalDensityLayer.
    Parameters are derived from ISM turbulence physics:

    1. R_cloud from (M_ecl, SFE, ρ_cl) via spherical geometry
    2. σ_v from Larson velocity-size relation
    3. Mach = σ_v / c_s
    4. b from ρ_cl (if not provided) via b_from_environment()
    5. σ_ln_ρ from Federrath+2010: σ² = ln(1 + b²M²)
    6. β from Kolmogorov↔Burgers interpolation
    7. χ from β (inverse mapping, awaiting Q(D) calibration)

    This provides a unified physical model where:
        BirthEnvironment → IMF (via env_to_imf_params)
        BirthEnvironment → FDF (via this function)

    Both derived from the same cluster mass, metallicity, and SFE.

    Parameters
    ----------
    log_mecl : array
        log₁₀(M_ecl / M☉), cluster stellar mass.
    sfe : array, optional
        Star formation efficiency. Default 0.33.
    log_rho_cl : array, optional
        log₁₀(ρ_cl / M☉ pc⁻³), cloud density. If None, computed from
        M_ecl, SFE, and Marks+2012 r_h-M scaling.
    b : float or None, optional
        Turbulence driving parameter. If None (default), derived from
        cloud density via b_from_environment(). Accepts 0.33 (solenoidal)
        to 1.0 (compressive).
    c_s : float, optional
        Sound speed [km/s]. Default 0.2.
    sigma_v0 : float, optional
        Larson normalization [km/s]. Default 1.0.
    alpha : float, optional
        Larson exponent. Default 0.5.
    base_profile : str, optional
        Base density profile: "uniform" or "plummer".
        Use "uniform" for CW04-comparable Q values.
    lambda_frac : float, optional
        Fractal blend fraction [0, 1]. Default 1.0 (full turbulent).
    virial_ratio : float, optional
        Target virial ratio for velocities. Default 0.5 (equilibrium).

    Returns
    -------
    FractalDensityLayer
        FDF parameters with physically motivated σ_ln_ρ and χ.

    Notes
    -----
    Parameter derivation (using Larson velocity-size relation):
        1. M_ecl, SFE → M_gas = M_ecl / SFE
        2. M_gas, ρ_cl → R_cloud = (3 M_gas / 4π ρ_cl)^(1/3)
        3. R_cloud → σ_v via Larson: σ_v = σ_v0 × R^α
        4. σ_v → Mach: M = σ_v / c_s
        5. b from ρ_cl via b_from_environment() (if not provided)
        6. Mach, b → σ_ln_ρ via Federrath+2010
        7. Mach → β via Kolmogorov/Burgers interpolation
        8. β → χ via inverse of chi-beta mapping

    This uses the PARENT CLOUD size (not stellar r_h) because turbulence
    imprints the fractal density structure before star formation.

    Expected ranges (with Larson relation):
        | M_ecl   | R_cloud | Mach | σ_ln_ρ | χ    |
        |---------|---------|------|--------|------|
        | 10³ M☉  | ~1.5 pc | ~6   | ~1.1   | ~2.7 |
        | 10⁴ M☉  | ~2.5 pc | ~8   | ~1.3   | ~2.8 |
        | 10⁵ M☉  | ~4.0 pc | ~10  | ~1.4   | ~2.8 |
        | 10⁶ M☉  | ~6.5 pc | ~13  | ~1.6   | ~2.8 |

    Warning
    -------
    The β→χ mapping (step 8) is TENTATIVE and awaits calibration against
    Cartwright & Whitworth (2004) Q(D) measurements. The physics chain
    (steps 1-7) is well-grounded.

    Examples
    --------
    >>> from progenax.cluster.fdf_config import env_to_fdf_layer
    >>> import jax.numpy as jnp
    >>>
    >>> # Young massive cluster (10⁴ M☉) - b derived from environment
    >>> layer = env_to_fdf_layer(log_mecl=jnp.array(4.0))
    >>> print(f"σ_ln_ρ = {layer.sigma_ln_rho:.2f}")  # Physically motivated!
    >>>
    >>> # Override b for specific driving mode
    >>> layer_solenoidal = env_to_fdf_layer(log_mecl=jnp.array(4.0), b=0.33)
    >>> layer_compressive = env_to_fdf_layer(log_mecl=jnp.array(4.0), b=1.0)
    >>>
    >>> # Can use same environment for IMF:
    >>> from progenax.imf.environment import BirthEnvironment, env_to_imf_params
    >>> env = BirthEnvironment.from_cluster_mass(M_ecl=1e4, FeH=0.0)
    >>> imf_params = env_to_imf_params(env)  # Consistent!
    """
    # Avoid circular import
    from progenax.cluster.fdf_density import FractalDensityLayer
    from progenax.imf.environment import compute_rho_cl

    # Default SFE
    if sfe is None:
        sfe = jnp.array(0.33)

    # Step 1: Cluster mass
    M_ecl = jnp.power(10.0, log_mecl)

    # Step 2: Cloud density (either from input or computed via Marks+2012)
    if log_rho_cl is not None:
        rho_cl = jnp.power(10.0, log_rho_cl)
        log_rho_for_b = log_rho_cl
    else:
        # Use Marks+2012 r_h-M relation to get cloud density
        rho_cl = compute_rho_cl(M_ecl, sfe)
        log_rho_for_b = jnp.log10(rho_cl)

    # Step 3: Cloud radius from spherical geometry
    R_cloud = cloud_radius_from_density(M_ecl, sfe, rho_cl)

    # Step 4: Mach number from Larson velocity-size relation
    mach = turbulent_mach_from_cloud(R_cloud, c_s, sigma_v0, alpha)

    # Step 5: Derive b from environment if not provided
    if b is None:
        b_derived = float(b_from_environment(log_rho_for_b))
    else:
        b_derived = b

    # Step 6: σ_ln_ρ from Federrath+2010
    sigma_ln_rho = sigma_ln_rho_from_mach(mach, b_derived)

    # Step 7: Spectral slope from turbulence regime
    beta = spectral_slope_from_mach(mach)

    # Step 8: Map β → χ (inverse of chi→beta mapping)
    # Current heuristic: β = beta_0 + beta_1 × (χ - 1.5)
    #                    β = 2.0 + 1.5 × (χ - 1.5) = 1.5χ - 0.25
    # Inverse: χ = (β + 0.25) / 1.5
    # NOTE: This mapping will be replaced after Q(D) calibration
    chi = (beta + 0.25) / 1.5
    chi = jnp.clip(chi, CHI_MIN, CHI_MAX)

    return FractalDensityLayer(
        chi=float(chi),
        sigma_ln_rho=float(sigma_ln_rho),
        lambda_frac=lambda_frac,
        virial_ratio=virial_ratio,
        base_profile=base_profile,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Physical constants (CALIBRATED)
    "G_KMS",
    "C_S_DEFAULT",
    "B_DEFAULT",
    "BETA_KOLMOGOROV",
    "BETA_BURGERS",
    "SIGMA_V0_DEFAULT",
    "ALPHA_LARSON",
    # Chi parameter bounds
    "CHI_MIN",
    "CHI_MAX",
    # Hyperparameter dataclasses (legacy)
    "FDFDensityHyperparams",
    "FDFDisplacementHyperparams",
    "FDF_DENSITY_DEFAULTS",
    "FDF_DISPLACEMENT_DEFAULTS",
    # Uncalibrated heuristics (QUARANTINED - use with caution)
    "FDFUncalibratedHeuristics",
    "FDF_HEURISTICS",
    # Turbulence physics helpers (Federrath+2010)
    "sigma_ln_rho_from_mach",
    "spectral_slope_from_mach",
    # Larson velocity-size relation (RECOMMENDED)
    "cloud_radius_from_density",
    "larson_sigma_v",
    "turbulent_mach_from_cloud",
    # Virial-based Mach (DEPRECATED)
    "turbulent_mach_from_virial",
    # Environment-dependent turbulence driving (TENTATIVE)
    "b_from_environment",
    # Environment mapping (CANONICAL entry point)
    "env_to_fdf_layer",
]

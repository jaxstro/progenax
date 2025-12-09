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

- **Mach number**: Estimated from virial equilibrium
    M = σ_v / c_s where σ_v = √(G M_ecl / r_h)
    c_s ~ 0.2 km/s for cold GMC (T ~ 10 K)

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


def turbulent_mach_from_virial(
    M_ecl: Float[Array, "..."],
    r_h: Float[Array, "..."],
    c_s: float = C_S_DEFAULT,
) -> Float[Array, "..."]:
    """Estimate turbulent Mach number from virial equilibrium.

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

    Notes
    -----
    Physical ranges for typical clusters:
        - Small OC (10³ M☉, 0.3 pc): σ_v ~ 1 km/s, M ~ 5
        - Large OC (10⁴ M☉, 0.5 pc): σ_v ~ 2 km/s, M ~ 10
        - YMC (10⁵ M☉, 1.0 pc): σ_v ~ 5 km/s, M ~ 25
        - GC (10⁶ M☉, 3.0 pc): σ_v ~ 10 km/s, M ~ 50

    Warning
    -------
    This is a simplified estimate. Real clouds have:
    - Non-virial initial states
    - Magnetic support
    - External pressure confinement
    - Varying temperature structure

    References
    ----------
    .. [1] Larson (1981) MNRAS 194, 809 - σ-R relation
    .. [2] Marks & Kroupa (2012) MNRAS 422, 2246 - r_h-M relation

    Examples
    --------
    >>> M_ecl = jnp.array(1e4)  # 10⁴ M☉ cluster
    >>> r_h = jnp.array(0.5)    # 0.5 pc
    >>> M = turbulent_mach_from_virial(M_ecl, r_h)
    >>> print(f"Mach = {float(M):.1f}")  # ~14
    """
    sigma_v = jnp.sqrt(G_KMS * M_ecl / r_h)
    return sigma_v / c_s


# =============================================================================
# BirthEnvironment → FDF Parameter Mapping
# =============================================================================


def env_to_fdf_layer(
    log_mecl: Float[Array, ""],
    sfe: Float[Array, ""] | None = None,
    b: float = B_DEFAULT,
    c_s: float = C_S_DEFAULT,
    base_profile: str = "uniform",
    lambda_frac: float = 1.0,
    virial_ratio: float = 0.5,
) -> "FractalDensityLayer":
    """Convert environment parameters to physically motivated FDF layer.

    Derives turbulent parameters from the same environmental conditions
    that determine the IMF (Marks+2012, Jerabkova+2018).

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
    b : float, optional
        Turbulence driving parameter. Default 0.4.
    c_s : float, optional
        Sound speed [km/s]. Default 0.2.
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
    Parameter derivation:
        1. M_ecl → r_h via Marks+2012 scaling
        2. (M_ecl, r_h) → σ_v via virial theorem
        3. σ_v → Mach via M = σ_v / c_s
        4. Mach → σ_ln_ρ via Federrath+2010
        5. Mach → β via Kolmogorov/Burgers interpolation
        6. β → χ via inverse of chi-beta mapping

    Expected ranges:
        | M_ecl   | σ_ln_ρ | χ    |
        |---------|--------|------|
        | 10³ M☉  | ~0.9   | ~2.6 |
        | 10⁴ M☉  | ~1.4   | ~2.8 |
        | 10⁵ M☉  | ~1.9   | ~2.8 |
        | 10⁶ M☉  | ~2.4   | ~2.8 |

    Warning
    -------
    The χ(β) mapping is preliminary. The inverse mapping:
        χ = (β + 0.25) / 1.5
    is derived from the uncalibrated β = 2.0 + 1.5×(χ - 1.5) heuristic.
    This will be updated after proper Q(D) calibration.

    Examples
    --------
    >>> from progenax.cluster.fdf_config import env_to_fdf_layer
    >>> # Young massive cluster (10⁴ M☉)
    >>> layer = env_to_fdf_layer(log_mecl=jnp.array(4.0))
    >>> print(f"σ_ln_ρ = {layer.sigma_ln_rho:.2f}")  # Physically motivated!
    >>>
    >>> # Can use same environment for IMF:
    >>> from progenax.imf.environment import BirthEnvironment, env_to_imf_params
    >>> env = BirthEnvironment.from_cluster_mass(M_ecl=1e4, FeH=0.0)
    >>> imf_params = env_to_imf_params(env)  # Consistent!
    """
    # Avoid circular import
    from progenax.cluster.fdf_density import FractalDensityLayer
    from progenax.imf.environment import compute_r_half

    # Default SFE
    if sfe is None:
        sfe = jnp.array(0.33)

    # Step 1: Cluster mass and radius
    M_ecl = jnp.power(10.0, log_mecl)
    r_h = compute_r_half(M_ecl)

    # Step 2: Mach number from virial equilibrium
    mach = turbulent_mach_from_virial(M_ecl, r_h, c_s)

    # Step 3: σ_ln_ρ from Federrath+2010
    sigma_ln_rho = sigma_ln_rho_from_mach(mach, b)

    # Step 4: Spectral slope from turbulence regime
    beta = spectral_slope_from_mach(mach)

    # Step 5: Map β → χ (inverse of chi→beta mapping)
    # Current heuristic: β = beta_0 + beta_1 × (χ - 1.5)
    #                    β = 2.0 + 1.5 × (χ - 1.5) = 1.5χ - 0.25
    # Inverse: χ = (β + 0.25) / 1.5
    # NOTE: This mapping will be replaced after Q(D) calibration
    chi = (beta + 0.25) / 1.5
    chi = jnp.clip(chi, 1.6, 3.0)

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
    # Physical constants
    "G_KMS",
    "C_S_DEFAULT",
    "B_DEFAULT",
    "BETA_KOLMOGOROV",
    "BETA_BURGERS",
    # Hyperparameter dataclasses
    "FDFDensityHyperparams",
    "FDFDisplacementHyperparams",
    "FDF_DENSITY_DEFAULTS",
    "FDF_DISPLACEMENT_DEFAULTS",
    # Turbulence physics helpers
    "sigma_ln_rho_from_mach",
    "spectral_slope_from_mach",
    "turbulent_mach_from_virial",
    # Environment mapping
    "env_to_fdf_layer",
]

# progenax/src/progenax/cluster/turbulence.py
"""Turbulence physics: Federrath+2010, Larson velocity-size relation.

This module implements the turbulence physics that determines gas density
structure in star-forming clouds. All functions are JAX-compatible.

Key Relations
-------------
- Federrath+2010 Eq. 19: σ²_ln_ρ = ln(1 + b²M²)  (with Eq. 18 σ_ρ/⟨ρ⟩ = bM)
- Larson (1981): σ_v = σ_v0 × R^α
- Kim & Ryu (2005): the DENSITY power-spectrum slope flattens with Mach

References
----------
- Federrath et al. (2010) A&A 512, A81
- Larson (1981) MNRAS 194, 809
- Solomon et al. (1987) ApJ 319, 730
- Kim & Ryu (2005) ApJ 630, L45 - density power spectrum of compressible turbulence
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from progenax.cluster.constants import (
    ALPHA_LARSON,
    B_DEFAULT,
    BETA_DENSITY_FLOOR,
    BETA_KOLMOGOROV,
    C_S_DEFAULT,
    G_KMS,
    KIMRYU_BETA_INTERCEPT,
    KIMRYU_BETA_LOGSLOPE,
    SIGMA_V0_DEFAULT,
)

# =============================================================================
# Density-Mach Relation (Federrath+2010)
# =============================================================================


def sigma_ln_rho_from_mach(
    mach: Float[Array, "..."],
    b: float = B_DEFAULT,
) -> Float[Array, "..."]:
    """Density contrast from Mach number (Federrath+2010 Eq. 19).

    The variance of the log-density field in supersonic turbulence:
        σ²_ln_ρ = ln(1 + b²M²)

    This is FK10 Eq. 19 (the log form of the linear relation Eq. 18,
    σ_ρ/⟨ρ⟩ = b·M); the mean is fixed by mass conservation to ⟨s⟩ = -σ²_ln_ρ/2
    (FK10 Eq. 11). See the per-paper note federrath-2010 for the equation-number history.

    Parameters
    ----------
    mach : array
        Turbulent Mach number M = σ_v / c_s.
    b : float, optional
        Turbulence driving parameter (default 0.4).
        - b ≈ 1/3: Solenoidal (incompressible) driving
        - b ≈ 1.0: Compressive driving
        - b ≈ 0.4: Natural mixture

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
    .. [1] Federrath et al. (2010) A&A 512, A81, Eq. 19
    """
    return jnp.sqrt(jnp.log(1.0 + b**2 * mach**2))


def spectral_slope_from_mach(mach: Float[Array, "..."]) -> Float[Array, "..."]:
    """DENSITY power-spectrum slope β from Mach number (Kim & Ryu 2005).

    Returns the slope of the 3D density power spectrum P_3D(k) ∝ k^{-β}. In supersonic
    turbulence the *density* spectrum FLATTENS as the Mach number rises (mass is swept
    into sheets and filaments), so β **decreases** with M — the opposite of the velocity
    (Kolmogorov→Burgers) cascade.

    Calibration: Kim & Ryu (2005, ApJ 630, L45) measure 3D density spectra E_ρ(k)∝k^{-s}
    with s = 1.73 (M=1.2, ≈Kolmogorov 5/3), 1.08 (3.4), 0.75 (7.3), 0.52 (12). Converting
    to the 3D power-spectral-density convention β = s + 2 and least-squares fitting in
    log10(M):
        β(M) = 3.788 - 1.203·log10(M),  clipped to [2.0, 11/3].
    The Kolmogorov ceiling (11/3) is the transonic limit; the floor (2.0) is the 1D
    strong-shock density limit P_ρ ∝ k^0.

    Parameters
    ----------
    mach : array
        Turbulent Mach number M = σ_v / c_s.

    Returns
    -------
    beta : array
        Density power spectrum slope P_3D(k) ∝ k^{-β} (decreasing in M).

    Notes
    -----
    This is a log-linear fit to Kim & Ryu's four measured 3D Mach points (1.2–12);
    treat it as a calibrated interpolation, not a first-principles law. It is a *density*
    slope — distinct from the velocity Kolmogorov/Burgers slopes. Magnetic fields and
    self-gravity (not modelled by Kim & Ryu) modify it further.

    References
    ----------
    .. [1] Kim & Ryu (2005) ApJ 630, L45 - density power spectrum vs Mach
    .. [2] Kolmogorov (1941) - transonic ceiling (E_ρ slope ≈ -5/3)
    """
    beta = KIMRYU_BETA_INTERCEPT + KIMRYU_BETA_LOGSLOPE * jnp.log10(mach)
    return jnp.clip(beta, BETA_DENSITY_FLOOR, BETA_KOLMOGOROV)


# =============================================================================
# Larson Velocity-Size Relation
# =============================================================================


def cloud_radius_from_density(
    M_ecl: Float[Array, "..."],
    sfe: Float[Array, "..."],
    rho_cl: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Parent cloud radius [pc] from cluster mass, SFE, and cloud density.

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
    The cloud radius is used for Larson velocity-size relation.
    This is NOT the stellar half-mass radius r_h.

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

    σ_v(R) = σ_v0 × (R / 1 pc)^α

    Parameters
    ----------
    R_cloud : array
        Cloud radius [pc].
    sigma_v0 : float, optional
        Normalization velocity [km/s] at 1 pc. Default 1.0.
    alpha : float, optional
        Power-law exponent. Default 0.5.

    Returns
    -------
    sigma_v : array
        Turbulent velocity dispersion [km/s].

    Notes
    -----
    Typical literature values:
        - Larson (1981): σ_v0 ≈ 1.1, α ≈ 0.38
        - Solomon+1987: σ_v0 ≈ 0.72, α ≈ 0.5

    References
    ----------
    .. [1] Larson (1981) MNRAS 194, 809
    .. [2] Solomon et al. (1987) ApJ 319, 730
    """
    return sigma_v0 * jnp.power(jnp.maximum(R_cloud, 1e-3), alpha)


def turbulent_mach_from_cloud(
    R_cloud: Float[Array, "..."],
    c_s: float = C_S_DEFAULT,
    sigma_v0: float = SIGMA_V0_DEFAULT,
    alpha: float = ALPHA_LARSON,
) -> Float[Array, "..."]:
    """Gas turbulent Mach number from Larson velocity-size relation.

    M = σ_v(R_cloud) / c_s

    This is the RECOMMENDED method for deriving Mach numbers.

    Parameters
    ----------
    R_cloud : array
        Parent cloud radius [pc].
    c_s : float, optional
        Sound speed [km/s]. Default 0.2.
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
    Expected ranges:
        | R_cloud | Mach |
        |---------|------|
        | 2 pc    | ~7   |
        | 3 pc    | ~9   |
        | 5 pc    | ~11  |
        | 10 pc   | ~16  |

    References
    ----------
    .. [1] Larson (1981) MNRAS 194, 809
    .. [2] Federrath et al. (2010) A&A 512, A81
    """
    sigma_v = larson_sigma_v(R_cloud, sigma_v0, alpha)
    return sigma_v / c_s


def turbulent_mach_from_virial(
    M_ecl: Float[Array, "..."],
    r_h: Float[Array, "..."],
    c_s: float = C_S_DEFAULT,
) -> Float[Array, "..."]:
    """DEPRECATED: Estimate turbulent Mach from virial equilibrium.

    .. deprecated::
        This uses stellar r_h, giving unrealistically high Mach numbers.
        Use `turbulent_mach_from_cloud()` instead.

    Parameters
    ----------
    M_ecl : array
        Stellar mass [M☉].
    r_h : array
        Half-mass radius [pc].
    c_s : float, optional
        Sound speed [km/s]. Default 0.2.

    Returns
    -------
    mach : array
        Turbulent Mach number (WARNING: unrealistically high).
    """
    sigma_v = jnp.sqrt(G_KMS * M_ecl / r_h)
    return sigma_v / c_s


# =============================================================================
# Environment-Dependent Driving Parameter
# =============================================================================


def b_from_environment(
    log_rho_cl: Float[Array, "..."],
    log_rho_transition: float = 4.0,
    b_low: float = 0.33,
    b_high: float = 0.7,
    width: float = 1.0,
) -> Float[Array, "..."]:
    """Turbulence driving parameter b from cloud density.

    Interpolates:
        - Low-density: solenoidal driving (b ≈ 0.33)
        - High-density: compressive driving (b ≈ 0.7)

    Parameters
    ----------
    log_rho_cl : array
        Log₁₀ of cloud density [M☉/pc³].
    log_rho_transition : float, optional
        Transition density (default 10⁴ M☉/pc³).
    b_low : float, optional
        Low-density b value (default 0.33).
    b_high : float, optional
        High-density b value (default 0.7).
    width : float, optional
        Transition width in dex (default 1.0).

    Returns
    -------
    b : array
        Turbulence driving parameter.

    Notes
    -----
    This mapping is TENTATIVE. The relationship between density and
    driving mode is not firmly established.

    References
    ----------
    .. [1] Federrath et al. (2010) A&A 512, A81
    .. [2] Federrath (2013) MNRAS 436, 1245
    """
    t = 0.5 * (1.0 + jnp.tanh((log_rho_cl - log_rho_transition) / width))
    return b_low * (1.0 - t) + b_high * t


__all__ = [
    "sigma_ln_rho_from_mach",
    "spectral_slope_from_mach",
    "cloud_radius_from_density",
    "larson_sigma_v",
    "turbulent_mach_from_cloud",
    "turbulent_mach_from_virial",
    "b_from_environment",
]

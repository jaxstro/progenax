# progenax/src/progenax/cluster/fdf_config.py
"""FDF configuration: environment mapping and phenomenological helpers.

This module provides:
1. env_to_fdf_layer(): Map environment to FDF parameters (CANONICAL entry point)
2. Phenomenological helpers for f_sub defaults (cluster types, D→f_sub)

Two-Layer Architecture
----------------------
**Layer 1: Turbulent Gas Density Field**
    Controlled by ``env_to_fdf_layer()`` which derives σ_ln_ρ and β
    from turbulence physics (Federrath+2010, Larson relation).

**Layer 2: Gravoturbulent Collapse Selection**
    Controlled by f_sub. Use physics-based ``tail_layer_from_env()``
    or phenomenological helpers for defaults.

See Also
--------
progenax.cluster.constants : Physical constants
progenax.cluster.turbulence : Turbulence physics helpers
progenax.cluster.gravoturbulent : GravoturbulentEnv, tail_layer_from_env

References
----------
- Federrath et al. (2010) A&A 512, A81 - Density-Mach relation
- Larson (1981) MNRAS 194, 809 - Velocity-size relation
- Burkhart & Mocz (2019) ApJ 879, 129 - BM19 framework
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jaxtyping import Array, Float

# Import from split modules (re-export for backward compatibility)
from progenax.cluster.constants import (
    G_KMS,
    C_S_DEFAULT,
    B_DEFAULT,
    BETA_KOLMOGOROV,
    BETA_BURGERS,
    SIGMA_V0_DEFAULT,
    ALPHA_LARSON,
    CHI_MIN,
    CHI_MAX,
)

from progenax.cluster.turbulence import (
    sigma_ln_rho_from_mach,
    spectral_slope_from_mach,
    cloud_radius_from_density,
    larson_sigma_v,
    turbulent_mach_from_cloud,
    turbulent_mach_from_virial,
    b_from_environment,
)

from progenax.cluster.fdf_hyperparams import (
    FDFDensityHyperparams,
    FDFDisplacementHyperparams,
    FDFUncalibratedHeuristics,
    FDF_DENSITY_DEFAULTS,
    FDF_DISPLACEMENT_DEFAULTS,
    FDF_HEURISTICS,
)

from progenax.cluster.gravoturbulent import (
    GravoturbulentEnv,
    TailSelectionConfig,
    GRAVOTURBULENT_PRESETS,
    env_from_preset,
    tail_layer_from_env,
)

if TYPE_CHECKING:
    from progenax.cluster.fdf_density import FractalDensityLayer, TailSubstructureLayer


# =============================================================================
# Environment → FDF Parameter Mapping (CANONICAL)
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

    Derives FractalDensityLayer from ISM turbulence physics:
    1. R_cloud from (M_ecl, SFE, ρ_cl)
    2. σ_v from Larson velocity-size relation
    3. Mach = σ_v / c_s
    4. σ_ln_ρ from Federrath+2010
    5. β from Kolmogorov↔Burgers interpolation

    Parameters
    ----------
    log_mecl : array
        log₁₀(M_ecl / M☉), cluster stellar mass.
    sfe : array, optional
        Star formation efficiency. Default 0.33.
    log_rho_cl : array, optional
        log₁₀(ρ_cl / M☉ pc⁻³), cloud density.
    b : float or None, optional
        Turbulence driving parameter. If None, derived from density.
    c_s : float, optional
        Sound speed [km/s]. Default 0.2.
    sigma_v0 : float, optional
        Larson normalization [km/s]. Default 1.0.
    alpha : float, optional
        Larson exponent. Default 0.5.
    base_profile : str, optional
        Base density profile: "uniform" or "plummer".
    lambda_frac : float, optional
        Fractal blend fraction [0, 1]. Default 1.0.
    virial_ratio : float, optional
        Target virial ratio. Default 0.5.

    Returns
    -------
    FractalDensityLayer
        FDF parameters with physically motivated σ_ln_ρ and χ.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> layer = env_to_fdf_layer(log_mecl=jnp.array(4.0))
    >>> print(f"σ_ln_ρ = {layer.sigma_ln_rho:.2f}")
    """
    from progenax.cluster.fdf_density import FractalDensityLayer
    from progenax.imf.environment import compute_rho_cl

    # Default SFE
    if sfe is None:
        sfe = jnp.array(0.33)

    # Cluster mass
    M_ecl = jnp.power(10.0, log_mecl)

    # Cloud density
    if log_rho_cl is not None:
        rho_cl = jnp.power(10.0, log_rho_cl)
        log_rho_for_b = log_rho_cl
    else:
        rho_cl = compute_rho_cl(M_ecl, sfe)
        log_rho_for_b = jnp.log10(rho_cl)

    # Cloud radius
    R_cloud = cloud_radius_from_density(M_ecl, sfe, rho_cl)

    # Mach number
    mach = turbulent_mach_from_cloud(R_cloud, c_s, sigma_v0, alpha)

    # Derive b if not provided
    if b is None:
        b_derived = float(b_from_environment(log_rho_for_b))
    else:
        b_derived = b

    # σ_ln_ρ from Federrath+2010
    sigma_ln_rho = sigma_ln_rho_from_mach(mach, b_derived)

    # Spectral slope
    beta = spectral_slope_from_mach(mach)

    # β → χ mapping (tentative)
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
# Phenomenological Helpers (Legacy Interface)
# =============================================================================


def default_f_sub_for_cluster_type(cluster_type: str) -> float:
    """Phenomenological dense-tail fraction by cluster type.

    Parameters
    ----------
    cluster_type : str
        One of: "assoc", "oc", "ymc", "gc" (case-insensitive).

    Returns
    -------
    f_sub : float
        Default dense-tail fraction in [0, 1].

    Notes
    -----
    Values: assoc=0.15, oc=0.30, ymc=0.55, gc=0.70
    These are NOT calibrated to simulations.
    """
    mapping = {
        "assoc": 0.15,
        "oc": 0.30,
        "ymc": 0.55,
        "gc": 0.70,
    }
    return mapping.get(cluster_type.lower(), 0.30)


def tail_layer_from_cluster_type(cluster_type: str) -> "TailSubstructureLayer":
    """Create TailSubstructureLayer with default f_sub for cluster type.

    Parameters
    ----------
    cluster_type : str
        One of: "assoc", "oc", "ymc", "gc".

    Returns
    -------
    TailSubstructureLayer
        Configured with default f_sub for that cluster type.
    """
    from progenax.cluster.fdf_density import TailSubstructureLayer

    return TailSubstructureLayer(f_sub=default_f_sub_for_cluster_type(cluster_type))


def f_sub_from_D(D: float) -> float:
    """Map GW-style fractal dimension D to dense tail fraction f_sub.

    Linear mapping: D=1.5 → f_sub=0.70, D=3.0 → f_sub=0.15

    WARNING: NOT calibrated to reproduce CW04 Q(D).

    Parameters
    ----------
    D : float
        Fractal dimension in [1.5, 3.0].

    Returns
    -------
    f_sub : float
        Dense tail fraction in [0.15, 0.70].
    """
    D_clamped = jnp.clip(D, 1.5, 3.0)
    f_high, f_low = 0.70, 0.15
    t = (3.0 - D_clamped) / (3.0 - 1.5)
    return f_low + (f_high - f_low) * t


def tail_layer_from_D(D: float) -> "TailSubstructureLayer":
    """Create TailSubstructureLayer from GW-style fractal dimension D.

    Parameters
    ----------
    D : float
        Fractal dimension in [1.5, 3.0].

    Returns
    -------
    TailSubstructureLayer
        Configured with f_sub derived from D.
    """
    from progenax.cluster.fdf_density import TailSubstructureLayer

    return TailSubstructureLayer(f_sub=float(f_sub_from_D(D)))


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Physical constants (from constants.py)
    "G_KMS",
    "C_S_DEFAULT",
    "B_DEFAULT",
    "BETA_KOLMOGOROV",
    "BETA_BURGERS",
    "SIGMA_V0_DEFAULT",
    "ALPHA_LARSON",
    "CHI_MIN",
    "CHI_MAX",
    # Hyperparameters (from fdf_hyperparams.py)
    "FDFDensityHyperparams",
    "FDFDisplacementHyperparams",
    "FDFUncalibratedHeuristics",
    "FDF_DENSITY_DEFAULTS",
    "FDF_DISPLACEMENT_DEFAULTS",
    "FDF_HEURISTICS",
    # Turbulence physics (from turbulence.py)
    "sigma_ln_rho_from_mach",
    "spectral_slope_from_mach",
    "cloud_radius_from_density",
    "larson_sigma_v",
    "turbulent_mach_from_cloud",
    "turbulent_mach_from_virial",
    "b_from_environment",
    # Gravoturbulent (from gravoturbulent.py)
    "GravoturbulentEnv",
    "TailSelectionConfig",
    "GRAVOTURBULENT_PRESETS",
    "env_from_preset",
    "tail_layer_from_env",
    # Environment mapping (this file)
    "env_to_fdf_layer",
    # Phenomenological helpers (this file)
    "default_f_sub_for_cluster_type",
    "tail_layer_from_cluster_type",
    "f_sub_from_D",
    "tail_layer_from_D",
]

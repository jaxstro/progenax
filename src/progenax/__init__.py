"""
progenax - Differentiable initial conditions for N-body simulations.

Part of the jaxstro ecosystem.
"""

# Configure JAX for 64-bit precision BEFORE any other imports
# This must be called before any JAX arrays are created
from jaxstro.jaxconfig import enable_high_precision as _enable_jax_hp

_enable_jax_hp()
del _enable_jax_hp  # avoid leaking into public API

from .defaults import DEFAULT_UNITS
from .protocols import SpatialProfile, VelocityDF, IMFProtocol
from .profiles import PlummerProfile, KingProfile, EFFProfile, solve_king_profile
from .kinematics import PlummerVelocityDF, KingVelocityDF, EFFVelocityDF
from .builders import (
    ICResult,
    build_spatial_ic,
    to_com_frame,
    virial_scale,
    compute_stellar_radii,
    compute_kinetic_energy,
    compute_potential_energy,
)
from .analytical import (
    AnalyticalIC,
    SOLAR_SYSTEM_PLANETS,
    get_planet,
    two_body_kepler,
    two_body_period,
    two_body_energy,
    three_body_figure_eight,
    figure_eight_period,
    harmonic_oscillator,
    harmonic_solution,
    earth_sun_2body,
    earth_sun_eccentric,
    sun_earth_jupiter_3body,
    solar_system_inner_4,
    solar_system_full,
)
from .binaries import (
    KeplerElements,
    compute_period,
    period_to_semimajor_axis,
    BinaryOrbitalState,
    batch_elements_to_resolved,
)
from .tidal import (
    jacobi_radius,
    jacobi_radius_isothermal,
    apply_tidal_truncation,
    fill_factor_to_r_h,
)
# Legacy GW2004 fractal (deprecated - use FDF instead)
from .cluster.fractal_gw_legacy import (
    generate_fractal_positions,
    rescale_fractal_to_target_radii,
    assign_velocities_and_virialize,
)
from .populations import (
    TwoComponentConfig,
    generate_two_component_cluster,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_UNITS",
    # Protocols
    "SpatialProfile",
    "VelocityDF",
    "IMFProtocol",
    # Spatial density profiles
    "PlummerProfile",
    "KingProfile",
    "EFFProfile",
    "solve_king_profile",
    # Velocity distribution functions
    "PlummerVelocityDF",
    "KingVelocityDF",
    "EFFVelocityDF",
    # Builders
    "ICResult",
    "build_spatial_ic",
    "to_com_frame",
    "virial_scale",
    "compute_stellar_radii",
    "compute_kinetic_energy",
    "compute_potential_energy",
    # Analytical test cases
    "AnalyticalIC",
    "SOLAR_SYSTEM_PLANETS",
    "get_planet",
    "two_body_kepler",
    "two_body_period",
    "two_body_energy",
    "three_body_figure_eight",
    "figure_eight_period",
    "harmonic_oscillator",
    "harmonic_solution",
    "earth_sun_2body",
    "earth_sun_eccentric",
    "sun_earth_jupiter_3body",
    "solar_system_inner_4",
    "solar_system_full",
    # Binary orbital mechanics
    "KeplerElements",
    "compute_period",
    "period_to_semimajor_axis",
    "BinaryOrbitalState",
    "batch_elements_to_resolved",
    # Tidal physics
    "jacobi_radius",
    "jacobi_radius_isothermal",
    "apply_tidal_truncation",
    "fill_factor_to_r_h",
    # Fractal substructure
    "generate_fractal_positions",
    "rescale_fractal_to_target_radii",
    "assign_velocities_and_virialize",
    # Two-component populations
    "TwoComponentConfig",
    "generate_two_component_cluster",
]

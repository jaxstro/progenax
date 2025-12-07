"""
progenax - Differentiable initial conditions for N-body simulations.

Part of the jaxstro ecosystem.
"""

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

__version__ = "0.1.0"

__all__ = [
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
]

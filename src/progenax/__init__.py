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
from .protocols import (
    SpatialProfile,
    VelocityDF,
    IMFProtocol,
    PeriodDistribution,
    EccentricityDistribution,
    ConditionalEccentricityDistribution,
    MassPeriodEccentricityDistribution,
)
from .profiles import (
    PlummerProfile, KingProfile, MichieProfile, EFFProfile,
    solve_king_profile, solve_michie_profile,
)
from .kinematics import PlummerVelocityDF, KingVelocityDF, MichieVelocityDF, EFFVelocityDF
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
    CartesianState,
    BinaryState,
    compute_period,
    period_to_semimajor_axis,
    BinaryOrbitalState,
    batch_elements_to_resolved,
    LogUniformPeriod,
    LogNormalPeriod,
    SanaOBPeriod,
    ThermalEccentricity,
    UniformEccentricity,
    LogisticThermalEccentricity,
    MoeEccentricity,
    sample_isotropic_orientations,
    RadialBinaryFraction,
    MassDependentBinaryConfig,
    sample_mass_dependent_orbits,
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
from .cluster import energy_sorted_segregation
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
    "PeriodDistribution",
    "EccentricityDistribution",
    "ConditionalEccentricityDistribution",
    "MassPeriodEccentricityDistribution",
    # Spatial density profiles
    "PlummerProfile",
    "KingProfile",
    "MichieProfile",
    "EFFProfile",
    "solve_king_profile",
    "solve_michie_profile",
    # Velocity distribution functions
    "PlummerVelocityDF",
    "KingVelocityDF",
    "MichieVelocityDF",
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
    "CartesianState",
    "BinaryState",
    "compute_period",
    "period_to_semimajor_axis",
    "BinaryOrbitalState",
    "batch_elements_to_resolved",
    # Binary population distributions
    "LogUniformPeriod",
    "LogNormalPeriod",
    "SanaOBPeriod",
    "ThermalEccentricity",
    "UniformEccentricity",
    "LogisticThermalEccentricity",
    "MoeEccentricity",
    "sample_isotropic_orientations",
    "RadialBinaryFraction",
    "MassDependentBinaryConfig",
    "sample_mass_dependent_orbits",
    # Tidal physics
    "jacobi_radius",
    "jacobi_radius_isothermal",
    "apply_tidal_truncation",
    "fill_factor_to_r_h",
    # Fractal substructure
    "generate_fractal_positions",
    "rescale_fractal_to_target_radii",
    "assign_velocities_and_virialize",
    # Mass segregation
    "energy_sorted_segregation",
    # Two-component populations
    "TwoComponentConfig",
    "generate_two_component_cluster",
]

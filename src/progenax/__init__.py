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
    BinaryFractionModel,
    CompanionModel,
)
from .profiles import (
    PlummerProfile, KingProfile, MichieProfile, EFFProfile,
    LIMEPYProfile,
    solve_king_profile, solve_michie_profile, solve_limepy_profile,
    solve_multimass_limepy, find_alpha_for_masses,
)
from .kinematics import (
    PlummerVelocityDF, KingVelocityDF, MichieVelocityDF, EFFVelocityDF,
    LIMEPYVelocityDF,
)
from .builders import (
    ICResult,
    Systems,
    Stars,
    TotalMass,
    build_spatial_ic,
    build_binary_cluster,
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
    CombinedBinaryFraction,
    MassDependentBinaryConfig,
    sample_mass_dependent_orbits,
    ResolvedBinaries,
    resolve_binary_components,
    CompanionElements,
    IndependentCompanions,
    MoeCompanions,
    relative_energy,
    find_bound_pairs,
    find_bound_multiples,
    primordial_survival,
    BinaryEnergyBudget,
    binary_energy_budget,
)
from .tidal import (
    jacobi_radius,
    jacobi_radius_isothermal,
    apply_tidal_truncation,
    fill_factor_to_r_h,
)
from .cluster import MultiComponentCluster, energy_sorted_segregation

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
    "BinaryFractionModel",
    "CompanionModel",
    # Spatial density profiles
    "PlummerProfile",
    "KingProfile",
    "MichieProfile",
    "EFFProfile",
    "LIMEPYProfile",
    "MultiComponentCluster",
    "solve_king_profile",
    "solve_michie_profile",
    "solve_limepy_profile",
    "solve_multimass_limepy",
    "find_alpha_for_masses",
    # Velocity distribution functions
    "PlummerVelocityDF",
    "KingVelocityDF",
    "MichieVelocityDF",
    "LIMEPYVelocityDF",
    "EFFVelocityDF",
    # Builders
    "ICResult",
    "Systems",
    "Stars",
    "TotalMass",
    "build_spatial_ic",
    "build_binary_cluster",
    "resolve_binary_components",
    "ResolvedBinaries",
    "CompanionElements",
    "IndependentCompanions",
    "MoeCompanions",
    "relative_energy",
    "find_bound_pairs",
    "find_bound_multiples",
    "primordial_survival",
    "BinaryEnergyBudget",
    "binary_energy_budget",
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
    "CombinedBinaryFraction",
    "MassDependentBinaryConfig",
    "sample_mass_dependent_orbits",
    # Tidal physics
    "jacobi_radius",
    "jacobi_radius_isothermal",
    "apply_tidal_truncation",
    "fill_factor_to_r_h",
    # Primordial (non-equilibrium) mass segregation
    "energy_sorted_segregation",
]

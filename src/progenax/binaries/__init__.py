"""Binary star orbital mechanics for progenax.

Kepler mechanics, binary orbital state, and population parameter distributions.
All functions take an explicit G parameter (NO get_G() defaults).

Modules:
    kepler:         KeplerElements (forward state machinery), CartesianState, BinaryState
    kepler_inverse: orbital_elements_from_state (rv -> elements)
    kepler_period:  compute_period, period_to_semimajor_axis (Kepler-III conversions)
    orbital_state:  BinaryOrbitalState, batch operations
    period:         LogUniformPeriod, LogNormalPeriod, SanaOBPeriod
    eccentricity:   ThermalEccentricity, UniformEccentricity, MoeEccentricity
    orientation:    sample_isotropic_orientations
    mass_dependent: RadialBinaryFraction, MassDependentBinaryConfig,
                    sample_mass_dependent_orbits
"""

from .assembly import ResolvedBinaries, resolve_binary_components
from .catalog import (
    CatalogedBinaryClusterIC,
    PrimordialSystemCatalog,
    periapsis_contact_margin,
    validate_primordial_system_catalog,
)
from .companions import (
    CompanionElements,
    IndependentCompanions,
    MoeCompanions,
)
from .diagnostics import (
    BinaryEnergyBudget,
    binary_energy_budget,
    find_bound_multiples,
    find_bound_pairs,
    primordial_survival,
    relative_energy,
)
from .eccentricity import (
    LogisticThermalEccentricity,
    MoeEccentricity,
    ThermalEccentricity,
    UniformEccentricity,
)
from .kepler import (
    BinaryState,
    CartesianState,
    KeplerElements,
)
from .kepler_period import compute_period, period_to_semimajor_axis
from .mass_dependent import (
    CombinedBinaryFraction,
    MassDependentBinaryConfig,
    RadialBinaryFraction,
    sample_mass_dependent_orbits,
)
from .orbital_state import (
    BinaryOrbitalState,
    batch_elements_to_resolved,
)
from .orientation import sample_isotropic_orientations
from .period import LogNormalPeriod, LogUniformPeriod, SanaOBPeriod

__all__ = [
    # Kepler mechanics
    "KeplerElements",
    "CartesianState",
    "BinaryState",
    "compute_period",
    "period_to_semimajor_axis",
    # Binary orbital state
    "BinaryOrbitalState",
    "batch_elements_to_resolved",
    # Period distributions
    "LogUniformPeriod",
    "LogNormalPeriod",
    "SanaOBPeriod",
    # Eccentricity distributions
    "ThermalEccentricity",
    "UniformEccentricity",
    "LogisticThermalEccentricity",
    "MoeEccentricity",
    # Orientation
    "sample_isotropic_orientations",
    # Radial / mass-dependent prescriptions
    "RadialBinaryFraction",
    "CombinedBinaryFraction",
    "MassDependentBinaryConfig",
    "sample_mass_dependent_orbits",
    # Binary -> spatial-IC connector
    "ResolvedBinaries",
    "resolve_binary_components",
    "CatalogedBinaryClusterIC",
    "PrimordialSystemCatalog",
    "periapsis_contact_margin",
    "validate_primordial_system_catalog",
    # Companion/orbit layer (Batch 4k)
    "CompanionElements",
    "IndependentCompanions",
    "MoeCompanions",
    # Dynamic binary diagnostics
    "relative_energy",
    "find_bound_pairs",
    "find_bound_multiples",
    "primordial_survival",
    "BinaryEnergyBudget",
    "binary_energy_budget",
]

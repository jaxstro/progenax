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

from .kepler import (
    KeplerElements,
    CartesianState,
    BinaryState,
)
from .kepler_period import compute_period, period_to_semimajor_axis

from .orbital_state import (
    BinaryOrbitalState,
    batch_elements_to_resolved,
)

from .period import LogUniformPeriod, LogNormalPeriod, SanaOBPeriod
from .eccentricity import (
    ThermalEccentricity,
    UniformEccentricity,
    LogisticThermalEccentricity,
    MoeEccentricity,
)
from .orientation import sample_isotropic_orientations
from .mass_dependent import (
    RadialBinaryFraction,
    MassDependentBinaryConfig,
    sample_mass_dependent_orbits,
)

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
    "MassDependentBinaryConfig",
    "sample_mass_dependent_orbits",
]

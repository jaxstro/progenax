"""Binary star orbital mechanics for progenax.

Combines Kepler mechanics and binary orbital state from gravax-legacy.
All functions take explicit G parameter (NO get_G() defaults).

Modules:
    kepler: KeplerElements, period conversions
    orbital_state: BinaryOrbitalState, batch operations
    population: Period, eccentricity, and orientation distributions
"""

from .kepler import (
    KeplerElements,
    compute_period,
    period_to_semimajor_axis,
)

from .orbital_state import (
    BinaryOrbitalState,
    make_elements_from_inputs,
    elements_to_resolved_state,
    batch_elements_to_resolved,
    elements_to_com_and_internal,
    batch_elements_to_com_and_internal,
    KeplerElements_IC,
)

from .population import (
    LogUniformPeriod,
    LogNormalPeriod,
    ThermalEccentricity,
    UniformEccentricity,
    sample_isotropic_orientations,
)

__all__ = [
    # Kepler mechanics
    "KeplerElements",
    "compute_period",
    "period_to_semimajor_axis",
    # Binary orbital state
    "BinaryOrbitalState",
    "make_elements_from_inputs",
    "elements_to_resolved_state",
    "batch_elements_to_resolved",
    "elements_to_com_and_internal",
    "batch_elements_to_com_and_internal",
    # Population distributions
    "LogUniformPeriod",
    "LogNormalPeriod",
    "ThermalEccentricity",
    "UniformEccentricity",
    "sample_isotropic_orientations",
    # Backwards compatibility
    "KeplerElements_IC",
]

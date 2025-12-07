# progenax/src/progenax/profiles/__init__.py
"""
Spatial density profiles for stellar systems.

Implements the SpatialProfile protocol with these density models:
- PlummerProfile: Plummer (1911) spherical profile
- KingProfile: King (1966) lowered isothermal model
- EFFProfile: Elson-Fall-Freeman (1987) truncated power-law

Mass segregation transforms:
- apply_mass_segregation: Primordial mass segregation (radial scaling)
- apply_mass_segregation_baumgardt: Energy-ranked orbit assignment (Baumgardt+2008)
- compute_mass_segregation_ratio: MSR diagnostic

All profiles provide:
- sample_positions(masses, key) -> (N, 3) positions
- characteristic_radius() -> scalar radius
"""

from progenax.profiles.plummer import PlummerProfile
from progenax.profiles.king import KingProfile, solve_king_profile
from progenax.profiles.eff import EFFProfile
from progenax.profiles.mass_segregation import (
    apply_mass_segregation,
    apply_mass_segregation_baumgardt,
    compute_mass_segregation_ratio,
)

__all__ = [
    "PlummerProfile",
    "KingProfile",
    "solve_king_profile",
    "EFFProfile",
    "apply_mass_segregation",
    "apply_mass_segregation_baumgardt",
    "compute_mass_segregation_ratio",
]

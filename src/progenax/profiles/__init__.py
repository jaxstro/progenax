# progenax/src/progenax/profiles/__init__.py
"""
Spatial density profiles for stellar systems.

This module provides density profile models for generating star cluster
initial conditions. Each profile implements the SpatialProfile protocol
for position sampling.

Profiles:
    PlummerProfile: Plummer (1911) spherical profile - analytic DF available
    KingProfile: King (1966) lowered isothermal model - analytic DF available
    EFFProfile: Elson-Fall-Freeman (1987) truncated power-law - no analytic DF
    UniformSphereProfile: Uniform density sphere - CW04 reference (Q ≈ 0.79)

Functional API:
    make_profile: Factory function for creating profile instances
    sample_density_profile: Sample positions from any supported profile
    compute_profile_potential: Compute analytic potential at positions

All profiles provide:
    - sample_positions(masses, key) -> (N, 3) positions
    - characteristic_radius() -> scalar radius
    - density(r) -> unnormalized density at radius r

Note:
    Mass segregation and substructure generation have been moved to the
    `progenax.cluster` module. Diagnostics (MST-based Λ_MSR) are in
    `progenax.diagnostics`.

References:
    Plummer (1911) MNRAS 71, 460
    King (1966) AJ 71, 64
    Elson, Fall & Freeman (1987) ApJ 323, 54
    Cartwright & Whitworth (2004) MNRAS 348, 589
"""

from progenax.profiles.plummer import PlummerProfile
from progenax.profiles.king import KingProfile, solve_king_profile
from progenax.profiles.michie import MichieProfile, solve_michie_profile
from progenax.profiles.limepy import LIMEPYProfile, solve_limepy_profile
from progenax.profiles.limepy_multimass import (
    MultiMassLIMEPY,
    solve_multimass_limepy,
    find_alpha_for_masses,
)
from progenax.profiles.eff import EFFProfile
from progenax.profiles.uniform import UniformSphereProfile
from progenax.profiles.api import (
    ProfileName,
    make_profile,
    sample_density_profile,
    compute_profile_potential,
)

__all__ = [
    # Profile classes
    "PlummerProfile",
    "KingProfile",
    "solve_king_profile",
    "MichieProfile",
    "solve_michie_profile",
    "LIMEPYProfile",
    "solve_limepy_profile",
    "MultiMassLIMEPY",
    "solve_multimass_limepy",
    "find_alpha_for_masses",
    "EFFProfile",
    "UniformSphereProfile",
    # Functional API
    "ProfileName",
    "make_profile",
    "sample_density_profile",
    "compute_profile_potential",
]

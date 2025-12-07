"""
Velocity distribution functions and transforms for initial conditions.

This module provides velocity sampling from various distribution functions:
- PlummerVelocityDF: Exact Plummer (1911) DF using Beta distribution
- KingVelocityDF: King (1966) lowered Maxwellian with escape velocity cutoff
- EFFVelocityDF: Isotropic Gaussian for EFF profiles

Velocity transforms:
- apply_osipkov_merritt: Osipkov-Merritt radial anisotropy

All classes implement the VelocityDF protocol for use with IC assembly.
"""

from progenax.kinematics.plummer_df import PlummerVelocityDF
from progenax.kinematics.king_df import KingVelocityDF
from progenax.kinematics.eff_df import EFFVelocityDF
from progenax.kinematics.anisotropy import apply_osipkov_merritt

__all__ = [
    "PlummerVelocityDF",
    "KingVelocityDF",
    "EFFVelocityDF",
    "apply_osipkov_merritt",
]

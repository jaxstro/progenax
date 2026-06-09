"""
Velocity distribution functions and transforms for initial conditions.

This module provides velocity sampling from various distribution functions:
- PlummerVelocityDF: Exact Plummer (1911) DF using Beta distribution
- KingVelocityDF: King (1966) lowered Maxwellian with escape velocity cutoff
- MichieVelocityDF: Michie-King self-consistent radially anisotropic model
- EFFVelocityDF: Ergodic DF for EFF profiles via Eddington inversion

Osipkov-Merritt radial anisotropy is an intrinsic DF property: pass
``anisotropy_radius`` (r_a) to a velocity DF (Plummer/EFF) for beta(r)=r^2/(r^2+r_a^2).
King's radially anisotropic generalisation is the separate MichieVelocityDF
(+ MichieProfile), whose density differs from the isotropic King model.

Velocity transforms:
- apply_solid_body_rotation: Solid body rotation
- apply_differential_rotation: Differential rotation with peaked curve

High-level API:
- VelocityModel: Complete velocity model specification (DF + rotation + target_Q)
- sample_velocities_pipeline: Full pipeline with optional virial rescaling

All classes implement the VelocityDF protocol for use with IC assembly.
"""

from progenax.kinematics.plummer_df import PlummerVelocityDF
from progenax.kinematics.king_df import KingVelocityDF
from progenax.kinematics.michie_df import MichieVelocityDF
from progenax.kinematics.limepy_df import LIMEPYVelocityDF
from progenax.kinematics.eff_df import EFFVelocityDF
from progenax.kinematics.rotation import apply_solid_body_rotation, apply_differential_rotation
from progenax.kinematics.api import (
    VelocityDF,
    RotationParams,
    VelocityModel,
    sample_velocities_pipeline,
)

__all__ = [
    # Distribution functions
    "PlummerVelocityDF",
    "KingVelocityDF",
    "MichieVelocityDF",
    "LIMEPYVelocityDF",
    "EFFVelocityDF",
    # Transforms
    "apply_solid_body_rotation",
    "apply_differential_rotation",
    # High-level API
    "VelocityDF",
    "RotationParams",
    "VelocityModel",
    "sample_velocities_pipeline",
]

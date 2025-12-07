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
]

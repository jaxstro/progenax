"""
progenax - Differentiable initial conditions for N-body simulations.

Part of the jaxstro ecosystem.
"""

from .protocols import SpatialProfile, VelocityDF, IMFProtocol
from .profiles import PlummerProfile, KingProfile, EFFProfile, solve_king_profile

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
]

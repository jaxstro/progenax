"""Convenience cluster-IC builders (thin, differentiable sugar over build_spatial_ic).

Public API: build_cluster, matched_velocity_df, RotationSpec, ClusterParams,
build_cluster_from_params, and the named aliases build_{plummer,king,eff,michie,limepy}_cluster.

Design: docs/plans/2026-06-14-cluster-builder-api-design.md (round-2 addendum).
"""
from __future__ import annotations
from typing import Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .defaults import DEFAULT_UNITS
from .protocols import SpatialProfile, VelocityDF
from .builders import ICResult, build_spatial_ic, virial_scale, compute_stellar_radii
from .profiles import (
    PlummerProfile, EFFProfile, KingProfile, MichieProfile, LIMEPYProfile,
)
from .kinematics import (
    PlummerVelocityDF, EFFVelocityDF, KingVelocityDF, MichieVelocityDF, LIMEPYVelocityDF,
    apply_solid_body_rotation, apply_differential_rotation,
)

_ZHAT = jnp.array([0.0, 0.0, 1.0])

__all__ = [
    "build_cluster", "matched_velocity_df", "RotationSpec", "ClusterParams",
    "build_cluster_from_params", "build_plummer_cluster", "build_king_cluster",
    "build_eff_cluster", "build_michie_cluster", "build_limepy_cluster",
]

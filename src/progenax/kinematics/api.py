"""
Velocity sampling API for progenax initial conditions.

Provides a unified pipeline for velocity generation:
    DF sampling → rotation → optional virial rescaling

This module combines the individual DF modules with optional rotation and an
optional virial rescale. Radial anisotropy (Osipkov-Merritt) is an intrinsic
property of the velocity DF itself (pass ``anisotropy_radius`` to the DF), not a
separate pipeline stage.

Example:
    >>> from progenax.kinematics import (
    ...     PlummerVelocityDF,
    ...     VelocityModel,
    ...     RotationParams,
    ...     sample_velocities_pipeline,
    ... )
    >>> import jax
    >>> import jax.numpy as jnp
    >>>
    >>> # Radially anisotropic Plummer DF + solid-body rotation, native equilibrium.
    >>> model = VelocityModel(
    ...     df=PlummerVelocityDF(r_h=1.0, anisotropy_radius=2.0),
    ...     rotation=RotationParams(solid_body=True, pattern_speed=0.1),
    ... )
    >>>
    >>> # Sample velocities
    >>> key = jax.random.PRNGKey(42)
    >>> positions = jnp.zeros((100, 3))  # from spatial sampling
    >>> masses = jnp.ones(100)
    >>> from jaxstro.units import STELLAR
    >>> velocities = sample_velocities_pipeline(
    ...     key, positions, masses, model=model, G=STELLAR.G
    ... )
"""

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, PRNGKeyArray

from progenax.dynamics.virial import rescale_velocities_to_virial
from progenax.kinematics.rotation import (
    apply_differential_rotation,
    apply_solid_body_rotation,
)

# Single source of truth for the VelocityDF protocol (audit A1): re-exported
# here so progenax.kinematics.api.VelocityDF stays importable, but defined once
# in progenax.protocols to prevent structural drift.
from progenax.protocols import VelocityDF


class RotationParams(eqx.Module):
    """Parameters for rotation velocity transforms.

    Supports both solid-body and differential rotation. Both can be
    applied together if desired.

    Attributes:
        solid_body: Whether to apply solid-body rotation.
        differential: Whether to apply differential rotation.
        pattern_speed: Angular velocity for solid-body rotation [rad/time].
            Maps to omega in apply_solid_body_rotation.
        v_peak: Peak rotation velocity for differential rotation [velocity units].
        r_peak: Radius of peak rotation [length units].
        axis: Rotation axis vector (3,). If None, uses z-axis [0, 0, 1].
            Will be normalized internally.

    Notes:
        Solid-body: v_rot = omega x r (constant angular velocity)
        Differential: v_phi(R) = v_peak * (R/R_peak) * exp(1 - R/R_peak)

    References:
        Lynden-Bell (1960) MNRAS 120, 204
        Binney & Tremaine (2008) Section 4.8
    """

    solid_body: bool = eqx.field(static=True, default=False)
    differential: bool = eqx.field(static=True, default=False)
    pattern_speed: float = 0.0
    v_peak: float = 0.0
    r_peak: float = 1.0
    # Default z-axis if None; static: a plain-Python config tuple, not a leaf
    axis: Optional[tuple[float, float, float]] = eqx.field(static=True, default=None)


class VelocityModel(eqx.Module):
    """Complete velocity model specification.

    Bundles a distribution function with optional rotation and a target virial
    ratio. Radial anisotropy is a property of the DF itself: pass
    ``anisotropy_radius`` to PlummerVelocityDF/EFFVelocityDF for an Osipkov-Merritt
    DF (beta(r)=r^2/(r^2+r_a^2)).

    Attributes:
        df: Velocity distribution function (PlummerVelocityDF, KingVelocityDF, etc.;
            anisotropic DFs supply their own radial anisotropy).
        rotation: Optional rotation parameters.
        target_Q: Target virial ratio Q = T / |V|, or None (default). The
            Plummer/King/EFF DFs are already sampled in detailed equilibrium, so
            target_Q=None keeps their native equilibrium (no rescale). Pass a float
            only to deliberately force a virial ratio (e.g. 0.5 for equilibrium,
            <0.5 subvirial, >0.5 supervirial), or when adding rotation / mixing an
            inconsistent profile+DF combination.

    Example:
        >>> model = VelocityModel(
        ...     df=PlummerVelocityDF(r_h=1.0, anisotropy_radius=2.0),
        ...     rotation=RotationParams(solid_body=True, pattern_speed=0.1),
        ... )
    """

    df: VelocityDF
    rotation: Optional[RotationParams] = None
    target_Q: Optional[float] = None


def sample_velocities_pipeline(
    key: PRNGKeyArray,
    positions: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    model: VelocityModel,
    G: float,
) -> Float[Array, "N 3"]:
    """Velocity pipeline: DF sampling -> rotation -> optional virial rescale.

    This is the main entry point for velocity generation in progenax. Radial
    anisotropy lives on the DF (pass ``anisotropy_radius``), so the pipeline only
    layers optional rotation and an optional virial rescale on top of the DF sample.

    Pipeline stages:
        1. Sample velocities from distribution function (model.df; carries any
           Osipkov-Merritt anisotropy intrinsically)
        2. Apply rotation (solid-body and/or differential) if configured
        3. Rescale to target virial ratio (only if model.target_Q is not None),
           then remove COM motion

    Args:
        key: JAX random key.
        positions: Particle positions (N, 3) [length units].
        masses: Particle masses (N,) [mass units].
        model: VelocityModel specifying DF + rotation + target Q.
        G: Gravitational constant (REQUIRED, explicit-units policy; e.g. ``STELLAR.G``).

    Returns:
        velocities: Particle velocities (N, 3) [velocity units].

    Example:
        >>> from progenax.kinematics import (
        ...     PlummerVelocityDF, VelocityModel, sample_velocities_pipeline
        ... )
        >>> import jax
        >>>
        >>> from jaxstro.units import STELLAR
        >>> model = VelocityModel(df=PlummerVelocityDF(r_h=1.0), target_Q=0.5)
        >>> key = jax.random.PRNGKey(42)
        >>> velocities = sample_velocities_pipeline(
        ...     key, positions, masses, model, G=STELLAR.G
        ... )

    Notes:
        - All stages are JAX-compatible and differentiable
        - Virial rescaling uses O(N^2) pairwise potential energy calculation
        - COM motion is removed after rescaling
    """
    # Stage 1: DF sampling (the DF is the only source of randomness in the pipeline).
    v = model.df.sample_velocities(positions, masses, key, G=G)

    # Stage 2: Rotation
    if model.rotation is not None:
        # Get rotation axis (default to z-axis if None)
        if model.rotation.axis is None:
            axis = jnp.array([0.0, 0.0, 1.0])
        else:
            axis = jnp.array(model.rotation.axis)

        # Solid-body rotation
        if model.rotation.solid_body and model.rotation.pattern_speed != 0.0:
            v = apply_solid_body_rotation(
                velocities=v,
                positions=positions,
                omega=model.rotation.pattern_speed,
                axis=axis,
            )

        # Differential rotation
        if model.rotation.differential and model.rotation.v_peak != 0.0:
            v = apply_differential_rotation(
                velocities=v,
                positions=positions,
                v_peak=model.rotation.v_peak,
                R_peak=model.rotation.r_peak,
                axis=axis,
            )

    # Stage 3: Optional virial rescaling. target_Q=None keeps the DF's native
    # equilibrium (Plummer/King/EFF are already sampled in detailed equilibrium);
    # pass a float only to deliberately force an overall virial ratio.
    #
    # NOTE (audit S3): this is an ISOTROPIC speed rescale. It does NOT restore
    # stationarity after a rotation overlay (Stage 2) — it cannot remove the
    # injected net angular momentum L_z, so the result stays non-stationary even
    # at Q=0.5. Rescaling-to-0.5 is not a fix for the rotation-broken equilibrium.
    if model.target_Q is not None:
        v = rescale_velocities_to_virial(
            positions=positions,
            velocities=v,
            masses=masses,
            G=G,
            target_Q=model.target_Q,
        )

    # Remove COM motion
    M_total = jnp.sum(masses)
    v_com = jnp.sum(masses[:, None] * v, axis=0) / M_total
    v = v - v_com

    return v


__all__ = [
    "VelocityDF",
    "RotationParams",
    "VelocityModel",
    "sample_velocities_pipeline",
]

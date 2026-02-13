"""
Velocity sampling API for progenax initial conditions.

Provides a unified pipeline for velocity generation:
    DF sampling → anisotropy → rotation → virial rescaling

This module combines the individual DF modules with optional transforms
to produce physically realistic velocity distributions.

Example:
    >>> from progenax.kinematics import (
    ...     PlummerVelocityDF,
    ...     VelocityModel,
    ...     AnisotropyParams,
    ...     RotationParams,
    ...     sample_velocities_pipeline,
    ... )
    >>> import jax
    >>> import jax.numpy as jnp
    >>>
    >>> # Create velocity model with Plummer DF + anisotropy + rotation
    >>> model = VelocityModel(
    ...     df=PlummerVelocityDF(r_h=1.0),
    ...     anisotropy=AnisotropyParams(use_osipkov_merritt=True, r_a=2.0),
    ...     rotation=RotationParams(solid_body=True, pattern_speed=0.1),
    ...     target_Q=0.5,
    ... )
    >>>
    >>> # Sample velocities
    >>> key = jax.random.PRNGKey(42)
    >>> positions = jnp.zeros((100, 3))  # from spatial sampling
    >>> masses = jnp.ones(100)
    >>> velocities = sample_velocities_pipeline(
    ...     key, positions, masses, model=model
    ... )
"""

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, PRNGKeyArray

from progenax import defaults
from progenax.kinematics.anisotropy import apply_osipkov_merritt
from progenax.kinematics.rotation import (
    apply_solid_body_rotation,
    apply_differential_rotation,
)
from progenax.dynamics.virial import rescale_velocities_to_virial


@runtime_checkable
class VelocityDF(Protocol):
    """Protocol for velocity distribution functions.

    All velocity DFs must implement this interface for use with the
    kinematics API pipeline.
    """

    def sample_velocities(
        self,
        positions: Float[Array, "N 3"],
        masses: Float[Array, "N"],
        key: PRNGKeyArray,
        G: float | None = None,
    ) -> Float[Array, "N 3"]:
        """Sample velocities from the distribution function.

        Args:
            positions: Particle positions (N, 3) [length units]
            masses: Particle masses (N,) [mass units]
            key: JAX random key
            G: Gravitational constant. If None, uses default.

        Returns:
            Cartesian velocities (N, 3) [velocity units]
        """
        ...


@dataclass(frozen=True)
class AnisotropyParams:
    """Parameters for velocity anisotropy transforms.

    Attributes:
        use_osipkov_merritt: Whether to apply Osipkov-Merritt anisotropy.
        r_a: Anisotropy radius [length units]. At r = r_a, beta = 0.5.
             Smaller r_a means stronger radial anisotropy.

    Notes:
        The Osipkov-Merritt profile gives:
            beta(r) = r^2 / (r^2 + r_a^2)

        where beta = 1 - sigma_t^2 / (2 sigma_r^2) is the anisotropy parameter:
            - beta = 0: isotropic
            - beta -> 1: purely radial
            - beta < 0: tangentially biased

    References:
        Osipkov (1979) Soviet Astronomy Letters 5, 42
        Merritt (1985) AJ 90, 1027
    """

    use_osipkov_merritt: bool = False
    r_a: float = 1.0


@dataclass(frozen=True)
class RotationParams:
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

    solid_body: bool = False
    differential: bool = False
    pattern_speed: float = 0.0
    v_peak: float = 0.0
    r_peak: float = 1.0
    axis: Optional[tuple[float, float, float]] = None  # Default z-axis if None


@dataclass(frozen=True)
class VelocityModel:
    """Complete velocity model specification.

    Bundles a distribution function with optional anisotropy and rotation
    transforms, plus a target virial ratio.

    Attributes:
        df: Velocity distribution function (PlummerVelocityDF, KingVelocityDF, etc.)
        anisotropy: Optional anisotropy parameters.
        rotation: Optional rotation parameters.
        target_Q: Target virial ratio Q = T / |V|. Default 0.5 for equilibrium.

    Example:
        >>> model = VelocityModel(
        ...     df=PlummerVelocityDF(r_h=1.0),
        ...     anisotropy=AnisotropyParams(use_osipkov_merritt=True, r_a=2.0),
        ...     rotation=RotationParams(solid_body=True, pattern_speed=0.1),
        ...     target_Q=0.5,
        ... )
    """

    df: VelocityDF
    anisotropy: Optional[AnisotropyParams] = None
    rotation: Optional[RotationParams] = None
    target_Q: float = 0.5


def sample_velocities_pipeline(
    key: PRNGKeyArray,
    positions: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    model: VelocityModel,
    G: float | None = None,
) -> Float[Array, "N 3"]:
    """Velocity pipeline: DF sampling -> anisotropy -> rotation -> virial rescale.

    This is the main entry point for velocity generation in progenax.
    It applies a sequence of transforms to produce physically realistic
    velocity distributions.

    Pipeline stages:
        1. Sample velocities from distribution function (model.df)
        2. Apply Osipkov-Merritt anisotropy if configured
        3. Apply rotation (solid-body and/or differential) if configured
        4. Rescale to target virial ratio and remove COM motion

    Args:
        key: JAX random key.
        positions: Particle positions (N, 3) [length units].
        masses: Particle masses (N,) [mass units].
        model: VelocityModel specifying DF + transforms + target Q.
        G: Gravitational constant. If None, uses progenax.DEFAULT_UNITS.G.

    Returns:
        velocities: Particle velocities (N, 3) [velocity units].

    Example:
        >>> from progenax.kinematics import (
        ...     PlummerVelocityDF, VelocityModel, sample_velocities_pipeline
        ... )
        >>> import jax
        >>>
        >>> model = VelocityModel(df=PlummerVelocityDF(r_h=1.0), target_Q=0.5)
        >>> key = jax.random.PRNGKey(42)
        >>> velocities = sample_velocities_pipeline(key, positions, masses, model)

    Notes:
        - All stages are JAX-compatible and differentiable
        - Virial rescaling uses O(N^2) pairwise potential energy calculation
        - COM motion is removed after rescaling
    """
    if G is None:
        G = defaults.DEFAULT_UNITS.G

    # Split key for each stage that needs randomness
    key_df, key_aniso = jax.random.split(key, 2)

    # Stage 1: DF sampling
    v = model.df.sample_velocities(positions, masses, key_df, G=G)

    # Stage 2: Anisotropy
    if model.anisotropy is not None and model.anisotropy.use_osipkov_merritt:
        v = apply_osipkov_merritt(
            velocities=v,
            positions=positions,
            key=key_aniso,
            r_a=model.anisotropy.r_a,
        )

    # Stage 3: Rotation
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

    # Stage 4: Virial rescaling + COM removal
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
    "AnisotropyParams",
    "RotationParams",
    "VelocityModel",
    "sample_velocities_pipeline",
]

"""Rotation velocity transforms for star cluster ICs.

Adds streaming rotation to velocity distributions.

References:
    Lynden-Bell (1960) MNRAS 120, 204 - Rotating stellar systems
    Binney & Tremaine (2008) "Galactic Dynamics" Section 4.8
"""

import jax.numpy as jnp
from jaxtyping import Array, Float


def apply_solid_body_rotation(
    velocities: Float[Array, "N 3"],
    positions: Float[Array, "N 3"],
    omega: float,
    axis: Float[Array, "3"],
) -> Float[Array, "N 3"]:
    """Add solid body rotation to velocities.

    Adds streaming velocity v_rot = omega x r to existing velocities.

    For solid body rotation, all particles rotate with the same
    angular velocity omega, giving v_phi = omega * R (cylindrical R).

    Args:
        velocities: Input velocities (N, 3)
        positions: Particle positions (N, 3)
        omega: Angular velocity magnitude [rad/time]
        axis: Rotation axis vector (3,), will be normalized. Must be nonzero.

    Returns:
        Velocities with rotation added (N, 3)

    Example:
        >>> # Add rotation around z-axis with omega = 0.1 rad/Myr
        >>> v_rot = apply_solid_body_rotation(
        ...     velocities, positions,
        ...     omega=0.1,
        ...     axis=jnp.array([0., 0., 1.])
        ... )

    Reference:
        Binney & Tremaine (2008) Section 4.8
    """
    # Normalize axis with epsilon safeguard
    axis_mag = jnp.linalg.norm(axis)
    # Use safe division - if axis is zero, result will be NaN which is appropriate
    axis_norm = axis / jnp.maximum(axis_mag, 1e-30)

    # Rotation velocity: v_rot = omega x r = omega * (axis x r)
    omega_vec = omega * axis_norm  # (3,)

    # Cross product omega x r for each particle
    v_rotation = jnp.cross(omega_vec, positions)  # (N, 3)

    return velocities + v_rotation


def apply_differential_rotation(
    velocities: Float[Array, "N 3"],
    positions: Float[Array, "N 3"],
    v_peak: float,
    R_peak: float,
    axis: Float[Array, "3"],
) -> Float[Array, "N 3"]:
    """Add differential rotation with peaked rotation curve.

    Rotation curve: v_phi(R) = v_peak * (R/R_peak) * exp(1 - R/R_peak)

    This gives:
        - v_phi(0) = 0 (no rotation at center)
        - v_phi(R_peak) = v_peak (maximum rotation)
        - v_phi -> 0 as R -> infinity (decreasing at large R)

    Args:
        velocities: Input velocities (N, 3)
        positions: Particle positions (N, 3)
        v_peak: Peak rotation velocity [velocity units]
        R_peak: Radius of peak rotation [length units]
        axis: Rotation axis vector (3,), will be normalized. Must be nonzero.

    Returns:
        Velocities with differential rotation added (N, 3)

    Note:
        The peaked form v_phi(R) = v_peak (R/R_peak) exp(1 - R/R_peak) is a
        *phenomenological* rotation curve (smooth rise to a single peak at R_peak,
        then decay) chosen for convenience -- it is NOT taken from a specific paper.
        Lynden-Bell (1960), MNRAS 120, 204 is the classic reference for rotating
        stellar systems in general, not for this functional form.
    """
    # Normalize axis with epsilon safeguard
    axis_mag = jnp.linalg.norm(axis)
    axis_norm = axis / jnp.maximum(axis_mag, 1e-30)

    # Compute cylindrical radius R (distance from rotation axis)
    # R^2 = |r|^2 - (r . axis)^2
    r_dot_axis = jnp.sum(positions * axis_norm, axis=1, keepdims=True)  # (N, 1)
    R_squared = jnp.sum(positions**2, axis=1, keepdims=True) - r_dot_axis**2
    R = jnp.sqrt(jnp.maximum(R_squared, 1e-20))  # (N, 1)

    # Rotation curve
    x = R / R_peak
    v_phi = v_peak * x * jnp.exp(1.0 - x)  # (N, 1)

    # Azimuthal direction: phi_hat = (axis x r) / |axis x r|
    axis_cross_r = jnp.cross(axis_norm, positions)  # (N, 3)
    axis_cross_r_mag = jnp.linalg.norm(axis_cross_r, axis=1, keepdims=True)
    phi_hat = axis_cross_r / jnp.maximum(axis_cross_r_mag, 1e-10)

    # Add rotation velocity
    v_rotation = v_phi * phi_hat

    return velocities + v_rotation


__all__ = ["apply_solid_body_rotation", "apply_differential_rotation"]

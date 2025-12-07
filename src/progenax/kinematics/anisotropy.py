"""Velocity anisotropy transforms for star cluster ICs.

Implements velocity distribution modifications to introduce radial/tangential bias.

References:
    Osipkov (1979) Soviet Astronomy Letters 5, 42
    Merritt (1985) AJ 90, 1027
    Binney & Tremaine (2008) "Galactic Dynamics" Section 4.3.2
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


def apply_osipkov_merritt(
    velocities: Float[Array, "N 3"],
    positions: Float[Array, "N 3"],
    key: PRNGKeyArray,
    r_a: float,
) -> Float[Array, "N 3"]:
    """Apply Osipkov-Merritt radial anisotropy to velocities.

    Transforms isotropic velocities to have radial anisotropy profile:

        beta(r) = r^2 / (r^2 + r_a^2)

    where beta = 1 - sigma_t^2/(2*sigma_r^2) is the anisotropy parameter:
        - beta = 0: isotropic
        - beta -> 1: purely radial
        - beta < 0: tangentially biased

    For Osipkov-Merritt:
        - beta(0) = 0 (isotropic at center)
        - beta(r_a) = 0.5
        - beta(inf) = 1 (radial at large r)

    Algorithm:
        1. Decompose v into radial and tangential components
        2. Compute target ratio based on beta(r)
        3. Redistribute speed between radial/tangential preserving |v|

    Args:
        velocities: Input velocities (N, 3)
        positions: Particle positions (N, 3)
        key: JAX random key (for random tangential direction)
        r_a: Anisotropy radius [length units]

    Returns:
        Transformed velocities (N, 3) with same |v| but anisotropic distribution

    Reference:
        Osipkov (1979) Soviet Astronomy Letters 5, 42
        Merritt (1985) AJ 90, 1027
    """
    N = positions.shape[0]

    # Compute radii
    r = jnp.linalg.norm(positions, axis=1, keepdims=True)  # (N, 1)
    r_safe = jnp.maximum(r, 1e-10)

    # Radial unit vector
    r_hat = positions / r_safe  # (N, 3)

    # Decompose velocity into radial and tangential
    v_r_scalar = jnp.sum(velocities * r_hat, axis=1, keepdims=True)  # (N, 1)
    v_r = v_r_scalar * r_hat  # (N, 3)
    v_t = velocities - v_r  # (N, 3)

    v_t_mag = jnp.linalg.norm(v_t, axis=1, keepdims=True)  # (N, 1)
    v_mag = jnp.linalg.norm(velocities, axis=1, keepdims=True)  # (N, 1)

    # Compute anisotropy parameter beta(r) = r^2 / (r^2 + r_a^2)
    beta = r**2 / (r**2 + r_a**2)  # (N, 1)

    # Target velocity ratio: for beta = 1 - sigma_t^2/(2*sigma_r^2)
    # We want: v_r^2 / v^2 = (1 + beta) / (3 - beta) for proper anisotropy
    # This ensures <v_r^2> / <v^2> matches Osipkov-Merritt profile
    f_r_sq = (1.0 + beta) / (3.0 - beta + 1e-10)  # Fraction of v^2 that is radial
    f_r_sq = jnp.clip(f_r_sq, 0.0, 1.0)

    # New radial and tangential magnitudes (preserving total speed)
    new_v_r_mag = v_mag * jnp.sqrt(f_r_sq)
    new_v_t_mag = v_mag * jnp.sqrt(1.0 - f_r_sq)

    # Preserve radial direction (sign)
    v_r_sign = jnp.sign(v_r_scalar)
    v_r_sign = jnp.where(v_r_sign == 0, 1.0, v_r_sign)
    new_v_r = new_v_r_mag * v_r_sign * r_hat

    # For tangential: preserve direction if nonzero, else random
    v_t_safe = jnp.maximum(v_t_mag, 1e-10)
    t_hat = v_t / v_t_safe  # (N, 3)

    # Generate random tangential direction for particles with v_t ~ 0
    key1, key2 = jax.random.split(key)
    random_vec = jax.random.normal(key1, (N, 3))
    random_vec = random_vec - jnp.sum(random_vec * r_hat, axis=1, keepdims=True) * r_hat
    random_vec_mag = jnp.linalg.norm(random_vec, axis=1, keepdims=True)
    random_t_hat = random_vec / jnp.maximum(random_vec_mag, 1e-10)

    # Use original direction if v_t is significant, else random
    use_original = (v_t_mag > 1e-10 * v_mag).astype(jnp.float64)
    final_t_hat = use_original * t_hat + (1.0 - use_original) * random_t_hat

    new_v_t = new_v_t_mag * final_t_hat

    return new_v_r + new_v_t


__all__ = ["apply_osipkov_merritt"]

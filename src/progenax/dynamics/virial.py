"""Virial ratio utilities.

Convention: Q = T / |V|

Physical interpretation:
    - Q = 0.5: Virial equilibrium (2T + V = 0)
    - Q < 0.5: Subvirial (cold), system will collapse
    - Q > 0.5: Supervirial (hot), system will expand/unbind
"""
import jax.numpy as jnp
from jaxtyping import Array, Float


def compute_kinetic_energy(
    velocities: Float[Array, "N 3"],
    masses: Float[Array, "N"],
) -> Float[Array, ""]:
    """Compute total kinetic energy: T = 0.5 * sum(m_i * v_i^2)."""
    v_squared = jnp.sum(velocities**2, axis=-1)
    return 0.5 * jnp.sum(masses * v_squared)


def compute_potential_energy(
    positions: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    G: float,
    softening: float = 0.0,
) -> Float[Array, ""]:
    """Compute total potential energy: V = -G * sum_{i<j}(m_i * m_j / r_ij).

    Returns negative value (bound systems have V < 0).
    """
    # Pairwise distances
    diff = positions[:, None, :] - positions[None, :, :]  # (N, N, 3)
    r2 = jnp.sum(diff**2, axis=-1) + softening**2  # (N, N)
    r = jnp.sqrt(r2)

    # Mass products
    m_prod = masses[:, None] * masses[None, :]  # (N, N)

    # Upper triangle only (i < j)
    V = -G * jnp.sum(jnp.triu(m_prod / (r + 1e-10), k=1))
    return V


def compute_virial_ratio(
    positions: Float[Array, "N 3"],
    velocities: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    G: float,
    softening: float = 0.0,
) -> Float[Array, ""]:
    """Compute virial ratio Q = T / |V|.

    Convention:
        Q = 0.5: Virial equilibrium (2T + V = 0)
        Q < 0.5: Subvirial (cold, collapsing)
        Q > 0.5: Supervirial (hot, expanding)

    Args:
        positions: Particle positions, shape (N, 3)
        velocities: Particle velocities, shape (N, 3)
        masses: Particle masses, shape (N,)
        G: Gravitational constant
        softening: Softening length for potential calculation

    Returns:
        Virial ratio Q = T / |V|
    """
    T = compute_kinetic_energy(velocities, masses)
    V = compute_potential_energy(positions, masses, G=G, softening=softening)
    return T / jnp.abs(V)


def rescale_velocities_to_virial(
    positions: Float[Array, "N 3"],
    velocities: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    G: float,
    target_Q: float = 0.5,
    softening: float = 0.0,
) -> Float[Array, "N 3"]:
    """Rescale velocities to achieve target virial ratio Q = T/|V|.

    Args:
        positions: Particle positions, shape (N, 3)
        velocities: Particle velocities, shape (N, 3)
        masses: Particle masses, shape (N,)
        G: Gravitational constant
        target_Q: Desired virial ratio (default 0.5 for equilibrium)
        softening: Softening length for potential calculation

    Returns:
        Rescaled velocities with Q = target_Q
    """
    Q_current = compute_virial_ratio(positions, velocities, masses, G=G, softening=softening)
    scale = jnp.sqrt(target_Q / (Q_current + 1e-10))
    return velocities * scale

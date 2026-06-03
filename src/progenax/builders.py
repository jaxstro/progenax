"""
IC builders for progenax.

Provides:
- ICResult: Dataclass for initial condition outputs
- build_spatial_ic: Generic builder using profile + velocity DF
- to_com_frame: Center-of-mass frame transformation
- virial_scale: Scale velocities to target virial ratio
- compute_stellar_radii: Estimate stellar radii from mass
- compute_kinetic_energy, compute_potential_energy: Energy helpers
"""

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .protocols import SpatialProfile, VelocityDF


@dataclass(frozen=True)
class ICResult:
    """
    Result from initial conditions generation.

    Immutable dataclass containing all particle data.
    No dependency on gravax - can be converted to any state format.

    Attributes:
        positions: Particle positions (N, 3) [length units]
        velocities: Particle velocities (N, 3) [velocity units]
        masses: Particle masses (N,) [M_sun]
        softening: Softening length [length units]
        stellar_radii: Stellar radii (N,) [R_sun]
        ids: Particle IDs (N,) or None
    """

    positions: Float[Array, "N 3"]
    velocities: Float[Array, "N 3"]
    masses: Float[Array, "N"]
    softening: float | Float[Array, ""]
    stellar_radii: Float[Array, "N"]
    ids: Optional[Float[Array, "N"]] = None


def compute_stellar_radii(masses: Float[Array, "N"]) -> Float[Array, "N"]:
    """
    Estimate stellar radii from mass (main-sequence + brown dwarfs).

    Returns radii in SOLAR RADII (R☉).

    R/R☉ relations (3 regimes):
    - M > 1 M☉: R ∝ M^0.8 (massive stars)
    - 0.08 ≤ M ≤ 1 M☉: R ∝ M^0.57 (low-mass main sequence)
    - M < 0.08 M☉: R ∝ M^0.08 (brown dwarfs, ~0.1 R☉)

    Args:
        masses: Particle masses (N,) [M☉]

    Returns:
        Radii in R☉
    """

    def radius_high_mass(m):
        return jnp.power(m, 0.8)

    def radius_low_mass(m):
        return jnp.power(m, 0.57)

    def radius_brown_dwarf(m):
        return 0.1 * jnp.power(m / 0.08, 0.08)

    radii = jax.vmap(
        lambda m: jax.lax.cond(
            m > 1.0,
            radius_high_mass,
            lambda mv: jax.lax.cond(
                mv >= 0.08, radius_low_mass, radius_brown_dwarf, mv
            ),
            m,
        )
    )(masses)

    return radii


def compute_kinetic_energy(
    velocities: Float[Array, "N 3"],
    masses: Float[Array, "N"],
) -> Float[Array, ""]:
    """
    Compute total kinetic energy: T = 0.5 * sum(m_i * v_i^2).

    Args:
        velocities: Particle velocities (N, 3)
        masses: Particle masses (N,)

    Returns:
        Total kinetic energy
    """
    v_squared = jnp.sum(velocities**2, axis=1)
    return 0.5 * jnp.sum(masses * v_squared)


def compute_potential_energy(
    positions: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    G: float,
    softening: float = 0.0,
) -> Float[Array, ""]:
    """
    Compute total potential energy: V = -G * sum_{i<j}(m_i * m_j / r_ij).

    Uses Plummer softening: r_ij -> sqrt(r_ij^2 + eps^2)

    Args:
        positions: Particle positions (N, 3)
        masses: Particle masses (N,)
        G: Gravitational constant
        softening: Softening length (default: 0)

    Returns:
        Total potential energy (negative)
    """
    N = positions.shape[0]

    # Pairwise distances (vectorized)
    diff = positions[:, None, :] - positions[None, :, :]  # (N, N, 3)
    r_squared = jnp.sum(diff**2, axis=2)  # (N, N)
    r_soft = jnp.sqrt(r_squared + softening**2)  # Plummer softening

    # Avoid self-interaction
    r_soft = jnp.where(jnp.eye(N, dtype=bool), jnp.inf, r_soft)

    # Mass products
    m_prod = masses[:, None] * masses[None, :]  # (N, N)

    # Sum upper triangle (i < j)
    V = -G * jnp.sum(jnp.triu(m_prod / r_soft, k=1))

    return V


def to_com_frame(
    positions: Float[Array, "N 3"],
    velocities: Float[Array, "N 3"],
    masses: Float[Array, "N"],
) -> tuple[Float[Array, "N 3"], Float[Array, "N 3"]]:
    """
    Transform to center-of-mass frame.

    Args:
        positions: Particle positions (N, 3)
        velocities: Particle velocities (N, 3)
        masses: Particle masses (N,)

    Returns:
        (positions_com, velocities_com): Transformed coordinates
    """
    M_total = jnp.sum(masses)

    # COM position and velocity
    r_com = jnp.sum(positions * masses[:, None], axis=0) / M_total
    v_com = jnp.sum(velocities * masses[:, None], axis=0) / M_total

    return positions - r_com, velocities - v_com


def virial_scale(
    positions: Float[Array, "N 3"],
    velocities: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    Q_target: float,
    G: float,
    softening: float = 0.0,
) -> Float[Array, "N 3"]:
    """
    Scale velocities to achieve target virial ratio Q = T/|V|.

    Physical interpretation:
        - Q = 0.5: Virial equilibrium (2T + V = 0)
        - Q < 0.5: Sub-virial (cold), system will collapse
        - Q > 0.5: Super-virial (hot), system will expand/unbind

    Args:
        positions: Particle positions (N, 3)
        velocities: Particle velocities (N, 3)
        masses: Particle masses (N,)
        Q_target: Target virial ratio (0.5 for equilibrium)
        G: Gravitational constant
        softening: Softening length (default: 0)

    Returns:
        Scaled velocities

    References:
        Goodwin & Whitworth (2004) A&A 413, 929 - Sub-virial clusters
        Baumgardt & Kroupa (2007) MNRAS 380, 1589 - Cluster dissolution
    """
    T = compute_kinetic_energy(velocities, masses)
    V = compute_potential_energy(positions, masses, G=G, softening=softening)

    # Q = T / |V| (NOT 2T / |V|)
    Q_current = T / jnp.abs(V)
    scale = jnp.sqrt(Q_target / Q_current)

    return velocities * scale


def build_spatial_ic(
    profile: SpatialProfile,
    masses: Float[Array, "N"],
    velocity_df: VelocityDF,
    key: PRNGKeyArray,
    G: float,
    Q: Optional[float] = 0.5,
    softening_factor: float = 0.01,
    softening_floor: Optional[float] = None,
    id_offset: int = 0,
) -> ICResult:
    """
    Build initial conditions from spatial profile and velocity DF.

    Args:
        profile: Spatial density profile (must implement SpatialProfile protocol)
        masses: Particle masses (N,) [M_sun]
        velocity_df: Velocity distribution function (must implement VelocityDF protocol)
        key: JAX random key
        G: Gravitational constant (REQUIRED - no default)
        Q: Virial ratio target Q = T/|V| (0.5 for equilibrium, None to disable)
        softening_factor: Softening as fraction of mean separation (default: 0.01)
        softening_floor: Minimum softening (default: None)
        id_offset: Offset for particle IDs (default: 0)

    Returns:
        ICResult with positions, velocities, masses, softening, stellar_radii
    """
    N = masses.shape[0]

    # Split keys
    key_pos, key_vel = jax.random.split(key)

    # 1. Sample positions
    positions = profile.sample_positions(masses, key_pos)

    # 2. Sample velocities (G must be threaded through — see audit C1; without it
    #    the DF silently falls back to DEFAULT_UNITS.G and ignores the caller's units)
    velocities = velocity_df.sample_velocities(positions, masses, key_vel, G=G)

    # 3. Compute softening
    r_char = profile.characteristic_radius()
    d_mean = r_char / jnp.power(N, 1.0 / 3.0)
    softening = softening_factor * d_mean
    if softening_floor is not None:
        softening = jnp.maximum(softening, softening_floor)
    # Keep softening as a JAX scalar so build_spatial_ic stays differentiable
    # (float() concretized a tracer -> broke jax.grad wrt r_h; audit CR-FU-2).
    # It is only stored on ICResult and passed to virial_scale, both array-safe.
    softening = jnp.asarray(softening)

    # 4. Compute stellar radii
    stellar_radii = compute_stellar_radii(masses)

    # 5. Transform to COM frame
    positions, velocities = to_com_frame(positions, velocities, masses)

    # 6. Apply virial scaling
    if Q is not None:
        velocities = virial_scale(positions, velocities, masses, Q, G, softening)

    # 7. Generate IDs
    ids = jnp.arange(id_offset, id_offset + N) if id_offset != 0 else None

    return ICResult(
        positions=positions,
        velocities=velocities,
        masses=masses,
        softening=softening,
        stellar_radii=stellar_radii,
        ids=ids,
    )


__all__ = [
    "ICResult",
    "compute_stellar_radii",
    "compute_kinetic_energy",
    "compute_potential_energy",
    "to_com_frame",
    "virial_scale",
    "build_spatial_ic",
]

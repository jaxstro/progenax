"""
Two-component cluster population generation.

Enables creating clusters with distinct spatial and kinematic populations
(e.g., extended halo + concentrated core, or mass-segregated components).

References:
    Vesperini & Heggie (1997) MNRAS 289, 898 - Multi-mass cluster dynamics
    Baumgardt et al. (2008) MNRAS 384, 1231 - Primordial mass segregation
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from .protocols import SpatialProfile, VelocityDF


@dataclass(frozen=True)
class TwoComponentConfig:
    """
    Configuration for two-component cluster generation.

    Defines two distinct populations with separate spatial profiles and
    velocity distribution functions. Common use cases:
    - Extended halo + concentrated core
    - Mass-segregated populations (heavy stars sink to center)
    - Dynamically distinct components (e.g., disk + bulge)

    Attributes:
        f_A: Fraction of particles in population A [0, 1]
        profile_A: Spatial density profile for population A (extended)
        profile_B: Spatial density profile for population B (concentrated)
        velocity_df_A: Velocity distribution function for population A
        velocity_df_B: Velocity distribution function for population B

    Examples:
        >>> from progenax.profiles import PlummerProfile
        >>> from progenax.kinematics import PlummerVelocityDF
        >>>
        >>> # Extended halo + concentrated core
        >>> profile_halo = PlummerProfile(r_h=2.0)
        >>> profile_core = PlummerProfile(r_h=0.5)
        >>> df_halo = PlummerVelocityDF(r_h=2.0)
        >>> df_core = PlummerVelocityDF(r_h=0.5)
        >>>
        >>> config = TwoComponentConfig(
        ...     f_A=0.3,  # 30% in extended halo
        ...     profile_A=profile_halo,
        ...     profile_B=profile_core,
        ...     velocity_df_A=df_halo,
        ...     velocity_df_B=df_core,
        ... )

    References:
        Vesperini & Heggie (1997) MNRAS 289, 898
        Baumgardt et al. (2008) MNRAS 384, 1231
    """

    f_A: float
    profile_A: SpatialProfile
    profile_B: SpatialProfile
    velocity_df_A: VelocityDF
    velocity_df_B: VelocityDF


def generate_two_component_cluster(
    masses: Float[Array, "N"],
    config: TwoComponentConfig,
    key: PRNGKeyArray,
    G: float,
    pop_mask: Optional[Bool[Array, "N"]] = None,
) -> Tuple[Float[Array, "N 3"], Float[Array, "N 3"], Float[Array, "N"]]:
    """
    Generate two-component cluster with distinct spatial/kinematic populations.

    Samples positions and velocities from two separate profiles and DFs,
    then combines them based on population membership. Population assignment
    can be random (default) or user-specified (via pop_mask).

    Args:
        masses: Particle masses (N,) [Msun]
        config: Two-component configuration
        key: JAX random key
        G: Gravitational constant
        pop_mask: Optional boolean mask for population A (True = pop A, False = pop B).
                  If None, randomly assigns f_A fraction to pop A.

    Returns:
        positions: Cartesian positions (N, 3) [length units]
        velocities: Cartesian velocities (N, 3) [velocity units]
        pop_id: Population ID array (N,) with values:
                0 = population A (extended)
                1 = population B (concentrated)

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from progenax.profiles import PlummerProfile
        >>> from progenax.kinematics import PlummerVelocityDF
        >>> from progenax.populations import TwoComponentConfig, generate_two_component_cluster
        >>>
        >>> # Setup
        >>> N = 1000
        >>> masses = jnp.ones(N)
        >>> key = jax.random.PRNGKey(42)
        >>>
        >>> # Create configuration
        >>> profile_A = PlummerProfile(r_h=2.0)  # Extended
        >>> profile_B = PlummerProfile(r_h=0.5)  # Concentrated
        >>> df_A = PlummerVelocityDF(r_h=2.0)
        >>> df_B = PlummerVelocityDF(r_h=0.5)
        >>>
        >>> config = TwoComponentConfig(
        ...     f_A=0.3,
        ...     profile_A=profile_A,
        ...     profile_B=profile_B,
        ...     velocity_df_A=df_A,
        ...     velocity_df_B=df_B,
        ... )
        >>>
        >>> # Generate cluster (random assignment)
        >>> from jaxstro.units import STELLAR
        >>> positions, velocities, pop_id = generate_two_component_cluster(
        ...     masses, config, key, G=STELLAR.G
        ... )
        >>>
        >>> # Generate cluster (custom assignment - most massive in core)
        >>> pop_mask = masses > jnp.median(masses)  # Heavy stars in core (pop B)
        >>> positions, velocities, pop_id = generate_two_component_cluster(
        ...     masses, config, key, G=STELLAR.G, pop_mask=~pop_mask
        ... )

    Notes:
        - Population A is typically extended (larger r_h)
        - Population B is typically concentrated (smaller r_h)
        - pop_id convention: 0 = A (extended), 1 = B (concentrated)
        - When pop_mask is provided, f_A is ignored
        - Fully JIT-compatible and differentiable

    References:
        Vesperini & Heggie (1997) MNRAS 289, 898 - Multi-mass dynamics
        Baumgardt et al. (2008) MNRAS 384, 1231 - Primordial segregation
    """
    N = len(masses)
    key_assign, key_pos_A, key_pos_B, key_vel_A, key_vel_B = jax.random.split(key, 5)

    # Determine population membership
    if pop_mask is None:
        # Random assignment: f_A fraction to pop A
        u = jax.random.uniform(key_assign, (N,))
        is_pop_A = u < config.f_A
    else:
        # User-specified assignment
        is_pop_A = pop_mask

    # Convert to integer pop_id (0 = A, 1 = B)
    pop_id = jnp.where(is_pop_A, 0, 1)

    # Sample positions for BOTH populations (all N particles)
    # We'll use jnp.where to select the right positions for each particle
    positions_A = config.profile_A.sample_positions(masses, key_pos_A)
    positions_B = config.profile_B.sample_positions(masses, key_pos_B)

    # Sample velocities for BOTH populations
    velocities_A = config.velocity_df_A.sample_velocities(
        positions_A, masses, key_vel_A, G=G
    )
    velocities_B = config.velocity_df_B.sample_velocities(
        positions_B, masses, key_vel_B, G=G
    )

    # Select positions and velocities based on population membership
    # is_pop_A is (N,), need to broadcast to (N, 3)
    is_pop_A_3d = is_pop_A[:, jnp.newaxis]  # (N, 1) for broadcasting

    positions = jnp.where(is_pop_A_3d, positions_A, positions_B)
    velocities = jnp.where(is_pop_A_3d, velocities_A, velocities_B)

    return positions, velocities, pop_id


__all__ = ["TwoComponentConfig", "generate_two_component_cluster"]

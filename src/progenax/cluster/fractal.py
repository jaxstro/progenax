# progenax/src/progenax/cluster/fractal.py
"""
Fractal substructure generation for star cluster initial conditions.

Implements the Goodwin-Whitworth (2004) / McLuster algorithm for generating
fractal distributions with tunable fractal dimension D.

Algorithm:
1. Build hierarchical octree fractal in cube [-1, 1]^3
2. Cut to unit sphere (r <= 1)
3. Subsample to N_stars
4. Profile mapping via rank-based radial rescaling
5. Velocity virial scaling

References:
    Goodwin & Whitworth (2004) A&A 413, 929 - Fractal star cluster structure
    Küpper et al. (2011) MNRAS 417, 2300 - McLuster implementation
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array, random
from jaxtyping import Bool, Float, Int, PRNGKeyArray


def _select_k_survivors(
    key: PRNGKeyArray,
    n_parents: int,
    n_children: int,
    k: int,
) -> Bool[Array, "n_parents n_children"]:
    """
    Select exactly k survivors per parent using sorting-based sampling.

    Gumbel-max trick: assign uniform priorities, take top-k by rank.

    Args:
        key: JAX PRNG key
        n_parents: Number of parent particles
        n_children: Number of children per parent
        k: Number of survivors to select per parent

    Returns:
        Boolean mask (n_parents, n_children) with exactly k True per row.
    """
    priorities = random.uniform(key, (n_parents, n_children))
    ranks = jnp.argsort(jnp.argsort(priorities, axis=1), axis=1)
    return ranks < k


def generate_fractal_positions(
    key: PRNGKeyArray,
    N_stars: int,
    D: float = 2.0,
    N_div: int = 2,
    g_max: int = 6,
    oversample_factor: float = 8.0,
    forced: bool = True,
    enforce_octant_symmetry: bool = True,
) -> Tuple[Float[Array, "N 3"], Float[Array, "N 3"], Int[Array, "N"]]:
    """
    Generate fractal star cluster in unit sphere (McLuster algorithm).

    Algorithm:
        1. Start with single particle at origin
        2. For each generation:
           a. Create N_div^3 children per parent at subcube centers
           b. Select survivors (forced-k or probabilistic)
           c. Shrink delta, add position and velocity noise
        3. Cut to unit sphere (r <= 1)
        4. Subsample to N_stars
        5. Remove COM drift

    Args:
        key: JAX PRNG key
        N_stars: Target number of stars
        D: Fractal dimension in [1.3, 3.0]. D=3 is uniform, D~1.6 is clumpy.
        N_div: Subdivision factor per axis (must be 2)
        g_max: Number of generations
        oversample_factor: Pool size multiplier for sphere cut
        forced: If True, exactly k=round(N_div^D) survivors per parent
        enforce_octant_symmetry: If True, keep all 8 children at level 0

    Returns:
        Tuple of (positions, velocities, generation_index):
            - positions: (N_stars, 3) in unit sphere (r <= 1), centered on COM
            - velocities: (N_stars, 3) raw from random walk (NOT virialized)
            - generation_index: (N_stars,) which generation each particle last
              survived into (0 = root, 1..g_max = subsequent levels).
              NOTE: This is NOT a clump ID - particles with same generation
              index are not necessarily siblings. For clump grouping, use
              spatial proximity instead.

    Raises:
        ValueError: If not enough survivors inside unit sphere
        AssertionError: If N_div != 2
    """
    assert N_div == 2, f"Only N_div=2 supported, got {N_div}"

    # Clamp D to valid range
    D_clamped = jnp.clip(D, 1.3, 3.0)

    # Survival probability and forced-k count
    p_survive = jnp.power(float(N_div), D_clamped - 3.0)
    k_survivors = jnp.rint(jnp.power(float(N_div), D_clamped)).astype(jnp.int32)
    n_children = N_div ** 3  # 8
    k_survivors = jnp.clip(k_survivors, 1, n_children)

    # Pool size with oversampling for sphere cut
    N_pool = int(min(oversample_factor * N_stars * 16, 1_000_000))

    # Octant center offsets: (±0.5, ±0.5, ±0.5) normalized
    # NOT corners! Centers of the 8 subcubes.
    octant_offsets = jnp.array([
        [-0.5, -0.5, -0.5],
        [-0.5, -0.5, +0.5],
        [-0.5, +0.5, -0.5],
        [-0.5, +0.5, +0.5],
        [+0.5, -0.5, -0.5],
        [+0.5, -0.5, +0.5],
        [+0.5, +0.5, -0.5],
        [+0.5, +0.5, +0.5],
    ], dtype=jnp.float32)  # Shape: (8, 3)

    def generate_level(carry, level_idx):
        """Generate one level of the fractal tree."""
        positions, velocities, ancestry, alive_mask, delta, key = carry
        key, key_survivors, key_pos_noise, key_vel_noise = random.split(key, 4)

        # 1. Build children at subcube centers (NOT corners!)
        # child_center = parent + delta * octant_offset
        centers = positions[:, None, :] + delta * octant_offsets[None, :, :]
        # Shape: (N_pool, 8, 3)

        # Inherit parent velocities
        children_vel = jnp.broadcast_to(
            velocities[:, None, :], (N_pool, n_children, 3)
        )

        # 2. Survivor selection
        if enforce_octant_symmetry:
            # Level 0: keep all 8 children (no culling)
            # Level 1+: apply normal survivor selection
            use_full_survival = level_idx == 0
        else:
            use_full_survival = False

        if forced:
            # Forced-k mode: exactly k survivors per parent
            survivor_mask = _select_k_survivors(
                key_survivors, N_pool, n_children, k_survivors
            )
            # Override for level 0 if enforce_octant_symmetry
            survivor_mask = jax.lax.cond(
                use_full_survival,
                lambda m: jnp.ones_like(m, dtype=bool),
                lambda m: m,
                survivor_mask,
            )
        else:
            # Probabilistic mode
            survival_samples = random.uniform(key_survivors, (N_pool, n_children))
            survivor_mask = survival_samples < p_survive
            survivor_mask = jax.lax.cond(
                use_full_survival,
                lambda m: jnp.ones_like(m, dtype=bool),
                lambda m: m,
                survivor_mask,
            )

        # 3. Parent alive mask repeated for each child
        parent_alive = jnp.repeat(alive_mask[:, None], n_children, axis=1)
        combined_mask = parent_alive & survivor_mask  # (N_pool, 8)

        # 4. Flatten and compact
        flat_pos = centers.reshape(-1, 3)
        flat_vel = children_vel.reshape(-1, 3)
        flat_mask = combined_mask.ravel()

        # Compact using sorted indices (JAX-compatible)
        max_idx = N_pool * n_children
        priorities = jnp.where(flat_mask, jnp.arange(max_idx), max_idx)
        sorted_indices = jnp.argsort(priorities)

        compact_indices = sorted_indices[:N_pool]
        compacted_pos = flat_pos[compact_indices]
        compacted_vel = flat_vel[compact_indices]

        n_alive = jnp.sum(flat_mask)
        new_alive_mask = jnp.arange(N_pool) < n_alive

        # 5. Shrink delta THEN add noise (AMUSE pattern)
        delta_new = delta / N_div

        # Position noise: uniform in [-delta_new, +delta_new]
        pos_noise = (
            2.0 * random.uniform(key_pos_noise, (N_pool, 3)) - 1.0
        ) * delta_new

        # Velocity noise: Gaussian scaled by delta_new (hierarchical random walk)
        vel_noise = random.normal(key_vel_noise, (N_pool, 3)) * delta_new

        # Apply noise only to alive particles
        new_positions = jnp.where(
            new_alive_mask[:, None], compacted_pos + pos_noise, compacted_pos
        )
        new_velocities = jnp.where(
            new_alive_mask[:, None], compacted_vel + vel_noise, compacted_vel
        )

        # Update ancestry with generation index
        new_ancestry = jnp.where(
            new_alive_mask, jnp.int32(level_idx + 1), ancestry
        )

        return (
            new_positions, new_velocities, new_ancestry,
            new_alive_mask, delta_new, key
        ), None

    # Initialize with root at origin
    initial_positions = jnp.zeros((N_pool, 3))
    initial_velocities = jnp.zeros((N_pool, 3))
    initial_ancestry = jnp.zeros(N_pool, dtype=jnp.int32)
    initial_alive_mask = jnp.zeros(N_pool, dtype=bool).at[0].set(True)
    initial_delta = 1.0  # Start with delta = 1.0 (full cube scale)

    # Run through generations
    (final_pos, final_vel, final_ancestry, final_alive, _, key), _ = jax.lax.scan(
        generate_level,
        (initial_positions, initial_velocities, initial_ancestry,
         initial_alive_mask, initial_delta, key),
        jnp.arange(g_max),
    )

    # =========================================================================
    # REMOVE COM DRIFT (BEFORE sphere cut to preserve r <= 1 constraint)
    # =========================================================================
    # COM is computed using only alive particles
    n_alive_total = jnp.sum(final_alive)
    com = jnp.sum(
        jnp.where(final_alive[:, None], final_pos, 0.0), axis=0
    ) / jnp.maximum(n_alive_total, 1)

    final_pos_centered = final_pos - com

    # =========================================================================
    # SPHERE CUT: Select particles with r <= 1.0
    # =========================================================================
    radii = jnp.linalg.norm(final_pos_centered, axis=1)
    in_sphere = (radii <= 1.0) & final_alive

    # Get indices of valid particles (inside sphere and alive)
    # Use size=N_pool to get fixed-size output for JIT compatibility
    # Invalid entries will be at the end after sorting
    priorities = jnp.where(in_sphere, jnp.arange(N_pool), N_pool)
    sorted_valid_indices = jnp.argsort(priorities)

    # Count valid particles
    N_valid = jnp.sum(in_sphere)

    # =========================================================================
    # SUBSAMPLE to N_stars
    # =========================================================================
    # Use categorical sampling from valid particles.
    # This works for both N_valid >= N_stars (sampling without much repetition)
    # and N_valid < N_stars (sampling with necessary repetition).
    key, subkey = random.split(key)

    # Create uniform weights over valid particles
    weights = jnp.where(
        jnp.arange(N_pool) < N_valid,
        1.0 / jnp.maximum(N_valid, 1.0),
        0.0
    )

    # Sample N_stars indices using categorical sampling
    # This automatically handles both cases:
    # - N_valid >= N_stars: low probability of duplication
    # - N_valid < N_stars: will duplicate as needed
    chosen_relative = random.choice(
        subkey,
        N_pool,
        shape=(N_stars,),
        replace=True,
        p=weights,
    )

    # Map to actual particle indices via sorted_valid_indices
    chosen = sorted_valid_indices[chosen_relative]

    positions_out = final_pos_centered[chosen]
    velocities_out = final_vel[chosen]
    ancestry_out = final_ancestry[chosen]

    return positions_out, velocities_out, ancestry_out


def rescale_fractal_to_target_radii(
    positions: Float[Array, "N 3"],
    target_radii: Float[Array, "N"],
) -> Float[Array, "N 3"]:
    """
    Rescale fractal positions to match target radial distribution.

    Rank-based mapping: i-th smallest fractal radius maps to i-th smallest
    target radius, preserving angular structure.

    Args:
        positions: (N, 3) positions from generate_fractal_positions
        target_radii: (N,) desired radii from profile sampling

    Returns:
        positions_rescaled: (N, 3) with matching radial distribution
    """
    r_frac = jnp.linalg.norm(positions, axis=1)
    N = r_frac.shape[0]

    # Sort indices
    idx_frac_sorted = jnp.argsort(r_frac)
    r_target_sorted = jnp.sort(target_radii)

    # Map: i-th smallest fractal radius gets i-th smallest target radius
    r_mapped = jnp.zeros(N, dtype=r_frac.dtype)
    r_mapped = r_mapped.at[idx_frac_sorted].set(r_target_sorted)

    # Scale positions preserving direction
    eps = 1e-10
    r_frac_safe = jnp.maximum(r_frac, eps)
    scale = r_mapped / r_frac_safe

    return positions * scale[:, None]


def rescale_velocities_to_virial(
    positions: Float[Array, "N 3"],
    velocities: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    target_Q_vir: float = 0.5,
    G: Optional[float] = None,
    softening: float = 1e-4,
) -> Float[Array, "N 3"]:
    """
    Rescale velocities to achieve target virial ratio Q = K/|U|.

    Algorithm:
        1. Remove COM velocity from input
        2. Compute potential energy U
        3. Compute current kinetic energy K_raw (in COM frame)
        4. Scale velocities by sqrt(K_target / K_raw)

    Args:
        positions: (N, 3) particle positions
        velocities: (N, 3) raw velocities from fractal random walk
        masses: (N,) particle masses
        target_Q_vir: Desired virial ratio (0.5 = equilibrium)
        G: Gravitational constant (default: jaxstro.units.STELLAR.G)
        softening: Softening length for potential calculation

    Returns:
        velocities_scaled: (N, 3) with Q = target_Q_vir (in COM frame)
    """
    if G is None:
        from jaxstro.units import STELLAR
        G = STELLAR.G

    from progenax.dynamics.virial import compute_potential_energy

    # Remove COM velocity FIRST (to measure K in COM frame)
    M_total = jnp.sum(masses)
    v_com = jnp.sum(masses[:, None] * velocities, axis=0) / M_total
    velocities_com_frame = velocities - v_com

    # Compute potential energy
    U = compute_potential_energy(positions, masses, G=G, softening=softening)

    # Current kinetic energy (in COM frame)
    K_raw = 0.5 * jnp.sum(masses[:, None] * velocities_com_frame**2)

    # Target kinetic energy
    K_target = target_Q_vir * jnp.abs(U)

    # Scale velocities (already in COM frame, so result stays in COM frame)
    scale = jnp.sqrt(K_target / jnp.maximum(K_raw, 1e-12))
    velocities_scaled = velocities_com_frame * scale

    return velocities_scaled


def assign_velocities_and_virialize(
    key: PRNGKeyArray,
    positions: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    target_Q_vir: float = 0.5,
    ancestry: Optional[Float[Array, "N"]] = None,
    coherent: bool = True,
    G: Optional[float] = None,
    softening: float = 1e-4,
) -> Float[Array, "N 3"]:
    """
    Assign velocities and scale to target virial ratio.

    DEPRECATED: Use rescale_velocities_to_virial with fractal-generated
    velocities instead. This function is kept for backward compatibility.

    Args:
        key: JAX random key
        positions: (N, 3) particle positions
        masses: (N,) particle masses
        target_Q_vir: Target virial ratio (0.5 = equilibrium)
        ancestry: Generation indices (unused in new implementation)
        coherent: Whether to use coherent velocities (ignored)
        G: Gravitational constant
        softening: Softening length

    Returns:
        velocities: (N, 3) scaled to target_Q_vir
    """
    if G is None:
        from jaxstro.units import STELLAR
        G = STELLAR.G

    N = masses.shape[0]

    # Generate random velocities
    velocities_raw = random.normal(key, (N, 3))

    # Use the new function
    return rescale_velocities_to_virial(
        positions, velocities_raw, masses,
        target_Q_vir=target_Q_vir, G=G, softening=softening,
    )


__all__ = [
    "generate_fractal_positions",
    "rescale_fractal_to_target_radii",
    "rescale_velocities_to_virial",
    "assign_velocities_and_virialize",
]

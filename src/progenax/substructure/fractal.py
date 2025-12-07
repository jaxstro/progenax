"""Fractal substructure generation for star cluster ICs.

Implements the Goodwin-Whitworth (2004) algorithm for generating fractal
distributions with tunable fractal dimension D. Uses static allocation
with masks for JIT compatibility.

References:
    Goodwin & Whitworth (2004) A&A 413, 929 - Fractal star cluster structure
    Kupper et al. (2011) MNRAS 417, 2300 - McLuster implementation
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


def generate_fractal_positions(
    n_stars: int,
    key: PRNGKeyArray,
    d_fractal: float,
    g_max: int = 6,
) -> Float[Array, "n_stars 3"]:
    """Generate fractal positions using Goodwin-Whitworth algorithm.

    Implements recursive subdivision where each parent spawns 8 children
    in a cube, with survival probability p = 2^(D-3). Uses adaptive depth
    and static allocation for JIT compatibility.

    Algorithm:
        1. Start with root cell at origin
        2. For each generation g:
            - Each parent spawns 8 children (cube subdivision)
            - Each child survives with probability p = 2^(D-3)
            - Child positions: parent ± offset/2^g in each dimension
        3. Continue for g_adapt = ceil(log₂(N)/D) + 1 generations (capped at g_max)
        4. Filter particles to unit sphere
        5. Downsample to exactly N particles

    Args:
        n_stars: Target number of particles
        key: JAX random key for reproducibility
        d_fractal: Fractal dimension D ∈ [1.5, 3.0]
            - D = 1.5: Very clumpy (p = 0.35)
            - D = 2.0: Moderately clumpy (p = 0.5)
            - D = 2.6: Weakly clumpy (p = 0.76)
            - D = 3.0: Uniform (p = 1.0)
        g_max: Maximum recursion depth (default 6, gives 8^6 = 262,144 cells)

    Returns:
        Fractal positions (n_stars, 3) in unit sphere

    References:
        Goodwin & Whitworth (2004) A&A 413, 929
        Kupper et al. (2011) MNRAS 417, 2300 - McLuster

    Note:
        Uses static allocation for 8^g_max particles with survival masks
        for JIT compatibility. Not differentiable due to random sampling
        and downsampling (use blend overlay for differentiable applications).

        For JIT compatibility, always uses g_max generations. For optimal
        performance, set g_max = ceil(log₂(N)/D) + 1, but g_max=6 works
        well for most applications (up to N~10,000).
    """
    # Use g_max directly for JIT compatibility
    # Adaptive depth would be: g = ceil(log₂(N)/D) + 1
    # But computing this inside requires concrete n_stars value
    g = g_max

    # Survival probability
    p_survive = jnp.power(2.0, d_fractal - 3.0)

    # Static allocation for maximum possible particles
    max_particles = 8**g

    # Initialize root particle at origin
    # We'll build up generations using scan
    def generate_level(carry, level_idx):
        """Generate one level of fractal subdivision."""
        positions, alive_mask, n_alive, key = carry
        key, subkey1, subkey2 = jax.random.split(key, 3)

        # Current generation size (number of alive parents)
        # Each alive parent spawns 8 children

        # Generate offsets for 8 children (±1 in each dimension)
        # Offsets shape: (8, 3)
        child_offsets = jnp.array([
            [-1, -1, -1],
            [-1, -1, +1],
            [-1, +1, -1],
            [-1, +1, +1],
            [+1, -1, -1],
            [+1, -1, +1],
            [+1, +1, -1],
            [+1, +1, +1],
        ], dtype=jnp.float32)

        # Scale factor for this generation (cell size halves each level)
        scale = 0.5 ** (level_idx + 1)

        # For each alive parent, generate 8 children
        # We'll use vmap to vectorize over parents
        def spawn_children(parent_pos):
            """Spawn 8 children from one parent."""
            # children shape: (8, 3)
            children = parent_pos[None, :] + scale * child_offsets
            return children

        # Get alive parent positions
        alive_positions = positions * alive_mask[:, None]

        # Generate all children (including from "dead" parents, we'll mask them)
        # We need to spawn 8 children per particle slot (including dead ones for static shape)
        # Then mask based on parent alive status

        # Reshape to (n_particles, 1, 3) for broadcasting
        parent_positions_broadcast = positions[:, None, :]  # (n_particles, 1, 3)
        child_offsets_broadcast = child_offsets[None, :, :]  # (1, 8, 3)

        # Generate children: (n_particles, 8, 3)
        all_children = parent_positions_broadcast + scale * child_offsets_broadcast

        # Flatten to (n_particles * 8, 3)
        all_children_flat = all_children.reshape(-1, 3)

        # Parent alive mask repeated 8 times: (n_particles * 8,)
        parent_alive_repeated = jnp.repeat(alive_mask, 8)

        # Child survival based on probability p_survive
        survival_samples = jax.random.uniform(subkey1, (max_particles * 8,))
        child_survives = survival_samples < p_survive

        # Combined mask: parent alive AND child survives
        new_alive_mask = parent_alive_repeated & child_survives

        # Pad/truncate to max_particles
        new_positions = all_children_flat[:max_particles]
        new_alive_mask = new_alive_mask[:max_particles]

        new_n_alive = jnp.sum(new_alive_mask)

        return (new_positions, new_alive_mask, new_n_alive, key), None

    # Initialize with root particle
    initial_positions = jnp.zeros((max_particles, 3))
    initial_alive = jnp.zeros(max_particles, dtype=bool).at[0].set(True)
    initial_n_alive = 1

    # Run through generations
    (final_positions, final_alive, final_n_alive, _), _ = jax.lax.scan(
        generate_level,
        (initial_positions, initial_alive, initial_n_alive, key),
        jnp.arange(g)
    )

    # Filter to alive particles and unit sphere using masking (JIT-compatible)
    # Compute radii for all particles
    radii = jnp.linalg.norm(final_positions, axis=1)

    # Combined mask: alive AND in unit sphere
    valid_mask = final_alive & (radii <= 1.0)

    # Use jnp.where to get valid indices (JIT-compatible)
    # This gives us indices of valid particles
    valid_indices = jnp.where(valid_mask, size=max_particles, fill_value=0)[0]
    n_valid = jnp.sum(valid_mask).astype(jnp.int32)

    # Take first n_valid particles (rest are duplicates of index 0)
    # We'll use jax.random.choice to select n_stars from the valid ones
    key, subkey = jax.random.split(key)

    # Sample from valid particles (with replacement if needed)
    # Use valid_indices[:n_valid] but this requires concrete n_valid
    # Instead, sample from all max_particles with weighted probability
    weights = valid_mask.astype(jnp.float32)
    weights = weights / jnp.maximum(jnp.sum(weights), 1.0)

    sampled_indices = jax.random.choice(
        subkey, max_particles, shape=(n_stars,), replace=True, p=weights
    )
    final_positions_out = final_positions[sampled_indices]

    return final_positions_out


def apply_fractal_overlay_radial(
    positions_smooth: Float[Array, "N 3"],
    key: PRNGKeyArray,
    d_fractal: float,
) -> Float[Array, "N 3"]:
    """Apply fractal overlay preserving radial distribution (McLuster-style).

    This is the PRIMARY overlay method. It preserves the input radial density
    profile exactly while replacing the angular structure with fractal clumps.

    Algorithm:
        1. Generate fractal positions for N particles
        2. Sort both smooth and fractal positions by radius
        3. Match by rank: smooth radius + fractal direction
        4. x_final[k] = r_smooth[k] * (x_fractal[k] / |x_fractal[k]|)

    This ensures:
        - Radial density profile unchanged (exact)
        - Angular distribution becomes fractal/clumpy
        - Total mass distribution preserved

    Args:
        positions_smooth: Input positions with desired radial profile (N, 3)
        key: JAX random key for fractal generation
        d_fractal: Fractal dimension D ∈ [1.5, 3.0]

    Returns:
        Positions with fractal angular structure (N, 3)

    References:
        Kupper et al. (2011) MNRAS 417, 2300 - McLuster implementation

    Note:
        This is the recommended method for most applications as it preserves
        the physics-motivated radial density profile (Plummer, King, etc.)
        while adding observationally-motivated clumpy structure.
    """
    N = positions_smooth.shape[0]

    # Generate fractal positions
    positions_fractal = generate_fractal_positions(N, key, d_fractal)

    # Compute radii for both distributions
    radii_smooth = jnp.linalg.norm(positions_smooth, axis=1)
    radii_fractal = jnp.linalg.norm(positions_fractal, axis=1)

    # Sort by radius
    idx_smooth = jnp.argsort(radii_smooth)
    idx_fractal = jnp.argsort(radii_fractal)

    # Match by rank: take smooth radius, fractal direction
    # For each rank k: r_final = r_smooth[k], direction from fractal[k]
    positions_smooth_sorted = positions_smooth[idx_smooth]
    positions_fractal_sorted = positions_fractal[idx_fractal]

    radii_smooth_sorted = radii_smooth[idx_smooth]
    radii_fractal_sorted = radii_fractal[idx_fractal]

    # Compute fractal unit directions
    fractal_directions = positions_fractal_sorted / jnp.maximum(
        radii_fractal_sorted[:, None], 1e-10
    )

    # Combine: smooth radii + fractal directions
    positions_combined_sorted = radii_smooth_sorted[:, None] * fractal_directions

    # Unsort back to original order
    # Create inverse permutation
    positions_out = jnp.zeros_like(positions_smooth)
    positions_out = positions_out.at[idx_smooth].set(positions_combined_sorted)

    return positions_out


def apply_fractal_overlay_blend(
    positions_smooth: Float[Array, "N 3"],
    key: PRNGKeyArray,
    d_fractal: float,
    lambda_frac: float,
) -> Float[Array, "N 3"]:
    """Apply fractal overlay using linear blending (experimental).

    This is an EXPERIMENTAL alternative that is fully differentiable but
    modifies the radial density profile.

    Algorithm:
        1. Generate fractal positions for N particles
        2. Scale fractal to match smooth radial extent
        3. Blend: x_final = (1-λ) * x_smooth + λ * x_fractal_scaled

    Args:
        positions_smooth: Input positions with desired radial profile (N, 3)
        key: JAX random key for fractal generation
        d_fractal: Fractal dimension D ∈ [1.5, 3.0]
        lambda_frac: Blending parameter λ ∈ [0, 1]
            - λ = 0: Pure smooth distribution (no fractal)
            - λ = 1: Pure fractal distribution (scaled)
            - 0 < λ < 1: Linear blend

    Returns:
        Blended positions (N, 3)

    Note:
        This method is fully differentiable (unlike radial overlay) but
        MODIFIES the radial density profile. Use for gradient-based
        applications where you can tolerate profile changes, or when
        you want smooth transitions between smooth and fractal.

        For most applications, prefer apply_fractal_overlay_radial() which
        preserves the radial profile exactly.
    """
    N = positions_smooth.shape[0]

    # Generate fractal positions
    positions_fractal = generate_fractal_positions(N, key, d_fractal)

    # Scale fractal to match smooth radial extent
    # Match RMS radius
    rms_smooth = jnp.sqrt(jnp.mean(jnp.sum(positions_smooth**2, axis=1)))
    rms_fractal = jnp.sqrt(jnp.mean(jnp.sum(positions_fractal**2, axis=1)))
    scale_factor = rms_smooth / jnp.maximum(rms_fractal, 1e-10)
    positions_fractal_scaled = positions_fractal * scale_factor

    # Linear blend
    positions_out = (1.0 - lambda_frac) * positions_smooth + lambda_frac * positions_fractal_scaled

    return positions_out


__all__ = [
    "generate_fractal_positions",
    "apply_fractal_overlay_radial",
    "apply_fractal_overlay_blend",
]

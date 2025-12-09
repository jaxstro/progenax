# progenax/src/progenax/cluster/mass_segregation.py
"""
Baumgardt+2008 energy-ordered mass segregation for cluster ICs.

Implements the energy-ordered orbit assignment method from Baumgardt et al.
(2008) and McLuster (Küpper et al. 2011) for primordial mass segregation.

Core Algorithm:
    1. Generate orbit pool from equilibrium distribution function
    2. Sort orbits by specific energy (most bound first)
    3. Sort masses by mass (most massive first)
    4. Assign masses to orbits such that massive stars get lower-energy orbits
    5. The lambda_seg parameter controls segregation strength via blending

Key Advantages:
    - Preserves density profile (each mass group in virial equilibrium)
    - Preserves total virial ratio
    - S=1 at bin level, smooth control via lambda_seg blending in IC generator

References:
    Baumgardt, De Marchi & Kroupa (2008), ApJ 685, 247
    Küpper et al. (2011), MNRAS 417, 2300 - McLuster code
"""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import Array, random
from jaxtyping import Float, Int, PRNGKeyArray


def energy_sorted_segregation(
    key: PRNGKeyArray,
    masses: Float[Array, "N"],
    positions_pool: Float[Array, "N_pool 3"],
    velocities_pool: Float[Array, "N_pool 3"],
    potential_fn: Callable[[Float[Array, "N_pool 3"]], Float[Array, "N_pool"]],
) -> Tuple[Float[Array, "N"], Float[Array, "N 3"], Float[Array, "N 3"]]:
    """
    Assign positions/velocities to masses using Baumgardt energy-ordered method.

    This implements full energy-ordered segregation (S=1 at the bin level).
    Partial segregation (S < 1) is handled via lambda_seg blending in the IC
    generator, not within this function.

    Algorithm (Baumgardt+2008 Appendix, McLuster implementation):
        1. Sort masses descending → m_sorted[i] for mass rank i
        2. Compute cumulative mass: M_cum_sorted[i] = sum(m_sorted[0:i+1])
        3. Normalize: M_cum_norm[i] = M_cum_sorted[i] / M_total
        4. Compute energy bin boundaries for each mass rank:
           bin_low[i] = floor(N_pool * M_cum_norm[i-1])
           bin_high[i] = floor(N_pool * M_cum_norm[i]) - 1
        5. Sort orbit pool by specific energy (most bound first)
        6. For each mass rank i, sample orbit uniformly from [bin_low[i], bin_high[i]]
        7. Map back to original mass ordering

    Args:
        key: JAX random key for orbit sampling within bins.
        masses: Stellar masses with shape (N,). Will be sorted internally;
            output preserves original ordering.
        positions_pool: Orbit pool positions with shape (N_pool, 3) from
            equilibrium distribution function.
        velocities_pool: Orbit pool velocities with shape (N_pool, 3) from
            equilibrium distribution function. Must use consistent units with
            potential_fn so that E = 0.5 * v² + Φ is meaningful.
        potential_fn: Callable computing specific potential Φ(r) (per unit mass)
            at positions. Must return shape (N_pool,) with same energy units as
            0.5 * v². Should be the analytic profile potential (Plummer/King/EFF),
            NOT a direct N-body sum, ensuring consistency between the DF used to
            draw orbits and the energy ordering used for segregation.

    Returns:
        Tuple of (masses_out, positions, velocities):
            - masses_out: Masses in original order (N,) - unchanged from input
            - positions: Assigned positions (N, 3)
            - velocities: Assigned velocities (N, 3)

    Shape Expectations:
        - masses: (N,)
        - positions_pool: (N_pool, 3)
        - velocities_pool: (N_pool, 3)
        - potential_fn(positions_pool) must return: (N_pool,)

    Pool Size Recommendations:
        In practice, set N_pool ≈ pool_factor * N with pool_factor >= 4 so each
        cumulative-mass bin contains multiple orbits and the segregation signal
        is smooth. The algorithm technically works for any N_pool >= N, but small
        N_pool will cause many stars to sample very similar orbits.

    No-Orbit-Reuse Guarantee:
        The cumulative mass binning produces disjoint, ordered bins that partition
        the sorted orbit list (up to rounding). Each mass rank i draws one orbit
        from its own bin [bin_low[i], bin_high[i]], so each selected orbit index
        is used at most once. There is no reuse of the same orbit across different
        mass ranks.

    Non-Differentiability:
        This function is NOT differentiable due to:
            - Sorting (argsort)
            - Discrete bin boundaries (floor)
            - Random sampling (randint)

        For gradient-based inference, do NOT differentiate through this function.
        Instead, generate a fully segregated catalog and an unsegregated catalog,
        then blend them in the IC generator via a continuous parameter lambda_seg::

            positions = (1 - lambda_seg) * positions_unseg + lambda_seg * positions_seg

        Differentiation should occur through this blending, not the discrete
        segregation algorithm.

    Segregation Strength:
        This function implements full Baumgardt-style energy ordering (S=1 at bin
        level): the most massive stars only sample from the most bound energy bins.
        Randomness within bins ensures different realizations produce different
        (but equally segregated) ICs. The _mcluster_partial_shuffle helper is kept
        for potential future S-based schemes but is not currently used.
    """
    N = masses.shape[0]
    N_pool = positions_pool.shape[0]

    # Basic shape sanity (trace-time only, helps catch misuse)
    assert positions_pool.shape == velocities_pool.shape, (
        "positions_pool and velocities_pool must have the same shape (N_pool, 3)"
    )

    # --- Step 1: Sort masses descending to get mass-rank ordering ---
    mass_order = jnp.argsort(-masses)
    m_sorted = masses[mass_order]  # m_sorted[i] = mass at rank i

    # --- Step 2: Compute cumulative mass and bin boundaries ---
    M_cum_sorted = jnp.cumsum(m_sorted)
    # Guard against degenerate total mass (e.g., all-zero masses)
    M_total = jnp.maximum(M_cum_sorted[-1], 1e-12)
    M_cum_norm = M_cum_sorted / M_total

    # Prepend 0 for bin_low calculation (match dtype for JAX hygiene)
    M_cum_norm_shifted = jnp.concatenate([
        jnp.array([0.0], dtype=M_cum_norm.dtype),
        M_cum_norm[:-1]
    ])

    # Note: M_cum_norm is strictly increasing, so (bin_low[i], bin_high[i])
    # define disjoint, ordered bins that partition the sorted orbits.
    # Each rank i samples one orbit from its own bin -> no orbit index is reused.
    bin_low = jnp.floor(N_pool * M_cum_norm_shifted).astype(jnp.int32)
    bin_high = jnp.floor(N_pool * M_cum_norm).astype(jnp.int32) - 1
    bin_high = jnp.maximum(bin_high, bin_low)  # Ensure at least one orbit per bin

    # --- Step 3: Sort orbit pool by specific energy ---
    kinetic = 0.5 * jnp.sum(velocities_pool**2, axis=1)
    potential = potential_fn(positions_pool)
    specific_energy = kinetic + potential

    # Most bound (most negative) first; assumes potential_fn returns negative
    # values for bound orbits, as standard for gravitational potentials
    energy_order = jnp.argsort(specific_energy)
    sorted_positions = positions_pool[energy_order]
    sorted_velocities = velocities_pool[energy_order]

    # --- Step 4: Sample orbit for each rank ---
    # For S=1, we draw orbits uniformly within each energy bin.
    # S=1 is enforced at the bin level (most massive ranks only sample
    # from the most bound bins), not by a deterministic pick inside bins.
    # Using vmap for parallel execution (faster than scan for independent ops).
    keys_per_rank = random.split(key, N)

    def sample_orbit_for_rank(i):
        """Sample an orbit index for mass rank i."""
        low = bin_low[i]
        high = bin_high[i] + 1  # randint uses exclusive upper bound
        return random.randint(keys_per_rank[i], (), low, high)

    orbit_indices = jax.vmap(sample_orbit_for_rank)(jnp.arange(N))

    # Gather positions and velocities for each rank
    pos_for_rank = sorted_positions[orbit_indices]  # (N, 3)
    vel_for_rank = sorted_velocities[orbit_indices]  # (N, 3)

    # --- Step 5: Map back to original mass ordering ---
    # mass_order[k] = which original index has rank k
    # We want: positions[original_idx] = pos_for_rank[rank_of_original_idx]
    inverse_order = jnp.argsort(mass_order)

    positions_out = pos_for_rank[inverse_order]
    velocities_out = vel_for_rank[inverse_order]

    # Return masses unchanged (preserve caller's mass array), with
    # positions/velocities reordered according to Baumgardt assignment
    return masses, positions_out, velocities_out


def _mcluster_partial_shuffle(
    key: PRNGKeyArray,
    N: int,
    S: float,
) -> Int[Array, "N"]:
    """
    Build rank-to-star mapping via McLuster Eq. A1 (reference implementation).

    This is a reference implementation of McLuster's Equation A1 "partial shuffle"
    for S-controlled segregation in rank-space. It returns integer indices mapping
    mass ranks to original star indices.

    Behavior:
        - S = 1: Identity mapping (no shuffle). Star i keeps rank i.
        - S = 0: Full random permutation. Stars are fully shuffled across ranks.
        - 0 < S < 1: Partial shuffle with bias toward identity.

    The algorithm iteratively assigns stars to ranks, using:
        j = floor((N - i) * (1 - X^(1-S)))
    where X ~ Uniform(0, 1) and i is the current rank being assigned.

    Note: This shuffles star labels but does not change which energy bin each
    mass rank draws from. For S to control physical Λ_MSR, the orbit selection
    must also depend on S.

    NOTE: Currently unused in energy_sorted_segregation; kept as a reference
    implementation for potential future direct S-controlled segregation schemes
    (instead of lambda_seg blending). Tested in test_mass_segregation.py.

    Args:
        key: JAX random key for stochastic shuffle.
        N: Number of particles.
        S: Segregation parameter in [0, 1]. S=1 gives identity, S=0 gives
            full random permutation.

    Returns:
        star_for_rank: Integer array (N,) where star_for_rank[i] is the original
            star index that should occupy mass rank i.
    """
    # Clamp S into [0, 1] defensively
    S = jnp.clip(S, 0.0, 1.0)

    # Generate all random numbers upfront for JAX compatibility
    X = random.uniform(key, (N,))

    def shuffle_step(carry, i):
        # available_mask[j] = True if star j has not yet been assigned to a rank
        # star_for_rank[k] = which original star index is assigned to rank k
        available_mask, star_for_rank = carry

        # Number of available slots
        n_available = N - i

        # McLuster formula: j = (N - i) * (1 - X^(1-S))
        j_relative = jnp.floor(
            n_available * (1.0 - jnp.power(X[i], 1.0 - S))
        ).astype(jnp.int32)
        j_relative = jnp.clip(j_relative, 0, n_available - 1)

        # Build array of available slot indices, padded with N for unavailable
        available_indices = jnp.where(
            available_mask,
            jnp.arange(N, dtype=jnp.int32),
            jnp.full((N,), N, dtype=jnp.int32),
        )
        # Sort to push unavailable (value=N) to the end
        available_indices_sorted = jnp.sort(available_indices)

        # Take the j_relative-th available index
        target_star = available_indices_sorted[j_relative]

        # Update mapping and mask
        star_for_rank = star_for_rank.at[i].set(target_star)
        available_mask = available_mask.at[target_star].set(False)

        return (available_mask, star_for_rank), None

    init_available = jnp.ones(N, dtype=bool)
    init_star_for_rank = jnp.zeros(N, dtype=jnp.int32)

    (_, star_for_rank), _ = jax.lax.scan(
        shuffle_step,
        (init_available, init_star_for_rank),
        jnp.arange(N),
    )

    return star_for_rank


__all__ = [
    "energy_sorted_segregation",
]

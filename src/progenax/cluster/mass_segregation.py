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
        2. Cumulative-mass coordinate M_cum_norm[i] = sum(m_sorted[0:i+1]) / M_total,
           and the bin-centre target energy-rank t[i] = floor(N_pool * M_cum_mid[i]),
           M_cum_mid[i] = (M_cum_norm[i-1] + M_cum_norm[i]) / 2
        3. Sort orbit pool by specific energy (most bound first)
        4. Assign each mass rank a DISTINCT orbit by isotonic-rounding the targets to a
           strictly increasing integer sequence in [0, N_pool-1]:
               idx[i] = max(idx[i-1] + 1, min(t[i], N_pool - N + i))
           (the per-rank upper clamp prevents overflow; the running max guarantees no
           reuse). Most massive rank 0 -> smallest index -> most bound orbit.
        5. Map back to original mass ordering

    This deterministic monotonic assignment replaced an earlier per-bin random sampler
    whose cumulative-mass bins collapsed below one orbit for steep IMFs, forcing many
    low-mass ranks onto the same orbit (coincident stars, V = -inf). See
    docs/website/50-validation/mass-segregation.md.

    Args:
        key: JAX random key (retained for API stability; the assignment is now
            deterministic — realisation variety comes from the random orbit pool).
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
        Set N_pool = pool_factor * N with pool_factor >= 4 so the assigned orbits are
        a well-spread mass-weighted subsample of the pool and the segregation signal is
        smooth. Any N_pool >= N yields a valid no-reuse assignment; small N_pool just
        leaves fewer distinct orbits between mass ranks.

    No-Orbit-Reuse Guarantee:
        The isotonic-rounding assignment (Step 4) produces a strictly increasing integer
        sequence in [0, N_pool-1], so every mass rank receives a DISTINCT orbit and no
        two stars are coincident — for ANY mass spectrum, including steep IMFs. (The
        previous per-bin sampler did NOT guarantee this: sub-orbit bins collapsed and
        reused orbits.) Verified for uniform/bimodal/Kroupa/extreme-steep spectra in
        tests/unit/cluster/test_mass_segregation.py.

    Non-Differentiability:
        This function is NOT differentiable (argsort, floor). For gradient-based
        inference, do NOT differentiate through it; instead blend a fully segregated and
        an unsegregated catalog in the IC generator via a continuous lambda_seg::

            positions = (1 - lambda_seg) * positions_unseg + lambda_seg * positions_seg

        Differentiate through the blend, not this discrete assignment. (Note: only
        lambda_seg in {0, 1} are exact equilibria; intermediate blends drift from
        per-mass-group virial balance — see per_group_virial_ratio and the validation
        page. The first-principles partial-equilibrium alternative is the multi-mass
        LIMEPY family.)

    Segregation Strength:
        Implements full Baumgardt-style energy ordering (S=1): the most massive stars
        occupy the most bound orbits. The assignment is deterministic given the pool;
        realisation variety comes from re-drawing the random orbit pool. The
        _mcluster_partial_shuffle helper is kept as a reference for future S-based
        schemes but is not used here.
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

    # --- Step 2: Cumulative-mass coordinate (Baumgardt energy ordering) ---
    M_cum_sorted = jnp.cumsum(m_sorted)
    # Guard against degenerate total mass (e.g., all-zero masses)
    M_total = jnp.maximum(M_cum_sorted[-1], 1e-12)
    M_cum_norm = M_cum_sorted / M_total
    M_cum_norm_shifted = jnp.concatenate([
        jnp.array([0.0], dtype=M_cum_norm.dtype),
        M_cum_norm[:-1]
    ])
    # Target energy-rank for each mass rank = N_pool * (bin-centre cumulative mass).
    # Monotonic in rank (masses > 0); most massive rank 0 -> smallest target -> most bound.
    M_cum_mid = 0.5 * (M_cum_norm + M_cum_norm_shifted)
    target = jnp.floor(N_pool * M_cum_mid).astype(jnp.int32)

    # --- Step 3: Sort orbit pool by specific energy ---
    kinetic = 0.5 * jnp.sum(velocities_pool**2, axis=1)
    potential = potential_fn(positions_pool)
    specific_energy = kinetic + potential

    # Most bound (most negative) first; assumes potential_fn returns negative
    # values for bound orbits, as standard for gravitational potentials
    energy_order = jnp.argsort(specific_energy)
    sorted_positions = positions_pool[energy_order]
    sorted_velocities = velocities_pool[energy_order]

    # --- Step 4: Deterministic monotonic DISTINCT orbit assignment (no reuse) ---
    # Each mass rank gets one orbit, indices strictly increasing in rank so the
    # most massive get the most bound orbits. We isotonic-round `target` to a
    # strictly increasing integer sequence in [0, N_pool-1]:
    #   idx[i] = max(idx[i-1] + 1, min(target[i], N_pool - N + i)).
    # The per-rank upper clamp N_pool-N+i guarantees idx[N-1] <= N_pool-1 (no
    # overflow), and the running max guarantees distinctness. Since N_pool >= N a
    # valid injective assignment always exists. This replaces the old per-bin
    # sampler, whose bins collapsed below one orbit for steep IMFs and forced many
    # low-mass ranks onto the SAME orbit (coincident stars -> V = -inf). `key` is
    # retained for API stability; the assignment is now deterministic (realization
    # variety comes from the random orbit pool).
    del key
    upper = (N_pool - N) + jnp.arange(N, dtype=jnp.int32)

    def assign_step(prev, t_u):
        t, u = t_u
        cur = jnp.maximum(prev + 1, jnp.minimum(t, u))
        return cur, cur

    _, orbit_indices = jax.lax.scan(
        assign_step, jnp.array(-1, dtype=jnp.int32), (target, upper)
    )

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

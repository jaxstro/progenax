# progenax/src/progenax/cluster/mass_segregation.py
"""
PRIMORDIAL (energy-ordered) mass segregation for cluster ICs.

Implements energy-ordered orbit assignment in the spirit of Baumgardt, De
Marchi & Kroupa (2008) and McLuster (Küpper et al. 2011): the most massive
stars are placed on the most bound orbits of an equilibrium pool. This is a
*primordial-segregation generator*, not an equilibrium construction — the
fully-ordered output is the one clean state (each mass group is an energy
shell of the parent equilibrium, hence individually virial, Q_j ~ 0.5;
validated in tests/validation/test_segregation_equilibrium_physics.py).

Departure from the published recipe (deliberate, documented): Baumgardt+2008
draw orbits randomly within cumulative-mass bins; here the assignment is a
deterministic isotonic rounding of the bin-centre targets. This guarantees a
DISTINCT orbit per star for ANY mass spectrum (the random per-bin sampler
collapsed below one orbit per bin for steep IMFs, producing coincident stars
and V = -inf). Realisation variety comes from re-drawing the random pool.

For a *differentiable* segregation knob, use the first-principles
equilibrium family instead: MultiComponentCluster.from_mass_segregation(delta)
(multi-mass lowered-isothermal, w_j = mu_j^-delta — a true shared-potential
equilibrium at every delta).

Core Algorithm:
    1. Generate orbit pool from an equilibrium distribution function
    2. Sort orbits by specific energy (most bound first)
    3. Sort masses descending
    4. Assign mass rank i the isotonic-rounded cumulative-mass orbit index
       (full S=1 energy ordering; strictly increasing, no orbit reuse)

References:
    Baumgardt, De Marchi & Kroupa (2008), ApJ 685, 247
    Küpper et al. (2011), MNRAS 417, 2300 - McLuster code
"""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, PRNGKeyArray


def energy_sorted_segregation(
    key: PRNGKeyArray,
    masses: Float[Array, "N"],
    positions_pool: Float[Array, "N_pool 3"],
    velocities_pool: Float[Array, "N_pool 3"],
    potential_fn: Callable[[Float[Array, "N_pool 3"]], Float[Array, "N_pool"]],
) -> Tuple[Float[Array, "N"], Float[Array, "N 3"], Float[Array, "N 3"]]:
    """
    Assign positions/velocities to masses using Baumgardt energy-ordered method.

    This implements full energy-ordered segregation (S=1 at the bin level) —
    a PRIMORDIAL generator. For partial/continuous segregation use the
    equilibrium family (MultiComponentCluster.from_mass_segregation), not a
    blend of this function's output.

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
        This function is NOT differentiable (argsort, floor) and is not meant to
        be: it is a discrete primordial-assignment generator. For gradient-based
        inference over segregation strength, use the first-principles equilibrium
        knob instead — MultiComponentCluster.from_mass_segregation(delta), which
        is differentiable in delta and a true shared-potential equilibrium at
        every value. (The historical lambda_seg catalog blend was retired: its
        intermediate states drift from per-mass-group virial balance — see
        per_group_virial_ratio and the mass-segregation validation page.)

    Segregation Strength:
        Implements full Baumgardt-style energy ordering (S=1): the most massive
        stars occupy the most bound orbits. The assignment is deterministic given
        the pool; realisation variety comes from re-drawing the random orbit pool.
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


__all__ = [
    "energy_sorted_segregation",
]

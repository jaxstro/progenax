"""Dynamic binary diagnostics — measure the *current* binary population from state.

A primordial pairing (set at t=0 by the IC builder) goes stale under dynamical
evolution: encounters ionize soft binaries, three-body captures form new ones, and
exchanges swap partners. So the binary population at any later snapshot must be
*measured* from the current (positions, velocities, masses), not read off the IC
labels.

- :func:`relative_energy` — the differentiable two-body primitive (E_rel < 0 bound).
- :func:`find_bound_pairs` — mutual-nearest-neighbour + bound pairs (NBODY/kira
  criterion). O(N^2), **eager** (dynamic-shape pair list — a measurement, not a hot
  path), and uses ``argmin`` so it is not differentiable. (`relative_energy` is.)
- :func:`primordial_survival` — compares the current pairing to the t=0 provenance.

`find_bound_pairs` is exactly the primitive a KS/chain regularization scheme needs to
identify which close pairs to regularize.

References:
    Aarseth (2003) "Gravitational N-Body Simulations" — binary detection in clusters.
    Heggie (1975) MNRAS 173, 729 — hard/soft binaries (E_rel sign).
"""

from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from ..dynamics.virial import compute_kinetic_energy, compute_potential_energy


def relative_energy(
    r_i: Float[Array, "3"],
    r_j: Float[Array, "3"],
    v_i: Float[Array, "3"],
    v_j: Float[Array, "3"],
    m_i: Float[Array, ""],
    m_j: Float[Array, ""],
    *,
    G: float,
) -> Float[Array, ""]:
    """The (internal) two-body orbital energy of the pair (i, j).

    E_rel = ½ μ |v_j - v_i|² − G m_i m_j / |r_j - r_i|,  μ = m_i m_j / (m_i + m_j).
    E_rel < 0 ⇒ the pair is gravitationally bound. For a bound orbit of semi-major
    axis a, E_rel = −G m_i m_j / (2a). Differentiable (separation guarded).
    """
    dr = r_j - r_i
    dv = v_j - v_i
    mu = m_i * m_j / (m_i + m_j)
    sep = jnp.sqrt(jnp.sum(dr**2))
    sep_safe = jnp.maximum(sep, 1e-30)
    return 0.5 * mu * jnp.sum(dv**2) - G * m_i * m_j / sep_safe


def find_bound_pairs(
    positions: Float[Array, "N 3"],
    velocities: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    *,
    G: float,
):
    """Detect bound two-body pairs: mutual nearest neighbours with E_rel < 0.

    Returns (pair_idx, E_rel) where pair_idx is (K, 2) with i < j per row, and
    E_rel is (K,) the pair binding energies. Each particle appears in at most one
    pair (mutual-NN is a matching). O(N^2); eager (dynamic K).

    **Scaling.** The full N×N separation matrix is materialized, so this is intended for
    N ≲ a few×10^3 (memory ~ N^2). For larger snapshots use an accelerated neighbour-list
    finder (cell list / kd-tree) — ticketed for gravax, which already has the Hermite-AC
    neighbour machinery (docs/notes/2026-06-04-accelerated-bound-finder-gravax-ticket.md).
    """
    N = positions.shape[0]
    idx = jnp.arange(N)

    dr = positions[:, None, :] - positions[None, :, :]
    sep = jnp.sqrt(jnp.sum(dr**2, axis=-1))
    sep = jnp.where(jnp.eye(N, dtype=bool), jnp.inf, sep)  # exclude self
    nn = jnp.argmin(sep, axis=1)
    mutual = nn[nn] == idx

    e_with_nn = jax.vmap(
        lambda i, j: relative_energy(
            positions[i], positions[j], velocities[i], velocities[j],
            masses[i], masses[j], G=G,
        )
    )(idx, nn)
    is_member = mutual & (e_with_nn < 0.0)
    keep = is_member & (idx < nn)  # keep the low index of each mutual pair once

    keep_idx = jnp.where(keep)[0]  # eager / dynamic
    pair_idx = jnp.stack([keep_idx, nn[keep_idx]], axis=1)
    return pair_idx, e_with_nn[keep_idx]


def find_bound_multiples(
    positions: Float[Array, "N 3"],
    velocities: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    *,
    G: float,
    max_levels: int = 3,
):
    """Detect hierarchical bound systems (binaries, triples, quadruples, …).

    Iteratively collapses mutual-nearest-neighbour bound pairs into COM
    pseudo-bodies and re-runs on the reduced set, to a fixed `max_levels` depth: a
    single bound to a binary-COM becomes a triple, two binaries a quadruple, etc.
    Mutual-NN is a matching, so all disjoint merges at a level happen at once.

    Fixed-shape (N body slots, bounded `lax.scan`) ⇒ jit-safe; uses `argmin`, so
    not differentiable (a diagnostic). O(N^2) per level (materializes the N×N separation
    matrix) — intended for N ≲ a few×10^3; an accelerated neighbour-list version is
    ticketed for gravax (docs/notes/2026-06-04-accelerated-bound-finder-gravax-ticket.md).

    Args:
        max_levels: hierarchy depth to resolve (default 3 ⇒ up to ~octuples).

    Returns:
        (system_id, multiplicity), each (N,): `system_id[i]` is the body slot of
        particle i (members of one hierarchy share it); `multiplicity[i]` is the
        number of particles in that system.
    """
    N = positions.shape[0]
    idx = jnp.arange(N)

    def _level(carry, _):
        bpos, bvel, bmass, alive, label = carry
        dr = bpos[:, None, :] - bpos[None, :, :]
        sep = jnp.sqrt(jnp.sum(dr**2, axis=-1))
        sep = jnp.where(jnp.eye(N, dtype=bool), jnp.inf, sep)
        sep = jnp.where(alive[:, None] & alive[None, :], sep, jnp.inf)
        nn = jnp.argmin(sep, axis=1)
        mutual = (nn[nn] == idx) & alive & alive[nn]
        e_nn = jax.vmap(
            lambda i, j: relative_energy(
                bpos[i], bpos[j], bvel[i], bvel[j], bmass[i], bmass[j], G=G
            )
        )(idx, nn)
        pair = mutual & (e_nn < 0.0)
        keeper = pair & (idx < nn)       # low index absorbs its partner
        absorbed = pair & (idx > nn)
        p = nn

        Mtot = bmass + jnp.where(keeper, bmass[p], 0.0)
        Mtot_safe = jnp.where(Mtot > 0.0, Mtot, 1.0)
        merged_pos = (bmass[:, None] * bpos + bmass[p][:, None] * bpos[p]) / Mtot_safe[:, None]
        merged_vel = (bmass[:, None] * bvel + bmass[p][:, None] * bvel[p]) / Mtot_safe[:, None]
        bpos = jnp.where(keeper[:, None], merged_pos, bpos)
        bvel = jnp.where(keeper[:, None], merged_vel, bvel)
        bmass = jnp.where(keeper, Mtot, bmass)
        alive = alive & (~absorbed)
        remap = jnp.where(absorbed, nn, idx)  # absorbed slot -> its keeper
        label = remap[label]
        return (bpos, bvel, bmass, alive, label), None

    carry0 = (positions, velocities, masses, jnp.ones(N, dtype=bool), idx)
    (_, _, _, _, label), _ = jax.lax.scan(_level, carry0, None, length=max_levels)

    system_id = label
    counts = jnp.zeros(N).at[label].add(1.0)
    multiplicity = counts[label].astype(jnp.int32)
    return system_id, multiplicity


def _primordial_pair_set(primordial_system_id: Int[Array, "N"]) -> set:
    """Set of (i<j) index tuples for primordial systems with exactly two members."""
    groups = defaultdict(list)
    for idx, sid in enumerate(primordial_system_id):
        groups[int(sid)].append(idx)
    return {tuple(sorted(m)) for m in groups.values() if len(m) == 2}


def primordial_survival(current_pairs, primordial_system_id: Int[Array, "N"]) -> dict:
    """Compare the current bound pairing to the t=0 primordial labelling.

    Args:
        current_pairs: (K, 2) index pairs from :func:`find_bound_pairs`.
        primordial_system_id: (N,) the IC-time `primordial_system_id` (paired
            particles share an id).

    Returns:
        dict with integer counts: ``survived`` (primordial binaries still bound),
        ``disrupted`` (primordial binaries no longer a current pair), and
        ``newly_formed`` (current pairs that were not primordial binaries).
    """
    prim = _primordial_pair_set(primordial_system_id)
    cur = {
        tuple(sorted((int(current_pairs[k, 0]), int(current_pairs[k, 1]))))
        for k in range(current_pairs.shape[0])
    }
    return {
        "survived": len(prim & cur),
        "disrupted": len(prim - cur),
        "newly_formed": len(cur - prim),
    }


class BinaryEnergyBudget(NamedTuple):
    """Two-scale energy budget of a primordial-binary cluster.

    Attributes:
        E_internal: total internal orbital energy of the primordial binaries
            (Σ ``relative_energy``; < 0 if bound). The separate "reservoir" that the
            global virial scaling (``Q``) does NOT touch. NB the ``softening`` passed
            to the bound-pair finders does NOT soften this internal binding energy —
            it only regularizes the inter-system potential (audit S18).
        T_com, W_com: bulk kinetic / gravitational energy of the *system COMs* — the
            scale the cluster is virialized on.
        Q_com: ``T_com / |W_com|`` — the virial ratio the cluster was scaled to
            (≈ the ``Q`` passed to ``build_binary_cluster``).
        Q_resolved: ``T / |W|`` on the *resolved* stars — the naive ratio that mixes
            the cluster and internal-binary scales. The deep internal binary binding
            dominates |W|, so Q_resolved is DEFLATED below the cluster Q (measured
            ≈ 0.31 vs the 0.5 the cluster was scaled to), NOT inflated (audit S10).
        n_binaries: number of primordial binaries (two positive-mass members).
    """

    E_internal: Float[Array, ""]
    T_com: Float[Array, ""]
    W_com: Float[Array, ""]
    Q_com: Float[Array, ""]
    Q_resolved: Float[Array, ""]
    n_binaries: int


def _system_pairs(system_id: Int[Array, "N"], masses: Float[Array, "N"]) -> Int[Array, "K 2"]:
    """(K, 2) index pairs for systems with exactly two positive-mass members (eager).

    Ghost secondaries (mass 0, from the masked `ResolvedBinaries`) are excluded, so the
    diagnostic gives identical results on the compacted and masked representations.
    """
    groups = defaultdict(list)
    for idx, sid in enumerate(system_id):
        groups[int(sid)].append(idx)
    pairs = [
        m
        for m in groups.values()
        if len(m) == 2 and float(masses[m[0]]) > 0.0 and float(masses[m[1]]) > 0.0
    ]
    if not pairs:
        return jnp.zeros((0, 2), dtype=jnp.int32)
    return jnp.asarray(pairs, dtype=jnp.int32)


def binary_energy_budget(
    positions: Float[Array, "N 3"],
    velocities: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    system_id: Int[Array, "N"],
    *,
    G: float,
    softening: float = 0.0,
) -> BinaryEnergyBudget:
    """Separate the cluster-COM virial from the internal binary binding energy.

    `build_binary_cluster` virializes the *system COMs* to `Q` (binaries as point
    masses) and leaves the internal binary binding energy as a separate reservoir
    (the McLuster scale-separation convention, Küpper+2011 §A8). This diagnostic makes
    that explicit: it reconstructs the system COMs (segment sums over `system_id`) and
    reports `Q_com` (recovers the virial target), `E_internal` (the reservoir), and the
    naive `Q_resolved` (which mixes the two scales and is therefore *not* ~`Q`).

    Keyed on `system_id`, so it accepts either the compacted `ICResult`
    (`primordial_system_id`) or the masked `ResolvedBinaries` (ghost secondaries m=0
    contribute exactly 0). The COM reconstruction is differentiable; the binary pairing
    is eager (like `find_bound_pairs`). Handles singles + binaries — systems with > 2
    members (triples/quadruples) are skipped (the higher-multiplicity seam).

    Args:
        positions, velocities, masses: resolved per-star state.
        system_id: per-star system label (members of one system share it).
        G: gravitational constant (REQUIRED).
        softening: softening for the COM/resolved PE (default 0 = exact, matching the
            collisional build default).

    Returns:
        :class:`BinaryEnergyBudget`.
    """
    n_sys = int(jnp.max(system_id)) + 1

    # System COMs as pseudo-bodies (segment sums; differentiable). Empty segments
    # (none for a contiguous system_id) are guarded to mass 1 and contribute 0.
    M_sys = jax.ops.segment_sum(masses, system_id, n_sys)
    M_safe = jnp.where(M_sys > 0.0, M_sys, 1.0)
    com_pos = jax.ops.segment_sum(masses[:, None] * positions, system_id, n_sys) / M_safe[:, None]
    com_vel = jax.ops.segment_sum(masses[:, None] * velocities, system_id, n_sys) / M_safe[:, None]

    T_com = compute_kinetic_energy(com_vel, M_sys)
    W_com = compute_potential_energy(com_pos, M_sys, G=G, softening=softening)
    Q_com = T_com / jnp.abs(W_com)

    # Internal binding energy: relative_energy per primordial binary (exact two-body
    # internal energy = -G m_i m_j / 2a for a bound orbit).
    pairs = _system_pairs(system_id, masses)
    if pairs.shape[0] == 0:
        E_internal = jnp.asarray(0.0)
    else:
        E_internal = jnp.sum(
            jax.vmap(
                lambda p: relative_energy(
                    positions[p[0]], positions[p[1]], velocities[p[0]], velocities[p[1]],
                    masses[p[0]], masses[p[1]], G=G,
                )
            )(pairs)
        )

    # Naive virial ratio on the resolved stars (mixes scales — the misleading number).
    T_res = compute_kinetic_energy(velocities, masses)
    W_res = compute_potential_energy(positions, masses, G=G, softening=softening)
    Q_resolved = T_res / jnp.abs(W_res)

    return BinaryEnergyBudget(
        E_internal=E_internal,
        T_com=T_com,
        W_com=W_com,
        Q_com=Q_com,
        Q_resolved=Q_resolved,
        n_binaries=pairs.shape[0],
    )


__all__ = [
    "relative_energy",
    "find_bound_pairs",
    "find_bound_multiples",
    "primordial_survival",
    "BinaryEnergyBudget",
    "binary_energy_budget",
]

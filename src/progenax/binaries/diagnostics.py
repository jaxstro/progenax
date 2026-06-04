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

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


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
    not differentiable (a diagnostic). O(N^2) per level.

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


__all__ = ["relative_energy", "find_bound_pairs", "find_bound_multiples", "primordial_survival"]

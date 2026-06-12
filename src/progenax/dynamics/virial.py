"""Virial ratio utilities.

Convention: Q = T / |V|

Physical interpretation:
    - Q = 0.5: Virial equilibrium (2T + V = 0)
    - Q < 0.5: Subvirial (cold), system will collapse
    - Q > 0.5: Supervirial (hot), system will expand/unbind
"""
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


def compute_kinetic_energy(
    velocities: Float[Array, "N 3"],
    masses: Float[Array, "N"],
) -> Float[Array, ""]:
    """Compute total kinetic energy: T = 0.5 * sum(m_i * v_i^2)."""
    v_squared = jnp.sum(velocities**2, axis=-1)
    return 0.5 * jnp.sum(masses * v_squared)


def _pad_rows(arr, block):
    """Pad axis 0 to a multiple of ``block`` with zeros (static shapes)."""
    pad = (-arr.shape[0]) % block
    if pad == 0:
        return arr
    widths = ((0, pad),) + ((0, 0),) * (arr.ndim - 1)
    return jnp.pad(arr, widths)


def compute_potential_energy(
    positions: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    G: float,
    softening: float = 0.0,
    block_size: int = 256,
) -> Float[Array, ""]:
    """Total potential energy V = -G * sum_{i<j} m_i m_j / r_ij (Plummer-softened).

    Blocked row-scan (``lax.scan`` over row blocks of ``block_size`` stars vs ALL
    columns): peak transient memory is O(block_size * N), not O(N^2), for the
    forward AND backward pass — the dense kernel measured 32.8 GB at N = 2e4
    (2026-06-10); blocked at the default 256 it is ~0.12 GB. The backward pass
    stays O(block_size * N) via ``jax.checkpoint`` rematerialization: each
    block's forward is recomputed during the vjp instead of stored, so no
    O(N^2) stacked residuals accumulate across scan iterations. Identical pair
    set and per-pair arithmetic; only float64 summation ORDER changes across
    blocks (re-association at the 1e-15 relative level). ``block_size`` is a
    Python int and must be static under jax.jit.

    Differentiable at ``softening=0``: the i<j mask feeds excluded entries
    (diagonal, lower triangle, padded rows) a safe value *before* ``sqrt`` so no
    masked-out ``sqrt(0)`` cotangent can NaN-poison the gradient. This is the
    single canonical energy implementation; ``progenax.builders`` re-exports it.
    """
    N = positions.shape[0]
    if N == 0:
        return jnp.zeros((), dtype=positions.dtype)
    block = int(min(block_size, N))
    pos_b = _pad_rows(positions, block).reshape(-1, block, 3)
    m_b = _pad_rows(masses, block).reshape(-1, block)
    idx_b = jnp.arange(pos_b.shape[0] * block).reshape(-1, block)
    col = jnp.arange(N)

    @jax.checkpoint
    def block_sum(pb, mb, ib):
        diff = pb[:, None, :] - positions[None, :, :]          # (block, N, 3)
        r2 = jnp.sum(diff**2, axis=2)                          # (block, N)
        upper = ib[:, None] < col[None, :]                     # i<j; padded rows all-False
        r_soft = jnp.sqrt(jnp.where(upper, r2 + softening**2, 1.0))
        pair = jnp.where(upper, (mb[:, None] * masses[None, :]) / r_soft, 0.0)
        return jnp.sum(pair)

    def body(acc, blk):
        pb, mb, ib = blk
        return acc + block_sum(pb, mb, ib), None

    V, _ = jax.lax.scan(body, jnp.zeros((), dtype=positions.dtype),
                        (pos_b, m_b, idx_b))
    return -G * V


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


def mass_group_masks(
    masses: Float[Array, "N"],
    n_groups: int,
) -> Bool[Array, "n_groups N"]:
    """Partition stars into ``n_groups`` equal-count mass-rank bins (light -> heavy).

    Returns a boolean array ``(n_groups, N)`` where row ``g`` selects the stars whose
    mass rank falls in group ``g`` (group 0 = lightest, group ``n_groups-1`` = heaviest).
    The groups are disjoint and cover all N stars (up to integer division of N).

    Used to ask whether each mass sub-population is individually in virial equilibrium
    (:func:`per_group_virial_ratio`) — the diagnostic that distinguishes a true
    multi-mass equilibrium from a globally-rescaled blend.
    """
    N = masses.shape[0]
    order = jnp.argsort(masses)  # ascending: lightest first
    rank = jnp.argsort(order)  # rank[i] = position of star i in ascending order
    group_of_rank = (rank * n_groups) // N  # 0..n_groups-1
    return jnp.stack([group_of_rank == g for g in range(n_groups)], axis=0)


def _accelerations(
    positions: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    G: float,
    softening: float = 0.0,
    block_size: int = 256,
) -> Float[Array, "N 3"]:
    """Direct-summation accelerations a_i = -G sum_k m_k (r_i - r_k) / |r_ik|^3.

    Blocked row-scan: O(block_size * N) transient memory for the forward AND
    backward pass (see :func:`compute_potential_energy`) — each block's forward
    is rematerialized via ``jax.checkpoint`` during the vjp instead of stored,
    so no O(N^2) stacked residuals accumulate across scan iterations.
    Plummer-softened; differentiable at ``softening=0`` (the interaction mask
    feeds excluded entries a safe value before the inverse-cube — diagonal AND
    padded rows, so no masked inf can NaN-poison the vjp through the discarded
    pad slice). ``block_size`` is a Python int and must be static under jax.jit.
    """
    N = positions.shape[0]
    if N == 0:
        return jnp.zeros((0, 3), dtype=positions.dtype)
    block = int(min(block_size, N))
    pos_b = _pad_rows(positions, block).reshape(-1, block, 3)
    idx_b = jnp.arange(pos_b.shape[0] * block).reshape(-1, block)
    col = jnp.arange(N)

    @jax.checkpoint
    def block_acc(pb, ib):
        diff = pb[:, None, :] - positions[None, :, :]            # (block, N, 3)
        r2 = jnp.sum(diff**2, axis=2)
        interact = (ib[:, None] != col[None, :]) & (ib[:, None] < N)
        inv_r3 = jnp.where(interact, 1.0, 0.0) * jnp.where(
            interact, r2 + softening**2, 1.0) ** (-1.5)
        return -G * jnp.sum(masses[None, :, None] * diff * inv_r3[:, :, None],
                            axis=1)

    def body(_, blk):
        pb, ib = blk
        return None, block_acc(pb, ib)

    _, a = jax.lax.scan(body, None, (pos_b, idx_b))
    return a.reshape(-1, 3)[:N]


def per_group_virial_ratio(
    positions: Float[Array, "N 3"],
    velocities: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    G: float,
    group_masks: Bool[Array, "n_groups N"],
    softening: float = 0.0,
) -> Float[Array, "n_groups"]:
    """Per-mass-group virial ratio Q_j = T_j / |W_j| in the TOTAL gravitational field.

    Scalar virial theorem for a subsystem in steady state: ``2 T_j + W_j = 0``, where

        T_j = 0.5 sum_{i in j} m_i v_i^2        (kinetic energy of the group)
        W_j = sum_{i in j} m_i r_i . a_i        (a_i = acceleration from ALL stars)

    so ``Q_j = T_j / |W_j| = 0.5`` for a group in equilibrium — the same convention as
    :func:`compute_virial_ratio`. By the Clausius identity ``sum_i m_i r_i . a_i = V``
    for ``1/r`` gravity, a single all-ones group reproduces the global virial ratio
    exactly (``W = V``), and the per-group ``W_j`` partition the global virial.

    Positions enter only through differences (origin-independent); velocities are
    measured relative to the mass-weighted mean (bulk motion removed) so a moving COM
    does not inflate ``T_j``.

    Args:
        positions: Particle positions, shape (N, 3).
        velocities: Particle velocities, shape (N, 3).
        masses: Particle masses, shape (N,).
        G: Gravitational constant (consistent units).
        group_masks: Boolean group membership, shape (n_groups, N) — e.g. from
            :func:`mass_group_masks`.
        softening: Plummer softening for the acceleration sum.

    Returns:
        Q_j for each group, shape (n_groups,).
    """
    v_com = jnp.average(velocities, axis=0, weights=masses)
    v = velocities - v_com[None, :]

    a = _accelerations(positions, masses, G, softening=softening)  # (N, 3)
    # Per-star contributions
    T_i = 0.5 * masses * jnp.sum(v**2, axis=1)  # (N,)
    W_i = masses * jnp.sum(positions * a, axis=1)  # (N,)  m_i r_i . a_i

    g = group_masks.astype(positions.dtype)  # (n_groups, N)
    T_j = g @ T_i  # (n_groups,)
    W_j = g @ W_i  # (n_groups,)
    return T_j / (jnp.abs(W_j) + 1e-30)


def _is_concrete(x) -> bool:
    """True iff x is a concrete value (codebase idiom; cf. king.py auto-sizing)."""
    try:
        float(x)
        return True
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError,
            TypeError):
        return False


def rescale_velocities_to_virial(
    positions: Float[Array, "N 3"],
    velocities: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    G: float,
    target_Q: float = 0.5,
    softening: float = 0.0,
) -> Float[Array, "N 3"]:
    """Rescale velocities to achieve target virial ratio Q = T/|V|.

    This is the single implementation of virial velocity rescaling;
    ``builders.virial_scale`` delegates here (audit J5 dedupe).

    Cold input (T=0 -> Q_current=0) makes the rescale undefined (0/0). A
    CONCRETE T=0 refuses loudly (ValueError); a TRACED T=0 yields NaN — the
    honest sentinel, since a traced scalar cannot gate a Python raise (audit J5;
    the previous +1e-10 hack silently mapped cold input to ~0 velocities).

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
    if _is_concrete(Q_current) and float(Q_current) <= 0.0:
        raise ValueError(
            "cannot rescale from zero kinetic energy (T=0): velocities are all "
            "zero, so the virial ratio is undefined. Provide non-zero velocities "
            "(e.g. sample from a velocity DF) before virial scaling."
        )
    # double-where keeps the untaken (Q<=0) branch finite; traced T=0 -> NaN.
    Q_safe = jnp.where(Q_current > 0.0, Q_current, 1.0)
    scale = jnp.where(Q_current > 0.0, jnp.sqrt(target_Q / Q_safe), jnp.nan)
    return velocities * scale

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


def compute_potential_energy(
    positions: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    G: float,
    softening: float = 0.0,
) -> Float[Array, ""]:
    """Compute total potential energy: V = -G * sum_{i<j}(m_i * m_j / r_ij).

    Uses Plummer softening: r_ij -> sqrt(r_ij^2 + eps^2). Returns a negative
    value (bound systems have V < 0).

    Differentiable at ``softening=0``: a double-``where`` feeds the diagonal a
    safe positive value *before* ``sqrt`` (otherwise the diagonal ``sqrt(0)``
    derivative is ``inf`` and ``0 * inf = nan`` survives a later ``where``), then
    sets the diagonal to ``inf`` so the ``i < j`` sum drops it. This is the single
    canonical energy implementation; ``progenax.builders`` re-exports it.
    """
    N = positions.shape[0]
    diff = positions[:, None, :] - positions[None, :, :]  # (N, N, 3)
    r_squared = jnp.sum(diff**2, axis=2)  # (N, N)
    eye = jnp.eye(N, dtype=bool)
    r_squared_safe = jnp.where(eye, 1.0, r_squared + softening**2)
    r_soft = jnp.where(eye, jnp.inf, jnp.sqrt(r_squared_safe))
    m_prod = masses[:, None] * masses[None, :]  # (N, N)
    V = -G * jnp.sum(jnp.triu(m_prod / r_soft, k=1))
    return V


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
) -> Float[Array, "N 3"]:
    """Direct-summation gravitational accelerations a_i = -G sum_k m_k r_ik / |r_ik|^3.

    Plummer-softened and differentiable at ``softening=0`` (the self term is masked
    before the inverse-cube so the diagonal never differentiates ``r^-3`` at 0).
    """
    N = positions.shape[0]
    diff = positions[:, None, :] - positions[None, :, :]  # (N, N, 3): r_i - r_k
    r_squared = jnp.sum(diff**2, axis=2)  # (N, N)
    eye = jnp.eye(N, dtype=bool)
    r_squared_safe = jnp.where(eye, 1.0, r_squared + softening**2)
    inv_r3 = jnp.where(eye, 0.0, r_squared_safe ** (-1.5))  # self term -> 0
    # a_i = -G sum_k m_k (r_i - r_k) / r_ik^3
    return -G * jnp.sum(masses[None, :, None] * diff * inv_r3[:, :, None], axis=1)


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


def rescale_velocities_to_virial(
    positions: Float[Array, "N 3"],
    velocities: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    G: float,
    target_Q: float = 0.5,
    softening: float = 0.0,
) -> Float[Array, "N 3"]:
    """Rescale velocities to achieve target virial ratio Q = T/|V|.

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
    scale = jnp.sqrt(target_Q / (Q_current + 1e-10))
    return velocities * scale

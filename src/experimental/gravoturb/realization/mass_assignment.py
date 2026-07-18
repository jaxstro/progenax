r"""Density-correlated mass placement for the gravoturbulent cluster IC (Tier C — primordial segregation).

By default ``build_cluster_ic`` pairs the input IMF masses with sampled positions in arbitrary
order — i.e. *no* primordial mass segregation (the clean Allison-2009 setup for studying *dynamical*
segregation). This module adds the physically-motivated alternative: massive stars form preferentially
in the **dense BM19 clumps** (competitive accretion / dense cores), with a tunable strength.

``correlated_mass_assignment(masses, local_density, lambda_corr, key)`` returns a *permutation* of
``masses`` aligned with the star positions such that

    λ_corr = 0  → random pairing                (no primordial segregation),
    λ_corr = 1  → density-rank ↔ mass-rank      (most massive star in the densest cell),
    0 < λ < 1   → partial correlation,

built on the McLuster Eq. A1 partial-shuffle (Küpper et al. 2011) applied to a **density** key rather
than an orbital-energy key. Mass-conserving (a pure permutation). Non-differentiable (categorical
ranking), consistent with the rest of star placement (spec §8); ``lambda_corr`` is the knob.

JAX-native (``jax.lax.scan``).
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def _partial_shuffle(key: jax.Array, n: int, strength: Float[Array, ""]) -> Array:
    r"""Rank→rank mapping via McLuster Eq. A1 (Küpper et al. 2011): ``strength``=1 → identity,
    ``strength``=0 → uniform random permutation. ``perm[i]`` is the rank assigned to slot ``i``.

    Self-contained JAX-native reimplementation (the experimental package does not depend on
    released-core private helpers). ``j = floor((n−i)·(1 − X^{1−strength}))`` over the remaining
    available ranks, X ~ U(0,1)."""
    s = jnp.clip(strength, 0.0, 1.0)
    X = jax.random.uniform(key, (n,))

    def step(carry, i):
        available, perm = carry
        n_avail = n - i
        j = jnp.floor(n_avail * (1.0 - jnp.power(X[i], 1.0 - s))).astype(jnp.int32)
        j = jnp.clip(j, 0, n_avail - 1)
        avail_idx = jnp.where(available, jnp.arange(n, dtype=jnp.int32),
                              jnp.full((n,), n, dtype=jnp.int32))
        target = jnp.sort(avail_idx)[j]          # j-th still-available rank
        perm = perm.at[i].set(target)
        available = available.at[target].set(False)
        return (available, perm), None

    init = (jnp.ones(n, dtype=bool), jnp.zeros(n, dtype=jnp.int32))
    (_, perm), _ = jax.lax.scan(step, init, jnp.arange(n))
    return perm


def correlated_mass_assignment(
    masses: Float[Array, "n"],
    local_density: Float[Array, "n"],
    lambda_corr: Float[Array, ""],
    key: jax.Array,
) -> Float[Array, "n"]:
    r"""Reorder ``masses`` so massive stars sit in dense cells with strength ``lambda_corr``.

    ``local_density`` is the density at each star's position (e.g. ``exp(s_total)`` at its cell).
    Returns a permutation of ``masses`` aligned with the positions: at ``lambda_corr=1`` the
    most-massive star takes the densest position; at ``lambda_corr=0`` the pairing is random.
    """
    n = int(masses.shape[0])
    density_rank = jnp.argsort(-local_density)         # density_rank[k] = star at density-rank k
    mass_desc = jnp.sort(masses)[::-1]                 # mass at mass-rank i (most massive first)
    perm = _partial_shuffle(key, n, lambda_corr)       # density-rank i ← mass-rank perm[i]
    return jnp.zeros_like(masses).at[density_rank].set(mass_desc[perm])


def _rank_normal(values: Float[Array, "n"]) -> Float[Array, "n"]:
    r"""Rank → standard-normal latent: z = Φ⁻¹((rank(values) + 0.5)/n) (a Gaussian-copula margin).

    Distribution-free (uses only ranks), so ``z`` is standard normal regardless of the input
    marginal. Non-differentiable (argsort), consistent with the rest of star placement."""
    n = values.shape[0]
    ranks = jnp.argsort(jnp.argsort(values))
    u = (ranks + 0.5) / n
    return jax.scipy.special.ndtri(u)


def blended_system_placement(
    local_density: Float[Array, "n"],
    mass_score: Float[Array, "n"],
    mult_score: Float[Array, "n"],
    *,
    lambda_corr: Float[Array, ""] | float,
    lambda_mult: Float[Array, ""] | float,
    key: jax.Array,
) -> Array:
    r"""Assign whole systems to fixed positions by a Gaussian-copula affinity (spec §C).

    Each *system* is a self-consistent (m1, m2, orbit) unit; ``mass_score`` ranks systems by mass
    (e.g. m_sys) and ``mult_score`` ranks them by multiplicity/compactness (binary & tighter →
    higher). The returned permutation ``perm`` aligns systems to positions so that, in rank space,
    system **mass** correlates with local density at strength ``lambda_corr`` and system
    **multiplicity** correlates with density at an INDEPENDENT strength ``lambda_mult`` — a
    controlled density→multiplicity birth-imprint *beyond* the emergent mass channel.

    Mechanism (Gaussian copula): system affinity ``a = λ_corr·z_m + λ_mult·z_u + σ·ε`` with
    ``z_m, z_u`` the rank-normal mass/mult latents, ``ε`` standard-normal noise, and
    ``σ = √(1 − λ_corr² − λ_mult²)`` (weights renormalized onto the unit circle if they exceed it).
    Positions (ranked by density) are paired rank-to-rank with systems (ranked by ``a``), so
    ``corr(z_m, density) ≈ λ_corr`` and ``corr(z_u, density) ≈ λ_mult`` (plus the mass–mult
    coupling). ``λ_corr = λ_mult = 0`` → random placement. Whole systems move together, so every
    marginal and the Moe internal (P, q, e) joint are preserved. Non-differentiable (rank op),
    consistent with ``correlated_mass_assignment``.

    Returns ``perm`` with ``perm[j]`` = the system index placed at position ``j`` (reorder the system
    arrays by ``perm`` to align them with the positions).
    """
    n = local_density.shape[0]
    lc = jnp.asarray(lambda_corr, dtype=float)
    lm = jnp.asarray(lambda_mult, dtype=float)
    w2 = lc**2 + lm**2
    renorm = jnp.where(w2 > 1.0, 1.0 / jnp.sqrt(jnp.maximum(w2, 1e-12)), 1.0)
    lc, lm = lc * renorm, lm * renorm
    sigma = jnp.sqrt(jnp.clip(1.0 - (lc**2 + lm**2), 0.0, 1.0))

    z_d = _rank_normal(local_density)
    z_m = _rank_normal(mass_score)
    z_u = _rank_normal(mult_score)
    eps = jax.random.normal(key, (n,))
    affinity = lc * z_m + lm * z_u + sigma * eps

    pos_order = jnp.argsort(z_d)       # positions ascending by density
    sys_order = jnp.argsort(affinity)  # systems ascending by affinity
    # position pos_order[k] (k-th lowest density) gets system sys_order[k] (k-th lowest affinity)
    return jnp.zeros(n, dtype=jnp.int32).at[pos_order].set(sys_order.astype(jnp.int32))

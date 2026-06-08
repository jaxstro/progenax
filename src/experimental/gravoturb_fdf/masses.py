r"""Density-correlated mass placement for the FDF cluster IC (Tier C — primordial segregation).

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

JAX-native (``jax.lax.scan``). See docs/plans/2026-06-07-gravoturb-fdf-methods-figures-and-segregation-validation-design.md.
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

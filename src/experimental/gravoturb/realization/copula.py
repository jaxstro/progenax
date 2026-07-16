"""Rank copulas: remap a Gaussian field to the BM19 log-density marginal.

Split out of the pre-rename ``field.py`` (Phase 0.5); function bodies are unchanged
(byte-identity gate: tests/experimental/unit/test_rename_byte_identity.py).

``rank_copula_field`` gives the faithful *volume* marginal (used for the tail);
``mass_conserving_copula_field`` assigns slab-mass-averaged densities at exact volume
quantiles so the realized dense-mass fraction matches BM19 f_dense to O(1/N) — the AC6
cornerstone. Both are monotone in the input field (rank structure preserved), so any
Gaussian source — an independent GRF or the Helmholtz-derived −∇·v field — carries its
spatial correlations through unchanged.

JAX-native.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from gravoturb.theory.density_cdf import (
    bm19_icdf,
    bm19_icdf_analytic,
    bm19_mass_cdf,
    bm19_mean_density,
)


def rank_to_uniform(values: Float[Array, "..."]) -> Float[Array, "..."]:
    r"""Empirical-CDF (rank) copula: u = (rank(values) + 0.5) / N.

    Double-argsort assigns each element its rank in [0, N); ``(rank+0.5)/N`` is a
    permutation of the uniform plotting positions, so ``u`` is **exactly** uniform on
    (0,1) regardless of the input's realized marginal. Shape-preserving.

    Non-differentiable in ``values`` (argsort); used on a frozen GRF realization, so
    downstream grads in the cloud parameters are unaffected.
    """
    flat = values.ravel()
    n = flat.size
    ranks = jnp.argsort(jnp.argsort(flat))
    u = (ranks + 0.5) / n
    return u.reshape(values.shape)


def rank_copula_field(
    g: Float[Array, "..."],
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
) -> Float[Array, "..."]:
    r"""Remap a GRF ``g`` to the BM19 volume marginal via the rank copula.

    ``u = (rank(g)+0.5)/N`` (distribution-free) → ``s = F_BM19^{-1}(u)``, then a
    constant shift enforces the ρ_0 convention ``⟨e^s⟩ = 1`` (ρ_0 = volume-mean
    density). The shift is a constant in (mach, b, alpha)-space, so the marginal shape
    is preserved and the result is differentiable in the cloud parameters through the
    smooth CDF table (the ranks are frozen).

    Returns ``s = ln(ρ/ρ_0)`` with the same shape as ``g``.
    """
    u = rank_to_uniform(g)
    s_raw = bm19_icdf(u.ravel(), mach, b, alpha).reshape(g.shape)
    # Enforce ρ_0 = volume mean: ⟨e^s⟩ = 1  ⇒  s = s_raw − ln⟨e^{s_raw}⟩.
    shift = jnp.log(jnp.mean(jnp.exp(s_raw)))
    return s_raw - shift


def mass_conserving_copula_field(
    g: Float[Array, "..."],
    mach: Float[Array, ""],
    b: Float[Array, ""],
    alpha: Float[Array, ""],
) -> Float[Array, "..."]:
    r"""Remap a GRF ``g`` to the BM19 marginal with **exact** mass conservation.

    Each cell (ranked by ``g``) is assigned the mass-averaged density over its
    volume-quantile slab rather than the point value ``e^{F^{-1}(u)}``:

        ρ_i / ρ_0 = N · ΔM_i · ⟨e^s⟩,   ΔM_i = M(s_{i+1}) − M(s_i),

    with ``M`` the normalized mass CDF and ``s_i = F^{-1}(i/N)`` the analytic volume
    iCDF. The realized tail mass fraction Σ_{tail} ΔM equals BM19 f_dense to O(1/N)
    (vs. the −2…−5.5% truncation bias of the point-value rank copula), because the
    extreme power-law tail mass is collected analytically into the top slab.

    The volume mean ⟨e^s⟩ = bm19_mean_density (≥1) is the BM19-consistent ρ_0 (not a
    forced 1). Monotone in ``g`` (order preserved); differentiable in (mach,b,alpha):
    interior slab edges are smooth, the 0/1 mass endpoints are constants.
    """
    flat = g.ravel()
    n = flat.size
    ranks = jnp.argsort(jnp.argsort(flat))

    u_inner = jnp.arange(1, n) / n  # interior edges in (0,1)
    s_inner = bm19_icdf_analytic(u_inner, mach, b, alpha)
    m_inner = bm19_mass_cdf(s_inner, mach, b, alpha)
    m_edges = jnp.concatenate([jnp.zeros(1), m_inner, jnp.ones(1)])  # M(0)=0, M(1)=1
    dM = jnp.diff(m_edges)  # N normalized slab masses, sum = 1

    rho_sorted = n * dM * bm19_mean_density(mach, b, alpha)  # ρ_i/ρ_0
    s_sorted = jnp.log(rho_sorted)
    return s_sorted[ranks].reshape(g.shape)

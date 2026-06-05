"""Rank-copula remap of a GRF to the BM19 volume marginal (spec §3.5 steps 2-4).

The rank copula maps any field g to exactly-uniform quantiles u=(rank+0.5)/N
(distribution-free, independent of g's realized marginal), then s=F_BM19^{-1}(u),
ρ=exp(s), normalized so the volume-mean density is unity (⟨e^s⟩=1, the definition
of ρ_0). Monotone → spatial rank/clump locations are preserved.

Tests: u is exactly uniform; the s-marginal matches BM19 (shift-invariant quantile
spacing); order is preserved; ⟨e^s⟩=1; grad flows through the cloud parameters.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.experimental


def test_rank_to_uniform_exactly_uniform():
    """u is a permutation of (i+0.5)/N — exactly uniform, independent of input shape."""
    from gravoturb_fdf.field.field import rank_to_uniform

    g = jax.random.normal(jax.random.PRNGKey(0), (8, 8, 8))
    u = rank_to_uniform(g)
    n = g.size
    assert u.shape == g.shape
    assert jnp.allclose(jnp.sort(u.ravel()), (jnp.arange(n) + 0.5) / n)


def test_rank_to_uniform_distribution_free():
    """Skewed (exp) input gives the same exactly-uniform u as Gaussian input."""
    from gravoturb_fdf.field.field import rank_to_uniform

    skew = jnp.exp(3.0 * jax.random.normal(jax.random.PRNGKey(1), (500,)))
    u = rank_to_uniform(skew)
    n = u.size
    assert jnp.allclose(jnp.sort(u), (jnp.arange(n) + 0.5) / n)


def test_rank_copula_preserves_order():
    """Monotone remap: argsort(s) == argsort(g) (clump locations preserved)."""
    from gravoturb_fdf.field.field import rank_copula_field

    g = jax.random.normal(jax.random.PRNGKey(2), (16, 16, 16))
    s = rank_copula_field(g, mach=6.0, b=0.4, alpha=1.8)
    assert jnp.array_equal(jnp.argsort(s.ravel()), jnp.argsort(g.ravel()))


def test_rank_copula_mean_density_unity():
    """Normalization enforces ⟨ρ⟩ = ⟨e^s⟩ = 1 (ρ_0 = volume mean)."""
    from gravoturb_fdf.field.field import rank_copula_field

    g = jax.random.normal(jax.random.PRNGKey(3), (100_000,))
    s = rank_copula_field(g, mach=6.0, b=0.4, alpha=1.8)
    assert float(jnp.mean(jnp.exp(s))) == pytest.approx(1.0, abs=1e-6)


def test_rank_copula_marginal_matches_bm19():
    """s-marginal matches BM19: quantile spacings equal the analytic iCDF (shift-free)."""
    from gravoturb_fdf.field.field import rank_copula_field
    from gravoturb_fdf.theory.pdf import bm19_icdf

    mach, b, alpha = 6.0, 0.4, 1.8
    g = jax.random.normal(jax.random.PRNGKey(4), (200_000,))
    s = np.asarray(rank_copula_field(g, mach=mach, b=b, alpha=alpha))
    probs = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.97])
    emp = np.quantile(s, np.asarray(probs))
    theo = np.asarray(bm19_icdf(probs, mach, b, alpha))
    # shift-invariant: compare quantile spacings (the normalization shift cancels)
    assert np.allclose(np.diff(emp), np.diff(theo), atol=0.05)


def test_rank_copula_differentiable_in_alpha():
    """Grad of an s-statistic w.r.t. alpha flows through the smooth CDF table and is finite."""
    from gravoturb_fdf.field.field import rank_copula_field

    g = jax.random.normal(jax.random.PRNGKey(5), (4096,))

    def stat(alpha):
        return jnp.mean(rank_copula_field(g, mach=6.0, b=0.4, alpha=alpha) ** 2)

    grad = float(jax.grad(stat)(1.8))
    assert jnp.isfinite(grad)

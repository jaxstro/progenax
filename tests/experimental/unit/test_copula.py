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


def test_mass_conserving_preserves_order():
    """Mass-conserving remap is monotone: argsort(s) == argsort(g)."""
    from gravoturb_fdf.field.field import mass_conserving_copula_field

    g = jax.random.normal(jax.random.PRNGKey(10), (16, 16, 16))
    s = mass_conserving_copula_field(g, mach=8.0, b=0.5, alpha=1.8)
    assert jnp.array_equal(jnp.argsort(s.ravel()), jnp.argsort(g.ravel()))


def test_mass_conserving_mean_density_matches_theory():
    """⟨e^s⟩ equals bm19_mean_density exactly (mass-conserving construction)."""
    from gravoturb_fdf.field.field import mass_conserving_copula_field
    from gravoturb_fdf.theory.pdf import bm19_mean_density

    g = jax.random.normal(jax.random.PRNGKey(11), (50_000,))
    s = mass_conserving_copula_field(g, mach=8.0, b=0.5, alpha=1.8)
    assert float(jnp.mean(jnp.exp(s))) == pytest.approx(
        float(bm19_mean_density(8.0, 0.5, 1.8)), rel=1e-6
    )


@pytest.mark.parametrize("mach,b,alpha", [(10.0, 0.4, 2.0), (12.0, 1 / 3, 1.6)])
def test_mass_conserving_f_dense_exact(mach, b, alpha):
    """Realized hard mass fraction reproduces BM19 f_dense to <0.5% (the AC6 fix)."""
    from gravoturb_fdf.field.field import mass_conserving_copula_field
    from gravoturb_fdf.theory.bm19 import (
        f_dense_bm19_full,
        sigma_s_squared,
        transition_density,
    )

    g = jax.random.normal(jax.random.PRNGKey(12), (200_000,))
    s = mass_conserving_copula_field(g, mach=mach, b=b, alpha=alpha)
    rho = jnp.exp(s)
    s_t = transition_density(alpha, sigma_s_squared(mach, b))
    realized = float(jnp.sum(jnp.where(s > s_t, rho, 0.0)) / jnp.sum(rho))
    theory = float(f_dense_bm19_full(mach, b, alpha))
    assert abs(realized - theory) / theory < 0.005


def test_mass_conserving_differentiable_in_alpha():
    """Grad of an s-statistic w.r.t. alpha is finite (analytic iCDF + mass CDF)."""
    from gravoturb_fdf.field.field import mass_conserving_copula_field

    g = jax.random.normal(jax.random.PRNGKey(13), (4096,))
    grad = float(jax.grad(
        lambda a: jnp.mean(mass_conserving_copula_field(g, 8.0, 0.5, a) ** 2)
    )(1.8))
    assert jnp.isfinite(grad)


def test_rank_copula_differentiable_in_alpha():
    """Grad of an s-statistic w.r.t. alpha flows through the smooth CDF table and is finite."""
    from gravoturb_fdf.field.field import rank_copula_field

    g = jax.random.normal(jax.random.PRNGKey(5), (4096,))

    def stat(alpha):
        return jnp.mean(rank_copula_field(g, mach=6.0, b=0.4, alpha=alpha) ** 2)

    grad = float(jax.grad(stat)(1.8))
    assert jnp.isfinite(grad)

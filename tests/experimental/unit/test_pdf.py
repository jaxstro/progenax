"""BM19 volume density PDF + CDF/inverse-CDF — physics tests.

The volume-weighted PDF p(s) (s = ln(rho/rho_0)) is a mass-conserving lognormal
body (mean s0 = -sigma_s^2/2) for s < s_t, joined continuously to a powerlaw tail
p_PL(s) = C e^{-alpha s} for s >= s_t, then renormalized so int p(s) ds = 1 (volume).

The inverse CDF s = F^{-1}(u) maps uniform u in (0,1) -> s and is the engine of the
rank-copula field realization in P2; it must be smooth in (mach, b, alpha) for grads.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.experimental

PARAMS = [(5.0, 0.4, 2.0), (10.0, 1.0 / 3, 1.6), (8.0, 0.5, 1.8)]


@pytest.mark.parametrize("mach,b,alpha", PARAMS)
def test_volume_pdf_normalized(mach, b, alpha):
    """int p(s) ds = 1 over the support (volume normalization)."""
    from gravoturb_fdf.theory.pdf import bm19_volume_pdf

    s = np.linspace(-30.0, 80.0, 2_000_000)
    p = np.asarray(bm19_volume_pdf(s, mach, b, alpha))
    assert np.trapezoid(p, s) == pytest.approx(1.0, abs=2e-3)


@pytest.mark.parametrize("mach,b,alpha", PARAMS)
def test_icdf_analytic_roundtrip(mach, b, alpha):
    """Analytic iCDF inverts the volume CDF exactly: F(F^{-1}(u)) = u, incl. deep tail."""
    from gravoturb_fdf.theory.pdf import bm19_icdf_analytic, build_bm19_cdf_table

    u = np.array([1e-6, 0.01, 0.2, 0.5, 0.8, 0.99, 1 - 1e-6, 1 - 5e-7])
    s = np.asarray(bm19_icdf_analytic(u, mach, b, alpha))
    # recover u via the analytic mass-independent volume CDF table (fine interp)
    s_grid, cdf = build_bm19_cdf_table(mach, b, alpha, n_nodes=200_000)
    u_back = np.interp(s, np.asarray(s_grid), np.asarray(cdf))
    assert np.allclose(u_back, u, atol=2e-4)


def test_icdf_analytic_differentiable():
    """Analytic iCDF is differentiable in alpha across body and tail (no where-nan)."""
    import jax
    from gravoturb_fdf.theory.pdf import bm19_icdf_analytic

    for u in (0.3, 0.95):
        g = float(
            jax.grad(
                lambda a: jnp.sum(bm19_icdf_analytic(jnp.array([u]), 6.0, 0.4, a))
            )(1.8)
        )
        assert jnp.isfinite(g)


@pytest.mark.parametrize("mach,b,alpha", PARAMS)
def test_mass_cdf_matches_numeric(mach, b, alpha):
    """Normalized mass CDF M(s)=∫_{-∞}^s e^{s'}p ds' / ⟨e^s⟩ matches numeric integration."""
    from gravoturb_fdf.theory.pdf import bm19_mass_cdf, bm19_volume_pdf

    s = np.linspace(-30.0, 90.0, 2_000_000)
    p = np.asarray(bm19_volume_pdf(s, mach, b, alpha))
    mass_density = np.exp(s) * p
    cum = np.concatenate(
        [[0.0], np.cumsum(0.5 * (mass_density[1:] + mass_density[:-1]) * np.diff(s))]
    )
    cum /= cum[-1]  # normalize to M(∞)=1
    grid = np.linspace(-10.0, 20.0, 40)
    m_analytic = np.asarray(bm19_mass_cdf(grid, mach, b, alpha))
    m_numeric = np.interp(grid, s, cum)
    assert np.allclose(m_analytic, m_numeric, atol=3e-3)


@pytest.mark.parametrize("mach,b,alpha", PARAMS)
def test_mass_cdf_at_transition_is_one_minus_f_dense(mach, b, alpha):
    """1 − M(s_t) equals BM19 f_dense (mass fraction above the transition)."""
    from gravoturb_fdf.theory.bm19 import (
        f_dense_bm19_full,
        sigma_s_squared,
        transition_density,
    )
    from gravoturb_fdf.theory.pdf import bm19_mass_cdf

    s_t = transition_density(alpha, sigma_s_squared(mach, b))
    f_dense = float(f_dense_bm19_full(mach, b, alpha))
    assert 1.0 - float(bm19_mass_cdf(s_t, mach, b, alpha)) == pytest.approx(
        f_dense, rel=1e-4
    )


@pytest.mark.parametrize("mach,b,alpha", PARAMS)
def test_mean_density_matches_numeric(mach, b, alpha):
    """bm19_mean_density = ⟨ρ/ρ_0⟩ = ∫ e^s p ds ≥ 1 (power-law tail adds mass)."""
    from gravoturb_fdf.theory.pdf import bm19_mean_density, bm19_volume_pdf

    s = np.linspace(-30.0, 90.0, 2_000_000)
    numeric = np.trapezoid(
        np.exp(s) * np.asarray(bm19_volume_pdf(s, mach, b, alpha)), s
    )
    val = float(bm19_mean_density(mach, b, alpha))
    assert val >= 1.0
    assert val == pytest.approx(numeric, rel=2e-3)


@pytest.mark.parametrize("mach,b,alpha", PARAMS)
def test_volume_tail_fraction_matches_integral(mach, b, alpha):
    """bm19_volume_tail_fraction = ∫_{s_t}^∞ p(s) ds (volume fraction above s_t)."""
    from gravoturb_fdf.theory.bm19 import sigma_s_squared, transition_density
    from gravoturb_fdf.theory.pdf import bm19_volume_pdf, bm19_volume_tail_fraction

    s_t = float(transition_density(alpha, sigma_s_squared(mach, b)))
    s = np.linspace(s_t, s_t + 200.0, 2_000_000)
    integral = np.trapezoid(np.asarray(bm19_volume_pdf(s, mach, b, alpha)), s)
    frac = float(bm19_volume_tail_fraction(mach, b, alpha))
    assert 0.0 < frac < 1.0
    assert frac == pytest.approx(integral, rel=1e-3)


@pytest.mark.parametrize("mach,b,alpha", PARAMS)
def test_volume_pdf_continuous_at_st(mach, b, alpha):
    """p(s) is continuous across the lognormal->powerlaw transition s_t."""
    from gravoturb_fdf.theory.bm19 import sigma_s_squared, transition_density
    from gravoturb_fdf.theory.pdf import bm19_volume_pdf

    s_t = float(transition_density(alpha, sigma_s_squared(mach, b)))
    lo = float(bm19_volume_pdf(s_t - 1e-6, mach, b, alpha))
    hi = float(bm19_volume_pdf(s_t + 1e-6, mach, b, alpha))
    assert lo == pytest.approx(hi, rel=1e-4)


@pytest.mark.parametrize("mach,b,alpha", PARAMS)
def test_cdf_monotone_and_bounds(mach, b, alpha):
    from gravoturb_fdf.theory.pdf import build_bm19_cdf_table

    s_grid, cdf = build_bm19_cdf_table(mach, b, alpha)
    s_grid, cdf = np.asarray(s_grid), np.asarray(cdf)
    assert np.all(np.diff(cdf) >= -1e-12)  # non-decreasing
    assert cdf[0] == pytest.approx(0.0, abs=1e-3)
    assert cdf[-1] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("mach,b,alpha", PARAMS)
def test_icdf_roundtrip(mach, b, alpha):
    """F(F^{-1}(u)) ~= u for u in (0,1)."""
    import jax.numpy as jnp
    from gravoturb_fdf.theory.pdf import bm19_icdf, build_bm19_cdf_table

    s_grid, cdf = build_bm19_cdf_table(mach, b, alpha)
    u = jnp.linspace(0.02, 0.98, 50)
    s = bm19_icdf(u, mach, b, alpha)
    # re-evaluate CDF at the recovered s by interpolating the table
    u_back = jnp.interp(s, s_grid, cdf)
    assert np.allclose(np.asarray(u_back), np.asarray(u), atol=1e-3)


def test_icdf_monotone_in_u():
    import jax.numpy as jnp
    from gravoturb_fdf.theory.pdf import bm19_icdf

    s = np.asarray(bm19_icdf(jnp.linspace(0.05, 0.95, 100), 5.0, 0.4, 2.0))
    assert np.all(np.diff(s) > 0)


def test_icdf_differentiable_in_params():
    """bm19_icdf is differentiable in alpha (grad-safe table) - enables P2 copula grads."""
    import jax.numpy as jnp
    from gravoturb_fdf.theory.pdf import bm19_icdf

    def mean_s(alpha):
        u = jnp.linspace(0.1, 0.9, 64)
        return jnp.mean(bm19_icdf(u, 5.0, 0.4, alpha))

    g = float(jax.grad(mean_s)(1.8))
    assert math.isfinite(g)

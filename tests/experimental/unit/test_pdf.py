"""BM19 volume density PDF + CDF/inverse-CDF — physics tests.

The volume-weighted PDF p(s) (s = ln(rho/rho_0)) is a mass-conserving lognormal
body (mean s0 = -sigma_s^2/2) for s < s_t, joined continuously to a powerlaw tail
p_PL(s) = C e^{-alpha s} for s >= s_t, then renormalized so int p(s) ds = 1 (volume).

The inverse CDF s = F^{-1}(u) maps uniform u in (0,1) -> s and is the engine of the
rank-copula field realization in P2; it must be smooth in (mach, b, alpha) for grads.
"""

import math

import jax
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
    assert np.all(np.diff(cdf) >= -1e-12)          # non-decreasing
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
    from gravoturb_fdf.theory.pdf import bm19_icdf
    import jax.numpy as jnp

    s = np.asarray(bm19_icdf(jnp.linspace(0.05, 0.95, 100), 5.0, 0.4, 2.0))
    assert np.all(np.diff(s) > 0)


def test_icdf_differentiable_in_params():
    """bm19_icdf is differentiable in alpha (grad-safe table) - enables P2 copula grads."""
    from gravoturb_fdf.theory.pdf import bm19_icdf
    import jax.numpy as jnp

    def mean_s(alpha):
        u = jnp.linspace(0.1, 0.9, 64)
        return jnp.mean(bm19_icdf(u, 5.0, 0.4, alpha))

    g = float(jax.grad(mean_s)(1.8))
    assert math.isfinite(g)

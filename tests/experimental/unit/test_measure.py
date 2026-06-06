"""Unit tests for gravoturb_fdf.validation.measure (the oracle-measurement side).

These utilities measure 2-point statistics from realization fields and build the
theory-consistent ``smooth_copula_field`` (s = bm19_icdf(Phi(g_hat)) - shift on an
EXACTLY unit-variance g_hat) used as the clean oracle for the Gaussianization series
(AC11). numpy/scipy are permitted here (validation path, non-differentiable).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.experimental


def test_autocovariance_3d_recovers_cosine_mode():
    """xi(r) of a single cosine mode = 0.5 cos(2 pi k0 dx / N) (Wiener-Khinchin)."""
    from gravoturb_fdf.validation.measure import autocovariance_3d

    n, k0 = 16, 2
    i = np.arange(n)
    f = np.cos(2 * np.pi * k0 * i / n)[:, None, None] * np.ones((n, n, n))
    xi = autocovariance_3d(f)
    dx = np.arange(n)
    expected = 0.5 * np.cos(2 * np.pi * k0 * dx / n)
    assert np.allclose(xi[:, 0, 0], expected, atol=1e-10)


def test_autocovariance_3d_zero_lag_is_variance():
    """xi(0) = Var(field)."""
    from gravoturb_fdf.validation.measure import autocovariance_3d

    rng = np.random.default_rng(0)
    f = rng.normal(size=(24, 24, 24))
    xi = autocovariance_3d(f)
    assert xi[0, 0, 0] == pytest.approx(float(np.var(f)), rel=1e-10)


def test_measured_2pt_white_noise_decorrelates():
    """White noise: rho(r>0) ~ 0 (within sampling noise); variance recovered."""
    from gravoturb_fdf.validation.measure import gaussian_correlation_measured

    rng = np.random.default_rng(1)
    g = rng.normal(size=(32, 32, 32))
    r, rho = gaussian_correlation_measured(g, n_bins=12)
    assert np.all(r > 0.0)  # zero-lag excluded
    assert np.max(np.abs(rho)) < 0.05  # no spurious correlation


def test_smooth_copula_field_marginal_lognormal_limit():
    """smooth_copula_field marginal: large alpha => Var(s) ~ sigma_s^2."""
    from gravoturb_fdf.validation.measure import smooth_copula_field
    from gravoturb_fdf.theory.bm19 import sigma_s_squared

    mach, b, alpha = 5.0, 0.4, 6.0
    sig2 = float(sigma_s_squared(mach, b))
    key = jax.random.PRNGKey(3)
    g = jax.random.normal(key, (48, 48, 48))
    s = smooth_copula_field(g, mach, b, alpha)
    assert float(np.var(s)) == pytest.approx(sig2, rel=0.03)


def test_smooth_copula_field_mean_density_unity():
    """smooth_copula_field marginal: <e^s> ~ 1 (alpha=3, finite 2nd moment)."""
    from gravoturb_fdf.validation.measure import smooth_copula_field

    key = jax.random.PRNGKey(4)
    g = jax.random.normal(key, (48, 48, 48))
    s = smooth_copula_field(g, 5.0, 0.4, 3.0)
    assert float(np.mean(np.exp(s))) == pytest.approx(1.0, rel=3e-2)

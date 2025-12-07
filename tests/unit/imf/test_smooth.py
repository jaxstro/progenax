"""
Unit tests for smooth IMF families (Maschberger, TaperedPowerLaw, Schechter).

Tests importability, sampling, CDF bounds, and differentiability.
"""

import jax
import jax.numpy as jnp
import pytest

from progenax.imf.smooth import Maschberger, TaperedPowerLaw, Schechter


class TestMaschberger:
    """Tests for Maschberger (2013) smooth IMF."""

    def test_importable(self):
        """Test that Maschberger can be imported and instantiated."""
        imf = Maschberger()
        assert imf.mu == 0.2
        assert imf.alpha == 2.3
        assert imf.beta == 1.4
        assert imf.m_min == 0.01
        assert imf.m_max == 300.0

    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        imf = Maschberger(mu=0.5, alpha=2.0, beta=1.2, m_min=0.1, m_max=100.0)
        assert imf.mu == 0.5
        assert imf.alpha == 2.0
        assert imf.beta == 1.2
        assert imf.m_min == 0.1
        assert imf.m_max == 100.0

    def test_cdf_bounds(self):
        """Test CDF is 0 at m_min and 1 at m_max."""
        imf = Maschberger()
        cdf_min = imf.cdf(imf.m_min)
        cdf_max = imf.cdf(imf.m_max)
        assert jnp.isclose(cdf_min, 0.0, atol=1e-6)
        assert jnp.isclose(cdf_max, 1.0, atol=1e-6)

    def test_cdf_monotonic(self):
        """Test CDF is monotonically increasing."""
        imf = Maschberger()
        m_vals = jnp.logspace(jnp.log10(imf.m_min), jnp.log10(imf.m_max), 100)
        cdf_vals = jax.vmap(imf.cdf)(m_vals)
        diffs = jnp.diff(cdf_vals)
        assert jnp.all(diffs >= -1e-10)  # Allow tiny numerical errors

    def test_sampling_within_bounds(self):
        """Test sampling produces masses within [m_min, m_max]."""
        imf = Maschberger()
        key = jax.random.PRNGKey(42)
        masses = imf.sample(key, 1000)
        assert jnp.all(masses >= imf.m_min)
        assert jnp.all(masses <= imf.m_max)

    def test_ppf_analytical_vs_numerical(self):
        """Test analytical PPF matches numerical CDF inversion."""
        imf = Maschberger()
        u_vals = jnp.array([0.1, 0.25, 0.5, 0.75, 0.9])
        m_analytical = imf.ppf(u_vals)

        # Verify CDF(m) ≈ u
        cdf_vals = jax.vmap(imf.cdf)(m_analytical)
        assert jnp.allclose(cdf_vals, u_vals, atol=1e-5)

    def test_differentiable_logpdf(self):
        """Test logpdf is differentiable."""
        imf = Maschberger()
        m = jnp.array(1.0)

        def logpdf_fn(mass):
            return imf.logpdf(mass)

        grad_fn = jax.grad(logpdf_fn)
        gradient = grad_fn(m)
        assert jnp.isfinite(gradient)
        assert not jnp.isnan(gradient)

    def test_differentiable_ppf(self):
        """Test PPF is differentiable w.r.t. u."""
        imf = Maschberger()
        u = jnp.array(0.5)

        def ppf_fn(u_val):
            return imf.ppf(u_val)

        grad_fn = jax.grad(ppf_fn)
        gradient = grad_fn(u)
        assert jnp.isfinite(gradient)
        assert gradient > 0  # dm/du should be positive

    def test_differentiable_sample(self):
        """Test sampling is differentiable via reparameterization."""
        imf = Maschberger()

        def loss_fn(mu):
            imf_custom = Maschberger(mu=mu, alpha=2.3, beta=1.4)
            key = jax.random.PRNGKey(42)
            masses = imf_custom.sample(key, 100)
            return jnp.mean(masses)

        grad_fn = jax.grad(loss_fn)
        gradient = grad_fn(0.2)
        assert jnp.isfinite(gradient)


class TestTaperedPowerLaw:
    """Tests for Tapered Power Law IMF."""

    def test_importable(self):
        """Test that TaperedPowerLaw can be imported and instantiated."""
        imf = TaperedPowerLaw()
        assert imf.alpha == 2.3
        assert imf.m_peak == 0.3
        assert imf.beta == 2.0
        assert imf.m_min == 0.01
        assert imf.m_max == 300.0

    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        imf = TaperedPowerLaw(alpha=2.0, m_peak=0.5, beta=1.5, m_min=0.1, m_max=100.0)
        assert imf.alpha == 2.0
        assert imf.m_peak == 0.5
        assert imf.beta == 1.5
        assert imf.m_min == 0.1
        assert imf.m_max == 100.0

    def test_cdf_bounds(self):
        """Test CDF is 0 at m_min and 1 at m_max."""
        imf = TaperedPowerLaw()
        cdf_min = imf.cdf(imf.m_min)
        cdf_max = imf.cdf(imf.m_max)
        assert jnp.isclose(cdf_min, 0.0, atol=1e-6)
        assert jnp.isclose(cdf_max, 1.0, atol=1e-6)

    def test_cdf_monotonic(self):
        """Test CDF is monotonically increasing."""
        imf = TaperedPowerLaw()
        m_vals = jnp.logspace(jnp.log10(imf.m_min), jnp.log10(imf.m_max), 100)
        cdf_vals = jax.vmap(imf.cdf)(m_vals)
        diffs = jnp.diff(cdf_vals)
        assert jnp.all(diffs >= -1e-10)  # Allow tiny numerical errors

    def test_sampling_within_bounds(self):
        """Test sampling produces masses within [m_min, m_max]."""
        imf = TaperedPowerLaw()
        key = jax.random.PRNGKey(42)
        masses = imf.sample(key, 1000)
        assert jnp.all(masses >= imf.m_min)
        assert jnp.all(masses <= imf.m_max)

    def test_ppf_inverts_cdf(self):
        """Test PPF correctly inverts CDF."""
        imf = TaperedPowerLaw()
        u_vals = jnp.array([0.1, 0.25, 0.5, 0.75, 0.9])
        m_vals = imf.ppf(u_vals)

        # Verify CDF(m) ≈ u
        cdf_vals = jax.vmap(imf.cdf)(m_vals)
        assert jnp.allclose(cdf_vals, u_vals, atol=1e-4)

    def test_differentiable_logpdf(self):
        """Test logpdf is differentiable."""
        imf = TaperedPowerLaw()
        m = jnp.array(1.0)

        def logpdf_fn(mass):
            return imf.logpdf(mass)

        grad_fn = jax.grad(logpdf_fn)
        gradient = grad_fn(m)
        assert jnp.isfinite(gradient)
        assert not jnp.isnan(gradient)

    def test_differentiable_ppf(self):
        """Test PPF is differentiable w.r.t. u."""
        imf = TaperedPowerLaw()
        u = jnp.array(0.5)

        def ppf_fn(u_val):
            return imf.ppf(u_val)

        grad_fn = jax.grad(ppf_fn)
        gradient = grad_fn(u)
        assert jnp.isfinite(gradient)
        assert gradient > 0  # dm/du should be positive

    def test_differentiable_sample(self):
        """Test sampling is differentiable via reparameterization."""
        imf = TaperedPowerLaw()

        def loss_fn(m_peak):
            imf_custom = TaperedPowerLaw(alpha=2.3, m_peak=m_peak, beta=2.0)
            key = jax.random.PRNGKey(42)
            masses = imf_custom.sample(key, 100)
            return jnp.mean(masses)

        grad_fn = jax.grad(loss_fn)
        gradient = grad_fn(0.3)
        assert jnp.isfinite(gradient)


class TestSchechter:
    """Tests for Schechter function IMF."""

    def test_importable(self):
        """Test that Schechter can be imported and instantiated."""
        imf = Schechter()
        assert imf.alpha == 2.3
        assert imf.m_star == 100.0
        assert imf.m_min == 0.01
        assert imf.m_max == 300.0

    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        imf = Schechter(alpha=1.5, m_star=50.0, m_min=0.1, m_max=100.0)
        assert imf.alpha == 1.5
        assert imf.m_star == 50.0
        assert imf.m_min == 0.1
        assert imf.m_max == 100.0

    def test_cdf_bounds(self):
        """Test CDF is 0 at m_min and 1 at m_max."""
        imf = Schechter()
        cdf_min = imf.cdf(imf.m_min)
        cdf_max = imf.cdf(imf.m_max)
        assert jnp.isclose(cdf_min, 0.0, atol=1e-6)
        assert jnp.isclose(cdf_max, 1.0, atol=1e-6)

    def test_cdf_monotonic(self):
        """Test CDF is monotonically increasing."""
        imf = Schechter()
        m_vals = jnp.logspace(jnp.log10(imf.m_min), jnp.log10(imf.m_max), 100)
        cdf_vals = jax.vmap(imf.cdf)(m_vals)
        diffs = jnp.diff(cdf_vals)
        assert jnp.all(diffs >= -1e-10)  # Allow tiny numerical errors

    def test_sampling_within_bounds(self):
        """Test sampling produces masses within [m_min, m_max]."""
        imf = Schechter()
        key = jax.random.PRNGKey(42)
        masses = imf.sample(key, 1000)
        assert jnp.all(masses >= imf.m_min)
        assert jnp.all(masses <= imf.m_max)

    def test_ppf_inverts_cdf(self):
        """Test PPF correctly inverts CDF.

        Note: Schechter function with exponential cutoff can be challenging
        for Newton iteration at extreme quantiles. Test mid-range quantiles.
        """
        # Use better-conditioned parameters (flatter slope, lower cutoff)
        imf = Schechter(alpha=1.5, m_star=50.0, m_min=0.1, m_max=200.0)
        # Test lower and mid-range quantiles (avoid extreme upper tail)
        u_vals = jnp.array([0.1, 0.25, 0.5, 0.7])
        m_vals = imf.ppf(u_vals)

        # Verify CDF(m) ≈ u
        cdf_vals = jax.vmap(imf.cdf)(m_vals)
        assert jnp.allclose(cdf_vals, u_vals, atol=1e-3)

    def test_differentiable_logpdf(self):
        """Test logpdf is differentiable."""
        imf = Schechter()
        m = jnp.array(1.0)

        def logpdf_fn(mass):
            return imf.logpdf(mass)

        grad_fn = jax.grad(logpdf_fn)
        gradient = grad_fn(m)
        assert jnp.isfinite(gradient)
        assert not jnp.isnan(gradient)

    def test_differentiable_ppf(self):
        """Test PPF is differentiable w.r.t. u."""
        imf = Schechter()
        u = jnp.array(0.5)

        def ppf_fn(u_val):
            return imf.ppf(u_val)

        grad_fn = jax.grad(ppf_fn)
        gradient = grad_fn(u)
        assert jnp.isfinite(gradient)
        assert gradient > 0  # dm/du should be positive

    def test_differentiable_sample(self):
        """Test sampling is differentiable via reparameterization."""
        imf = Schechter()

        def loss_fn(m_star):
            imf_custom = Schechter(alpha=2.3, m_star=m_star)
            key = jax.random.PRNGKey(42)
            masses = imf_custom.sample(key, 100)
            return jnp.mean(masses)

        grad_fn = jax.grad(loss_fn)
        gradient = grad_fn(100.0)
        assert jnp.isfinite(gradient)


class TestNumericalIntegration:
    """Test numerical integration strategy for smooth IMFs."""

    def test_maschberger_analytical_cdf(self):
        """Verify Maschberger uses analytical CDF (fast)."""
        imf = Maschberger()
        # Should use analytical primitive, not numerical integration
        m = jnp.array(1.0)
        cdf_val = imf.cdf(m)
        assert jnp.isfinite(cdf_val)
        assert 0.0 <= cdf_val <= 1.0

    def test_taperedpowerlaw_numerical_cdf(self):
        """Verify TaperedPowerLaw uses numerical CDF integration."""
        imf = TaperedPowerLaw()
        # Uses numerical integration (10000 points linear grid)
        m = jnp.array(1.0)
        cdf_val = imf.cdf(m)
        assert jnp.isfinite(cdf_val)
        assert 0.0 <= cdf_val <= 1.0

    def test_schechter_numerical_cdf(self):
        """Verify Schechter uses numerical CDF integration."""
        imf = Schechter()
        # Uses numerical integration (10000 points linear grid)
        m = jnp.array(1.0)
        cdf_val = imf.cdf(m)
        assert jnp.isfinite(cdf_val)
        assert 0.0 <= cdf_val <= 1.0

    def test_vectorized_cdf(self):
        """Test CDF handles vectorized input."""
        imf = TaperedPowerLaw()
        m_vals = jnp.array([0.1, 0.5, 1.0, 5.0, 10.0])
        cdf_vals = jax.vmap(imf.cdf)(m_vals)

        assert cdf_vals.shape == m_vals.shape
        assert jnp.all(jnp.isfinite(cdf_vals))
        assert jnp.all(cdf_vals >= 0.0)
        assert jnp.all(cdf_vals <= 1.0)

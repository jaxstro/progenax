"""Tests for binary population parameter sampling."""

import jax
import jax.numpy as jnp
import pytest

from progenax.binaries.population import (
    LogUniformPeriod,
    LogNormalPeriod,
    ThermalEccentricity,
    UniformEccentricity,
)


class TestLogUniformPeriod:
    """Tests for Öpik's law: log-uniform period distribution."""

    def test_sample_shape(self):
        """Samples have correct shape."""
        dist = LogUniformPeriod(log_P_min=0.0, log_P_max=8.0)
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 1000)
        assert samples.shape == (1000,)

    def test_samples_in_range(self):
        """All samples within [P_min, P_max]."""
        dist = LogUniformPeriod(log_P_min=0.0, log_P_max=8.0)
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 10000)
        log_samples = jnp.log10(samples)
        assert jnp.all(log_samples >= 0.0 - 1e-6)
        assert jnp.all(log_samples <= 8.0 + 1e-6)

    def test_log_uniform_distribution(self):
        """log10(P) is uniformly distributed (Öpik's law)."""
        dist = LogUniformPeriod(log_P_min=0.0, log_P_max=8.0)
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 100000)
        log_samples = jnp.log10(samples)
        mean_log_P = jnp.mean(log_samples)
        assert jnp.abs(mean_log_P - 4.0) < 0.1

    def test_pdf_normalization(self):
        """PDF integrates to 1."""
        dist = LogUniformPeriod(log_P_min=0.0, log_P_max=4.0)
        P_vals = jnp.logspace(0, 4, 1000)
        pdf_vals = dist.pdf(P_vals)
        integral = jnp.trapezoid(pdf_vals, P_vals)
        assert jnp.abs(integral - 1.0) < 0.01


class TestLogNormalPeriod:
    """Tests for log-normal period distribution."""

    def test_sample_shape(self):
        """Samples have correct shape."""
        dist = LogNormalPeriod(mu_log_P=4.0, sigma_log_P=2.0)
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 1000)
        assert samples.shape == (1000,)

    def test_mean_log_period(self):
        """Mean of log10(P) matches mu parameter."""
        dist = LogNormalPeriod(mu_log_P=4.0, sigma_log_P=2.0)
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 100000)
        log_samples = jnp.log10(samples)
        mean_log_P = jnp.mean(log_samples)
        assert jnp.abs(mean_log_P - 4.0) < 0.05

    def test_std_log_period(self):
        """Std of log10(P) matches sigma parameter."""
        dist = LogNormalPeriod(mu_log_P=4.0, sigma_log_P=2.0)
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 100000)
        log_samples = jnp.log10(samples)
        std_log_P = jnp.std(log_samples)
        assert jnp.abs(std_log_P - 2.0) < 0.1

    def test_positive_periods(self):
        """All sampled periods are positive."""
        dist = LogNormalPeriod(mu_log_P=2.0, sigma_log_P=1.5)
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 10000)
        assert jnp.all(samples > 0)


class TestThermalEccentricity:
    """Tests for thermal eccentricity distribution f(e) = 2e."""

    def test_sample_shape(self):
        """Samples have correct shape."""
        dist = ThermalEccentricity()
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 1000)
        assert samples.shape == (1000,)

    def test_samples_in_range(self):
        """All samples in [0, 1)."""
        dist = ThermalEccentricity()
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 10000)
        assert jnp.all(samples >= 0.0)
        assert jnp.all(samples < 1.0)

    def test_thermal_mean(self):
        """Mean eccentricity for thermal is (2/3) * e_max."""
        dist = ThermalEccentricity(e_max=0.99)
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 100000)
        mean_e = jnp.mean(samples)
        expected_mean = (2.0 / 3.0) * 0.99
        assert jnp.abs(mean_e - expected_mean) < 0.01

    def test_thermal_cdf(self):
        """CDF is e² / e_max² for thermal distribution."""
        dist = ThermalEccentricity(e_max=0.99)
        e_vals = jnp.array([0.0, 0.5, 0.7, 0.99])
        expected = (e_vals / 0.99) ** 2
        computed = dist.cdf(e_vals)
        assert jnp.allclose(computed, expected, atol=1e-10)

    def test_ppf_inverse_cdf(self):
        """PPF is inverse of CDF."""
        dist = ThermalEccentricity()
        u_vals = jnp.array([0.0, 0.25, 0.49, 0.81, 1.0])
        e_vals = dist.ppf(u_vals)
        u_recovered = dist.cdf(e_vals)
        assert jnp.allclose(u_recovered, u_vals, atol=1e-10)


class TestUniformEccentricity:
    """Tests for uniform eccentricity distribution."""

    def test_sample_in_range(self):
        """Samples in [e_min, e_max]."""
        dist = UniformEccentricity(e_min=0.0, e_max=0.8)
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 10000)
        assert jnp.all(samples >= 0.0 - 1e-6)
        assert jnp.all(samples <= 0.8 + 1e-6)

    def test_uniform_mean(self):
        """Mean eccentricity is (e_min + e_max) / 2."""
        dist = UniformEccentricity(e_min=0.1, e_max=0.9)
        key = jax.random.PRNGKey(42)
        samples = dist.sample(key, 100000)
        mean_e = jnp.mean(samples)
        assert jnp.abs(mean_e - 0.5) < 0.01

"""Tests for binary population parameter sampling."""

import jax
import jax.numpy as jnp
import pytest

from progenax.binaries.population import (
    LogUniformPeriod,
    LogNormalPeriod,
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

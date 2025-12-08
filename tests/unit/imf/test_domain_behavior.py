"""Tests for domain behavior of IMF logpdf and cdf."""
import jax.numpy as jnp
import pytest
from progenax.imf import ChabrierIMF


class TestLogPDFDomain:
    """logpdf returns -inf outside [m_min, m_max]."""

    def test_logpdf_below_m_min(self):
        """logpdf returns -inf for m < m_min."""
        imf = ChabrierIMF()  # m_min = 0.08
        m_below = jnp.array([0.01, 0.05, 0.079])
        logpdf_vals = imf.logpdf(m_below)
        assert jnp.all(jnp.isneginf(logpdf_vals)), \
            f"logpdf should be -inf below m_min, got {logpdf_vals}"

    def test_logpdf_above_m_max(self):
        """logpdf returns -inf for m > m_max."""
        imf = ChabrierIMF()  # m_max = 100
        m_above = jnp.array([101.0, 150.0, 1000.0])
        logpdf_vals = imf.logpdf(m_above)
        assert jnp.all(jnp.isneginf(logpdf_vals)), \
            f"logpdf should be -inf above m_max, got {logpdf_vals}"

    def test_logpdf_inside_domain_finite(self):
        """logpdf returns finite values inside domain."""
        imf = ChabrierIMF()
        m_inside = jnp.array([0.1, 1.0, 10.0, 50.0])
        logpdf_vals = imf.logpdf(m_inside)
        assert jnp.all(jnp.isfinite(logpdf_vals)), \
            f"logpdf should be finite inside domain, got {logpdf_vals}"


class TestCDFDomain:
    """cdf returns 0 below m_min, 1 above m_max."""

    def test_cdf_below_m_min(self):
        """cdf returns 0 for m <= m_min."""
        imf = ChabrierIMF()
        m_below = jnp.array([0.01, 0.05, 0.08])
        cdf_vals = imf.cdf(m_below)
        assert jnp.allclose(cdf_vals, 0.0, atol=1e-6), \
            f"cdf should be 0 at/below m_min, got {cdf_vals}"

    def test_cdf_above_m_max(self):
        """cdf returns 1 for m >= m_max."""
        imf = ChabrierIMF()
        m_above = jnp.array([100.0, 101.0, 1000.0])
        cdf_vals = imf.cdf(m_above)
        assert jnp.allclose(cdf_vals, 1.0, atol=1e-6), \
            f"cdf should be 1 at/above m_max, got {cdf_vals}"

    def test_cdf_inside_domain_valid(self):
        """cdf returns values in (0, 1) inside domain."""
        imf = ChabrierIMF()
        m_inside = jnp.array([0.1, 1.0, 10.0, 50.0])
        cdf_vals = imf.cdf(m_inside)
        assert jnp.all((cdf_vals > 0) & (cdf_vals < 1)), \
            f"cdf should be in (0,1) inside domain, got {cdf_vals}"

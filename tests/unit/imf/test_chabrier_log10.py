"""Tests for Chabrier (2003) log₁₀-based system IMF."""
import jax
import jax.numpy as jnp
import pytest
from progenax.imf import ChabrierIMF


class TestChabrierLog10:
    """Verify Chabrier uses log₁₀ (not ln) as per Chabrier (2003)."""

    def test_lognormal_jacobian_factor(self):
        """Low-mass PDF has correct 1/(m ln 10) Jacobian factor.

        At m = m_c (peak of lognormal in log-space), the exponential = 1,
        so unnormalized pdf at m_c should be A_ln / (m_c * ln(10)).
        """
        imf = ChabrierIMF()
        m_c = imf.m_c  # 0.08

        # Get unnormalized PDF at m_c
        pdf_at_mc = imf._lognormal_pdf_unnorm(jnp.array(m_c))

        # Expected: A_ln / (m_c * ln(10))
        expected = imf.A_ln / (m_c * jnp.log(10.0))

        rel_error = jnp.abs(pdf_at_mc - expected) / expected
        assert rel_error < 0.01, \
            f"PDF at m_c: {float(pdf_at_mc):.6f}, expected {float(expected):.6f}"

    def test_lognormal_integral_matches_numerical(self):
        """Analytical lognormal integral matches numerical integration."""
        imf = ChabrierIMF()

        # Analytical integral
        integral_analytical = imf._lognormal_integral(imf.m_min, imf.m_trans)

        # Numerical integration
        m_grid = jnp.linspace(imf.m_min, imf.m_trans, 10000)
        pdf_grid = imf._lognormal_pdf_unnorm(m_grid)
        integral_numerical = jnp.trapezoid(pdf_grid, m_grid)

        rel_error = jnp.abs(integral_analytical - integral_numerical) / integral_numerical
        assert rel_error < 0.01, \
            f"Analytical: {float(integral_analytical):.6f}, Numerical: {float(integral_numerical):.6f}"


class TestChabrierContinuity:
    """Verify continuity at m_trans."""

    def test_continuous_at_m_trans(self):
        """PDF is continuous at transition mass."""
        imf = ChabrierIMF()
        eps = 1e-6

        pdf_below = jnp.exp(imf.logpdf(jnp.array(imf.m_trans - eps)))
        pdf_above = jnp.exp(imf.logpdf(jnp.array(imf.m_trans + eps)))

        rel_diff = jnp.abs(pdf_below - pdf_above) / pdf_below
        assert rel_diff < 0.01, \
            f"Discontinuity at m_trans: {float(pdf_below):.6f} vs {float(pdf_above):.6f}"

    def test_A_pl_ensures_continuity(self):
        """A_pl is computed for continuity at m_trans."""
        imf = ChabrierIMF()

        ln_pdf = imf._lognormal_pdf_unnorm(jnp.array(imf.m_trans))
        pl_pdf = imf._powerlaw_pdf_unnorm(jnp.array(imf.m_trans))

        rel_diff = jnp.abs(ln_pdf - pl_pdf) / ln_pdf
        assert rel_diff < 1e-6, \
            f"At m_trans: lognormal={float(ln_pdf):.6e}, powerlaw={float(pl_pdf):.6e}"

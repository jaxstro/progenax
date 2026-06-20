"""
Parametrized unit tests for IMF mathematical correctness.

Tests core properties that EVERY IMF must satisfy:
- CDF bounds: CDF(m_min)=0, CDF(m_max)=1
- CDF monotonicity: CDF increases with m
- PPF inverse: PPF(CDF(m)) ≈ m
- PDF normalization: ∫ PDF dm = 1
- Mean mass convergence

Uses pytest parametrization to test all IMF classes uniformly.
"""

import jax
import jax.numpy as jnp
import pytest

from progenax.imf import (
    ChabrierIMF,
    Maschberger,
    PowerLawIMF,
    TruncatedIMF,
)

# =============================================================================
# IMF Factory Fixtures
# =============================================================================


def create_salpeter():
    """Salpeter (1955) single power-law IMF."""
    return PowerLawIMF(exponents=[2.35], breakpoints=[], m_min=0.1, m_max=100.0)


def create_kroupa():
    """Kroupa (2001) three-segment power-law IMF."""
    return PowerLawIMF(
        exponents=[0.3, 1.3, 2.3],
        breakpoints=[0.08, 0.5],
        m_min=0.01,
        m_max=100.0,
    )


def create_chabrier():
    """Chabrier (2003) lognormal + power-law IMF."""
    return ChabrierIMF()


def create_maschberger():
    """Maschberger (2013) smooth IMF."""
    return Maschberger()


def create_truncated_chabrier():
    """Truncated Chabrier IMF (narrower mass range)."""
    return TruncatedIMF(ChabrierIMF(), m_min=0.1, m_max=50.0)


IMF_FACTORIES = [
    ("Salpeter", create_salpeter),
    ("Kroupa", create_kroupa),
    ("Chabrier", create_chabrier),
    ("Maschberger", create_maschberger),
    ("TruncatedChabrier", create_truncated_chabrier),
]

IMF_NAMES = [name for name, _ in IMF_FACTORIES]


@pytest.fixture(params=IMF_FACTORIES, ids=IMF_NAMES)
def imf(request):
    """Parametrized fixture providing each IMF instance."""
    _, factory = request.param
    return factory()


# =============================================================================
# CDF Properties (Mathematical)
# =============================================================================


class TestCDFProperties:
    """CDF must satisfy mathematical constraints."""

    def test_cdf_at_m_min(self, imf):
        """CDF(m_min) = 0."""
        cdf_min = imf.cdf(jnp.array(imf.m_min))
        assert abs(float(cdf_min)) < 1e-6, (
            f"CDF(m_min={imf.m_min}) = {float(cdf_min)}, expected 0"
        )

    def test_cdf_at_m_max(self, imf):
        """CDF(m_max) = 1."""
        cdf_max = imf.cdf(jnp.array(imf.m_max))
        assert abs(float(cdf_max) - 1.0) < 1e-6, (
            f"CDF(m_max={imf.m_max}) = {float(cdf_max)}, expected 1"
        )

    def test_cdf_monotonic_increasing(self, imf):
        """CDF is monotonically increasing."""
        m_grid = jnp.linspace(imf.m_min, imf.m_max, 200)
        cdf_grid = imf.cdf(m_grid)

        diffs = jnp.diff(cdf_grid)
        assert jnp.all(diffs >= -1e-3), (
            f"CDF not monotonic: min diff = {float(jnp.min(diffs))}"
        )


# =============================================================================
# PPF Inverse Property (Mathematical)
# =============================================================================


class TestPPFInverse:
    """PPF must be inverse of CDF."""

    def test_ppf_cdf_roundtrip(self, imf):
        """PPF(CDF(m)) recovers original mass."""
        m_test = jnp.linspace(imf.m_min * 1.01, imf.m_max * 0.99, 10)

        u_values = imf.cdf(m_test)
        m_recovered = imf.ppf(u_values)

        relative_errors = jnp.abs(m_recovered - m_test) / m_test
        max_error = float(jnp.max(relative_errors))

        assert max_error < 0.10, (
            f"PPF(CDF(m)) roundtrip error: {max_error * 100:.2f}% (expected <10%)"
        )


# =============================================================================
# PDF Normalization (Physics)
# =============================================================================


class TestPDFNormalization:
    """PDF must integrate to 1 over [m_min, m_max]."""

    def test_pdf_integrates_to_one(self, imf):
        """∫_{m_min}^{m_max} PDF(m) dm = 1."""
        log_m_grid = jnp.linspace(jnp.log(imf.m_min + 1e-6), jnp.log(imf.m_max), 10000)
        m_grid = jnp.exp(log_m_grid)
        pdf_grid = jnp.exp(imf.logpdf(m_grid))

        # For log-spaced grid: ∫f dm = ∫f(m) m d(ln m)
        integrand = pdf_grid * m_grid
        integral = jnp.trapezoid(integrand, log_m_grid)

        assert abs(float(integral) - 1.0) < 0.15, (
            f"PDF integral = {float(integral):.4f}, expected 1.0"
        )


# =============================================================================
# Statistical Properties
# =============================================================================


class TestStatisticalProperties:
    """Statistical properties of IMF samples."""

    def test_sample_mean_converges(self, imf, key):
        """Sample mean converges to theoretical mean."""
        masses = imf.sample(key, 10000)
        sample_mean = float(jnp.mean(masses))
        theory_mean = float(imf.mean_mass())

        relative_error = abs(sample_mean - theory_mean) / theory_mean

        assert relative_error < 0.20, (
            f"Sample mean {sample_mean:.4f} vs theory {theory_mean:.4f}, error {relative_error * 100:.1f}%"
        )


# =============================================================================
# JIT Compatibility (keep 1 test)
# =============================================================================


class TestJITCompatibility:
    """IMF methods must be JIT-compilable."""

    def test_sample_jits(self, imf, key):
        """sample() can be JIT-compiled."""

        @jax.jit
        def sample_wrapper(k):
            return imf.sample(k, 100)

        masses = sample_wrapper(key)
        assert jnp.all(jnp.isfinite(masses))


# =============================================================================
# Chabrier-Specific Tests
# =============================================================================


class TestChabrierSpecific:
    """Chabrier-specific tests for lognormal + power-law structure."""

    def test_A_pl_continuity(self):
        """A_pl ensures continuity at m_trans (lognormal → power-law)."""
        imf = ChabrierIMF()

        ln_pdf = imf._lognormal_pdf_unnorm(imf.m_trans)
        pl_pdf = imf._powerlaw_pdf_unnorm(imf.m_trans)

        assert jnp.abs(ln_pdf - pl_pdf) / ln_pdf < 0.01, (
            f"Discontinuity at m_trans: lognormal={ln_pdf}, powerlaw={pl_pdf}"
        )

    def test_lognormal_jacobian_factor(self):
        """Low-mass PDF has correct 1/(m ln 10) Jacobian factor (Chabrier 2003).

        At m = m_c (peak of lognormal in log-space), the exponential = 1,
        so unnormalized pdf at m_c should be A_ln / (m_c * ln(10)).
        """
        imf = ChabrierIMF()
        m_c = imf.m_c

        pdf_at_mc = imf._lognormal_pdf_unnorm(jnp.array(m_c))
        expected = imf.A_ln / (m_c * jnp.log(10.0))

        rel_error = jnp.abs(pdf_at_mc - expected) / expected
        assert rel_error < 0.01, (
            f"PDF at m_c: {float(pdf_at_mc):.6f}, expected {float(expected):.6f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

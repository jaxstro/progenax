"""Tests for smooth IMF families (Maschberger, TaperedPowerLaw, Schechter).

Coverage-gap tests for the numerical integration helpers and the
unnormalized log-PDF / CDF building blocks that the base-class
normalization and PPF solver rely on.

These assert real behavior:
- `_linear_trapz_integrate` reproduces an analytic power-law integral.
- `_scalar_cdf_unnorm` honors its m < m_min boundary (returns 0).
- TaperedPowerLaw / Schechter `_logpdf_unnorm` stay finite at edges.
- `_cdf_unnorm` (array branch) is shape-preserving and monotonic.
- The full normalized CDF round-trips with the PPF (cdf(ppf(u)) ~= u).
"""

import jax
import jax.numpy as jnp
import pytest

from progenax.imf.smooth import (
    Maschberger,
    Schechter,
    TaperedPowerLaw,
    _linear_trapz_integrate,
    _scalar_cdf_unnorm,
)


# =============================================================================
# Numerical integration helpers
# =============================================================================


class TestLinearTrapzIntegrate:
    """Verify the dense trapezoid integrator against an analytic integral."""

    def test_matches_powerlaw_integral(self):
        """integral of m^-alpha over [a,b] = (b^(1-a) - a^(1-a))/(1-alpha)."""
        alpha = 2.3
        m_min, m_max = 0.1, 50.0

        # log_pdf for a pure power law m^(-alpha)
        def log_pdf(m):
            return -alpha * jnp.log(m)

        numeric = _linear_trapz_integrate(log_pdf, m_min, m_max, n_points=20000)

        p = 1.0 - alpha  # exponent of the antiderivative
        analytic = (m_max**p - m_min**p) / p

        rel_err = jnp.abs(numeric - analytic) / jnp.abs(analytic)
        assert rel_err < 1e-3, (
            f"trapz integral {float(numeric):.5f} vs analytic "
            f"{float(analytic):.5f} (rel err {float(rel_err):.2e})"
        )

    def test_matches_flat_integral(self):
        """A flat PDF (log_pdf=0) integrates to (m_max - m_min) exactly."""
        m_min, m_max = 0.5, 3.5

        def log_pdf(m):
            return jnp.zeros_like(m)

        numeric = _linear_trapz_integrate(log_pdf, m_min, m_max, n_points=5000)
        assert jnp.isclose(numeric, m_max - m_min, rtol=1e-6)

    def test_zero_width_interval(self):
        """An interval [m_min, m_min] integrates to zero."""
        def log_pdf(m):
            return -2.3 * jnp.log(m)

        numeric = _linear_trapz_integrate(log_pdf, 1.0, 1.0, n_points=1000)
        assert jnp.abs(numeric) < 1e-12


class TestScalarCdfUnnorm:
    """Cover the m <= m_min boundary branch in _scalar_cdf_unnorm."""

    def _log_pdf(self, m):
        return -2.3 * jnp.log(m + 1e-30)

    def test_below_m_min_is_zero(self):
        """m < m_min returns exactly 0 (jnp.where short-circuit branch)."""
        val = _scalar_cdf_unnorm(self._log_pdf, jnp.array(0.005), m_min=0.01)
        assert float(val) == 0.0

    def test_at_m_min_is_zero(self):
        """m == m_min returns 0 (boundary of the <= comparison)."""
        val = _scalar_cdf_unnorm(self._log_pdf, jnp.array(0.01), m_min=0.01)
        assert float(val) == 0.0

    def test_above_m_min_is_positive_and_increasing(self):
        """m > m_min returns a positive, increasing partial integral."""
        m_min = 0.01
        v1 = _scalar_cdf_unnorm(self._log_pdf, jnp.array(0.5), m_min=m_min)
        v2 = _scalar_cdf_unnorm(self._log_pdf, jnp.array(2.0), m_min=m_min)
        assert float(v1) > 0.0
        assert float(v2) > float(v1), "CDF integral must grow with upper limit"


# =============================================================================
# TaperedPowerLaw
# =============================================================================


class TestTaperedPowerLaw:
    """Cover TaperedPowerLaw _logpdf_unnorm edges and _cdf_unnorm array branch."""

    def test_logpdf_finite_low_mass(self):
        """Low-mass taper (small-argument branch) stays finite, never NaN/-inf."""
        imf = TaperedPowerLaw(alpha=2.3, m_peak=0.3, beta=2.0, m_min=0.01)
        # Masses well below m_peak trigger the taper_arg < 0.01 branch
        m_low = jnp.array([1e-3, 5e-3, 0.01, 0.02])
        lp = imf._logpdf_unnorm(m_low)
        assert jnp.all(jnp.isfinite(lp)), f"non-finite low-mass logpdf: {lp}"

    def test_logpdf_finite_across_range(self):
        """logpdf is finite across the full domain (both taper branches)."""
        imf = TaperedPowerLaw()
        m = jnp.logspace(-2.0, jnp.log10(300.0), 200)
        lp = imf._logpdf_unnorm(m)
        assert jnp.all(jnp.isfinite(lp))

    def test_taper_suppresses_low_mass(self):
        """The taper suppresses the PDF below m_peak relative to pure power law.

        At m << m_peak the factor (1 - exp(-(m/m_peak)^beta)) -> (m/m_peak)^beta,
        so logpdf_unnorm should be ABOVE the bare -alpha*log(m) power law would
        be by ~beta*log(x) (x<1 -> negative), i.e. taper REDUCES the density.
        """
        imf = TaperedPowerLaw(alpha=2.3, m_peak=0.3, beta=2.0)
        m = jnp.array(0.03)  # well below m_peak
        lp_tapered = imf._logpdf_unnorm(m)
        lp_powerlaw = -imf.alpha * jnp.log(m + 1e-30)
        # Taper term is negative for x<1 -> tapered logpdf below bare power law
        assert float(lp_tapered) < float(lp_powerlaw)

    def test_cdf_unnorm_array_shape_and_monotonic(self):
        """_cdf_unnorm(array): shape preserved, non-decreasing (vmap branch).

        Monotonicity tolerance note: _cdf_unnorm re-grids [m_min, m_val] with a
        fixed n_points for EACH upper limit (see _linear_trapz_integrate). For
        the steep m^-alpha integrand (mass piled up just above m_min), adjacent
        upper limits sample the low-mass spike at slightly different nodes,
        giving O(1e-4)-relative quadrature wiggle that shrinks as n_points grows
        (verified: 200k points -> strictly increasing). We therefore require
        non-decreasing to a quadrature-aware RELATIVE floor (1e-3 of the max),
        which still fails hard for a broken/constant/sign-flipped CDF.
        """
        imf = TaperedPowerLaw(m_min=0.01, m_max=100.0)
        m = jnp.linspace(0.01, 100.0, 40)
        F = imf._cdf_unnorm(m)
        assert F.shape == m.shape
        assert jnp.all(jnp.isfinite(F))
        # Non-decreasing up to quadrature noise (relative floor)
        F_scale = jnp.max(jnp.abs(F))
        assert jnp.all(jnp.diff(F) >= -1e-3 * F_scale), (
            "tapered _cdf_unnorm decreased beyond quadrature noise"
        )
        # Discriminating: the integral genuinely grows (start near 0, end ~ max)
        assert F[0] < 1e-6 * F_scale
        assert F[-1] >= 0.99 * F_scale
        # Reshape branch: 2D input preserved
        m2d = m.reshape(8, 5)
        F2d = imf._cdf_unnorm(m2d)
        assert F2d.shape == (8, 5)

    def test_cdf_unnorm_scalar_branch(self):
        """_cdf_unnorm scalar input returns a scalar (is_scalar branch)."""
        imf = TaperedPowerLaw()
        F = imf._cdf_unnorm(jnp.array(1.0))
        assert jnp.ndim(F) == 0
        assert float(F) > 0.0

    def test_full_cdf_ppf_roundtrip(self):
        """Normalized cdf(ppf(u)) ~= u via the BaseIMF Newton solver."""
        imf = TaperedPowerLaw(alpha=2.3, m_peak=0.3, beta=2.0, m_min=0.01, m_max=100.0)
        u = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        m = imf.ppf(u)
        assert jnp.all(m >= imf.m_min - 1e-6)
        assert jnp.all(m <= imf.m_max + 1e-6)
        u_round = imf.cdf(m)
        assert jnp.allclose(u_round, u, atol=1e-3), (
            f"cdf(ppf(u)) != u: {u_round}"
        )


# =============================================================================
# Schechter
# =============================================================================


class TestSchechter:
    """Cover Schechter _logpdf_unnorm edges and _cdf_unnorm array branch."""

    def test_logpdf_finite_low_mass(self):
        """m -> 0 (with +1e-30 guard) keeps logpdf finite (no log(0))."""
        imf = Schechter(alpha=2.3, m_star=100.0, m_min=0.01)
        m = jnp.array([1e-4, 1e-3, 0.01])
        lp = imf._logpdf_unnorm(m)
        assert jnp.all(jnp.isfinite(lp))

    def test_logpdf_at_m_star(self):
        """At m = m_star, logpdf = -alpha*log(m_star) - 1 (exact, finite)."""
        imf = Schechter(alpha=2.3, m_star=50.0)
        lp = imf._logpdf_unnorm(jnp.array(50.0))
        expected = -2.3 * jnp.log(50.0 + 1e-30) - 1.0  # -m/m_star = -1
        assert jnp.isclose(lp, expected, rtol=1e-6)
        assert jnp.isfinite(lp)

    def test_logpdf_large_mass_decays(self):
        """Exponential cutoff drives logpdf strongly negative for m >> m_star."""
        imf = Schechter(alpha=1.35, m_star=10.0, m_max=300.0)
        lp_small = imf._logpdf_unnorm(jnp.array(10.0))
        lp_large = imf._logpdf_unnorm(jnp.array(200.0))  # 20x cutoff
        assert jnp.isfinite(lp_large)
        # The exp(-m/m_star) cutoff dominates: logpdf must drop sharply
        assert float(lp_large) < float(lp_small) - 10.0

    def test_cdf_unnorm_array_shape_and_monotonic(self):
        """_cdf_unnorm(array): shape preserved, non-decreasing (vmap branch).

        Same quadrature-noise consideration as TaperedPowerLaw: require
        non-decreasing to a relative floor (1e-3 of max) that still fails a
        broken CDF by orders of magnitude.
        """
        imf = Schechter(alpha=1.35, m_star=10.0, m_min=0.01, m_max=100.0)
        m = jnp.linspace(0.01, 100.0, 40)
        F = imf._cdf_unnorm(m)
        assert F.shape == m.shape
        assert jnp.all(jnp.isfinite(F))
        F_scale = jnp.max(jnp.abs(F))
        assert jnp.all(jnp.diff(F) >= -1e-3 * F_scale), (
            "Schechter _cdf_unnorm decreased beyond quadrature noise"
        )
        assert F[0] < 1e-6 * F_scale
        assert F[-1] >= 0.99 * F_scale
        # 2D reshape branch
        F2d = imf._cdf_unnorm(m.reshape(5, 8))
        assert F2d.shape == (5, 8)

    def test_cdf_unnorm_scalar_branch(self):
        """_cdf_unnorm scalar input returns a scalar."""
        imf = Schechter()
        F = imf._cdf_unnorm(jnp.array(1.0))
        assert jnp.ndim(F) == 0
        assert float(F) > 0.0

    def test_full_cdf_ppf_roundtrip(self):
        """Normalized cdf(ppf(u)) ~= u via the BaseIMF Newton solver."""
        imf = Schechter(alpha=1.35, m_star=10.0, m_min=0.01, m_max=100.0)
        u = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        m = imf.ppf(u)
        assert jnp.all(m >= imf.m_min - 1e-6)
        assert jnp.all(m <= imf.m_max + 1e-6)
        u_round = imf.cdf(m)
        assert jnp.allclose(u_round, u, atol=1e-3)


# =============================================================================
# Maschberger (analytic primitive path - light coverage of the shared family)
# =============================================================================


class TestMaschbergerSanity:
    """Light checks that the analytic-primitive Maschberger stays consistent."""

    def test_ppf_in_domain_and_cdf_roundtrip(self):
        """Analytic ppf stays in [m_min, m_max] and inverts the CDF."""
        imf = Maschberger(mu=0.2, alpha=2.3, beta=1.4, m_min=0.01, m_max=300.0)
        u = jnp.array([0.05, 0.25, 0.5, 0.75, 0.95])
        m = imf.ppf(u)
        assert jnp.all(m >= imf.m_min - 1e-6)
        assert jnp.all(m <= imf.m_max + 1e-6)
        # Analytic primitive CDF should invert tightly
        u_round = imf.cdf(m)
        assert jnp.allclose(u_round, u, atol=1e-4)

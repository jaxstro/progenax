"""Tests for smooth IMF families (Maschberger, TaperedPowerLaw, Schechter).

Coverage-gap tests for the unnormalized log-PDF / CDF building blocks that the
base-class normalization and PPF solver rely on.

These assert real behavior:
- `_shared_grid_cdf_unnorm` reproduces an analytic power-law CDF and its boundaries.
- TaperedPowerLaw / Schechter `_logpdf_unnorm` stay finite at edges.
- `_cdf_unnorm` is shape-preserving and EXACTLY monotone (shared-grid construction).
- The full normalized CDF round-trips with the PPF (cdf(ppf(u)) ~= u).
"""

import jax
import jax.numpy as jnp
import pytest

from progenax.imf.smooth import (
    Maschberger,
    Schechter,
    TaperedPowerLaw,
    _shared_grid_cdf_unnorm,
)


# =============================================================================
# Shared-grid cumulative CDF helper
# =============================================================================


class TestSharedGridCdfUnnorm:
    """The shared-grid cumulative CDF matches an analytic power-law integral and is
    monotone + boundary-correct on the production path."""

    def test_matches_powerlaw_cdf(self):
        """int_{m_min}^m s^-alpha ds = (m^(1-a) - m_min^(1-a))/(1-alpha)."""
        alpha = 2.3
        m_min, m_max = 0.1, 50.0

        def log_pdf(m):
            return -alpha * jnp.log(m)

        m = jnp.linspace(0.2, 50.0, 25)
        numeric = _shared_grid_cdf_unnorm(log_pdf, m, m_min, m_max, n_points=8000)
        p = 1.0 - alpha
        analytic = (m**p - m_min**p) / p
        rel_err = jnp.abs(numeric - analytic) / jnp.abs(analytic)
        assert jnp.all(rel_err < 2e-3), f"max rel err {float(jnp.max(rel_err)):.2e}"

    def test_boundaries(self):
        """m <= m_min -> 0; m >= m_max -> the full integral; monotone throughout."""
        def log_pdf(m):
            return -2.3 * jnp.log(m + 1e-30)

        assert float(_shared_grid_cdf_unnorm(log_pdf, jnp.array(0.005), 0.01, 100.0)) == 0.0
        full = float(_shared_grid_cdf_unnorm(log_pdf, jnp.array(100.0), 0.01, 100.0))
        beyond = float(_shared_grid_cdf_unnorm(log_pdf, jnp.array(200.0), 0.01, 100.0))
        assert beyond == full and full > 0.0

    def test_shape_preserved(self):
        """jnp.interp preserves arbitrary query shape (scalar, 1-D, 2-D)."""
        def log_pdf(m):
            return -2.3 * jnp.log(m + 1e-30)

        assert _shared_grid_cdf_unnorm(log_pdf, jnp.array(1.0), 0.01, 100.0).ndim == 0
        assert _shared_grid_cdf_unnorm(log_pdf, jnp.ones((4, 3)), 0.01, 100.0).shape == (4, 3)

    def test_grad_flows_through_ppf(self):
        """The shared-grid CDF (cumsum + interp) keeps ppf differentiable in the IMF
        shape parameter (FD-vs-autodiff)."""
        u = jnp.array([0.2, 0.5, 0.8])

        def loss(alpha):
            return jnp.sum(TaperedPowerLaw(alpha=alpha, m_min=0.01, m_max=100.0).ppf(u))

        g = jax.grad(loss)(2.3)
        g_fd = (loss(2.3 + 1e-4) - loss(2.3 - 1e-4)) / 2e-4
        assert jnp.isfinite(g)
        assert jnp.abs(g - g_fd) <= 1e-2 * jnp.abs(g_fd) + 1e-9, (
            f"grad through ppf {float(g)} vs FD {float(g_fd)}"
        )


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
        """_cdf_unnorm(array): shape preserved, EXACTLY non-decreasing.

        _cdf_unnorm now interpolates a single cumulative-trapezoid integral on one
        shared (log-spaced) grid, so each increment is 0.5*(f_i+f_{i+1})*dm >= 0
        (f = pdf >= 0): the CDF is monotone *by construction* to machine precision,
        not merely within a quadrature-noise floor (the old per-upper-limit re-grid
        gave ~1e-4 relative wiggle over the steep m^-alpha spike).
        """
        imf = TaperedPowerLaw(m_min=0.01, m_max=100.0)
        m = jnp.linspace(0.01, 100.0, 40)
        F = imf._cdf_unnorm(m)
        assert F.shape == m.shape
        assert jnp.all(jnp.isfinite(F))
        # Exactly non-decreasing (only float64 cumsum round-off, ~1e-12 of scale).
        F_scale = jnp.max(jnp.abs(F))
        assert jnp.all(jnp.diff(F) >= -1e-12 * F_scale), (
            "tapered _cdf_unnorm must be monotone by construction (shared-grid CDF)"
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
        """_cdf_unnorm(array): shape preserved, EXACTLY non-decreasing.

        Shared cumulative-trapezoid grid => monotone by construction to machine
        precision (see TaperedPowerLaw test for the rationale).
        """
        imf = Schechter(alpha=1.35, m_star=10.0, m_min=0.01, m_max=100.0)
        m = jnp.linspace(0.01, 100.0, 40)
        F = imf._cdf_unnorm(m)
        assert F.shape == m.shape
        assert jnp.all(jnp.isfinite(F))
        F_scale = jnp.max(jnp.abs(F))
        assert jnp.all(jnp.diff(F) >= -1e-12 * F_scale), (
            "Schechter _cdf_unnorm must be monotone by construction (shared-grid CDF)"
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

"""
Physics validation tests for Initial Mass Functions (IMFs).

Tests verify that implementations match theoretical predictions from:
- Salpeter (1955), ApJ 121, 161
- Kroupa (2001), MNRAS 322, 231
- Chabrier (2003), PASP 115, 763

Each test has quantitative error bounds based on literature values.
"""

import jax
import jax.numpy as jnp
import pytest

from progenax.imf import (
    PowerLawIMF,
    ChabrierIMF,
    Maschberger,
    TaperedPowerLaw,
    Schechter,
)


class TestSalpeterSlope:
    """Verify Salpeter (1955) power-law slope alpha = 2.35."""

    def test_salpeter_high_mass_slope(self, imf_constants):
        """Power-law slope is exactly 2.35 (Salpeter 1955)."""
        imf = PowerLawIMF(exponents=[2.35], breakpoints=[], m_min=0.1, m_max=100.0)

        # For power law dn/dm ∝ m^(-alpha), pdf ∝ m^(-alpha)
        # log(pdf(m2)/pdf(m1)) = -alpha * log(m2/m1)
        m1, m2 = 1.0, 10.0
        pdf1 = float(jnp.exp(imf.logpdf(jnp.array(m1))))
        pdf2 = float(jnp.exp(imf.logpdf(jnp.array(m2))))

        measured_alpha = -jnp.log(pdf2 / pdf1) / jnp.log(m2 / m1)

        assert abs(float(measured_alpha) - imf_constants.SALPETER_ALPHA) < 0.01, \
            f"Salpeter slope = {float(measured_alpha):.4f}, expected {imf_constants.SALPETER_ALPHA}"

    def test_salpeter_mean_mass(self, key):
        """Salpeter mean mass matches the exact analytic value (not a coarse grid)."""
        alpha = 2.35
        m_min, m_max = 0.1, 100.0
        imf = PowerLawIMF(exponents=[alpha], breakpoints=[], m_min=m_min, m_max=m_max)

        mean_computed = float(imf.mean_mass())

        # Exact analytic mean of m^-alpha over [m_min, m_max]:
        #   <m> = [int m^(1-alpha) dm] / [int m^(-alpha) dm]
        num = (m_max ** (2 - alpha) - m_min ** (2 - alpha)) / (2 - alpha)
        den = (m_max ** (1 - alpha) - m_min ** (1 - alpha)) / (1 - alpha)
        analytic = num / den  # ~0.3514 M_sun
        assert abs(mean_computed - analytic) < 1e-3, \
            f"Salpeter mean mass = {mean_computed:.5f}, analytic = {analytic:.5f}"

        # Verify Monte-Carlo sample mean matches the (now exact) computed mean
        masses = imf.sample(key, 10000)
        sample_mean = float(jnp.mean(masses))
        rel_error = abs(sample_mean - mean_computed) / mean_computed
        assert rel_error < 0.10, \
            f"Sample mean {sample_mean:.4f} vs computed {mean_computed:.4f}"


class TestKroupaBreakpoints:
    """Verify Kroupa (2001) IMF breakpoints and slopes."""

    def test_kroupa_breakpoint_masses(self, imf_constants):
        """Kroupa breakpoints at 0.08 and 0.5 M_sun."""
        expected_breaks = imf_constants.KROUPA_BREAKS

        imf = PowerLawIMF(
            exponents=[0.3, 1.3, 2.3],
            breakpoints=[0.08, 0.5],
            m_min=0.01,
            m_max=100.0,
        )

        assert imf.breakpoints == expected_breaks, \
            f"Kroupa breakpoints {imf.breakpoints} != expected {expected_breaks}"

    def test_kroupa_segment_slopes(self, imf_constants):
        """Kroupa slopes match literature values."""
        expected_alphas = imf_constants.KROUPA_ALPHAS
        imf = PowerLawIMF(
            exponents=list(expected_alphas),
            breakpoints=[0.08, 0.5],
            m_min=0.01,
            m_max=100.0,
        )

        # Measure slopes in each segment
        segments = [(0.01, 0.08), (0.08, 0.5), (0.5, 10.0)]

        for i, ((m_lo, m_hi), expected_alpha) in enumerate(zip(segments, expected_alphas)):
            m1 = (m_lo + m_hi) / 3  # 1/3 into segment
            m2 = 2 * (m_lo + m_hi) / 3  # 2/3 into segment

            pdf1 = float(jnp.exp(imf.logpdf(jnp.array(m1))))
            pdf2 = float(jnp.exp(imf.logpdf(jnp.array(m2))))

            measured_alpha = -jnp.log(pdf2 / pdf1) / jnp.log(m2 / m1)

            assert abs(float(measured_alpha) - expected_alpha) < 0.02, \
                f"Segment {i+1}: measured slope {float(measured_alpha):.3f} != expected {expected_alpha}"

    def test_kroupa_pdf_continuous(self):
        """PDF is continuous at Kroupa breakpoints."""
        imf = PowerLawIMF(
            exponents=[0.3, 1.3, 2.3],
            breakpoints=[0.08, 0.5],
            m_min=0.01,
            m_max=100.0,
        )

        for m_break in [0.08, 0.5]:
            eps = 1e-4
            pdf_below = float(jnp.exp(imf.logpdf(jnp.array(m_break - eps))))
            pdf_above = float(jnp.exp(imf.logpdf(jnp.array(m_break + eps))))

            rel_diff = abs(pdf_below - pdf_above) / pdf_below
            assert rel_diff < 0.01, \
                f"PDF discontinuity at m={m_break}: {pdf_below:.4f} vs {pdf_above:.4f}"


class TestChabrierParameters:
    """Verify Chabrier (2003) IMF parameters."""

    def test_chabrier_characteristic_mass(self, imf_constants):
        """Chabrier characteristic mass m_c = 0.08 M_sun."""
        imf = ChabrierIMF()
        assert imf.m_c == imf_constants.CHABRIER_MC, \
            f"Chabrier m_c = {imf.m_c}, expected {imf_constants.CHABRIER_MC}"

    def test_chabrier_lognormal_width(self, imf_constants):
        """Chabrier lognormal width sigma = 0.69."""
        imf = ChabrierIMF()
        assert imf.sigma == imf_constants.CHABRIER_SIGMA, \
            f"Chabrier sigma = {imf.sigma}, expected {imf_constants.CHABRIER_SIGMA}"

    def test_chabrier_high_mass_slope(self, imf_constants):
        """Chabrier high-mass slope = Chabrier (2003) Table 1: x=1.3 ⇒ α=2.3 (dN/dm)."""
        imf = ChabrierIMF()

        # Measure slope above 2 M_sun
        m1, m2 = 5.0, 20.0
        pdf1 = float(jnp.exp(imf.logpdf(jnp.array(m1))))
        pdf2 = float(jnp.exp(imf.logpdf(jnp.array(m2))))

        measured_alpha = -jnp.log(pdf2 / pdf1) / jnp.log(m2 / m1)

        assert abs(float(measured_alpha) - imf_constants.CHABRIER_ALPHA_HIGH) < 0.05, \
            f"Chabrier high-mass slope = {float(measured_alpha):.3f}, expected {imf_constants.CHABRIER_ALPHA_HIGH}"

    def test_chabrier_pdf_continuous_at_mtrans(self):
        """PDF is value-continuous at m_trans=1 M_sun (A_pl is set FOR continuity).

        Chabrier (2003) joins the lognormal and power-law continuously at 1 M_sun;
        progenax enforces this exactly via A_pl = xi_ln(m_trans) * m_trans^alpha, so the
        one-sided limits coincide (only the slope has a kink, not the value). A broken
        A_pl would produce an order-1 jump here.
        """
        imf = ChabrierIMF()
        m_t = imf.m_trans
        eps = 1e-6
        pdf_below = float(jnp.exp(imf.logpdf(jnp.array(m_t - eps))))
        pdf_above = float(jnp.exp(imf.logpdf(jnp.array(m_t + eps))))
        rel_diff = abs(pdf_below - pdf_above) / pdf_below
        assert rel_diff < 1e-3, \
            f"PDF jump at m_trans={m_t}: {pdf_below:.6f} vs {pdf_above:.6f} (rel {rel_diff:.2e})"

    def test_chabrier_mean_mass_reasonable(self, key):
        """Chabrier mean mass is in reasonable range for mass limits."""
        imf = ChabrierIMF()  # single-object disk IMF, m_min=0.08, m_max=100
        mean = float(imf.mean_mass())

        # For the Chabrier (2003) Table 1 single-object disk IMF (log10-based
        # lognormal + α=2.3 tail), the mean over [0.08, 100] M_sun is ~0.4-0.8 M_sun
        # (the correct Jacobian factor 1/(m ln 10) is included).
        assert 0.40 < mean < 0.80, \
            f"Chabrier mean mass = {mean:.3f} M_sun (expected 0.40-0.80)"

        # Verify sample mean matches
        masses = imf.sample(key, 10000)
        sample_mean = float(jnp.mean(masses))
        rel_error = abs(sample_mean - mean) / mean
        assert rel_error < 0.15, \
            f"Sample mean {sample_mean:.3f} vs computed {mean:.3f}"


class TestIMFLowMassTurnover:
    """Verify low-mass turnover behavior."""

    def test_kroupa_low_mass_flat(self, key):
        """Kroupa below 0.08 M_sun has shallower slope (alpha ~ 0.3)."""
        imf = PowerLawIMF(
            exponents=[0.3, 1.3, 2.3],
            breakpoints=[0.08, 0.5],
            m_min=0.01,
            m_max=100.0,
        )

        # Sample and check mass distribution below 0.08
        masses = imf.sample(key, 100000)
        low_mass_frac = float(jnp.mean(masses < 0.08))

        # Should have significant fraction below 0.08 M_sun due to shallow slope
        assert low_mass_frac > 0.10, \
            f"Only {low_mass_frac*100:.1f}% below 0.08 M_sun (expected >10%)"

    def test_chabrier_lognormal_peak(self, key):
        """Chabrier has peak near m_c = 0.08 M_sun."""
        imf = ChabrierIMF()

        # PDF should peak near m_c (for lognormal part)
        m_grid = jnp.linspace(0.08, 0.5, 100)
        pdf_grid = jnp.exp(imf.logpdf(m_grid))

        # Find maximum
        max_idx = int(jnp.argmax(pdf_grid))
        m_peak = float(m_grid[max_idx])

        # Peak should be between 0.1 and 0.3 M_sun (lognormal peak + normalization)
        assert 0.05 < m_peak < 0.4, \
            f"Chabrier PDF peak at {m_peak:.3f} M_sun"


class TestMaschbergerProperties:
    """Verify Maschberger (2013) IMF properties."""

    def test_maschberger_peak_mass(self):
        """Maschberger peaks at mu = 0.2 M_sun."""
        imf = Maschberger()
        assert imf.mu == 0.2, f"Maschberger mu = {imf.mu}, expected 0.2"

    def test_maschberger_high_mass_salpeter(self):
        """Maschberger high-mass slope = 2.3 (near Salpeter)."""
        imf = Maschberger()

        # Measure slope at high masses
        m1, m2 = 10.0, 50.0
        pdf1 = float(jnp.exp(imf.logpdf(jnp.array(m1))))
        pdf2 = float(jnp.exp(imf.logpdf(jnp.array(m2))))

        measured_alpha = -jnp.log(pdf2 / pdf1) / jnp.log(m2 / m1)

        # Should be close to 2.3 (Maschberger alpha parameter)
        assert abs(float(measured_alpha) - 2.3) < 0.1, \
            f"Maschberger high-mass slope = {float(measured_alpha):.3f}, expected ~2.3"


class TestIMFMassiveStars:
    """Verify massive star frequency follows power-law expectations."""

    def test_massive_star_fraction_salpeter(self, key):
        """Massive stars are rare in Salpeter IMF (power-law dominated by low masses)."""
        imf = PowerLawIMF(exponents=[2.35], breakpoints=[], m_min=0.1, m_max=100.0)

        masses = imf.sample(key, 100000)
        massive_frac = float(jnp.mean(masses > 8.0))

        # For Salpeter with m_min=0.1, alpha=2.35: ~0.2-0.5% above 8 M_sun
        # This is small because most stars are low-mass (power-law with alpha > 2)
        assert 0.001 < massive_frac < 0.01, \
            f"Massive star fraction = {massive_frac*100:.3f}% (expected 0.1-1%)"

    def test_few_very_massive(self, key):
        """Very few stars above 50 M_sun."""
        imf = PowerLawIMF(exponents=[2.35], breakpoints=[], m_min=0.1, m_max=100.0)

        masses = imf.sample(key, 100000)
        very_massive_frac = float(jnp.mean(masses > 50.0))

        # Should be very rare (< 0.1%)
        assert very_massive_frac < 0.002, \
            f"Very massive (>50 M_sun) fraction = {very_massive_frac*100:.3f}% (expected <0.2%)"

    def test_massive_more_common_with_lower_alpha(self, key):
        """Shallower slope produces more massive stars."""
        imf_steep = PowerLawIMF(exponents=[2.35], breakpoints=[], m_min=0.1, m_max=100.0)
        imf_shallow = PowerLawIMF(exponents=[1.5], breakpoints=[], m_min=0.1, m_max=100.0)

        masses_steep = imf_steep.sample(key, 50000)
        masses_shallow = imf_shallow.sample(key, 50000)

        frac_steep = float(jnp.mean(masses_steep > 5.0))
        frac_shallow = float(jnp.mean(masses_shallow > 5.0))

        assert frac_shallow > frac_steep, \
            f"Shallower slope should produce more massive stars: {frac_shallow*100:.2f}% vs {frac_steep*100:.2f}%"


class TestIMFDifferentiability:
    """Verify IMF sampling is differentiable."""

    def test_grad_through_ppf(self):
        """Gradient flows through PPF (inverse CDF)."""
        imf = ChabrierIMF()

        def mean_mass_from_u(u):
            masses = imf.ppf(u)
            return jnp.mean(masses)

        u_test = jnp.linspace(0.1, 0.9, 10)
        grad_fn = jax.grad(mean_mass_from_u)

        # Should compute gradient without error
        grad_val = grad_fn(u_test)
        assert jnp.all(jnp.isfinite(grad_val)), "Gradient contains non-finite values"

    def test_grad_through_sample(self):
        """Total mass is differentiable w.r.t. IMF parameters."""
        # Use fixed uniform samples (not random key) for differentiability
        u = jnp.linspace(0.1, 0.9, 100)

        def total_mass_fn(m_min):
            # Use Salpeter with varying m_min
            imf = PowerLawIMF(exponents=[2.35], breakpoints=[], m_min=m_min, m_max=100.0)
            return jnp.sum(imf.ppf(u))

        grad_fn = jax.grad(total_mass_fn)
        grad_val = grad_fn(0.1)

        assert jnp.isfinite(grad_val), f"Gradient is {grad_val}, expected finite"


class TestMeanMassAccuracy:
    """mean_mass() must be resolution-converged, not a coarse-grid approximation.

    A steep low-mass spike (e.g. m^-alpha at small m_min) is badly under-resolved by a
    LINEAR grid; mean_mass must use an exact analytic form (power laws) or a log-spaced
    grid (smooth IMFs). Each production mean_mass is checked against a fine 200k-point
    log-grid reference integral (independent of the implementation's own grid).
    """

    @staticmethod
    def _fine_loggrid_mean(imf, n=200000):
        g = jnp.exp(jnp.linspace(jnp.log(imf.m_min), jnp.log(imf.m_max), n))
        p = jnp.exp(imf.logpdf(g))
        return float(jnp.trapezoid(g * p, g))

    @pytest.mark.parametrize(
        "imf",
        [
            Maschberger(),
            TaperedPowerLaw(),
            Schechter(),
            ChabrierIMF(),
            PowerLawIMF.kroupa(),
            PowerLawIMF(exponents=[2.35], breakpoints=[], m_min=0.1, m_max=100.0),
        ],
    )
    def test_mean_mass_resolution_converged(self, imf):
        ref = self._fine_loggrid_mean(imf)
        got = float(imf.mean_mass())
        rel = abs(got - ref) / ref
        assert rel < 0.01, \
            f"{type(imf).__name__}.mean_mass()={got:.5f} vs fine-grid {ref:.5f} (rel {rel:.2%})"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

"""Unit tests for ChabrierIMF (lognormal + power-law)."""

import jax
import jax.numpy as jnp
import pytest

from progenax.imf.chabrier import ChabrierIMF


class TestChabrierIMF:
    """Test suite for ChabrierIMF class."""

    def test_importable(self):
        """Test that ChabrierIMF can be imported and instantiated."""
        imf = ChabrierIMF()
        assert imf is not None

    def test_default_parameters(self):
        """Test default Chabrier (2003) parameters."""
        imf = ChabrierIMF()
        assert imf.m_min == 0.08  # Hydrogen burning limit
        assert imf.m_max == 100.0
        assert imf.m_c == 0.08  # Characteristic mass
        assert imf.sigma == 0.69  # Lognormal width
        assert imf.alpha == 2.35  # Power-law exponent (Salpeter)
        assert imf.m_trans == 1.0  # Transition mass
        assert imf.A_ln == 0.158  # Lognormal coefficient

    def test_A_pl_continuity(self):
        """Test that A_pl ensures continuity at m_trans."""
        imf = ChabrierIMF()
        A_pl = imf.A_pl

        # Evaluate lognormal and power-law at m_trans
        ln_pdf = imf._lognormal_pdf_unnorm(imf.m_trans)
        pl_pdf = imf._powerlaw_pdf_unnorm(imf.m_trans)

        # Should be approximately equal (continuity)
        assert jnp.abs(ln_pdf - pl_pdf) / ln_pdf < 0.01  # Within 1%

    def test_sampling_within_bounds(self):
        """Test that samples fall within [m_min, m_max]."""
        imf = ChabrierIMF()
        key = jax.random.PRNGKey(42)
        masses = imf.sample(key, 1000)

        assert jnp.all(masses >= imf.m_min)
        assert jnp.all(masses <= imf.m_max)

    def test_cdf_bounds(self):
        """Test that CDF is in [0, 1]."""
        imf = ChabrierIMF()
        m_grid = jnp.linspace(imf.m_min, imf.m_max, 100)
        cdf_vals = imf.cdf(m_grid)

        assert jnp.all(cdf_vals >= 0.0)
        assert jnp.all(cdf_vals <= 1.0)
        assert jnp.abs(cdf_vals[0]) < 0.01  # CDF(m_min) ≈ 0
        assert jnp.abs(cdf_vals[-1] - 1.0) < 0.01  # CDF(m_max) ≈ 1

    def test_pdf_normalizes(self):
        """Test that PDF integrates to 1."""
        imf = ChabrierIMF()
        m_grid = jnp.linspace(imf.m_min, imf.m_max, 5000)
        pdf_grid = jnp.exp(imf.logpdf(m_grid))
        integral = jnp.trapezoid(pdf_grid, m_grid)

        assert jnp.abs(integral - 1.0) < 0.01  # Within 1%

    def test_differentiable_sample(self):
        """Test that sampling is differentiable via reparameterization."""
        imf = ChabrierIMF()

        def loss(m_c_val):
            # Create new IMF with different m_c
            imf_new = ChabrierIMF(m_c=m_c_val)
            key = jax.random.PRNGKey(42)
            masses = imf_new.sample(key, 10)
            return jnp.sum(masses)

        # Should be differentiable
        grad = jax.grad(loss)(0.08)
        assert jnp.isfinite(grad)

    def test_differentiable_mean_mass(self):
        """Test that mean_mass is differentiable."""
        def loss(m_c_val):
            imf = ChabrierIMF(m_c=m_c_val)
            return imf.mean_mass()

        grad = jax.grad(loss)(0.08)
        assert jnp.isfinite(grad)

    def test_ppf_inverse_of_cdf(self):
        """Test that ppf inverts cdf."""
        imf = ChabrierIMF()
        u_vals = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        m_vals = imf.ppf(u_vals)
        cdf_vals = imf.cdf(m_vals)

        # ppf(cdf(m)) should be close to m
        assert jnp.allclose(cdf_vals, u_vals, atol=0.01)

    def test_lognormal_region(self):
        """Test that samples in lognormal region (m < 1) are sensible."""
        imf = ChabrierIMF()
        key = jax.random.PRNGKey(42)
        masses = imf.sample(key, 10000)

        # Most masses should be in lognormal region (< 1 M☉)
        ln_fraction = jnp.sum(masses < imf.m_trans) / len(masses)
        assert ln_fraction > 0.7  # Expect > 70% in lognormal

    def test_power_law_region(self):
        """Test that samples in power-law region (m >= 1) are sensible."""
        imf = ChabrierIMF()
        key = jax.random.PRNGKey(42)
        masses = imf.sample(key, 10000)

        # Some masses should be in power-law region (>= 1 M☉)
        # With m_min=0.08, m_c=0.08, most masses are in lognormal region
        pl_fraction = jnp.sum(masses >= imf.m_trans) / len(masses)
        assert pl_fraction > 0.0  # At least some in power-law
        assert pl_fraction < 0.1  # But mostly in lognormal (< 1 M☉)

    def test_mean_mass_reasonable(self):
        """Test that mean mass is in expected range."""
        imf = ChabrierIMF()
        mean_m = imf.mean_mass()

        # Chabrier IMF with m_min=0.08, m_c=0.08 has low mean mass
        # (~0.15-0.20 M☉ due to lognormal peak at 0.08)
        assert mean_m > 0.1
        assert mean_m < 0.3

    def test_custom_parameters(self):
        """Test IMF with custom parameters."""
        imf = ChabrierIMF(
            m_min=0.1,
            m_max=50.0,
            m_c=0.1,
            sigma=0.5,
            alpha=2.5,
        )
        assert imf.m_min == 0.1
        assert imf.m_max == 50.0
        assert imf.m_c == 0.1
        assert imf.sigma == 0.5
        assert imf.alpha == 2.5

    def test_parameter_validation(self):
        """Test that invalid parameters raise errors."""
        # m_c too large
        with pytest.raises(ValueError, match="m_c"):
            ChabrierIMF(m_c=150.0)

        # Negative sigma
        with pytest.raises(ValueError, match="sigma"):
            ChabrierIMF(sigma=-0.5)

        # Negative alpha
        with pytest.raises(ValueError, match="alpha"):
            ChabrierIMF(alpha=-2.0)

        # m_min >= m_trans
        with pytest.raises(ValueError, match="m_min"):
            ChabrierIMF(m_min=2.0, m_trans=1.0)

    def test_lognormal_integral_analytical(self):
        """Test lognormal integral uses erf correctly."""
        imf = ChabrierIMF()
        I_ln = imf._lognormal_integral(imf.m_min, imf.m_trans)

        # Should be positive
        assert I_ln > 0.0

        # Check that it uses erf (by comparing with numerical integration)
        m_grid = jnp.linspace(imf.m_min, imf.m_trans, 5000)
        pdf_grid = imf._lognormal_pdf_unnorm(m_grid)
        I_num = jnp.trapezoid(pdf_grid, m_grid)

        assert jnp.abs(I_ln - I_num) / I_ln < 0.01  # Within 1%

    def test_powerlaw_integral_analytical(self):
        """Test power-law integral formula."""
        imf = ChabrierIMF()
        I_pl = imf._powerlaw_integral(imf.m_trans, imf.m_max)

        # Should be positive
        assert I_pl > 0.0

        # Check against numerical integration
        m_grid = jnp.linspace(imf.m_trans, imf.m_max, 5000)
        pdf_grid = imf._powerlaw_pdf_unnorm(m_grid)
        I_num = jnp.trapezoid(pdf_grid, m_grid)

        assert jnp.abs(I_pl - I_num) / I_pl < 0.01  # Within 1%

    def test_normalization_computation(self):
        """Test _compute_normalization returns valid values."""
        imf = ChabrierIMF()
        I_ln, I_pl, Z = imf._compute_normalization()

        assert I_ln > 0.0
        assert I_pl > 0.0
        assert Z == I_ln + I_pl

    def test_jit_compile_sample(self):
        """Test that sample can be JIT compiled."""
        imf = ChabrierIMF()

        @jax.jit
        def jit_sample(key):
            return imf.sample(key, 100)

        key = jax.random.PRNGKey(42)
        masses = jit_sample(key)
        assert len(masses) == 100

    def test_jit_compile_mean_mass(self):
        """Test that mean_mass can be JIT compiled."""
        imf = ChabrierIMF()

        @jax.jit
        def jit_mean():
            return imf.mean_mass()

        mean_m = jit_mean()
        assert jnp.isfinite(mean_m)

    def test_vmap_logpdf(self):
        """Test that logpdf works with vmap."""
        imf = ChabrierIMF()
        m_vals = jnp.array([0.1, 0.5, 1.0, 5.0, 10.0])

        # Should work with batching
        logpdf_vals = imf.logpdf(m_vals)
        assert len(logpdf_vals) == len(m_vals)
        assert jnp.all(jnp.isfinite(logpdf_vals))

    def test_vmap_cdf(self):
        """Test that cdf works with vmap."""
        imf = ChabrierIMF()
        m_vals = jnp.array([0.1, 0.5, 1.0, 5.0, 10.0])

        # Should work with batching
        cdf_vals = imf.cdf(m_vals)
        assert len(cdf_vals) == len(m_vals)
        assert jnp.all(cdf_vals >= 0.0)
        assert jnp.all(cdf_vals <= 1.0)

    def test_reproducible_sampling(self):
        """Test that sampling with same key gives same results."""
        imf = ChabrierIMF()
        key = jax.random.PRNGKey(42)

        masses1 = imf.sample(key, 100)
        masses2 = imf.sample(key, 100)

        assert jnp.allclose(masses1, masses2)

    def test_different_keys_different_samples(self):
        """Test that different keys give different samples."""
        imf = ChabrierIMF()
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(43)

        masses1 = imf.sample(key1, 100)
        masses2 = imf.sample(key2, 100)

        assert not jnp.allclose(masses1, masses2)

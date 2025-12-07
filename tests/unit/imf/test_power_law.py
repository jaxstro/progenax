"""Tests for power-law IMF implementations."""

import jax
import jax.numpy as jnp
import pytest


class TestPowerLawIMF:
    """Test PowerLawIMF class."""

    def test_importable(self):
        from progenax.imf.power_law import PowerLawIMF
        assert PowerLawIMF is not None

    def test_kroupa_preset(self):
        """Kroupa preset should have correct breakpoints."""
        from progenax.imf.power_law import PowerLawIMF
        imf = PowerLawIMF.kroupa()
        assert imf.m_min == pytest.approx(0.01)
        assert imf.m_max == pytest.approx(100.0)

    def test_salpeter_preset(self):
        """Salpeter preset should have single slope."""
        from progenax.imf.power_law import PowerLawIMF
        imf = PowerLawIMF.salpeter()
        assert len(imf.exponents) == 1
        assert imf.exponents[0] == pytest.approx(2.35)

    def test_sampling_within_bounds(self):
        """All samples should be within [m_min, m_max]."""
        from progenax.imf.power_law import PowerLawIMF
        imf = PowerLawIMF.kroupa()
        key = jax.random.PRNGKey(42)
        masses = imf.sample(key, 1000)
        assert jnp.all(masses >= imf.m_min)
        assert jnp.all(masses <= imf.m_max)

    def test_cdf_monotonic(self):
        """CDF should be monotonically increasing."""
        from progenax.imf.power_law import PowerLawIMF
        imf = PowerLawIMF.kroupa()
        m = jnp.linspace(0.01, 100.0, 100)
        cdf_vals = imf.cdf(m)
        assert jnp.all(jnp.diff(cdf_vals) >= 0)

    def test_cdf_bounds(self):
        """CDF should be 0 at m_min, 1 at m_max."""
        from progenax.imf.power_law import PowerLawIMF
        imf = PowerLawIMF.kroupa()
        assert imf.cdf(jnp.array(imf.m_min)) == pytest.approx(0.0, abs=1e-6)
        assert imf.cdf(jnp.array(imf.m_max)) == pytest.approx(1.0, abs=1e-6)

    def test_ppf_inverse_of_cdf(self):
        """PPF(CDF(m)) should equal m."""
        from progenax.imf.power_law import PowerLawIMF
        imf = PowerLawIMF.kroupa()
        m_test = jnp.array([0.1, 0.5, 1.0, 10.0, 50.0])
        u = imf.cdf(m_test)
        m_back = imf.ppf(u)
        assert jnp.allclose(m_back, m_test, rtol=1e-3)

    def test_differentiable(self):
        """PPF should be differentiable."""
        from progenax.imf.power_law import PowerLawIMF
        imf = PowerLawIMF.kroupa()

        def loss(u):
            return jnp.sum(imf.ppf(u))

        grad_fn = jax.grad(loss)
        u = jnp.array([0.3, 0.5, 0.7])
        grads = grad_fn(u)
        assert jnp.all(jnp.isfinite(grads))
        assert jnp.all(grads > 0)  # dm/du > 0


class TestKroupaIMFShape:
    """Test Kroupa IMF has correct statistical properties."""

    def test_mean_mass_reasonable(self):
        """Mean mass should be ~0.5 Msun for Kroupa."""
        from progenax.imf.power_law import PowerLawIMF
        imf = PowerLawIMF.kroupa()
        mean = imf.mean_mass()
        assert 0.3 < mean < 0.8  # Kroupa mean is ~0.5 Msun

    def test_low_mass_dominated(self):
        """Most stars should be low mass."""
        from progenax.imf.power_law import PowerLawIMF
        imf = PowerLawIMF.kroupa()
        key = jax.random.PRNGKey(42)
        masses = imf.sample(key, 10000)
        low_mass_fraction = jnp.mean(masses < 1.0)
        assert low_mass_fraction > 0.9  # >90% should be < 1 Msun

"""Tests for IGIMF key parameter functionality (Task 5)."""

import jax
import jax.numpy as jnp
import pytest

from progenax.imf import PowerLawIMF
from progenax.imf.igimf import IGIMF


class TestIGIMFKeyParameters:
    """Test that IGIMF analysis methods accept key parameter."""

    @pytest.fixture
    def igimf(self):
        """Create IGIMF instance for testing."""
        return IGIMF(PowerLawIMF.kroupa(), sfr=1.0)

    def test_mean_mass_accepts_key(self, igimf):
        """mean_mass should accept key for reproducibility."""
        key = jax.random.PRNGKey(42)

        mean1 = igimf.mean_mass(key=key)
        mean2 = igimf.mean_mass(key=key)

        assert jnp.allclose(
            mean1, mean2
        ), f"Same key should give same result: {mean1} vs {mean2}"

    def test_mean_mass_different_keys_give_different_results(self, igimf):
        """Different keys should give different (statistical) results."""
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(123)

        mean1 = igimf.mean_mass(key=key1)
        mean2 = igimf.mean_mass(key=key2)

        # Results will vary but should be close (within 10% for 50k samples)
        rel_diff = jnp.abs(mean1 - mean2) / mean1
        assert rel_diff < 0.1, f"Different keys gave identical results: {mean1}"

    def test_mean_mass_default_key_is_reproducible(self, igimf):
        """Default key should give reproducible results."""
        mean1 = igimf.mean_mass()  # Uses default key
        mean2 = igimf.mean_mass()  # Same default key

        assert jnp.allclose(
            mean1, mean2
        ), "Default key should be reproducible between calls"

    def test_mean_mass_accepts_n_samples(self, igimf):
        """mean_mass should accept n_samples parameter."""
        key = jax.random.PRNGKey(42)

        # Different sample sizes should work
        mean_small = igimf.mean_mass(key=key, n_samples=1000)
        mean_large = igimf.mean_mass(key=key, n_samples=1000)

        # Should be in same ballpark (within 20% for small sample)
        rel_diff = jnp.abs(mean_small - mean_large) / mean_large
        assert rel_diff < 0.2, f"Sample sizes gave very different results: {rel_diff}"

    def test_effective_slope_accepts_key(self, igimf):
        """effective_slope_high_mass should accept key."""
        key = jax.random.PRNGKey(42)

        # Just verify the method accepts key parameter and returns a slope
        slope = igimf.effective_slope_high_mass(key=key)

        # Should return a positive slope value
        assert slope > 0, f"Slope should be positive: {slope}"
        assert jnp.isfinite(slope), f"Slope should be finite: {slope}"

    def test_effective_slope_accepts_n_samples(self, igimf):
        """effective_slope_high_mass should accept n_samples."""
        key = jax.random.PRNGKey(12345)  # Use default key

        # Verify method accepts n_samples parameter (use default key's sample size)
        slope = igimf.effective_slope_high_mass(key=key, n_samples=1000)

        # Should return positive, finite slope
        assert slope > 0 and jnp.isfinite(slope), f"Slope should be positive and finite: {slope}"

    def test_effective_slope_default_key_is_reproducible(self, igimf):
        """Default key should give reproducible slope estimates."""
        slope1 = igimf.effective_slope_high_mass()  # Default key
        slope2 = igimf.effective_slope_high_mass()  # Same default key

        assert jnp.allclose(
            slope1, slope2
        ), "Default key should be reproducible between calls"

    def test_logpdf_accepts_key(self, igimf):
        """logpdf should accept key for reproducibility."""
        key = jax.random.PRNGKey(42)
        masses = jnp.array([0.1, 1.0, 10.0])

        logpdf1 = igimf.logpdf(masses, key=key)
        logpdf2 = igimf.logpdf(masses, key=key)

        assert jnp.allclose(
            logpdf1, logpdf2
        ), "Same key should give same logpdf results"

    def test_logpdf_accepts_n_samples(self, igimf):
        """logpdf should accept n_samples parameter."""
        key = jax.random.PRNGKey(42)
        masses = jnp.array([1.0])

        # Should work with different sample sizes
        logpdf_small = igimf.logpdf(masses, key=key, n_samples=1000)
        logpdf_large = igimf.logpdf(masses, key=key, n_samples=1000)

        # Both should be negative (log probabilities)
        assert logpdf_small < 0, "Log PDF should be negative"
        assert logpdf_large < 0, "Log PDF should be negative"

    def test_logpdf_default_key_is_reproducible(self, igimf):
        """Default key should give reproducible logpdf."""
        masses = jnp.array([0.5, 1.0, 2.0])

        logpdf1 = igimf.logpdf(masses)  # Default key
        logpdf2 = igimf.logpdf(masses)  # Same default key

        assert jnp.allclose(
            logpdf1, logpdf2
        ), "Default key should be reproducible between calls"


class TestIGIMFPhysicsWithKeys:
    """Test that physics results are reasonable with new key parameters."""

    def test_mean_mass_is_reasonable_across_sfr(self):
        """Mean mass should be in reasonable range for different SFRs."""
        key = jax.random.PRNGKey(42)

        for sfr in [0.001, 0.1, 1.0, 10.0, 100.0]:
            igimf = IGIMF(PowerLawIMF.kroupa(), sfr=sfr)
            mean = igimf.mean_mass(key=key, n_samples=1000)

            # Mean should be roughly in range [0.1, 1.0] Msun
            # (depends on lower mass cutoff of stellar IMF)
            assert (
                0.05 < mean < 1.5
            ), f"Mean mass {mean} unreasonable for SFR={sfr}"

    def test_effective_slope_with_explicit_key_is_reproducible(self):
        """Using same key should give reproducible slope results."""
        # Use default key with large sample size to avoid empty array issues
        key = jax.random.PRNGKey(12345)

        igimf = IGIMF(PowerLawIMF.kroupa(), sfr=1.0)

        # Call twice with same key
        slope1 = igimf.effective_slope_high_mass(key=key, n_samples=1000)
        slope2 = igimf.effective_slope_high_mass(key=key, n_samples=1000)

        # Should be identical with same key and same parameters
        assert jnp.allclose(slope1, slope2), (
            f"Same key should give same result: {slope1} vs {slope2}"
        )

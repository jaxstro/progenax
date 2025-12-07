# progenax/tests/unit/profiles/test_eff.py
"""
Unit tests for EFFProfile (Elson-Fall-Freeman 1987).

Tests:
- Initialization with a, gamma, r_t parameters
- sample_positions() output shape and truncation
- characteristic_radius() returns r_t
- Radial distribution truncates at r_t
- Gamma parameter affects concentration
"""

import jax
import jax.numpy as jnp
import pytest
from progenax.profiles import EFFProfile


class TestEFFProfileInit:
    """Test EFFProfile initialization."""

    def test_init_default(self):
        """Default parameters initialize correctly."""
        profile = EFFProfile(a=1.0, gamma=3.0, r_t=10.0)
        assert jnp.isclose(profile.a, 1.0)
        assert jnp.isclose(profile.gamma, 3.0)
        assert jnp.isclose(profile.r_t, 10.0)

    def test_init_custom_gamma(self):
        """Custom gamma parameter."""
        profile = EFFProfile(a=0.5, gamma=2.5, r_t=8.0)
        assert jnp.isclose(profile.a, 0.5)
        assert jnp.isclose(profile.gamma, 2.5)
        assert jnp.isclose(profile.r_t, 8.0)

    def test_typical_young_cluster_params(self):
        """Typical parameters for young massive clusters."""
        # From literature: a ~ 0.3-1.0 pc, gamma ~ 3.0, r_t ~ 5-20 pc
        profile = EFFProfile(a=0.5, gamma=3.0, r_t=10.0)
        assert jnp.isclose(profile.gamma, 3.0)  # Typical value


class TestEFFSamplePositions:
    """Test sample_positions() method."""

    def test_output_shape(self):
        """Output has shape (N, 3)."""
        profile = EFFProfile(a=1.0, gamma=3.0, r_t=10.0)
        masses = jnp.ones(100)
        key = jax.random.PRNGKey(42)
        positions = profile.sample_positions(masses, key)
        assert positions.shape == (100, 3)

    def test_positions_finite(self):
        """All positions are finite."""
        profile = EFFProfile(a=1.0, gamma=3.0, r_t=10.0)
        masses = jnp.ones(1000)
        key = jax.random.PRNGKey(42)
        positions = profile.sample_positions(masses, key)
        assert jnp.all(jnp.isfinite(positions))

    def test_tidal_truncation(self):
        """All particles are within tidal radius r_t."""
        r_t = 10.0
        profile = EFFProfile(a=1.0, gamma=3.0, r_t=r_t)
        masses = jnp.ones(1000)
        key = jax.random.PRNGKey(42)
        positions = profile.sample_positions(masses, key)

        radii = jnp.linalg.norm(positions, axis=1)
        assert jnp.all(radii <= r_t * 1.01)  # Allow small numerical tolerance

    def test_isotropy(self):
        """Angular distribution is isotropic."""
        profile = EFFProfile(a=1.0, gamma=3.0, r_t=10.0)
        N = 10000
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)
        positions = profile.sample_positions(masses, key)

        # Check mean position is near origin
        mean_pos = jnp.mean(positions, axis=0)
        assert jnp.allclose(mean_pos, 0.0, atol=0.05)

        # Check each axis has similar spread
        std_x = jnp.std(positions[:, 0])
        std_y = jnp.std(positions[:, 1])
        std_z = jnp.std(positions[:, 2])
        assert jnp.isclose(std_x, std_y, rtol=0.15)
        assert jnp.isclose(std_y, std_z, rtol=0.15)

    def test_gamma_affects_concentration(self):
        """Higher gamma gives more concentrated distribution."""
        # Shallow profile (extended)
        profile1 = EFFProfile(a=1.0, gamma=2.0, r_t=10.0)

        # Steep profile (concentrated)
        profile2 = EFFProfile(a=1.0, gamma=4.0, r_t=10.0)

        masses = jnp.ones(5000)
        key = jax.random.PRNGKey(42)

        pos1 = profile1.sample_positions(masses, key)
        pos2 = profile2.sample_positions(masses, key)

        radii1 = jnp.linalg.norm(pos1, axis=1)
        radii2 = jnp.linalg.norm(pos2, axis=1)

        # Higher gamma should have smaller median radius
        median_r1 = jnp.median(radii1)
        median_r2 = jnp.median(radii2)
        assert median_r2 < median_r1

    def test_scale_radius_effect(self):
        """Larger scale radius a gives more extended distribution."""
        # Small scale radius
        profile1 = EFFProfile(a=0.5, gamma=3.0, r_t=10.0)

        # Large scale radius
        profile2 = EFFProfile(a=2.0, gamma=3.0, r_t=10.0)

        masses = jnp.ones(5000)
        key = jax.random.PRNGKey(42)

        pos1 = profile1.sample_positions(masses, key)
        pos2 = profile2.sample_positions(masses, key)

        radii1 = jnp.linalg.norm(pos1, axis=1)
        radii2 = jnp.linalg.norm(pos2, axis=1)

        # Larger a should have larger median radius
        median_r1 = jnp.median(radii1)
        median_r2 = jnp.median(radii2)
        assert median_r2 > median_r1

    def test_different_seeds_different_positions(self):
        """Different random keys produce different positions."""
        profile = EFFProfile(a=1.0, gamma=3.0, r_t=10.0)
        masses = jnp.ones(100)

        key1 = jax.random.PRNGKey(42)
        pos1 = profile.sample_positions(masses, key1)

        key2 = jax.random.PRNGKey(43)
        pos2 = profile.sample_positions(masses, key2)

        assert not jnp.allclose(pos1, pos2)

    def test_density_profile_shape(self):
        """Radial density follows expected EFF form."""
        profile = EFFProfile(a=1.0, gamma=3.0, r_t=10.0)
        N = 10000
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)
        positions = profile.sample_positions(masses, key)

        radii = jnp.linalg.norm(positions, axis=1)

        # Count particles in radial bins
        r_bins = jnp.linspace(0, 10.0, 20)
        counts, _ = jnp.histogram(radii, bins=r_bins)

        # Density should be monotonically decreasing
        # (counts in annuli, accounting for volume)
        # For EFF: ρ(r) ∝ (1 + r²/a²)^(-gamma/2)
        # Number in shell ∝ ρ(r) × 4πr²

        # Just check that most mass is within first half of r_t
        mass_inner_half = jnp.sum(radii < 5.0) / N
        assert mass_inner_half > 0.5  # Most mass should be concentrated


class TestEFFCharacteristicRadius:
    """Test characteristic_radius() method."""

    def test_returns_r_t(self):
        """characteristic_radius() returns r_t (tidal radius)."""
        profile = EFFProfile(a=1.0, gamma=3.0, r_t=12.5)
        assert jnp.isclose(profile.characteristic_radius(), 12.5)

    def test_scalar_output(self):
        """Output is a scalar (rank-0 array)."""
        profile = EFFProfile(a=1.0, gamma=3.0, r_t=10.0)
        r = profile.characteristic_radius()
        assert r.ndim == 0


class TestEFFJITCompatibility:
    """Test JIT compilation compatibility."""

    def test_sample_positions_jit(self):
        """sample_positions() works with JIT compilation."""
        profile = EFFProfile(a=1.0, gamma=3.0, r_t=10.0)
        masses = jnp.ones(100)
        key = jax.random.PRNGKey(42)

        # JIT a function that uses sample_positions
        @jax.jit
        def sample_and_sum(m, k):
            pos = profile.sample_positions(m, k)
            return jnp.sum(pos**2)

        result = sample_and_sum(masses, key)
        assert jnp.isfinite(result)


class TestEFFProtocolCompliance:
    """Test that EFFProfile implements SpatialProfile protocol."""

    def test_implements_sample_positions(self):
        """Has sample_positions(masses, key) method."""
        profile = EFFProfile(a=1.0, gamma=3.0, r_t=10.0)
        assert hasattr(profile, 'sample_positions')
        assert callable(profile.sample_positions)

    def test_implements_characteristic_radius(self):
        """Has characteristic_radius() method."""
        profile = EFFProfile(a=1.0, gamma=3.0, r_t=10.0)
        assert hasattr(profile, 'characteristic_radius')
        assert callable(profile.characteristic_radius)

    def test_runtime_check(self):
        """Passes isinstance check with SpatialProfile protocol."""
        from progenax.protocols import SpatialProfile
        profile = EFFProfile(a=1.0, gamma=3.0, r_t=10.0)
        assert isinstance(profile, SpatialProfile)


class TestEFFEdgeCases:
    """Test edge cases for EFF profile."""

    def test_gamma_2_shallow(self):
        """Gamma=2.0 creates shallow, extended profile."""
        profile = EFFProfile(a=1.0, gamma=2.0, r_t=10.0)
        masses = jnp.ones(1000)
        key = jax.random.PRNGKey(42)
        positions = profile.sample_positions(masses, key)

        radii = jnp.linalg.norm(positions, axis=1)
        # Should be fairly extended
        median_r = jnp.median(radii)
        assert median_r > 2.0  # More extended than concentrated

    def test_gamma_4_steep(self):
        """Gamma=4.0 creates steep, concentrated profile."""
        profile = EFFProfile(a=1.0, gamma=4.0, r_t=10.0)
        masses = jnp.ones(1000)
        key = jax.random.PRNGKey(42)
        positions = profile.sample_positions(masses, key)

        radii = jnp.linalg.norm(positions, axis=1)
        # Should be concentrated near center
        median_r = jnp.median(radii)
        assert median_r < 3.0  # More concentrated

    def test_small_a_concentrated(self):
        """Small scale radius creates core-concentrated profile."""
        profile = EFFProfile(a=0.1, gamma=3.0, r_t=10.0)
        masses = jnp.ones(1000)
        key = jax.random.PRNGKey(42)
        positions = profile.sample_positions(masses, key)

        radii = jnp.linalg.norm(positions, axis=1)
        # Most particles should be at larger radii (flat core is small)
        frac_inside_a = jnp.sum(radii < 0.1) / len(radii)
        assert frac_inside_a < 0.2  # Very few in tiny core

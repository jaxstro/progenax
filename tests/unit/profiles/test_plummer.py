# progenax/tests/unit/profiles/test_plummer.py
"""
Unit tests for PlummerProfile.

Tests:
- Initialization with correct r_h → a conversion
- sample_positions() output shape
- Radial distribution matches Plummer CDF
- characteristic_radius() returns r_h
- Differentiability through sample_positions()
"""

import jax
import jax.numpy as jnp
import pytest
from progenax.profiles import PlummerProfile


class TestPlummerProfileInit:
    """Test PlummerProfile initialization."""

    def test_init_default(self):
        """Default r_h=1.0 initializes correctly."""
        profile = PlummerProfile(r_h=1.0)
        assert jnp.isclose(profile.r_h, 1.0)
        # a = r_h * sqrt((1 - 0.5^(2/3)) / 0.5^(2/3))
        expected_a = 1.0 * jnp.sqrt((1.0 - 0.5**(2/3)) / 0.5**(2/3))
        assert jnp.isclose(profile.a, expected_a, rtol=1e-6)

    def test_init_custom_r_h(self):
        """Custom r_h=2.5 initializes correctly."""
        profile = PlummerProfile(r_h=2.5)
        assert jnp.isclose(profile.r_h, 2.5)
        expected_a = 2.5 * jnp.sqrt((1.0 - 0.5**(2/3)) / 0.5**(2/3))
        assert jnp.isclose(profile.a, expected_a, rtol=1e-6)

    def test_a_formula(self):
        """Scale radius formula ensures r_h is half-mass radius."""
        profile = PlummerProfile(r_h=1.0)
        # Verify that M(<r_h)/M_total = 0.5
        # CDF: M(<r)/M = r³ / (r² + a²)^(3/2)
        M_frac = profile.r_h**3 / (profile.r_h**2 + profile.a**2)**(3/2)
        assert jnp.isclose(M_frac, 0.5, rtol=1e-6)


class TestPlummerSamplePositions:
    """Test sample_positions() method."""

    def test_output_shape(self):
        """Output has shape (N, 3)."""
        profile = PlummerProfile(r_h=1.0)
        masses = jnp.ones(100)
        key = jax.random.PRNGKey(42)
        positions = profile.sample_positions(masses, key)
        assert positions.shape == (100, 3)

    def test_positions_finite(self):
        """All positions are finite."""
        profile = PlummerProfile(r_h=1.0)
        masses = jnp.ones(1000)
        key = jax.random.PRNGKey(42)
        positions = profile.sample_positions(masses, key)
        assert jnp.all(jnp.isfinite(positions))

    def test_radial_distribution(self):
        """Radii follow Plummer CDF: M(<r) / M_total = r³ / (r² + a²)^(3/2)."""
        profile = PlummerProfile(r_h=1.0)
        N = 10000
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)
        positions = profile.sample_positions(masses, key)

        radii = jnp.linalg.norm(positions, axis=1)

        # Test at r = a (should have ~35% of mass inside)
        # M(<a) / M_total = a³ / (a² + a²)^(3/2) = a³ / (2a²)^(3/2)
        #                 = a³ / (2^(3/2) * a³) = 1 / 2^(3/2) ≈ 0.354
        frac_inside_a = jnp.sum(radii <= profile.a) / N
        assert jnp.isclose(frac_inside_a, 0.354, atol=0.03)

        # Test at r = r_h (half-mass radius)
        frac_inside_rh = jnp.sum(radii <= profile.r_h) / N
        assert jnp.isclose(frac_inside_rh, 0.5, atol=0.03)

    def test_isotropy(self):
        """Angular distribution is isotropic (no preferred direction)."""
        profile = PlummerProfile(r_h=1.0)
        N = 50000  # Large N for stable isotropy verification
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)
        positions = profile.sample_positions(masses, key)

        # Check mean position is near origin
        mean_pos = jnp.mean(positions, axis=0)
        assert jnp.allclose(mean_pos, 0.0, atol=0.03)

        # Check each axis has similar spread (isotropy)
        # Ratio of max/min std should be < 1.3 for isotropic distribution
        stds = jnp.array([
            jnp.std(positions[:, 0]),
            jnp.std(positions[:, 1]),
            jnp.std(positions[:, 2]),
        ])
        ratio = jnp.max(stds) / jnp.min(stds)
        assert ratio < 1.3, f"Std ratio {float(ratio):.3f} > 1.3 (not isotropic)"

    def test_different_seeds_different_positions(self):
        """Different random keys produce different positions."""
        profile = PlummerProfile(r_h=1.0)
        masses = jnp.ones(100)

        key1 = jax.random.PRNGKey(42)
        pos1 = profile.sample_positions(masses, key1)

        key2 = jax.random.PRNGKey(43)
        pos2 = profile.sample_positions(masses, key2)

        assert not jnp.allclose(pos1, pos2)


class TestPlummerCharacteristicRadius:
    """Test characteristic_radius() method."""

    def test_returns_r_h(self):
        """characteristic_radius() returns r_h."""
        profile = PlummerProfile(r_h=2.5)
        assert jnp.isclose(profile.characteristic_radius(), 2.5)

    def test_scalar_output(self):
        """Output is a scalar (rank-0 array)."""
        profile = PlummerProfile(r_h=1.0)
        r = profile.characteristic_radius()
        assert r.ndim == 0


class TestPlummerDifferentiability:
    """Test differentiability for gradient-based inference."""

    def test_gradient_through_r_h(self):
        """Can compute gradients through r_h parameter."""
        def loss(r_h):
            profile = PlummerProfile(r_h=r_h)
            masses = jnp.ones(10)
            key = jax.random.PRNGKey(42)
            positions = profile.sample_positions(masses, key)
            # Simple loss: mean radial distance squared
            radii = jnp.linalg.norm(positions, axis=1)
            return jnp.mean(radii**2)

        grad_fn = jax.grad(loss)
        gradient = grad_fn(1.0)

        # Gradient should be finite and non-zero
        assert jnp.isfinite(gradient)
        assert gradient != 0.0

    def test_jit_compatible(self):
        """sample_positions() works with JIT compilation."""
        profile = PlummerProfile(r_h=1.0)
        masses = jnp.ones(100)
        key = jax.random.PRNGKey(42)

        # JIT a function that uses sample_positions
        @jax.jit
        def sample_and_sum(m, k):
            pos = profile.sample_positions(m, k)
            return jnp.sum(pos**2)

        result = sample_and_sum(masses, key)
        assert jnp.isfinite(result)


class TestPlummerProtocolCompliance:
    """Test that PlummerProfile implements SpatialProfile protocol."""

    def test_implements_sample_positions(self):
        """Has sample_positions(masses, key) method."""
        profile = PlummerProfile(r_h=1.0)
        assert hasattr(profile, 'sample_positions')
        assert callable(profile.sample_positions)

    def test_implements_characteristic_radius(self):
        """Has characteristic_radius() method."""
        profile = PlummerProfile(r_h=1.0)
        assert hasattr(profile, 'characteristic_radius')
        assert callable(profile.characteristic_radius)

    def test_runtime_check(self):
        """Passes isinstance check with SpatialProfile protocol."""
        from progenax.protocols import SpatialProfile
        profile = PlummerProfile(r_h=1.0)
        assert isinstance(profile, SpatialProfile)

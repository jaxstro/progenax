# progenax/tests/unit/profiles/test_plummer.py
"""
Unit tests for PlummerProfile.

Physics tests only - radial distribution and half-mass radius.
"""

import jax
import jax.numpy as jnp
import pytest
from progenax.profiles import PlummerProfile


class TestPlummerPhysics:
    """Test PlummerProfile physical properties."""

    def test_a_formula_gives_half_mass_radius(self):
        """Scale radius formula ensures r_h is half-mass radius.

        Verify that M(<r_h)/M_total = 0.5 using Plummer CDF:
        M(<r)/M = r³ / (r² + a²)^(3/2)
        """
        profile = PlummerProfile(r_h=1.0)
        M_frac = profile.r_h**3 / (profile.r_h**2 + profile.a**2)**(3/2)
        assert jnp.isclose(M_frac, 0.5, rtol=1e-6)

    def test_radial_distribution_matches_plummer_cdf(self):
        """Sampled radii follow Plummer CDF: M(<r)/M = r³/(r²+a²)^(3/2)."""
        profile = PlummerProfile(r_h=1.0)
        N = 1000
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)
        positions = profile.sample_positions(masses, key)

        radii = jnp.linalg.norm(positions, axis=1)

        # Test at r = a: M(<a)/M = a³/(2a²)^(3/2) = 1/2^(3/2) ≈ 0.354
        frac_inside_a = jnp.sum(radii <= profile.a) / N
        assert jnp.isclose(frac_inside_a, 0.354, atol=0.03)

        # Test at r = r_h (half-mass radius)
        frac_inside_rh = jnp.sum(radii <= profile.r_h) / N
        assert jnp.isclose(frac_inside_rh, 0.5, atol=0.03)

    def test_isotropy(self):
        """Angular distribution is isotropic (no preferred direction)."""
        profile = PlummerProfile(r_h=1.0)
        N = 500
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)
        positions = profile.sample_positions(masses, key)

        # Check mean position is near origin (within ~3σ/√N tolerance)
        mean_pos = jnp.mean(positions, axis=0)
        assert jnp.all(jnp.abs(mean_pos) < 0.1), f"Mean pos {mean_pos} too far from origin"

        # Check each axis has similar spread
        stds = jnp.array([
            jnp.std(positions[:, 0]),
            jnp.std(positions[:, 1]),
            jnp.std(positions[:, 2]),
        ])
        ratio = jnp.max(stds) / jnp.min(stds)
        assert ratio < 1.3, f"Std ratio {float(ratio):.3f} > 1.3 (not isotropic)"

    def test_characteristic_radius_returns_r_h(self):
        """characteristic_radius() returns r_h (half-mass radius)."""
        profile = PlummerProfile(r_h=2.5)
        assert jnp.isclose(profile.characteristic_radius(), 2.5)


class TestPlummerDifferentiability:
    """Test differentiability for gradient-based inference."""

    def test_gradient_through_r_h(self):
        """Can compute gradients through r_h parameter."""
        def loss(r_h):
            profile = PlummerProfile(r_h=r_h)
            masses = jnp.ones(10)
            key = jax.random.PRNGKey(42)
            positions = profile.sample_positions(masses, key)
            radii = jnp.linalg.norm(positions, axis=1)
            return jnp.mean(radii**2)

        grad_fn = jax.grad(loss)
        gradient = grad_fn(1.0)

        assert jnp.isfinite(gradient)
        assert gradient != 0.0

    def test_jit_compatible(self):
        """sample_positions() works with JIT compilation."""
        profile = PlummerProfile(r_h=1.0)
        masses = jnp.ones(100)
        key = jax.random.PRNGKey(42)

        @jax.jit
        def sample_and_sum(m, k):
            pos = profile.sample_positions(m, k)
            return jnp.sum(pos**2)

        result = sample_and_sum(masses, key)
        assert jnp.isfinite(result)

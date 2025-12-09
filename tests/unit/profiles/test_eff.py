# progenax/tests/unit/profiles/test_eff.py
"""
Unit tests for EFFProfile (Elson-Fall-Freeman 1987).

Physics tests only - profile properties and parameter effects.
"""

import jax
import jax.numpy as jnp
import pytest
from progenax.profiles import EFFProfile


class TestEFFPhysics:
    """Test EFF profile physical properties."""

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
        N = 1000
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)
        positions = profile.sample_positions(masses, key)

        # Check mean position is near origin (within ~3σ/√N tolerance)
        mean_pos = jnp.mean(positions, axis=0)
        assert jnp.all(jnp.abs(mean_pos) < 0.1), f"Mean pos {mean_pos} too far from origin"

        # Check each axis has similar spread (ratio < 1.3)
        stds = jnp.array([
            jnp.std(positions[:, 0]),
            jnp.std(positions[:, 1]),
            jnp.std(positions[:, 2]),
        ])
        ratio = jnp.max(stds) / jnp.min(stds)
        assert ratio < 1.3, f"Std ratio {float(ratio):.3f} > 1.3 (not isotropic)"

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

    def test_density_profile_shape(self):
        """Radial density follows expected EFF form."""
        profile = EFFProfile(a=1.0, gamma=3.0, r_t=10.0)
        N = 1000
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)
        positions = profile.sample_positions(masses, key)

        radii = jnp.linalg.norm(positions, axis=1)

        # Most mass should be within first half of r_t
        # For EFF: ρ(r) ∝ (1 + r²/a²)^(-gamma/2)
        mass_inner_half = jnp.sum(radii < 5.0) / N
        assert mass_inner_half > 0.5  # Most mass should be concentrated

    def test_characteristic_radius_returns_r_t(self):
        """characteristic_radius() returns r_t (tidal radius)."""
        profile = EFFProfile(a=1.0, gamma=3.0, r_t=12.5)
        assert jnp.isclose(profile.characteristic_radius(), 12.5)

    def test_jit_compatible(self):
        """sample_positions() works with JIT compilation."""
        profile = EFFProfile(a=1.0, gamma=3.0, r_t=10.0)
        masses = jnp.ones(100)
        key = jax.random.PRNGKey(42)

        @jax.jit
        def sample_and_sum(m, k):
            pos = profile.sample_positions(m, k)
            return jnp.sum(pos**2)

        result = sample_and_sum(masses, key)
        assert jnp.isfinite(result)

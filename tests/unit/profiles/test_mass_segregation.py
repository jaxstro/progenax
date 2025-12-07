"""Tests for mass segregation transforms."""

import jax
import jax.numpy as jnp
import pytest

from progenax.profiles.mass_segregation import apply_mass_segregation, compute_mass_segregation_ratio


class TestMassSegregation:
    """Tests for primordial mass segregation."""

    def test_output_shape(self):
        """Output positions have same shape as input."""
        key = jax.random.PRNGKey(42)
        positions = jax.random.normal(key, (100, 3))
        masses = jax.random.uniform(jax.random.PRNGKey(0), (100,), minval=0.1, maxval=10.0)

        pos_out = apply_mass_segregation(positions, masses, eta=0.5, m_ref=1.0)
        assert pos_out.shape == (100, 3)

    def test_massive_stars_move_inward(self):
        """Massive stars (m > m_ref) move closer to center."""
        key = jax.random.PRNGKey(42)
        N = 1000
        positions = jax.random.normal(key, (N, 3)) * 2.0  # Scale ~ 2

        # Half low mass, half high mass
        masses = jnp.concatenate([
            jnp.ones(N//2) * 0.5,  # Low mass
            jnp.ones(N//2) * 5.0,  # High mass
        ])

        pos_out = apply_mass_segregation(positions, masses, eta=0.5, m_ref=1.0)

        r_in = jnp.linalg.norm(positions, axis=1)
        r_out = jnp.linalg.norm(pos_out, axis=1)

        # High mass stars should have smaller radii after segregation
        high_mass_mask = masses > 1.0
        r_in_high = jnp.mean(r_in[high_mass_mask])
        r_out_high = jnp.mean(r_out[high_mass_mask])

        assert r_out_high < r_in_high

    def test_low_mass_stars_move_outward(self):
        """Low mass stars (m < m_ref) move further from center."""
        key = jax.random.PRNGKey(42)
        N = 1000
        positions = jax.random.normal(key, (N, 3)) * 2.0

        masses = jnp.concatenate([
            jnp.ones(N//2) * 0.5,
            jnp.ones(N//2) * 5.0,
        ])

        pos_out = apply_mass_segregation(positions, masses, eta=0.5, m_ref=1.0)

        r_in = jnp.linalg.norm(positions, axis=1)
        r_out = jnp.linalg.norm(pos_out, axis=1)

        low_mass_mask = masses < 1.0
        r_in_low = jnp.mean(r_in[low_mass_mask])
        r_out_low = jnp.mean(r_out[low_mass_mask])

        assert r_out_low > r_in_low

    def test_eta_zero_no_change(self):
        """eta=0 leaves positions unchanged."""
        key = jax.random.PRNGKey(42)
        positions = jax.random.normal(key, (100, 3))
        masses = jax.random.uniform(jax.random.PRNGKey(0), (100,), minval=0.1, maxval=10.0)

        pos_out = apply_mass_segregation(positions, masses, eta=0.0, m_ref=1.0)

        assert jnp.allclose(pos_out, positions, rtol=1e-10)

    def test_preserves_direction(self):
        """Radial direction is preserved (only magnitude changes)."""
        key = jax.random.PRNGKey(42)
        positions = jax.random.normal(key, (100, 3))
        masses = jax.random.uniform(jax.random.PRNGKey(0), (100,), minval=0.1, maxval=10.0)

        pos_out = apply_mass_segregation(positions, masses, eta=0.5, m_ref=1.0)

        # Unit vectors should be the same
        r_in = jnp.linalg.norm(positions, axis=1, keepdims=True)
        r_out = jnp.linalg.norm(pos_out, axis=1, keepdims=True)

        r_hat_in = positions / jnp.maximum(r_in, 1e-10)
        r_hat_out = pos_out / jnp.maximum(r_out, 1e-10)

        assert jnp.allclose(r_hat_in, r_hat_out, atol=1e-10)

    def test_stronger_segregation_with_larger_eta(self):
        """Larger eta gives stronger mass segregation."""
        key = jax.random.PRNGKey(42)
        N = 1000
        positions = jax.random.normal(key, (N, 3)) * 2.0
        masses = jnp.concatenate([jnp.ones(N//2) * 0.5, jnp.ones(N//2) * 5.0])

        pos_weak = apply_mass_segregation(positions, masses, eta=0.2, m_ref=1.0)
        pos_strong = apply_mass_segregation(positions, masses, eta=0.8, m_ref=1.0)

        high_mass = masses > 1.0
        r_weak = jnp.mean(jnp.linalg.norm(pos_weak[high_mass], axis=1))
        r_strong = jnp.mean(jnp.linalg.norm(pos_strong[high_mass], axis=1))

        # Stronger segregation should put massive stars even closer to center
        assert r_strong < r_weak


class TestMassSegregationRatio:
    """Tests for MSR diagnostic."""

    def test_msr_increases_with_segregation(self):
        """MSR > 1 after applying segregation."""
        key = jax.random.PRNGKey(42)
        N = 1000
        positions = jax.random.normal(key, (N, 3)) * 2.0
        masses = jnp.concatenate([jnp.ones(N//2) * 0.5, jnp.ones(N//2) * 5.0])

        # Before segregation
        msr_before = compute_mass_segregation_ratio(positions, masses, mass_threshold=2.0)

        # After segregation
        pos_seg = apply_mass_segregation(positions, masses, eta=0.5, m_ref=1.0)
        msr_after = compute_mass_segregation_ratio(pos_seg, masses, mass_threshold=2.0)

        # MSR should increase after segregation
        assert msr_after > msr_before
        assert msr_after > 1.0  # Segregated system has MSR > 1

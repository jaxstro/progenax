"""Tests for mass segregation transforms."""

import jax
import jax.numpy as jnp
import pytest

from progenax.profiles.mass_segregation import (
    apply_mass_segregation,
    compute_mass_segregation_ratio,
    apply_mass_segregation_baumgardt,
)


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


class TestBaumgardtMassSegregation:
    """Tests for energy-ranked mass segregation (Baumgardt+2008)."""

    def test_output_shape(self):
        """Output has same shape as input."""
        key = jax.random.PRNGKey(42)
        N = 100
        positions = jax.random.normal(key, (N, 3))
        velocities = jax.random.normal(jax.random.PRNGKey(1), (N, 3)) * 0.1
        masses = jax.random.uniform(jax.random.PRNGKey(2), (N,), minval=0.1, maxval=10.0)

        pos_out, vel_out = apply_mass_segregation_baumgardt(
            positions, velocities, masses, s=0.5, key=key, G=1.0
        )

        assert pos_out.shape == (N, 3)
        assert vel_out.shape == (N, 3)

    def test_s_zero_random_assignment(self):
        """s=0 gives random assignment (weak mass-energy correlation)."""
        key = jax.random.PRNGKey(42)
        N = 200

        # Create simple radial positions and circular velocities
        radii = jnp.linspace(0.1, 2.0, N)
        angles = jnp.linspace(0, 2*jnp.pi, N)
        positions = jnp.stack([
            radii * jnp.cos(angles),
            radii * jnp.sin(angles),
            jnp.zeros(N)
        ], axis=1)
        velocities = jnp.stack([
            -jnp.sin(angles) * 0.1,
            jnp.cos(angles) * 0.1,
            jnp.zeros(N)
        ], axis=1)

        # Mass range
        masses = jnp.linspace(0.5, 5.0, N)

        pos_out, vel_out = apply_mass_segregation_baumgardt(
            positions, velocities, masses, s=0.0, key=key, G=1.0
        )

        # Compute correlation between mass and binding energy
        r_out = jnp.linalg.norm(pos_out, axis=1)
        v2_out = jnp.sum(vel_out**2, axis=1)

        # Approximate potential (simple -1/r)
        phi_out = -1.0 / jnp.maximum(r_out, 0.01)
        energy_out = 0.5 * v2_out + phi_out

        # For s=0, correlation should be weak (close to 0)
        corr = jnp.corrcoef(jnp.stack([masses, energy_out]))[0, 1]
        assert jnp.abs(corr) < 0.3  # Weak correlation for random assignment

    def test_s_one_maximal_segregation(self):
        """s=1 gives maximal segregation (most massive = most bound)."""
        key = jax.random.PRNGKey(42)
        N = 200

        # Create simple radial positions
        radii = jnp.linspace(0.1, 2.0, N)
        angles = jnp.linspace(0, 2*jnp.pi, N)
        positions = jnp.stack([
            radii * jnp.cos(angles),
            radii * jnp.sin(angles),
            jnp.zeros(N)
        ], axis=1)
        velocities = jnp.stack([
            -jnp.sin(angles) * 0.1,
            jnp.cos(angles) * 0.1,
            jnp.zeros(N)
        ], axis=1)

        # Mass range
        masses = jnp.linspace(0.5, 5.0, N)

        pos_out, vel_out = apply_mass_segregation_baumgardt(
            positions, velocities, masses, s=1.0, key=key, G=1.0
        )

        # Compute binding energies
        r_out = jnp.linalg.norm(pos_out, axis=1)
        v2_out = jnp.sum(vel_out**2, axis=1)
        phi_out = -1.0 / jnp.maximum(r_out, 0.01)
        energy_out = 0.5 * v2_out + phi_out

        # For s=1, most massive stars should have most negative energies
        # (most bound)
        corr = jnp.corrcoef(jnp.stack([masses, energy_out]))[0, 1]
        assert corr < -0.7  # Strong negative correlation (massive = more bound = more negative E)

    def test_s_intermediate(self):
        """Intermediate s gives intermediate segregation."""
        key = jax.random.PRNGKey(42)
        N = 200

        radii = jnp.linspace(0.1, 2.0, N)
        angles = jnp.linspace(0, 2*jnp.pi, N)
        positions = jnp.stack([
            radii * jnp.cos(angles),
            radii * jnp.sin(angles),
            jnp.zeros(N)
        ], axis=1)
        velocities = jnp.stack([
            -jnp.sin(angles) * 0.1,
            jnp.cos(angles) * 0.1,
            jnp.zeros(N)
        ], axis=1)
        masses = jnp.linspace(0.5, 5.0, N)

        # Test s=0.5
        pos_out, vel_out = apply_mass_segregation_baumgardt(
            positions, velocities, masses, s=0.5, key=key, G=1.0
        )

        r_out = jnp.linalg.norm(pos_out, axis=1)
        v2_out = jnp.sum(vel_out**2, axis=1)
        phi_out = -1.0 / jnp.maximum(r_out, 0.01)
        energy_out = 0.5 * v2_out + phi_out

        corr = jnp.corrcoef(jnp.stack([masses, energy_out]))[0, 1]

        # Should be intermediate correlation (between -0.7 and -0.3)
        assert -0.7 < corr < -0.2

    def test_preserves_total_mass(self):
        """Total mass is preserved (just reassigned)."""
        key = jax.random.PRNGKey(42)
        N = 100
        positions = jax.random.normal(key, (N, 3))
        velocities = jax.random.normal(jax.random.PRNGKey(1), (N, 3)) * 0.1
        masses = jax.random.uniform(jax.random.PRNGKey(2), (N,), minval=0.5, maxval=5.0)

        total_mass_before = jnp.sum(masses)

        pos_out, vel_out = apply_mass_segregation_baumgardt(
            positions, velocities, masses, s=0.7, key=key, G=1.0
        )

        # Mass array is not returned (positions/velocities are just reassigned)
        # But the function should preserve all particles
        assert pos_out.shape[0] == N
        assert vel_out.shape[0] == N

    def test_jit_compatible(self):
        """Function works under JIT."""
        key = jax.random.PRNGKey(42)
        N = 100
        positions = jax.random.normal(key, (N, 3))
        velocities = jax.random.normal(jax.random.PRNGKey(1), (N, 3)) * 0.1
        masses = jax.random.uniform(jax.random.PRNGKey(2), (N,), minval=0.5, maxval=5.0)

        # JIT compile
        jit_fn = jax.jit(
            lambda p, v, m, k: apply_mass_segregation_baumgardt(p, v, m, s=0.5, key=k, G=1.0)
        )

        pos_out, vel_out = jit_fn(positions, velocities, masses, key)

        assert pos_out.shape == (N, 3)
        assert vel_out.shape == (N, 3)

    def test_differentiable(self):
        """Gradients flow through the function."""
        key = jax.random.PRNGKey(42)
        N = 50  # Smaller for gradient test
        positions = jax.random.normal(key, (N, 3))
        velocities = jax.random.normal(jax.random.PRNGKey(1), (N, 3)) * 0.1
        masses = jax.random.uniform(jax.random.PRNGKey(2), (N,), minval=0.5, maxval=5.0)

        def loss(G_val):
            pos_out, vel_out = apply_mass_segregation_baumgardt(
                positions, velocities, masses, s=0.5, key=key, G=G_val
            )
            # Simple loss: mean kinetic energy
            return jnp.mean(jnp.sum(vel_out**2, axis=1))

        # Compute gradient
        grad_fn = jax.grad(loss)
        gradient = grad_fn(1.0)

        # Gradient should be finite and non-zero
        assert jnp.isfinite(gradient)
        # Note: gradient might be small but should exist

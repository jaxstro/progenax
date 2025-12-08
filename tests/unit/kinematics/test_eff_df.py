"""
Tests for EFF (Elson-Fall-Freeman) velocity distribution function.

Physics tests only - isotropy and virial scaling.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxstro.units import STELLAR
from progenax.kinematics.eff_df import EFFVelocityDF

G = STELLAR.G


class TestEFFVelocityDFPhysics:
    """Test EFF velocity DF physical properties."""

    def test_isotropic_distribution(self):
        """Velocities are isotropically distributed."""
        df = EFFVelocityDF(a=1.0, gamma=3.0, r_t=10.0)
        N = 1000
        positions = jax.random.normal(jax.random.PRNGKey(0), (N, 3))
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        velocities = df.sample_velocities(positions, masses, key, G=G)

        vx2_mean = jnp.mean(velocities[:, 0]**2)
        vy2_mean = jnp.mean(velocities[:, 1]**2)
        vz2_mean = jnp.mean(velocities[:, 2]**2)

        assert jnp.abs(vx2_mean - vy2_mean) / vx2_mean < 0.15
        assert jnp.abs(vy2_mean - vz2_mean) / vy2_mean < 0.15

    def test_mean_velocity_zero(self):
        """Mean velocity is zero (no bulk motion)."""
        df = EFFVelocityDF(a=1.0, gamma=3.0, r_t=10.0)
        N = 1000
        positions = jax.random.normal(jax.random.PRNGKey(0), (N, 3))
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        velocities = df.sample_velocities(positions, masses, key, G=G)

        mean_vel = jnp.mean(velocities, axis=0)
        v_std = jnp.std(velocities, axis=0)

        for i in range(3):
            assert jnp.abs(mean_vel[i]) < 3 * v_std[i] / jnp.sqrt(N)

    def test_velocity_scales_with_mass(self):
        """Velocity dispersion scales as σ ∝ √M (virial equilibrium)."""
        df = EFFVelocityDF(a=1.0, gamma=3.0, r_t=10.0)
        N = 500
        positions = jax.random.normal(jax.random.PRNGKey(0), (N, 3))

        masses_low = jnp.ones(N) * 0.5   # Total = 2500
        masses_high = jnp.ones(N) * 2.0  # Total = 10000

        key_low, key_high = jax.random.split(jax.random.PRNGKey(42))

        velocities_low = df.sample_velocities(positions, masses_low, key_low, G=G)
        velocities_high = df.sample_velocities(positions, masses_high, key_high, G=G)

        sigma_low = jnp.std(velocities_low[:, 0])
        sigma_high = jnp.std(velocities_high[:, 0])

        assert sigma_high > sigma_low

        ratio = sigma_high / sigma_low
        expected_ratio = jnp.sqrt(10000.0 / 2500.0)  # = 2.0
        assert jnp.abs(ratio - expected_ratio) / expected_ratio < 0.20

    def test_velocity_scales_with_size(self):
        """Velocity dispersion scales as σ ∝ 1/√a (virial equilibrium)."""
        N = 500
        positions = jax.random.normal(jax.random.PRNGKey(0), (N, 3))
        masses = jnp.ones(N)

        df_small = EFFVelocityDF(a=0.5, gamma=3.0, r_t=10.0)
        df_large = EFFVelocityDF(a=2.0, gamma=3.0, r_t=10.0)

        key_small, key_large = jax.random.split(jax.random.PRNGKey(42))

        velocities_small = df_small.sample_velocities(positions, masses, key_small, G=G)
        velocities_large = df_large.sample_velocities(positions, masses, key_large, G=G)

        sigma_small = jnp.std(velocities_small[:, 0])
        sigma_large = jnp.std(velocities_large[:, 0])

        # Smaller size → higher dispersion
        assert sigma_small > sigma_large

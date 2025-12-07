"""
Tests for EFF (Elson-Fall-Freeman) velocity distribution function.

Tests the EFFVelocityDF class which samples isotropic Gaussian velocities
with a velocity scale from virial equilibrium.
"""

import jax
import jax.numpy as jnp
import pytest

from progenax.kinematics.eff_df import EFFVelocityDF


class TestEFFVelocityDFInit:
    """Test EFFVelocityDF initialization."""

    def test_init_default(self):
        """Test default initialization."""
        df = EFFVelocityDF()
        assert df.a == 1.0  # Default scale radius
        assert df.gamma == 3.0  # Default power-law index
        assert df.r_t == 10.0  # Default tidal radius

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        df = EFFVelocityDF(a=2.5, gamma=2.5, r_t=15.0)
        assert df.a == 2.5
        assert df.gamma == 2.5
        assert df.r_t == 15.0


class TestEFFVelocityDFSampling:
    """Test velocity sampling from EFF DF."""

    def test_sample_velocities_shape(self):
        """Test that sample_velocities returns correct shape."""
        df = EFFVelocityDF(a=1.0, gamma=3.0, r_t=10.0)
        N = 100
        positions = jax.random.normal(jax.random.PRNGKey(0), (N, 3))
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        velocities = df.sample_velocities(positions, masses, key, G=1.0)

        assert velocities.shape == (N, 3)

    def test_sample_velocities_isotropic(self):
        """Test that velocities are isotropically distributed."""
        df = EFFVelocityDF(a=1.0, gamma=3.0, r_t=10.0)
        N = 10000
        positions = jax.random.normal(jax.random.PRNGKey(0), (N, 3))
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        velocities = df.sample_velocities(positions, masses, key, G=1.0)

        # Check isotropy: <vx²> ≈ <vy²> ≈ <vz²>
        vx2_mean = jnp.mean(velocities[:, 0]**2)
        vy2_mean = jnp.mean(velocities[:, 1]**2)
        vz2_mean = jnp.mean(velocities[:, 2]**2)

        # All should be within ~5% of each other (statistical test)
        assert jnp.abs(vx2_mean - vy2_mean) / vx2_mean < 0.05
        assert jnp.abs(vy2_mean - vz2_mean) / vy2_mean < 0.05
        assert jnp.abs(vz2_mean - vx2_mean) / vz2_mean < 0.05

    def test_sample_velocities_mean_zero(self):
        """Test that mean velocity is zero (no bulk motion)."""
        df = EFFVelocityDF(a=1.0, gamma=3.0, r_t=10.0)
        N = 10000
        positions = jax.random.normal(jax.random.PRNGKey(0), (N, 3))
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        velocities = df.sample_velocities(positions, masses, key, G=1.0)

        # Mean velocity should be close to zero
        mean_vel = jnp.mean(velocities, axis=0)
        v_std = jnp.std(velocities, axis=0)

        # Each component within 3σ/√N of zero (statistical test)
        for i in range(3):
            assert jnp.abs(mean_vel[i]) < 3 * v_std[i] / jnp.sqrt(N)

    def test_sample_velocities_reproducible(self):
        """Test that sampling is reproducible with same key."""
        df = EFFVelocityDF(a=1.0, gamma=3.0, r_t=10.0)
        N = 100
        positions = jax.random.normal(jax.random.PRNGKey(0), (N, 3))
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        velocities1 = df.sample_velocities(positions, masses, key, G=1.0)
        velocities2 = df.sample_velocities(positions, masses, key, G=1.0)

        assert jnp.allclose(velocities1, velocities2)

    def test_sample_velocities_different_key(self):
        """Test that different keys give different samples."""
        df = EFFVelocityDF(a=1.0, gamma=3.0, r_t=10.0)
        N = 100
        positions = jax.random.normal(jax.random.PRNGKey(0), (N, 3))
        masses = jnp.ones(N)

        velocities1 = df.sample_velocities(positions, masses, jax.random.PRNGKey(42), G=1.0)
        velocities2 = df.sample_velocities(positions, masses, jax.random.PRNGKey(43), G=1.0)

        assert not jnp.allclose(velocities1, velocities2)

    def test_sample_velocities_finite(self):
        """Test that all velocities are finite."""
        df = EFFVelocityDF(a=1.0, gamma=3.0, r_t=10.0)
        N = 1000
        positions = jax.random.normal(jax.random.PRNGKey(0), (N, 3))
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        velocities = df.sample_velocities(positions, masses, key, G=1.0)

        assert jnp.all(jnp.isfinite(velocities))


class TestEFFVelocityDFStatistics:
    """Test statistical properties of EFF velocity distribution."""

    def test_velocity_scale_with_mass(self):
        """Test that velocity scale increases with total mass."""
        df = EFFVelocityDF(a=1.0, gamma=3.0, r_t=10.0)
        N = 5000
        positions = jax.random.normal(jax.random.PRNGKey(0), (N, 3))

        # Two different total masses
        masses_low = jnp.ones(N) * 0.5  # Total = 2500
        masses_high = jnp.ones(N) * 2.0  # Total = 10000

        key_low, key_high = jax.random.split(jax.random.PRNGKey(42))

        velocities_low = df.sample_velocities(positions, masses_low, key_low, G=1.0)
        velocities_high = df.sample_velocities(positions, masses_high, key_high, G=1.0)

        # Velocity dispersion (1D)
        sigma_low = jnp.std(velocities_low[:, 0])
        sigma_high = jnp.std(velocities_high[:, 0])

        # Higher mass should give higher velocity dispersion
        assert sigma_high > sigma_low

        # Check scaling: σ ∝ √M
        ratio = sigma_high / sigma_low
        expected_ratio = jnp.sqrt(10000.0 / 2500.0)  # = 2.0
        assert jnp.abs(ratio - expected_ratio) / expected_ratio < 0.10

    def test_velocity_scale_with_size(self):
        """Test that velocity scale decreases with size."""
        N = 5000
        positions = jax.random.normal(jax.random.PRNGKey(0), (N, 3))
        masses = jnp.ones(N)

        # Two different scale radii
        df_small = EFFVelocityDF(a=0.5, gamma=3.0, r_t=10.0)
        df_large = EFFVelocityDF(a=2.0, gamma=3.0, r_t=10.0)

        key_small, key_large = jax.random.split(jax.random.PRNGKey(42))

        velocities_small = df_small.sample_velocities(positions, masses, key_small, G=1.0)
        velocities_large = df_large.sample_velocities(positions, masses, key_large, G=1.0)

        # Velocity dispersion (1D)
        sigma_small = jnp.std(velocities_small[:, 0])
        sigma_large = jnp.std(velocities_large[:, 0])

        # Smaller size should give higher velocity dispersion (σ ∝ 1/√a)
        assert sigma_small > sigma_large

    def test_protocol_compliance(self):
        """Test that EFFVelocityDF implements VelocityDF protocol."""
        from progenax.protocols import VelocityDF
        df = EFFVelocityDF(a=1.0, gamma=3.0, r_t=10.0)
        assert isinstance(df, VelocityDF)

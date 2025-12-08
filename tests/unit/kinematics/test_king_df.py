"""
Tests for King (1966) velocity distribution function.

Physics tests only - isotropy and dispersion profile.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxstro.units import STELLAR
from progenax.kinematics.king_df import KingVelocityDF

G = STELLAR.G


class TestKingVelocityDFPhysics:
    """Test King velocity DF physical properties."""

    def test_isotropic_distribution(self):
        """Velocities are isotropically distributed."""
        df = KingVelocityDF(W0=5.0, r_c=1.0, r_t=10.0)
        N = 1000
        r = 2.0
        positions = jnp.array([[r, 0.0, 0.0]] * N)
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        velocities = df.sample_velocities(positions, masses, key, G=G)

        vx2_mean = jnp.mean(velocities[:, 0]**2)
        vy2_mean = jnp.mean(velocities[:, 1]**2)
        vz2_mean = jnp.mean(velocities[:, 2]**2)

        assert jnp.abs(vx2_mean - vy2_mean) / vx2_mean < 0.20
        assert jnp.abs(vy2_mean - vz2_mean) / vy2_mean < 0.20

    def test_mean_velocity_zero(self):
        """Mean velocity is zero (no bulk motion)."""
        df = KingVelocityDF(W0=5.0, r_c=1.0, r_t=10.0)
        N = 1000
        r = 2.0
        positions = jnp.array([[r, 0.0, 0.0]] * N)
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        velocities = df.sample_velocities(positions, masses, key, G=G)

        mean_vel = jnp.mean(velocities, axis=0)
        v_std = jnp.std(velocities, axis=0)

        for i in range(3):
            assert jnp.abs(mean_vel[i]) < 3 * v_std[i] / jnp.sqrt(N)

    def test_dispersion_decreases_outward(self):
        """Velocity dispersion decreases with radius."""
        df = KingVelocityDF(W0=7.0, r_c=1.0, r_t=10.0)
        N = 500

        r_inner = 1.0
        r_outer = 5.0
        positions_inner = jnp.array([[r_inner, 0.0, 0.0]] * N)
        positions_outer = jnp.array([[r_outer, 0.0, 0.0]] * N)
        masses = jnp.ones(N)

        key_inner, key_outer = jax.random.split(jax.random.PRNGKey(42))

        velocities_inner = df.sample_velocities(positions_inner, masses, key_inner, G=G)
        velocities_outer = df.sample_velocities(positions_outer, masses, key_outer, G=G)

        sigma_inner = jnp.std(velocities_inner[:, 0])
        sigma_outer = jnp.std(velocities_outer[:, 0])

        assert sigma_inner > sigma_outer

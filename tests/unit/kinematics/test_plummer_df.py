"""
Tests for Plummer velocity distribution function.

Physics tests only - velocity bounds, isotropy, dispersion relations.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxstro.units import STELLAR
from progenax.kinematics.plummer_df import PlummerVelocityDF

G = STELLAR.G


class TestPlummerVelocityDFPhysics:
    """Test Plummer velocity DF physical properties."""

    def test_velocities_below_escape(self):
        """All velocities are below escape velocity."""
        df = PlummerVelocityDF(r_h=1.0)
        N = 1000
        positions = 0.1 * jax.random.normal(jax.random.PRNGKey(0), (N, 3))
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        velocities = df.sample_velocities(positions, masses, key, G=G)

        radii = jnp.linalg.norm(positions, axis=1)
        M_total = jnp.sum(masses)
        v_esc = jnp.sqrt(2.0 * G * M_total / jnp.sqrt(radii**2 + df.a**2))
        v_mag = jnp.linalg.norm(velocities, axis=1)

        assert jnp.all(v_mag < v_esc)

    def test_isotropic_distribution(self):
        """Velocities are isotropically distributed."""
        df = PlummerVelocityDF(r_h=1.0)
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

    def test_velocity_dispersion_relation(self):
        """Plummer dispersion: σ²(r) = GM/(6√(r²+a²))."""
        df = PlummerVelocityDF(r_h=1.0)
        N = 1000
        r = 0.5
        positions = jnp.array([[r, 0.0, 0.0]] * N)
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)
        M_total = N

        velocities = df.sample_velocities(positions, masses, key, G=G)

        sigma_measured = jnp.std(velocities[:, 0])
        sigma_theory = jnp.sqrt(G * M_total / (6.0 * jnp.sqrt(r**2 + df.a**2)))

        relative_error = jnp.abs(sigma_measured - sigma_theory) / sigma_theory
        assert relative_error < 0.15

    def test_beta_distribution_q_squared(self):
        """Velocity ratios q² = (v/v_esc)² follow Beta(3/2, 9/2).

        Mean of q² should be a/(a+b) = 1.5/(1.5+4.5) = 0.25.
        """
        df = PlummerVelocityDF(r_h=1.0)
        N = 1000
        positions = jax.random.normal(jax.random.PRNGKey(0), (N, 3))
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        velocities = df.sample_velocities(positions, masses, key, G=G)

        radii = jnp.linalg.norm(positions, axis=1)
        M_total = jnp.sum(masses)
        v_esc = jnp.sqrt(2.0 * G * M_total / jnp.sqrt(radii**2 + df.a**2))
        v_mag = jnp.linalg.norm(velocities, axis=1)
        q = v_mag / v_esc

        q2_mean = jnp.mean(q**2)
        assert jnp.abs(q2_mean - 0.25) < 0.01

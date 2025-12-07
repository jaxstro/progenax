"""Tests for rotation velocity transforms."""

import jax
import jax.numpy as jnp
import pytest

from progenax.kinematics.rotation import apply_solid_body_rotation, apply_differential_rotation


class TestSolidBodyRotation:
    """Tests for solid body rotation."""

    def test_output_shape(self):
        """Output velocities have same shape as input."""
        positions = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        velocities = jax.random.normal(jax.random.PRNGKey(1), (100, 3))

        v_out = apply_solid_body_rotation(
            velocities, positions, omega=1.0, axis=jnp.array([0., 0., 1.])
        )
        assert v_out.shape == (100, 3)

    def test_adds_rotation(self):
        """Rotation adds tangential velocity component."""
        N = 100
        # Particles in xy-plane
        R = 2.0
        theta = jnp.linspace(0, 2*jnp.pi, N, endpoint=False)
        positions = jnp.stack([R * jnp.cos(theta), R * jnp.sin(theta), jnp.zeros(N)], axis=1)
        velocities = jnp.zeros((N, 3))

        omega = 0.5
        v_out = apply_solid_body_rotation(
            velocities, positions, omega=omega, axis=jnp.array([0., 0., 1.])
        )

        # Expected tangential velocity: v = omega x R
        expected_v_mag = omega * R
        v_mag = jnp.linalg.norm(v_out, axis=1)
        assert jnp.allclose(v_mag, expected_v_mag, rtol=1e-10)

    def test_rotation_direction(self):
        """Rotation is in correct direction (right-hand rule)."""
        # Single particle at (1, 0, 0)
        positions = jnp.array([[1.0, 0.0, 0.0]])
        velocities = jnp.zeros((1, 3))

        # Rotation around z-axis, omega > 0
        v_out = apply_solid_body_rotation(
            velocities, positions, omega=1.0, axis=jnp.array([0., 0., 1.])
        )

        # Should give velocity in +y direction
        assert jnp.abs(v_out[0, 0]) < 1e-10  # No x velocity
        assert v_out[0, 1] > 0.9    # Positive y velocity
        assert jnp.abs(v_out[0, 2]) < 1e-10  # No z velocity

    def test_preserves_parallel_velocity(self):
        """Velocities parallel to rotation axis are preserved."""
        positions = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        # All velocities along z-axis
        velocities = jnp.zeros((100, 3))
        velocities = velocities.at[:, 2].set(1.0)

        v_out = apply_solid_body_rotation(
            velocities, positions, omega=1.0, axis=jnp.array([0., 0., 1.])
        )

        # z-component should be preserved
        assert jnp.allclose(v_out[:, 2], 1.0, rtol=1e-10)

    def test_zero_omega_no_change(self):
        """omega=0 leaves velocities unchanged."""
        positions = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        velocities = jax.random.normal(jax.random.PRNGKey(1), (100, 3))

        v_out = apply_solid_body_rotation(
            velocities, positions, omega=0.0, axis=jnp.array([0., 0., 1.])
        )

        assert jnp.allclose(v_out, velocities, rtol=1e-10)

    def test_arbitrary_axis(self):
        """Works with non-z rotation axis."""
        positions = jnp.array([[1.0, 0.0, 0.0]])
        velocities = jnp.zeros((1, 3))

        # Rotation around x-axis
        v_out = apply_solid_body_rotation(
            velocities, positions, omega=1.0, axis=jnp.array([1., 0., 0.])
        )

        # Particle at (1,0,0) rotating around x: no velocity added
        # (position is along rotation axis)
        assert jnp.allclose(v_out, 0.0, atol=1e-10)


class TestDifferentialRotation:
    """Tests for differential rotation."""

    def test_output_shape(self):
        """Output velocities have same shape as input."""
        positions = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        velocities = jax.random.normal(jax.random.PRNGKey(1), (100, 3))

        v_out = apply_differential_rotation(
            velocities, positions,
            v_peak=1.0, R_peak=1.0,
            axis=jnp.array([0., 0., 1.])
        )
        assert v_out.shape == (100, 3)

    def test_peak_at_R_peak(self):
        """Maximum rotation velocity at R = R_peak."""
        N = 100
        R_peak = 2.0
        v_peak = 1.5

        # Particles at exactly R_peak in xy-plane
        theta = jnp.linspace(0, 2*jnp.pi, N, endpoint=False)
        positions = jnp.stack([
            R_peak * jnp.cos(theta),
            R_peak * jnp.sin(theta),
            jnp.zeros(N)
        ], axis=1)
        velocities = jnp.zeros((N, 3))

        v_out = apply_differential_rotation(
            velocities, positions,
            v_peak=v_peak, R_peak=R_peak,
            axis=jnp.array([0., 0., 1.])
        )

        v_mag = jnp.linalg.norm(v_out, axis=1)
        assert jnp.allclose(v_mag, v_peak, rtol=1e-10)

    def test_decreases_at_large_R(self):
        """Rotation velocity decreases beyond R_peak."""
        R_peak = 1.0
        v_peak = 1.0

        # Particle at R = 5 * R_peak
        positions = jnp.array([[5.0, 0.0, 0.0]])
        velocities = jnp.zeros((1, 3))

        v_out = apply_differential_rotation(
            velocities, positions,
            v_peak=v_peak, R_peak=R_peak,
            axis=jnp.array([0., 0., 1.])
        )

        v_mag = jnp.linalg.norm(v_out, axis=1)
        # At R = 5*R_peak: v = v_peak * 5 * exp(1-5) ~ 0.09
        assert v_mag[0] < v_peak / 2  # Much less than peak

    def test_zero_at_center(self):
        """No rotation at center (on axis)."""
        positions = jnp.array([[0.0, 0.0, 1.0]])  # On z-axis
        velocities = jnp.zeros((1, 3))

        v_out = apply_differential_rotation(
            velocities, positions,
            v_peak=1.0, R_peak=1.0,
            axis=jnp.array([0., 0., 1.])
        )

        assert jnp.allclose(v_out, 0.0, atol=1e-10)

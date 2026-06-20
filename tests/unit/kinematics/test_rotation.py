"""Tests for rotation velocity transforms - physics only."""

import jax
import jax.numpy as jnp
import pytest

from progenax.kinematics.rotation import (
    apply_differential_rotation,
    apply_solid_body_rotation,
)


class TestSolidBodyRotation:
    """Tests for solid body rotation physics."""

    def test_adds_rotation(self):
        """Rotation adds tangential velocity: v = ω × r."""
        N = 100
        R = 2.0
        theta = jnp.linspace(0, 2 * jnp.pi, N, endpoint=False)
        positions = jnp.stack(
            [R * jnp.cos(theta), R * jnp.sin(theta), jnp.zeros(N)], axis=1
        )
        velocities = jnp.zeros((N, 3))

        omega = 0.5
        v_out = apply_solid_body_rotation(
            velocities, positions, omega=omega, axis=jnp.array([0.0, 0.0, 1.0])
        )

        expected_v_mag = omega * R
        v_mag = jnp.linalg.norm(v_out, axis=1)
        assert jnp.allclose(v_mag, expected_v_mag, rtol=1e-10)

    def test_rotation_direction(self):
        """Rotation follows right-hand rule around axis."""
        positions = jnp.array([[1.0, 0.0, 0.0]])
        velocities = jnp.zeros((1, 3))

        v_out = apply_solid_body_rotation(
            velocities, positions, omega=1.0, axis=jnp.array([0.0, 0.0, 1.0])
        )

        # At (1,0,0) with z-rotation: velocity should be in +y direction
        assert jnp.abs(v_out[0, 0]) < 1e-10
        assert v_out[0, 1] > 0.9
        assert jnp.abs(v_out[0, 2]) < 1e-10

    def test_preserves_parallel_velocity(self):
        """Velocities parallel to rotation axis are preserved."""
        positions = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        velocities = jnp.zeros((100, 3))
        velocities = velocities.at[:, 2].set(1.0)

        v_out = apply_solid_body_rotation(
            velocities, positions, omega=1.0, axis=jnp.array([0.0, 0.0, 1.0])
        )

        assert jnp.allclose(v_out[:, 2], 1.0, rtol=1e-10)

    def test_zero_omega_no_change(self):
        """ω=0 leaves velocities unchanged."""
        positions = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        velocities = jax.random.normal(jax.random.PRNGKey(1), (100, 3))

        v_out = apply_solid_body_rotation(
            velocities, positions, omega=0.0, axis=jnp.array([0.0, 0.0, 1.0])
        )

        assert jnp.allclose(v_out, velocities, rtol=1e-10)


class TestDifferentialRotation:
    """Tests for differential rotation physics."""

    def test_peak_at_R_peak(self):
        """Maximum rotation velocity at R = R_peak."""
        N = 100
        R_peak = 2.0
        v_peak = 1.5

        theta = jnp.linspace(0, 2 * jnp.pi, N, endpoint=False)
        positions = jnp.stack(
            [R_peak * jnp.cos(theta), R_peak * jnp.sin(theta), jnp.zeros(N)], axis=1
        )
        velocities = jnp.zeros((N, 3))

        v_out = apply_differential_rotation(
            velocities,
            positions,
            v_peak=v_peak,
            R_peak=R_peak,
            axis=jnp.array([0.0, 0.0, 1.0]),
        )

        v_mag = jnp.linalg.norm(v_out, axis=1)
        assert jnp.allclose(v_mag, v_peak, rtol=1e-10)

    def test_decreases_at_large_R(self):
        """Rotation velocity decreases beyond R_peak."""
        R_peak = 1.0
        v_peak = 1.0

        positions = jnp.array([[5.0, 0.0, 0.0]])
        velocities = jnp.zeros((1, 3))

        v_out = apply_differential_rotation(
            velocities,
            positions,
            v_peak=v_peak,
            R_peak=R_peak,
            axis=jnp.array([0.0, 0.0, 1.0]),
        )

        v_mag = jnp.linalg.norm(v_out, axis=1)
        assert v_mag[0] < v_peak / 2

    def test_zero_at_center(self):
        """No rotation on axis (R=0)."""
        positions = jnp.array([[0.0, 0.0, 1.0]])
        velocities = jnp.zeros((1, 3))

        v_out = apply_differential_rotation(
            velocities,
            positions,
            v_peak=1.0,
            R_peak=1.0,
            axis=jnp.array([0.0, 0.0, 1.0]),
        )

        assert jnp.allclose(v_out, 0.0, atol=1e-10)


class TestZeroAxisRefused:
    """Audit S15: a concrete zero rotation axis has no rotation direction; the
    old code silently no-op'd (axis/max(mag,1e-30)=0) under a stale 'NaN'
    comment. Concrete zero axis now raises."""

    def test_solid_body_zero_axis_raises(self):
        v = jnp.zeros((5, 3))
        pos = jax.random.normal(jax.random.PRNGKey(0), (5, 3))
        with pytest.raises(ValueError, match="zero vector|rotation direction"):
            apply_solid_body_rotation(v, pos, omega=0.1, axis=jnp.zeros(3))

    def test_differential_zero_axis_raises(self):
        v = jnp.zeros((5, 3))
        pos = jax.random.normal(jax.random.PRNGKey(1), (5, 3))
        with pytest.raises(ValueError, match="zero vector|rotation direction"):
            apply_differential_rotation(
                v, pos, v_peak=1.0, R_peak=1.0, axis=jnp.zeros(3)
            )

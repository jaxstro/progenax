"""Tests for velocity anisotropy transforms."""

import jax
import jax.numpy as jnp
import pytest

from progenax.kinematics.anisotropy import apply_osipkov_merritt


class TestOsipkovMerritt:
    """Tests for Osipkov-Merritt radial anisotropy."""

    def test_output_shape(self):
        """Output velocities have same shape as input."""
        key = jax.random.PRNGKey(42)
        N = 100
        positions = jax.random.normal(key, (N, 3))
        velocities = jax.random.normal(jax.random.PRNGKey(0), (N, 3))

        key_transform = jax.random.PRNGKey(1)
        v_out = apply_osipkov_merritt(velocities, positions, key_transform, r_a=1.0)
        assert v_out.shape == (N, 3)

    def test_preserves_speed(self):
        """Transformation preserves |v| for each particle."""
        key = jax.random.PRNGKey(42)
        N = 1000
        positions = jax.random.normal(key, (N, 3))
        velocities = jax.random.normal(jax.random.PRNGKey(0), (N, 3))

        speed_in = jnp.linalg.norm(velocities, axis=1)

        key_transform = jax.random.PRNGKey(1)
        v_out = apply_osipkov_merritt(velocities, positions, key_transform, r_a=1.0)
        speed_out = jnp.linalg.norm(v_out, axis=1)

        assert jnp.allclose(speed_in, speed_out, rtol=1e-10)

    def test_increases_radial_bias(self):
        """At r >> r_a, velocities become more radial."""
        key = jax.random.PRNGKey(42)
        N = 10000

        # Particles at large radii
        radii = jnp.ones(N) * 10.0
        theta = jax.random.uniform(key, (N,), minval=0, maxval=jnp.pi)
        phi = jax.random.uniform(jax.random.PRNGKey(1), (N,), minval=0, maxval=2*jnp.pi)

        positions = jnp.stack([
            radii * jnp.sin(theta) * jnp.cos(phi),
            radii * jnp.sin(theta) * jnp.sin(phi),
            radii * jnp.cos(theta),
        ], axis=1)

        # Start with isotropic velocities
        velocities = jax.random.normal(jax.random.PRNGKey(2), (N, 3))

        # Apply Osipkov-Merritt with r_a = 1 (so r >> r_a)
        key_transform = jax.random.PRNGKey(3)
        v_out = apply_osipkov_merritt(velocities, positions, key_transform, r_a=1.0)

        # Compute radial velocity fraction
        r_hat = positions / jnp.linalg.norm(positions, axis=1, keepdims=True)
        v_r = jnp.sum(v_out * r_hat, axis=1)
        v_mag = jnp.linalg.norm(v_out, axis=1)

        # Check that |v_r|/|v| is larger than isotropic (which would be ~0.577)
        mean_radial_fraction = jnp.mean(jnp.abs(v_r) / v_mag)
        assert mean_radial_fraction > 0.7  # Should be more radial than isotropic

    def test_isotropic_at_center(self):
        """At r << r_a, velocities remain nearly isotropic."""
        key = jax.random.PRNGKey(42)
        N = 10000

        # Particles at small radii
        positions = 0.01 * jax.random.normal(key, (N, 3))
        velocities = jax.random.normal(jax.random.PRNGKey(0), (N, 3))

        key_transform = jax.random.PRNGKey(1)
        v_out = apply_osipkov_merritt(velocities, positions, key_transform, r_a=10.0)

        # For isotropic, each component has equal variance
        var_x = jnp.var(v_out[:, 0])
        var_y = jnp.var(v_out[:, 1])
        var_z = jnp.var(v_out[:, 2])

        # Should be roughly equal (within 20%)
        mean_var = (var_x + var_y + var_z) / 3
        assert jnp.abs(var_x - mean_var) / mean_var < 0.2
        assert jnp.abs(var_y - mean_var) / mean_var < 0.2
        assert jnp.abs(var_z - mean_var) / mean_var < 0.2

    def test_jit_compatible(self):
        """Function works under JIT compilation."""
        @jax.jit
        def apply_jit(v, p, k, r_a):
            return apply_osipkov_merritt(v, p, k, r_a=r_a)

        key = jax.random.PRNGKey(42)
        positions = jax.random.normal(key, (100, 3))
        velocities = jax.random.normal(jax.random.PRNGKey(0), (100, 3))

        result = apply_jit(velocities, positions, jax.random.PRNGKey(1), 1.0)
        assert result.shape == (100, 3)

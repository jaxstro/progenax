"""Tests for explicit gravitational units, radius validation, and isotropic sampling."""

import jax
import jax.numpy as jnp
import pytest
from jaxstro.units import STELLAR

from progenax.kinematics.plummer_df import PlummerVelocityDF

G = STELLAR.G


class TestPlummerVelocityDFPhysics:
    """Test Plummer velocity DF physical properties."""

    def test_isotropic_distribution(self):
        """Velocities are isotropically distributed."""
        df = PlummerVelocityDF(r_h=1.0)
        N = 1000
        positions = jax.random.normal(jax.random.PRNGKey(0), (N, 3))
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        velocities = df.sample_velocities(positions, masses, key, G=G)

        vx2_mean = jnp.mean(velocities[:, 0] ** 2)
        vy2_mean = jnp.mean(velocities[:, 1] ** 2)
        vz2_mean = jnp.mean(velocities[:, 2] ** 2)

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


class TestPlummerVelocityDFExplicitG:
    """Explicit-units policy (audit A2): G is REQUIRED, no DEFAULT_UNITS fallback."""

    def test_g_is_required(self):
        """Omitting G must raise (required arg, no silent default-unit resolution)."""
        df = PlummerVelocityDF(r_h=1.0)
        N = 16
        positions = jnp.zeros((N, 3))
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(123)

        with pytest.raises(TypeError):
            df.sample_velocities(positions, masses, key)


@pytest.mark.parametrize("r_h", [0.0, -1.0])
def test_plummer_df_rejects_nonpositive_radius(r_h):
    """The velocity DF requires the same positive length as its spatial profile."""
    with pytest.raises(ValueError, match="PlummerVelocityDF r_h"):
        PlummerVelocityDF(r_h=r_h)

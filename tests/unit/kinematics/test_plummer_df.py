"""
Tests for Plummer velocity distribution function.

Covers the unit-specific guards: velocity isotropy, the σ²(r) dispersion
relation, and the default-units (G=None) resolution path.

Bound-velocity (v < v_esc) and Beta(3/2, 9/2) speed-statistics physics is
covered more thoroughly in the validation tier:
tests/validation/test_plummer_physics.py (TestPlummerBoundParticles,
TestPlummerBetaDistribution — equilibrium-sampled positions, ⟨q²⟩ mean+variance,
100%-bound assertion). The redundant unit duplicates were removed in the 2026-06
pre-release cross-tier test consolidation.
"""

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


class TestPlummerVelocityDFDefaults:
    """Tests for default-unit behavior."""

    def test_default_units_used_when_g_none(self, monkeypatch):
        """G=None should use progenax.defaults.DEFAULT_UNITS.G."""
        from jaxstro.units import CGS
        from progenax import defaults

        df = PlummerVelocityDF(r_h=1.0)
        N = 16
        positions = jnp.zeros((N, 3))
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(123)

        monkeypatch.setattr(defaults, "DEFAULT_UNITS", CGS)
        velocities_default = df.sample_velocities(positions, masses, key, G=None)
        velocities_explicit = df.sample_velocities(positions, masses, key, G=CGS.G)

        assert jnp.allclose(velocities_default, velocities_explicit)

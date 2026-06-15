"""
Tests for EFF (Elson-Fall-Freeman) velocity distribution function.

Covers the unit-specific guards: the virial σ∝√M and σ∝1/√a scaling trends.

Velocity-isotropy and zero-bulk-motion physics is covered more thoroughly in
the validation tier: tests/validation/test_eff_physics.py (TestEFFVelocityDF::
test_velocity_isotropy with equilibrium-sampled positions, ::test_zero_bulk_velocity
with a per-component 5σ/√N bound). The redundant unit duplicates were removed in
the 2026-06 pre-release cross-tier test consolidation.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxstro.units import STELLAR
from progenax.kinematics.eff_df import EFFVelocityDF

G = STELLAR.G


class TestEFFVelocityDFPhysics:
    """Test EFF velocity DF physical properties."""

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

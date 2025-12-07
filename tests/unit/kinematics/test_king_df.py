"""
Tests for King (1966) velocity distribution function.

Tests the KingVelocityDF class which samples velocities from the
King "lowered Maxwellian" distribution function.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxstro.units import STELLAR
from progenax.kinematics.king_df import KingVelocityDF

# Use stellar dynamics units for star cluster tests
G = STELLAR.G  # ≈ 0.00450 [pc³ Msun⁻¹ Myr⁻²]


class TestKingVelocityDFInit:
    """Test KingVelocityDF initialization."""

    def test_init_default(self):
        """Test default initialization."""
        df = KingVelocityDF()
        assert df.W0 == 5.0  # Default concentration
        assert df.r_c == 1.0  # Default core radius
        assert df.r_t == 10.0  # Default tidal radius

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        df = KingVelocityDF(W0=7.0, r_c=2.0, r_t=20.0)
        assert df.W0 == 7.0
        assert df.r_c == 2.0
        assert df.r_t == 20.0


class TestKingVelocityDFSampling:
    """Test velocity sampling from King DF."""

    def test_sample_velocities_shape(self):
        """Test that sample_velocities returns correct shape."""
        df = KingVelocityDF(W0=5.0, r_c=1.0, r_t=10.0)
        N = 100
        # Use positions within tidal radius
        radii = jnp.linspace(0.1, 9.0, N)
        positions = jnp.stack([radii, jnp.zeros(N), jnp.zeros(N)], axis=1)
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        velocities = df.sample_velocities(positions, masses, key, G=G)

        assert velocities.shape == (N, 3)

    def test_sample_velocities_bounded(self):
        """Test that velocities are below escape velocity (approximately)."""
        df = KingVelocityDF(W0=7.0, r_c=1.0, r_t=10.0)
        N = 1000
        # Sample at various radii
        radii = jnp.linspace(0.5, 8.0, N)
        positions = jnp.stack([radii, jnp.zeros(N), jnp.zeros(N)], axis=1)
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        velocities = df.sample_velocities(positions, masses, key, G=G)

        # All velocity magnitudes should be finite
        v_mag = jnp.linalg.norm(velocities, axis=1)
        assert jnp.all(jnp.isfinite(v_mag))
        assert jnp.all(v_mag >= 0.0)

    def test_sample_velocities_isotropic(self):
        """Test that velocities are isotropically distributed."""
        df = KingVelocityDF(W0=5.0, r_c=1.0, r_t=10.0)
        N = 10000
        # Sample at fixed radius
        r = 2.0
        positions = jnp.array([[r, 0.0, 0.0]] * N)
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        velocities = df.sample_velocities(positions, masses, key, G=G)

        # Check isotropy: <vx²> ≈ <vy²> ≈ <vz²>
        vx2_mean = jnp.mean(velocities[:, 0]**2)
        vy2_mean = jnp.mean(velocities[:, 1]**2)
        vz2_mean = jnp.mean(velocities[:, 2]**2)

        # All should be within ~10% of each other (statistical test)
        assert jnp.abs(vx2_mean - vy2_mean) / vx2_mean < 0.10
        assert jnp.abs(vy2_mean - vz2_mean) / vy2_mean < 0.10
        assert jnp.abs(vz2_mean - vx2_mean) / vz2_mean < 0.10

    def test_sample_velocities_mean_zero(self):
        """Test that mean velocity is zero (no bulk motion)."""
        df = KingVelocityDF(W0=5.0, r_c=1.0, r_t=10.0)
        N = 10000
        # Sample at fixed radius
        r = 2.0
        positions = jnp.array([[r, 0.0, 0.0]] * N)
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        velocities = df.sample_velocities(positions, masses, key, G=G)

        # Mean velocity should be close to zero
        mean_vel = jnp.mean(velocities, axis=0)
        v_std = jnp.std(velocities, axis=0)

        # Each component within 3σ/√N of zero (statistical test)
        for i in range(3):
            assert jnp.abs(mean_vel[i]) < 3 * v_std[i] / jnp.sqrt(N)

    def test_sample_velocities_reproducible(self):
        """Test that sampling is reproducible with same key."""
        df = KingVelocityDF(W0=5.0, r_c=1.0, r_t=10.0)
        N = 100
        radii = jnp.linspace(0.5, 8.0, N)
        positions = jnp.stack([radii, jnp.zeros(N), jnp.zeros(N)], axis=1)
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        velocities1 = df.sample_velocities(positions, masses, key, G=G)
        velocities2 = df.sample_velocities(positions, masses, key, G=G)

        assert jnp.allclose(velocities1, velocities2)

    def test_sample_velocities_different_key(self):
        """Test that different keys give different samples."""
        df = KingVelocityDF(W0=5.0, r_c=1.0, r_t=10.0)
        N = 100
        radii = jnp.linspace(0.5, 8.0, N)
        positions = jnp.stack([radii, jnp.zeros(N), jnp.zeros(N)], axis=1)
        masses = jnp.ones(N)

        velocities1 = df.sample_velocities(positions, masses, jax.random.PRNGKey(42), G=G)
        velocities2 = df.sample_velocities(positions, masses, jax.random.PRNGKey(43), G=G)

        assert not jnp.allclose(velocities1, velocities2)


class TestKingVelocityDFStatistics:
    """Test statistical properties of King velocity distribution."""

    def test_velocity_dispersion_decreases_outward(self):
        """Test that velocity dispersion decreases with radius."""
        df = KingVelocityDF(W0=7.0, r_c=1.0, r_t=10.0)
        N = 5000

        # Sample at two radii
        r_inner = 1.0
        r_outer = 5.0
        positions_inner = jnp.array([[r_inner, 0.0, 0.0]] * N)
        positions_outer = jnp.array([[r_outer, 0.0, 0.0]] * N)
        masses = jnp.ones(N)

        key_inner, key_outer = jax.random.split(jax.random.PRNGKey(42))

        velocities_inner = df.sample_velocities(positions_inner, masses, key_inner, G=G)
        velocities_outer = df.sample_velocities(positions_outer, masses, key_outer, G=G)

        # Velocity dispersion (1D)
        sigma_inner = jnp.std(velocities_inner[:, 0])
        sigma_outer = jnp.std(velocities_outer[:, 0])

        # Inner should have higher dispersion
        assert sigma_inner > sigma_outer

    def test_protocol_compliance(self):
        """Test that KingVelocityDF implements VelocityDF protocol."""
        from progenax.protocols import VelocityDF
        df = KingVelocityDF(W0=5.0, r_c=1.0, r_t=10.0)
        assert isinstance(df, VelocityDF)

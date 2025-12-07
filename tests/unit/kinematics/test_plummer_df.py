"""
Tests for Plummer velocity distribution function.

Tests the PlummerVelocityDF class which samples velocities from the
exact Plummer (1911) distribution function using Beta distribution sampling.
"""

import jax
import jax.numpy as jnp
import pytest

from progenax.kinematics.plummer_df import PlummerVelocityDF


class TestPlummerVelocityDFInit:
    """Test PlummerVelocityDF initialization."""

    def test_init_default(self):
        """Test default initialization (r_h=1.0)."""
        df = PlummerVelocityDF()
        assert df.r_h == 1.0
        # Check scale radius: a = r_h / sqrt(2^(2/3) - 1)
        expected_a = 1.0 / jnp.sqrt(2**(2/3) - 1)
        assert jnp.allclose(df.a, expected_a)

    def test_init_custom_r_h(self):
        """Test initialization with custom r_h."""
        df = PlummerVelocityDF(r_h=2.5)
        assert df.r_h == 2.5
        expected_a = 2.5 / jnp.sqrt(2**(2/3) - 1)
        assert jnp.allclose(df.a, expected_a)


class TestPlummerVelocityDFSampling:
    """Test velocity sampling from Plummer DF."""

    def test_sample_velocities_shape(self):
        """Test that sample_velocities returns correct shape."""
        df = PlummerVelocityDF(r_h=1.0)
        N = 100
        positions = jax.random.normal(jax.random.PRNGKey(0), (N, 3))
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        velocities = df.sample_velocities(positions, masses, key, G=1.0)

        assert velocities.shape == (N, 3)

    def test_sample_velocities_bounded(self):
        """Test that all velocities are below escape velocity."""
        df = PlummerVelocityDF(r_h=1.0)
        N = 1000
        # Use small radii to get high v_esc
        positions = 0.1 * jax.random.normal(jax.random.PRNGKey(0), (N, 3))
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)
        G = 1.0

        velocities = df.sample_velocities(positions, masses, key, G=G)

        # Compute escape velocity at each position
        radii = jnp.linalg.norm(positions, axis=1)
        M_total = jnp.sum(masses)
        v_esc = jnp.sqrt(2.0 * G * M_total / jnp.sqrt(radii**2 + df.a**2))

        # Check all velocities are below escape velocity
        v_mag = jnp.linalg.norm(velocities, axis=1)
        assert jnp.all(v_mag < v_esc)

    def test_sample_velocities_isotropic(self):
        """Test that velocities are isotropically distributed."""
        df = PlummerVelocityDF(r_h=1.0)
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
        df = PlummerVelocityDF(r_h=1.0)
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
        df = PlummerVelocityDF(r_h=1.0)
        N = 100
        positions = jax.random.normal(jax.random.PRNGKey(0), (N, 3))
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        velocities1 = df.sample_velocities(positions, masses, key, G=1.0)
        velocities2 = df.sample_velocities(positions, masses, key, G=1.0)

        assert jnp.allclose(velocities1, velocities2)

    def test_sample_velocities_different_key(self):
        """Test that different keys give different samples."""
        df = PlummerVelocityDF(r_h=1.0)
        N = 100
        positions = jax.random.normal(jax.random.PRNGKey(0), (N, 3))
        masses = jnp.ones(N)

        velocities1 = df.sample_velocities(positions, masses, jax.random.PRNGKey(42), G=1.0)
        velocities2 = df.sample_velocities(positions, masses, jax.random.PRNGKey(43), G=1.0)

        assert not jnp.allclose(velocities1, velocities2)


class TestPlummerVelocityDFStatistics:
    """Test statistical properties of Plummer velocity distribution."""

    def test_velocity_dispersion_relation(self):
        """
        Test the Plummer velocity dispersion relation.

        For Plummer sphere: σ²(r) = GM/(6√(r²+a²))
        And: v_esc²(r) = 2GM/√(r²+a²)
        Therefore: v_esc = sqrt(12) × σ (exact relation)
        """
        df = PlummerVelocityDF(r_h=1.0)
        N = 10000
        # Sample at specific radius
        r = 0.5
        positions = jnp.array([[r, 0.0, 0.0]] * N)
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)
        G = 1.0
        M_total = N

        velocities = df.sample_velocities(positions, masses, key, G=G)

        # Measured velocity dispersion (1D)
        sigma_measured = jnp.std(velocities[:, 0])

        # Theoretical velocity dispersion at radius r
        sigma_theory = jnp.sqrt(G * M_total / (6.0 * jnp.sqrt(r**2 + df.a**2)))

        # Should match within ~5% (statistical test with N=10000)
        relative_error = jnp.abs(sigma_measured - sigma_theory) / sigma_theory
        assert relative_error < 0.05

    def test_beta_distribution_sampling(self):
        """
        Test that velocity magnitudes follow Beta(3/2, 9/2) for q² = (v/v_esc)².

        The Plummer DF gives g(q) ∝ q²(1-q²)^(7/2) for q = v/v_esc.
        This corresponds to q² ~ Beta(3/2, 9/2).
        """
        df = PlummerVelocityDF(r_h=1.0)
        N = 10000
        positions = jax.random.normal(jax.random.PRNGKey(0), (N, 3))
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)
        G = 1.0

        velocities = df.sample_velocities(positions, masses, key, G=G)

        # Compute q = v/v_esc for each particle
        radii = jnp.linalg.norm(positions, axis=1)
        M_total = jnp.sum(masses)
        v_esc = jnp.sqrt(2.0 * G * M_total / jnp.sqrt(radii**2 + df.a**2))
        v_mag = jnp.linalg.norm(velocities, axis=1)
        q = v_mag / v_esc

        # q² should have mean = a/(a+b) for Beta(a,b)
        # For Beta(3/2, 9/2): mean(q²) = 1.5/(1.5+4.5) = 0.25
        q2_mean = jnp.mean(q**2)
        assert jnp.abs(q2_mean - 0.25) < 0.01  # Within 1% (tight!)

    def test_jit_compilation(self):
        """Test that sample_velocities works with JIT (already decorated)."""
        df = PlummerVelocityDF(r_h=1.0)
        N = 100
        positions = jax.random.normal(jax.random.PRNGKey(0), (N, 3))
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        # sample_velocities is already @jax.jit decorated
        velocities = df.sample_velocities(positions, masses, key, G=1.0)

        assert velocities.shape == (N, 3)

    def test_protocol_compliance(self):
        """Test that PlummerVelocityDF implements VelocityDF protocol."""
        from progenax.protocols import VelocityDF
        df = PlummerVelocityDF(r_h=1.0)
        assert isinstance(df, VelocityDF)

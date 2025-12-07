# progenax/tests/unit/test_populations.py
"""
Unit tests for two-component cluster populations.

Tests:
- TwoComponentConfig creation and validation
- generate_two_component_cluster output shapes
- Population fractions match f_A
- Population A is more extended than B (spatial separation)
- Custom population mask override
- JIT compatibility
"""

import jax
import jax.numpy as jnp
import pytest
from progenax.populations import TwoComponentConfig, generate_two_component_cluster
from progenax.profiles import PlummerProfile
from progenax.kinematics import PlummerVelocityDF


class TestTwoComponentConfig:
    """Test TwoComponentConfig dataclass."""

    def test_config_creation(self):
        """Config can be created with valid parameters."""
        profile_A = PlummerProfile(r_h=2.0)  # Extended
        profile_B = PlummerProfile(r_h=0.5)  # Concentrated
        df_A = PlummerVelocityDF(r_h=2.0)
        df_B = PlummerVelocityDF(r_h=0.5)

        config = TwoComponentConfig(
            f_A=0.3,
            profile_A=profile_A,
            profile_B=profile_B,
            velocity_df_A=df_A,
            velocity_df_B=df_B,
        )

        assert config.f_A == 0.3
        assert config.profile_A is profile_A
        assert config.profile_B is profile_B
        assert config.velocity_df_A is df_A
        assert config.velocity_df_B is df_B

    def test_config_immutable(self):
        """Config is frozen (immutable)."""
        profile_A = PlummerProfile(r_h=2.0)
        profile_B = PlummerProfile(r_h=0.5)
        df_A = PlummerVelocityDF(r_h=2.0)
        df_B = PlummerVelocityDF(r_h=0.5)

        config = TwoComponentConfig(
            f_A=0.3,
            profile_A=profile_A,
            profile_B=profile_B,
            velocity_df_A=df_A,
            velocity_df_B=df_B,
        )

        with pytest.raises(Exception):  # FrozenInstanceError or similar
            config.f_A = 0.5


class TestGenerateTwoComponentCluster:
    """Test generate_two_component_cluster function."""

    def test_output_shapes(self):
        """Output shapes are correct: (N,3), (N,3), (N,)."""
        N = 100
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        profile_A = PlummerProfile(r_h=2.0)
        profile_B = PlummerProfile(r_h=0.5)
        df_A = PlummerVelocityDF(r_h=2.0)
        df_B = PlummerVelocityDF(r_h=0.5)

        config = TwoComponentConfig(
            f_A=0.3,
            profile_A=profile_A,
            profile_B=profile_B,
            velocity_df_A=df_A,
            velocity_df_B=df_B,
        )

        positions, velocities, pop_id = generate_two_component_cluster(
            masses, config, key, G=1.0
        )

        assert positions.shape == (N, 3)
        assert velocities.shape == (N, 3)
        assert pop_id.shape == (N,)

    def test_population_fractions(self):
        """Population ID distribution matches f_A fraction."""
        N = 1000
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        profile_A = PlummerProfile(r_h=2.0)
        profile_B = PlummerProfile(r_h=0.5)
        df_A = PlummerVelocityDF(r_h=2.0)
        df_B = PlummerVelocityDF(r_h=0.5)

        config = TwoComponentConfig(
            f_A=0.3,
            profile_A=profile_A,
            profile_B=profile_B,
            velocity_df_A=df_A,
            velocity_df_B=df_B,
        )

        _, _, pop_id = generate_two_component_cluster(masses, config, key, G=1.0)

        # Count population A (pop_id == 0)
        n_A = jnp.sum(pop_id == 0)
        f_A_measured = n_A / N

        # Should be close to 0.3 (within 3 sigma for binomial)
        assert jnp.abs(f_A_measured - 0.3) < 0.05  # ~3 sigma for N=1000

    def test_population_A_more_extended(self):
        """Population A has larger mean radius than B."""
        N = 1000
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        # A is extended (r_h=2.0), B is concentrated (r_h=0.5)
        profile_A = PlummerProfile(r_h=2.0)
        profile_B = PlummerProfile(r_h=0.5)
        df_A = PlummerVelocityDF(r_h=2.0)
        df_B = PlummerVelocityDF(r_h=0.5)

        config = TwoComponentConfig(
            f_A=0.5,  # Equal populations for clearer comparison
            profile_A=profile_A,
            profile_B=profile_B,
            velocity_df_A=df_A,
            velocity_df_B=df_B,
        )

        positions, _, pop_id = generate_two_component_cluster(
            masses, config, key, G=1.0
        )

        # Compute mean radius for each population
        radii = jnp.linalg.norm(positions, axis=1)
        r_mean_A = jnp.mean(jnp.where(pop_id == 0, radii, 0.0)) / jnp.mean(pop_id == 0)
        r_mean_B = jnp.mean(jnp.where(pop_id == 1, radii, 0.0)) / jnp.mean(pop_id == 1)

        # Population A should be more extended
        assert r_mean_A > r_mean_B
        # Ratio should be roughly r_h_A / r_h_B = 2.0 / 0.5 = 4.0 (within factor ~2)
        assert r_mean_A / r_mean_B > 2.0

    def test_custom_population_mask(self):
        """Custom pop_mask overrides random assignment."""
        N = 100
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        profile_A = PlummerProfile(r_h=2.0)
        profile_B = PlummerProfile(r_h=0.5)
        df_A = PlummerVelocityDF(r_h=2.0)
        df_B = PlummerVelocityDF(r_h=0.5)

        config = TwoComponentConfig(
            f_A=0.3,  # This should be ignored when pop_mask is provided
            profile_A=profile_A,
            profile_B=profile_B,
            velocity_df_A=df_A,
            velocity_df_B=df_B,
        )

        # Custom mask: first 20 stars in pop A, rest in pop B
        pop_mask = jnp.arange(N) < 20

        positions, velocities, pop_id = generate_two_component_cluster(
            masses, config, key, G=1.0, pop_mask=pop_mask
        )

        # Verify pop_id matches pop_mask
        # pop_id == 0 means pop A, pop_id == 1 means pop B
        expected_pop_id = jnp.where(pop_mask, 0, 1)
        assert jnp.allclose(pop_id, expected_pop_id)

    def test_pop_id_values(self):
        """Population ID is 0 or 1."""
        N = 100
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        profile_A = PlummerProfile(r_h=2.0)
        profile_B = PlummerProfile(r_h=0.5)
        df_A = PlummerVelocityDF(r_h=2.0)
        df_B = PlummerVelocityDF(r_h=0.5)

        config = TwoComponentConfig(
            f_A=0.3,
            profile_A=profile_A,
            profile_B=profile_B,
            velocity_df_A=df_A,
            velocity_df_B=df_B,
        )

        _, _, pop_id = generate_two_component_cluster(masses, config, key, G=1.0)

        # All values should be 0 or 1
        assert jnp.all((pop_id == 0) | (pop_id == 1))

    def test_jit_compatible(self):
        """Function works under JIT via wrapper."""
        N = 50
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        profile_A = PlummerProfile(r_h=2.0)
        profile_B = PlummerProfile(r_h=0.5)
        df_A = PlummerVelocityDF(r_h=2.0)
        df_B = PlummerVelocityDF(r_h=0.5)

        config = TwoComponentConfig(
            f_A=0.3,
            profile_A=profile_A,
            profile_B=profile_B,
            velocity_df_A=df_A,
            velocity_df_B=df_B,
        )

        # Closure captures config (JIT-compatible because profiles/DFs are PyTrees)
        @jax.jit
        def jitted_fn(masses, key, G):
            return generate_two_component_cluster(masses, config, key, G)

        positions, velocities, pop_id = jitted_fn(masses, key, 1.0)

        assert positions.shape == (N, 3)
        assert velocities.shape == (N, 3)
        assert pop_id.shape == (N,)

    def test_different_seeds_different_results(self):
        """Different random seeds give different results."""
        N = 100
        masses = jnp.ones(N)

        profile_A = PlummerProfile(r_h=2.0)
        profile_B = PlummerProfile(r_h=0.5)
        df_A = PlummerVelocityDF(r_h=2.0)
        df_B = PlummerVelocityDF(r_h=0.5)

        config = TwoComponentConfig(
            f_A=0.3,
            profile_A=profile_A,
            profile_B=profile_B,
            velocity_df_A=df_A,
            velocity_df_B=df_B,
        )

        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(123)

        pos1, vel1, pop1 = generate_two_component_cluster(masses, config, key1, G=1.0)
        pos2, vel2, pop2 = generate_two_component_cluster(masses, config, key2, G=1.0)

        # Should not be identical
        assert not jnp.allclose(pos1, pos2)
        assert not jnp.allclose(vel1, vel2)

    def test_reproducibility(self):
        """Same seed gives same results."""
        N = 100
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        profile_A = PlummerProfile(r_h=2.0)
        profile_B = PlummerProfile(r_h=0.5)
        df_A = PlummerVelocityDF(r_h=2.0)
        df_B = PlummerVelocityDF(r_h=0.5)

        config = TwoComponentConfig(
            f_A=0.3,
            profile_A=profile_A,
            profile_B=profile_B,
            velocity_df_A=df_A,
            velocity_df_B=df_B,
        )

        pos1, vel1, pop1 = generate_two_component_cluster(masses, config, key, G=1.0)
        pos2, vel2, pop2 = generate_two_component_cluster(masses, config, key, G=1.0)

        assert jnp.allclose(pos1, pos2)
        assert jnp.allclose(vel1, vel2)
        assert jnp.allclose(pop1, pop2)

    def test_both_populations_present(self):
        """Both populations are present (not degenerate case)."""
        N = 100
        masses = jnp.ones(N)
        key = jax.random.PRNGKey(42)

        profile_A = PlummerProfile(r_h=2.0)
        profile_B = PlummerProfile(r_h=0.5)
        df_A = PlummerVelocityDF(r_h=2.0)
        df_B = PlummerVelocityDF(r_h=0.5)

        config = TwoComponentConfig(
            f_A=0.3,
            profile_A=profile_A,
            profile_B=profile_B,
            velocity_df_A=df_A,
            velocity_df_B=df_B,
        )

        _, _, pop_id = generate_two_component_cluster(masses, config, key, G=1.0)

        # Both populations should be present
        assert jnp.any(pop_id == 0)  # Pop A exists
        assert jnp.any(pop_id == 1)  # Pop B exists

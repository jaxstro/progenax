# progenax/tests/unit/test_populations.py
"""
Unit tests for two-component cluster populations.

Physics-focused tests:
- Population fractions match f_A (statistical test)
- Population A is more extended than B (spatial separation)
"""

import jax
import jax.numpy as jnp
from jaxstro.units import STELLAR
from progenax.populations import TwoComponentConfig, generate_two_component_cluster
from progenax.profiles import PlummerProfile
from progenax.kinematics import PlummerVelocityDF

# Use stellar dynamics units for star cluster tests
G = STELLAR.G  # ≈ 0.00450 [pc³ Msun⁻¹ Myr⁻²]


def test_population_fractions():
    """Population ID distribution matches f_A fraction (statistical test)."""
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

    _, _, pop_id = generate_two_component_cluster(masses, config, key, G=G)

    # Count population A (pop_id == 0)
    n_A = jnp.sum(pop_id == 0)
    f_A_measured = n_A / N

    # Should be close to 0.3 (within 3 sigma for binomial)
    assert jnp.abs(f_A_measured - 0.3) < 0.05  # ~3 sigma for N=1000


def test_population_A_more_extended():
    """Population A has larger mean radius than B (spatial separation test)."""
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
        masses, config, key, G=G
    )

    # Compute mean radius for each population
    radii = jnp.linalg.norm(positions, axis=1)
    r_mean_A = jnp.mean(jnp.where(pop_id == 0, radii, 0.0)) / jnp.mean(pop_id == 0)
    r_mean_B = jnp.mean(jnp.where(pop_id == 1, radii, 0.0)) / jnp.mean(pop_id == 1)

    # Population A should be more extended
    assert r_mean_A > r_mean_B
    # Ratio should be roughly r_h_A / r_h_B = 2.0 / 0.5 = 4.0 (within factor ~2)
    assert r_mean_A / r_mean_B > 2.0


def test_generate_two_component_jit_compatible():
    """Function works under JIT (single JAX compatibility test)."""
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

    # Verify output structure is correct
    assert positions.shape == (N, 3)
    assert velocities.shape == (N, 3)
    assert pop_id.shape == (N,)

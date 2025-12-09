# progenax/tests/unit/imf/test_environment_v2.py
"""Unit tests for BirthEnvironment differentiable dataclass."""

import jax
import jax.numpy as jnp
import pytest


class TestBirthEnvironmentBasic:
    """Test BirthEnvironment creation and attributes."""

    def test_create_with_required_fields(self):
        """Can create BirthEnvironment with log_density and metallicity."""
        from progenax.imf.environment_v2 import BirthEnvironment

        env = BirthEnvironment(
            log_density=jnp.array(4.0),
            metallicity=jnp.array(-0.5),
        )

        assert jnp.isclose(env.log_density, 4.0)
        assert jnp.isclose(env.metallicity, -0.5)

    def test_default_sfr(self):
        """SFR defaults to 1.0 M_sun/yr."""
        from progenax.imf.environment_v2 import BirthEnvironment

        env = BirthEnvironment(
            log_density=jnp.array(4.0),
            metallicity=jnp.array(0.0),
        )

        assert jnp.isclose(env.sfr, 1.0)


class TestBirthEnvironmentJAXCompatibility:
    """Test BirthEnvironment works with JAX transformations."""

    def test_is_valid_pytree(self):
        """BirthEnvironment can be flattened as pytree."""
        from progenax.imf.environment_v2 import BirthEnvironment

        env = BirthEnvironment(
            log_density=jnp.array(4.0),
            metallicity=jnp.array(-0.5),
        )

        leaves, treedef = jax.tree_util.tree_flatten(env)

        assert len(leaves) == 3  # log_density, metallicity, sfr
        assert all(isinstance(leaf, jax.Array) for leaf in leaves)

    def test_grad_through_log_density(self):
        """Can compute gradient through log_density."""
        from progenax.imf.environment_v2 import BirthEnvironment

        def loss(log_density):
            env = BirthEnvironment(
                log_density=log_density,
                metallicity=jnp.array(0.0),
            )
            return env.log_density ** 2

        grad_fn = jax.grad(loss)
        gradient = grad_fn(jnp.array(4.0))

        assert jnp.isclose(gradient, 8.0)  # d/dx(x^2) = 2x


class TestEnvToIMFParams:
    """Test env_to_imf_params mapping function."""

    def test_universal_kroupa_ignores_environment(self):
        """universal_kroupa model returns standard Kroupa regardless of env."""
        from progenax.imf.environment_v2 import BirthEnvironment, env_to_imf_params

        env1 = BirthEnvironment(log_density=jnp.array(2.0), metallicity=jnp.array(-1.0))
        env2 = BirthEnvironment(log_density=jnp.array(6.0), metallicity=jnp.array(+0.5))

        params1 = env_to_imf_params(env1, model="universal_kroupa")
        params2 = env_to_imf_params(env2, model="universal_kroupa")

        assert jnp.isclose(params1.alpha_high, 2.3)
        assert jnp.isclose(params2.alpha_high, 2.3)
        assert jnp.isclose(params1.alpha_high, params2.alpha_high)

    def test_marks2012_density_dependence(self):
        """marks2012_like: higher density → lower alpha_high (top-heavy)."""
        from progenax.imf.environment_v2 import BirthEnvironment, env_to_imf_params

        env_low = BirthEnvironment(log_density=jnp.array(3.0), metallicity=jnp.array(0.0))
        env_high = BirthEnvironment(log_density=jnp.array(6.0), metallicity=jnp.array(0.0))

        params_low = env_to_imf_params(env_low, model="marks2012_like")
        params_high = env_to_imf_params(env_high, model="marks2012_like")

        # Higher density → more top-heavy → lower alpha
        assert params_high.alpha_high < params_low.alpha_high
        assert params_low.alpha_high <= 2.3  # At threshold, should be ~Kroupa

    def test_jerabkova2018_metallicity_effect(self):
        """jerabkova2018_like: metallicity affects alpha_high."""
        from progenax.imf.environment_v2 import BirthEnvironment, env_to_imf_params

        # Same density, different metallicity
        env_metal_poor = BirthEnvironment(log_density=jnp.array(5.0), metallicity=jnp.array(-1.0))
        env_metal_rich = BirthEnvironment(log_density=jnp.array(5.0), metallicity=jnp.array(+0.5))

        params_poor = env_to_imf_params(env_metal_poor, model="jerabkova2018_like")
        params_rich = env_to_imf_params(env_metal_rich, model="jerabkova2018_like")

        # Metallicity should have some effect (direction depends on model)
        assert params_poor.alpha_high != params_rich.alpha_high

    def test_gradient_through_env_to_params(self):
        """Can compute gradient of alpha_high wrt log_density."""
        from progenax.imf.environment_v2 import BirthEnvironment, env_to_imf_params

        def get_alpha(log_density):
            env = BirthEnvironment(
                log_density=log_density,
                metallicity=jnp.array(0.0),
            )
            params = env_to_imf_params(env, model="marks2012_like")
            return params.alpha_high

        grad_fn = jax.grad(get_alpha)
        gradient = grad_fn(jnp.array(5.0))

        assert jnp.isfinite(gradient)
        # Marks model: higher density → lower alpha, so gradient < 0
        assert gradient < 0

    def test_invalid_model_raises(self):
        """Unknown model name raises ValueError."""
        from progenax.imf.environment_v2 import BirthEnvironment, env_to_imf_params

        env = BirthEnvironment(log_density=jnp.array(4.0), metallicity=jnp.array(0.0))

        with pytest.raises(ValueError, match="Unknown model"):
            env_to_imf_params(env, model="nonexistent_model")

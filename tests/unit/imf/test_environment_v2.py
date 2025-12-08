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

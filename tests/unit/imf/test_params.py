"""Unit tests for IMFParams differentiable dataclass."""

import jax
import jax.numpy as jnp
import pytest


class TestIMFParamsBasic:
    """Test IMFParams creation and attributes."""

    def test_kroupa_factory_creates_standard_values(self):
        """IMFParams.kroupa() returns canonical Kroupa (2001) values."""
        from progenax.imf.params import IMFParams

        params = IMFParams.kroupa()

        assert jnp.isclose(params.alpha_low, 0.3, atol=1e-6)
        assert jnp.isclose(params.alpha_mid, 1.3, atol=1e-6)
        assert jnp.isclose(params.alpha_high, 2.3, atol=1e-6)

    def test_fixed_breaks_are_correct(self):
        """Mass breaks are fixed at 0.08 and 0.5 M_sun."""
        from progenax.imf.params import IMFParams

        params = IMFParams.kroupa()

        assert params.m_break1 == 0.08
        assert params.m_break2 == 0.50

    def test_custom_alpha_high(self):
        """Can create IMFParams with custom alpha_high."""
        from progenax.imf.params import IMFParams

        params = IMFParams(
            alpha_low=jnp.array(0.3),
            alpha_mid=jnp.array(1.3),
            alpha_high=jnp.array(2.7),
        )

        assert jnp.isclose(params.alpha_high, 2.7)


class TestIMFParamsJAXCompatibility:
    """Test IMFParams works with JAX transformations."""

    def test_is_valid_pytree(self):
        """IMFParams can be flattened/unflattened as pytree."""
        from progenax.imf.params import IMFParams

        params = IMFParams.kroupa()
        leaves, treedef = jax.tree_util.tree_flatten(params)

        # Should have 3 leaves (alpha_low, alpha_mid, alpha_high)
        assert len(leaves) == 3
        assert all(isinstance(leaf, jax.Array) for leaf in leaves)

    def test_jit_compatible(self):
        """IMFParams works inside JIT-compiled function."""
        from progenax.imf.params import IMFParams

        @jax.jit
        def get_alpha_sum(params):
            return params.alpha_low + params.alpha_mid + params.alpha_high

        params = IMFParams.kroupa()
        result = get_alpha_sum(params)

        assert jnp.isclose(result, 0.3 + 1.3 + 2.3)

    def test_grad_through_alpha_high(self):
        """Can compute gradient through alpha_high."""
        from progenax.imf.params import IMFParams

        def loss(alpha_high):
            params = IMFParams(
                alpha_low=jnp.array(0.3),
                alpha_mid=jnp.array(1.3),
                alpha_high=alpha_high,
            )
            return params.alpha_high ** 2

        grad_fn = jax.grad(loss)
        gradient = grad_fn(jnp.array(2.3))

        # d/dx(x^2) = 2x, so gradient should be 2*2.3 = 4.6
        assert jnp.isclose(gradient, 4.6)

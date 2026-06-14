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

        assert jnp.isclose(params.alpha0, 0.3, atol=1e-6)
        assert jnp.isclose(params.alpha1, 1.3, atol=1e-6)
        assert jnp.isclose(params.alpha2, 2.3, atol=1e-6)
        assert jnp.isclose(params.alpha3, 2.3, atol=1e-6)

    def test_fixed_breaks_are_correct(self):
        """Mass breaks are fixed at 0.08, 0.50, and 1.00 M_sun."""
        from progenax.imf.params import IMFParams

        params = IMFParams.kroupa()

        assert params.m_break0 == 0.08
        assert params.m_break1 == 0.50
        assert params.m_break2 == 1.00

    def test_custom_alpha3(self):
        """Can create IMFParams with custom alpha3."""
        from progenax.imf.params import IMFParams

        params = IMFParams(
            alpha0=jnp.array(0.3),
            alpha1=jnp.array(1.3),
            alpha2=jnp.array(2.3),
            alpha3=jnp.array(2.7),
        )

        assert jnp.isclose(params.alpha3, 2.7)


class TestIMFParamsJAXCompatibility:
    """Test IMFParams works with JAX transformations."""

    def test_is_valid_pytree(self):
        """IMFParams can be flattened/unflattened as pytree."""
        from progenax.imf.params import IMFParams

        params = IMFParams.kroupa()
        leaves, treedef = jax.tree_util.tree_flatten(params)

        # Should have 4 leaves (alpha0, alpha1, alpha2, alpha3)
        assert len(leaves) == 4
        assert all(isinstance(leaf, jax.Array) for leaf in leaves)

    def test_jit_compatible(self):
        """IMFParams works inside JIT-compiled function."""
        from progenax.imf.params import IMFParams

        @jax.jit
        def get_alpha_sum(params):
            return params.alpha0 + params.alpha1 + params.alpha2 + params.alpha3

        params = IMFParams.kroupa()
        result = get_alpha_sum(params)

        assert jnp.isclose(result, 0.3 + 1.3 + 2.3 + 2.3)

    # The former test_grad_through_alpha3 was a trivial d/dx(x^2)==2x PyTree-leaf sanity
    # check (it grad'd params.alpha3**2, exercising NO IMF graph) -- pure smoke, removed
    # (audit T6). The real params->summary alpha3 gradient through the IMF likelihood is
    # FD-audited by the grad-audit registry (tests/validation/grad_audit/registry.py ::
    # IMFParams.log_prob_nll [alpha3]); see
    # docs/website/50-validation/differentiability-audit.md (registry is SoT).

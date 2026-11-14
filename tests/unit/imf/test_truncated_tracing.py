"""Traced truncation bounds; three compiled value/forward/reverse configurations."""

import jax
import jax.numpy as jnp
import pytest

from progenax.imf import PowerLawIMF, TruncatedIMF

_INNER = PowerLawIMF([0.0], [], m_min=1.0, m_max=10.0)


def _quantiles(bounds):
    lower, upper = bounds
    return TruncatedIMF(_INNER, lower, upper).ppf(jnp.array([0.25, 0.75]))


_quantiles_jit = jax.jit(_quantiles)
_quantiles_forward = jax.jit(jax.jacfwd(_quantiles))
_quantiles_reverse = jax.jit(jax.jacrev(_quantiles))


def test_truncated_uniform_bound_derivatives_and_runtime_guard():
    """Catch Python max/min on traced cutoffs; verify uniform quantiles and reject empty support."""
    bounds = jnp.array([2.0, 8.0])
    expected = jnp.array([3.5, 6.5])
    expected_jacobian = jnp.array([[0.75, 0.25], [0.25, 0.75]])
    # Uniform quantiles are affine functions of the support bounds, so only
    # floating-point arithmetic contributes error at these O(1) values.
    assert jnp.allclose(_quantiles_jit(bounds), expected, rtol=2e-14, atol=0.0)
    assert jnp.allclose(
        _quantiles_forward(bounds), expected_jacobian, rtol=2e-14, atol=0.0
    )
    assert jnp.allclose(
        _quantiles_reverse(bounds), expected_jacobian, rtol=2e-14, atol=0.0
    )
    with pytest.raises(ValueError, match="Invalid truncation bounds"):
        TruncatedIMF(_INNER, 8.0, 2.0)
    with pytest.raises((ValueError, RuntimeError), match="Invalid truncation bounds"):
        _quantiles_jit(jnp.array([8.0, 2.0])).block_until_ready()

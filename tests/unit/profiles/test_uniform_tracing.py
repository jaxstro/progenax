"""Uniform-sphere constructor tracing; two compiled value/AD configurations."""

import jax
import jax.numpy as jnp
import pytest

from progenax.profiles.uniform import UniformSphereProfile


def _quantiles(radius):
    return UniformSphereProfile(radius)._sample_radii(jnp.array([0.125, 0.5, 1.0]))


_quantiles_jit = jax.jit(_quantiles)
_quantiles_jacobian = jax.jit(jax.jacfwd(_quantiles))


def test_uniform_radius_is_traced_through_constructor():
    """Catch a Python boolean guard on R; verify r=R u^(1/3) and dr/dR analytically."""
    radius = 2.0
    expected_derivative = jnp.array([0.5, 0.5 ** (1 / 3), 1.0])
    assert jnp.allclose(
        _quantiles_jit(radius), radius * expected_derivative, rtol=2e-15, atol=0.0
    )
    assert jnp.allclose(
        _quantiles_jacobian(radius), expected_derivative, rtol=2e-15, atol=0.0
    )
    with pytest.raises(ValueError, match="must be > 0"):
        UniformSphereProfile(-1.0)

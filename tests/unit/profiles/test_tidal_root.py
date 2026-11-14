"""Shared tidal-root values and derivatives across saved-node crossings."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progenax.profiles.king import _find_tidal_radius


@pytest.mark.parametrize("vacuum", [False, True])
def test_root_crossing_has_correct_value_and_derivative(vacuum):
    x = jnp.linspace(0.5, 2.5, 65)

    def root(a):
        return _find_tidal_radius(x, 1 / x - a if vacuum else a - x)

    pars = jnp.array([1 - 1e-5, 1.0, 1 + 1e-5])
    vals = jax.jit(jax.vmap(root))(pars)
    reverse = jax.jit(jax.vmap(jax.grad(root)))(pars)
    forward = jax.jit(jax.vmap(jax.jacfwd(root)))(pars)
    np.testing.assert_allclose(vals, 1 / pars if vacuum else pars, rtol=1e-10)
    np.testing.assert_allclose(
        reverse, -1 / pars**2 if vacuum else jnp.ones(3), rtol=1e-6
    )
    np.testing.assert_allclose(forward, reverse, rtol=1e-12)


@pytest.mark.parametrize(
    "y,expected",
    [
        ([4.0, 3.0, 2.0, 1.0], 2.0),
        ([0.0, -1.0, -2.0, -3.0], 0.5),
        ([1.0, 1.0, 1.0, 1.0], 2.0),
        ([0.0, 0.0, 0.0, 0.0], 0.5),
    ],
)
def test_root_fallback_has_finite_gradient(y, expected):
    x = jnp.array([0.5, 1.0, 1.5, 2.0])
    value, gradient = jax.jit(jax.value_and_grad(lambda z: _find_tidal_radius(x, z)))(
        jnp.array(y)
    )
    assert float(value) == expected
    assert bool(jnp.all(jnp.isfinite(gradient)))

"""Angular normalization and derivatives against independent angular integration."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

from progenax.profiles.limepy import _angle_integral_T


def test_isotropic_value_and_derivative():
    value, derivative = jax.jit(jax.value_and_grad(_angle_integral_T))(0.0)
    np.testing.assert_allclose(value, 2.0, rtol=1e-12)
    np.testing.assert_allclose(derivative, -4.0 / 3.0, rtol=1e-12)


def test_values_and_derivatives_against_original_coordinate_integral():
    beta = np.array(
        [1e-12, 0.1, 1.0, 25 - 1e-8, 25.0, 25 + 1e-8, 100.0, 1e4, 399440.9657416734]
    )
    reference = []
    for b in beta:
        points = sorted({1 - a / b for a in (1.0, 4.0, 16.0, 64.0) if b > a})
        row = []
        for derivative in (False, True):

            def integrand(c):
                s = (1 - c) * (1 + c)
                return 2 * np.exp(-b * s) * (-s if derivative else 1.0)

            value, _ = quad(
                integrand,
                0.0,
                1.0,
                points=points,
                epsabs=1e-22,
                epsrel=1e-12,
                limit=200,
            )
            row.append(value)
        reference.append(row)
    reference = np.asarray(reference)
    values = jax.jit(_angle_integral_T)(jnp.asarray(beta))
    reverse = jax.jit(jax.vmap(jax.grad(_angle_integral_T)))(jnp.asarray(beta))
    forward = jax.jit(jax.vmap(jax.jacfwd(_angle_integral_T)))(jnp.asarray(beta))
    np.testing.assert_allclose(values, reference[:, 0], rtol=1e-5, atol=0.0)
    np.testing.assert_allclose(reverse, reference[:, 1], rtol=0.003, atol=0.0)
    np.testing.assert_allclose(forward, reverse, rtol=1e-12, atol=0.0)


def test_array_shape_and_scalar_agreement():
    beta = jnp.array([[0.0, 1.0], [25.0, 1e4]])
    actual = jax.jit(_angle_integral_T)(beta)
    expected = jax.vmap(_angle_integral_T)(beta.ravel()).reshape(beta.shape)
    assert actual.shape == beta.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-12)

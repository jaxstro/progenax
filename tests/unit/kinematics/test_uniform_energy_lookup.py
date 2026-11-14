"""Uniform-grid optimization preserves the generic interpolation contract."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progenax.kinematics.eddington import _interp_uniform, sample_speed_from_f_table


@pytest.mark.parametrize("size", [2, 17, 1000])
def test_lookup_value_and_derivatives(size):
    assert jax.config.jax_enable_x64
    knots = jnp.linspace(0.1, 2.0, size)
    queries = jnp.concatenate(
        (knots, (knots[:-1] + knots[1:]) / 2, jnp.array([-0.2, 2.2]))
    )

    def evaluate(theta, interp):
        grid = theta[0] * knots + theta[1]
        return interp(queries * theta[2], grid, jnp.sin(grid * theta[3]))

    theta = jnp.array([1.0, 0.0, 1.0, 0.7])
    for transform in (lambda f: f, jax.jacfwd, jax.jacrev):
        actual = jax.jit(transform(lambda t: evaluate(t, _interp_uniform)))(theta)
        expected = jax.jit(transform(lambda t: evaluate(t, jnp.interp)))(theta)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("nodes", [256, 4096])
def test_speed_value_and_derivatives(nodes):
    key = jax.random.PRNGKey(1)

    def evaluate(theta, uniform):
        energy = theta[0] * jnp.linspace(0.001, 0.999, 1000)
        values = energy ** theta[1]
        psi = theta[0] * jnp.array([0.0, 1e-7, 0.2, 0.9, 1.0])
        return jax.vmap(
            lambda p: sample_speed_from_f_table(
                key, p, energy, values, n_w=nodes, uniform_energy_grid=uniform
            )
        )(psi)

    theta = jnp.array([1.0, 2.0])
    for transform in (lambda f: f, jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(
            jax.jit(transform(lambda t: evaluate(t, True)))(theta),
            jax.jit(transform(lambda t: evaluate(t, False)))(theta),
            rtol=1e-12,
            atol=1e-12,
        )


def test_default_sampler_accepts_nonuniform_energy_grid():
    from jaxstro.numerics.sampling import inverse_cdf_draw

    energy = jnp.array([0.001, 0.02, 0.3, 0.6, 1.0])
    values = energy**2
    key = jax.random.PRNGKey(4)
    speed_grid = jnp.linspace(0.0, jnp.sqrt(1.6), 256)
    weight = speed_grid**2 * jnp.interp(0.8 - speed_grid**2 / 2, energy, values)
    expected = inverse_cdf_draw(weight, speed_grid, jax.random.uniform(key))
    actual = sample_speed_from_f_table(key, 0.8, energy, values)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

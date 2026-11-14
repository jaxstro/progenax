"""Factored density boundary law and isotropic symmetry."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progenax.profiles.limepy import _aniso_density_scalar, limepy_density_hat
from progenax.profiles.limepy_tables import AnisoDensityTable


@pytest.mark.parametrize("g", [0.0, 0.5, 1.0, 3.5])
def test_isotropic_limit_and_origin_sensitivity(g):
    table = jax.jit(lambda: AnisoDensityTable.build(12.0, 80.0, g))()
    potential = jnp.array([1e-6, 0.1, 7.0])
    exact = jnp.sqrt(2 * jnp.pi) * limepy_density_hat(potential, g)
    density = jax.jit(jax.vmap(lambda w: table.evaluate(w, 0.0)))(potential)
    direct = jax.jit(jax.vmap(lambda w: _aniso_density_scalar(w, 0.0, g)))(potential)
    np.testing.assert_allclose(density, exact, rtol=1e-5, atol=0.0)
    np.testing.assert_allclose(direct, exact, rtol=1e-5, atol=0.0)
    dp = jax.jit(jax.vmap(lambda w: jax.grad(lambda p: table.evaluate(w, p))(0.0)))(
        potential
    )
    np.testing.assert_array_equal(dp, jnp.zeros_like(potential))
    for w in [0.0, -1.0]:
        value, dw = jax.value_and_grad(lambda v: table.evaluate(v, 1.0))(w)
        assert float(value) == 0.0
        assert float(dw) == 0.0

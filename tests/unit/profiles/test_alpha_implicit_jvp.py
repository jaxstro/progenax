"""Independent normalized-mass identity for the implicit fraction tangent."""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

from progenax.profiles.limepy_multimass import find_alpha_for_masses


def test_unsegregated_fractions_and_mass_tangent():
    masses = jnp.array([0.6, 0.4])

    def fit(target, adjoint=None):
        return find_alpha_for_masses(
            jnp.array([1.0, 4.0]),
            target,
            5.0,
            1.0,
            0.0,
            n_iter=30,
            xi_max=300.0,
            n_points=500,
            adjoint=adjoint,
        )

    value, residual = jax.jit(fit)(masses)
    np.testing.assert_allclose(value, masses / jnp.sum(masses), rtol=1e-10)
    assert float(residual) < 1e-6
    expected = jax.jacfwd(lambda m: m / jnp.sum(m))(masses)
    forward = jax.jit(jax.jacfwd(lambda m: fit(m, diffrax.ForwardMode())[0]))(masses)
    reverse = jax.jit(jax.jacrev(lambda m: fit(m)[0]))(masses)
    # Existing exact-arithmetic tolerance: equal velocity scales imply equal shapes.
    np.testing.assert_allclose(forward, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(reverse, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(jnp.sum(forward, axis=0), 0.0, atol=1e-10)

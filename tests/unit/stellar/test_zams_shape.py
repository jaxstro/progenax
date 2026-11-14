"""Inverse ZAMS shape regression; two compiled programs for values and AD modes."""

import jax
import jax.numpy as jnp

from progenax.stellar import inverse_zams_luminosity, zams_luminosity

_inverse = jax.jit(inverse_zams_luminosity)


def _derivatives(luminosities):
    return jax.jacfwd(inverse_zams_luminosity)(luminosities), jax.jacrev(
        inverse_zams_luminosity
    )(luminosities)


_inverse_derivatives = jax.jit(_derivatives)


def test_inverse_zams_preserves_matrix_shape_and_inverse_derivatives():
    """Catch passing vector rows to a scalar Newton slope; check inverse identity per entry."""
    masses = jnp.array([[0.5, 1.0], [2.0, 5.0]])
    luminosities = zams_luminosity(masses)
    recovered = _inverse(luminosities)
    assert recovered.shape == masses.shape
    # Twenty Newton steps settle these interior masses to float64 roundoff.
    assert jnp.allclose(recovered, masses, rtol=3e-14, atol=0.0)
    # Independent inverse-function identity: dM/dL=1/(dL/dM), with no
    # cross-element couplings in this elementwise empirical relation.
    slope = jax.vmap(jax.grad(zams_luminosity))(masses.reshape(-1))
    expected = jnp.diag(1 / slope).reshape((2, 2, 2, 2))
    forward, reverse = _inverse_derivatives(luminosities)
    assert jnp.allclose(forward, expected, rtol=5e-13, atol=0.0)
    assert jnp.allclose(reverse, expected, rtol=5e-13, atol=0.0)

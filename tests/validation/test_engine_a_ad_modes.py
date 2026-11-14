"""Forward AD through public Engine A construction and radial observables."""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progenax.cluster._radial_cdf import radial_cdf, radial_quantile
from progenax.cluster.multicomponent import MultiComponentCluster


@pytest.mark.parametrize("g", [0.0, 1.0])
def test_engine_a_forward_reverse_and_finite_difference(g):
    def values(theta, adjoint=None):
        model = MultiComponentCluster.from_mass_segregation(
            alpha_j=[0.6, 0.4],
            m_j=[1.0, 4.0],
            W0=5.0,
            g=theta[0],
            delta=theta[1],
            r_a=4.0,
            eta=0.0,
            r_c=1.0,
            xi_max=800.0,
            n_ode_points=2000,
            n_grid=600,
            rtol=1e-12,
            atol=1e-14,
            adjoint=adjoint,
        )
        x, y, c = model._r_grid, model.engine_a.radial_pdf_j, model._cdf_j
        return jnp.array(
            [
                model.r_t,
                model.mu_tot,
                *[radial_cdf(0.5, x, y[i], c[i]) for i in range(2)],
                *[radial_quantile(0.025, x, y[i], c[i]) for i in range(2)],
            ]
        )

    theta = jnp.array([g, 0.4])
    forward = jax.jit(jax.jacfwd(lambda t: values(t, diffrax.ForwardMode())))(theta)
    reverse = jax.jit(jax.jacrev(values))(theta)
    assert bool(jnp.all(jnp.isfinite(forward)))
    # Adaptive adjoints differ numerically: use the retained derivative budget,
    # not the exact-arithmetic criterion used for simple algebraic identities.
    np.testing.assert_allclose(forward, reverse, rtol=0.003, atol=0.0)
    both = lambda t: values(t, diffrax.DirectAdjoint())
    for transform in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(
            jax.jit(transform(both))(theta), forward, rtol=0.003, atol=0.0
        )
    evaluate = jax.jit(values)
    for i in range(2):
        h = 1e-5
        shift = jnp.eye(2)[i] * h
        lower = theta if i == 0 and g == 0 else theta - shift
        denominator = h if i == 0 and g == 0 else 2 * h
        fd = (evaluate(theta + shift) - evaluate(lower)) / denominator
        # Retained Engine A derivative criterion; g=0 uses the right derivative.
        np.testing.assert_allclose(forward[:, i], fd, rtol=0.003, atol=0.0)

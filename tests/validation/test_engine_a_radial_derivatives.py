"""Engine A radial derivatives with explicit potential-interval boundaries."""

import jax
import jax.numpy as jnp
import numpy as np

from progenax.cluster._radial_cdf import radial_cdf, radial_quantile
from progenax.cluster.multicomponent import MultiComponentCluster


def test_radial_derivatives_with_potential_boundaries():
    """Former worst case: both components, CDF and inverse, both parameters."""
    theta = jnp.array([4.0, 0.3])

    def evaluate(t):
        model = MultiComponentCluster.from_mass_segregation(
            alpha_j=[0.6, 0.4],
            m_j=[1.0, 4.0],
            W0=7.0,
            g=1.0,
            eta=0.0,
            r_c=1.0,
            r_a=t[0],
            delta=t[1],
            xi_max=800.0,
            n_ode_points=2000,
            n_grid=600,
            rtol=1e-12,
            atol=1e-14,
        )
        x, y, m = model._r_grid, model.engine_a.radial_pdf_j, model._cdf_j
        values = jnp.concatenate(
            (
                jax.vmap(lambda p, c: radial_cdf(0.5, x, p, c))(y, m),
                jax.vmap(lambda p, c: radial_quantile(0.025, x, p, c))(y, m),
            )
        )
        # Every saved potential boundary lies on the integration grid.
        edges = jnp.minimum(model.xi_grid * model.r_c, model.r_t)
        indices = jnp.clip(jnp.searchsorted(x, edges), 0, x.size - 1)
        return values, x[indices] - edges

    forward = jax.jit(evaluate)
    ad = jax.jit(jax.jacrev(lambda t: evaluate(t)[0]))(theta)
    _, boundary_error = forward(theta)
    np.testing.assert_array_equal(boundary_error, 0.0)
    for i in range(2):
        # Existing smaller step, fixed before this regression was added.
        h = 1e-5 * max(float(abs(theta[i])), 1.0)
        shift = jnp.eye(2)[i] * h
        plus, plus_error = forward(theta + shift)
        minus, minus_error = forward(theta - shift)
        np.testing.assert_array_equal(plus_error, 0.0)
        np.testing.assert_array_equal(minus_error, 0.0)
        fd = (plus - minus) / (2 * h)
        assert bool(jnp.all(jnp.isfinite(ad[:, i])))
        np.testing.assert_allclose(ad[:, i], fd, rtol=0.003, atol=0.0)

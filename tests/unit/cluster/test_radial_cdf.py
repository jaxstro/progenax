"""Analytic oracles for the Engine A piecewise-linear radial density."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progenax.cluster._radial_cdf import radial_cdf, radial_quantile


@pytest.mark.parametrize("slope", [-1.0, 0.0, 1.0])
def test_linear_density_has_analytic_cdf_and_inverse(slope):
    # p(r) = 1 + slope * (2r - 1), normalized on [0, 1].
    grid = jnp.array([0.0, 0.13, 0.51, 0.87, 1.0])
    density = 1 + slope * (2 * grid - 1)
    mass = (1 - slope) * grid + slope * grid**2
    radius = jnp.linspace(0, 1, 1025)
    expected = (1 - slope) * radius + slope * radius**2
    np.testing.assert_allclose(
        radial_cdf(radius, grid, density, mass), expected, atol=1e-14
    )
    recovered = radial_quantile(expected, grid, density, mass)
    np.testing.assert_allclose(recovered, radius, atol=1e-12)
    assert float(recovered[0]) == 0.0
    assert float(recovered[-1]) == 1.0
    # Implicit inverse derivative: dr/du = 1 / p(r).
    u = jnp.array([0.02, 0.3, 0.8, 0.98])
    r = radial_quantile(u, grid, density, mass)
    derivative = jax.vmap(jax.grad(lambda q: radial_quantile(q, grid, density, mass)))(
        u
    )
    np.testing.assert_allclose(derivative, 1 / (1 + slope * (2 * r - 1)), rtol=1e-12)


def test_zero_density_plateau_and_support():
    grid = jnp.arange(5.0)
    density = jnp.array([0.0, 1.0, 0.0, 0.0, 1.0])
    mass = jnp.array([0.0, 0.5, 1.0, 1.0, 1.5])
    u = jnp.linspace(0, 1, 1025)
    radius = radial_quantile(u, grid, density, mass)
    assert bool(jnp.all(jnp.diff(radius) >= 0))
    np.testing.assert_allclose(radial_cdf(radius, grid, density, mass), u, atol=1e-14)
    np.testing.assert_array_equal(
        radial_quantile(jnp.array([-1.0, 2.0]), grid, density, mass), [0.0, 4.0]
    )


def test_moving_support_derivative():
    # p(r) proportional to r on [0, scale], so r(u)=scale*sqrt(u).
    def inverse(scale):
        grid = scale * jnp.array([0.0, 0.2, 0.7, 1.0])
        return radial_quantile(0.3, grid, grid, grid**2 / 2)

    np.testing.assert_allclose(jax.grad(inverse)(2.0), jnp.sqrt(0.3), rtol=1e-12)


@pytest.mark.parametrize("r_a", [None, 4.0])
def test_engine_a_sampled_radii_follow_integrated_density(r_a):
    from progenax.cluster.multicomponent import MultiComponentCluster

    model = MultiComponentCluster.from_mass_segregation(
        alpha_j=[0.6, 0.4],
        m_j=[1.0, 4.0],
        W0=5.0,
        g=1.0,
        delta=0.4,
        eta=0.0,
        r_c=1.0,
        r_a=r_a,
        n_grid=600,
    )
    key = jax.random.PRNGKey(13)
    _, position_key, _, _, _ = jax.random.split(key, 5)
    u = jax.random.uniform(position_key, (400,))
    sample = model.sample_cluster(key, 400, G=1.0)
    radius = jnp.linalg.norm(sample.positions, axis=1)
    recovered = jax.vmap(
        lambda r, c: radial_cdf(
            r, model._r_grid, model.engine_a.radial_pdf_j[c], model._cdf_j[c]
        )
    )(radius, sample.component_id)
    # Checks the production route and component assignment against the forward CDF.
    np.testing.assert_allclose(recovered, u, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("tidal_radius", [0.3, 1.0, 2.7])
def test_knot_grid_minimum_resolution_and_padded_derivatives(tidal_radius):
    from progenax.cluster._radial_cdf import radial_grid

    potential = jnp.array([0.01, 0.2, 0.7, 1.5, 3.0])
    grid = radial_grid(potential, tidal_radius, 19)
    assert int(jnp.sum(jnp.diff(grid) > 0)) >= 19
    for edge in jnp.minimum(potential, tidal_radius):
        assert bool(jnp.any(grid == edge))

    def inverse(scale):
        x = radial_grid(potential, scale, 19)
        return radial_quantile(0.3, x, x, x * x / 2)

    np.testing.assert_allclose(
        jax.grad(inverse)(tidal_radius), jnp.sqrt(0.3), rtol=1e-12
    )
    x = grid
    u = jnp.linspace(0.0, 1.0, 1025)
    r = radial_quantile(u, x, x, x * x / 2)
    np.testing.assert_allclose(radial_cdf(r, x, x, x * x / 2), u, atol=1e-14)


def test_padded_grid_support_endpoint_derivatives():
    from progenax.cluster._radial_cdf import radial_grid

    def endpoints(scale):
        x = radial_grid(jnp.array([0.01, 0.2, 0.7, 1.5, 3.0]), scale, 19)
        m = x * x / 2
        return jnp.array(
            [
                radial_cdf(0.0, x, x, m),
                radial_cdf(scale, x, x, m),
                radial_quantile(0.0, x, x, m),
                radial_quantile(1.0, x, x, m),
            ]
        )

    np.testing.assert_allclose(
        jax.jit(jax.jacrev(endpoints))(1.0), [0.0, 0.0, 0.0, 1.0], atol=1e-12
    )


def test_indexed_component_inverse_matches_scalar_values_and_ad():
    from progenax.cluster._radial_cdf import component_radial_quantile

    probability = jnp.concatenate(
        (jnp.array([-0.1, 0.0, 1.0, 1.1]), jnp.linspace(0.001, 0.999, 1025))
    )
    component = jnp.arange(probability.size) % 2

    def evaluate(theta, indexed):
        grid = theta[0] * jnp.array([0.0, 0.2, 0.2, 0.5, 0.8, 1.0])
        density = jnp.stack(
            (
                jnp.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
                jnp.array([1.0, 1.0, 1.0, 2.0, 1.0, 0.0])
                * jnp.exp(theta[1] * jnp.arange(6)),
            )
        )
        area = jnp.diff(grid) * (density[:, 1:] + density[:, :-1]) / 2
        cumulative = jnp.concatenate(
            (jnp.zeros((2, 1)), jnp.cumsum(area, axis=1)), axis=1
        )
        if indexed:
            return component_radial_quantile(
                probability, component, grid, density, cumulative
            )
        return jax.vmap(
            lambda p, c: radial_quantile(p, grid, density[c], cumulative[c])
        )(probability, component)

    theta = jnp.array([2.0, 0.3])
    reference = lambda t: evaluate(t, False)
    indexed = lambda t: evaluate(t, True)
    np.testing.assert_array_equal(indexed(theta), reference(theta))
    np.testing.assert_allclose(
        jax.jacfwd(indexed)(theta), jax.jacfwd(reference)(theta), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        jax.grad(lambda t: jnp.sum(indexed(t) ** 2))(theta),
        jax.grad(lambda t: jnp.sum(reference(t) ** 2))(theta),
        rtol=1e-12,
        atol=1e-12,
    )

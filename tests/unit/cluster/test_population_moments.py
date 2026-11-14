"""Exact radial integration and explicit population-moment contract."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progenax.cluster._radial_cdf import (
    inverse_cdf_second_moments,
    radial_second_moments,
)


def test_inverse_cdf_moment_plateaus_tails_and_derivatives():
    # Two uniform radial intervals separated by a zero-probability gap,
    # with constant interpolation tails. The weight moves probability
    # between intervals, independently of the length scale.
    def moment(theta):
        scale, weight = theta
        grid = scale * jnp.array([1.0, 2.0, 3.0, 4.0])
        cdf = jnp.array([[0.1, weight, weight, 0.9]])
        return inverse_cdf_second_moments(grid, cdf)[0]

    def reference(theta):
        scale, weight = theta
        return scale**2 * (
            0.1 + (weight - 0.1) * 7 / 3 + (0.9 - weight) * 37 / 3 + 0.1 * 16
        )

    theta = jnp.array([2.0, 0.4])
    np.testing.assert_allclose(moment(theta), reference(theta), rtol=1e-12)
    for ad in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(ad(moment)(theta), ad(reference)(theta), rtol=1e-12)


from progenax.cluster.multicomponent import MultiComponentCluster


def test_linear_density_second_moment_and_scale_derivative():
    # Uniform and linearly increasing PDFs on [0, L], including a repeated node.
    def moments(scale):
        grid = scale * jnp.array([0.0, 0.25, 0.25, 1.0])
        density = jnp.stack((jnp.ones(4), grid / scale))
        return radial_second_moments(grid, density)

    expected = jnp.array([1 / 3, 1 / 2])
    np.testing.assert_allclose(moments(2.0), 4 * expected, rtol=1e-12)
    np.testing.assert_allclose(jax.jacfwd(moments)(2.0), 4 * expected, rtol=1e-12)
    np.testing.assert_allclose(jax.jacrev(moments)(2.0), 4 * expected, rtol=1e-12)


def test_population_moment_contract_rejects_unsupported_inputs():
    # Contract checks precede numerical work; no equilibrium solve is needed.
    from types import SimpleNamespace

    call = MultiComponentCluster.estimate_population_moments
    for field in ["n_stars", "n_realizations", "draws_per_component"]:
        for invalid in [0, -1, 1.5, True]:
            args = dict(n_stars=10, n_realizations=2, draws_per_component=10)
            args[field] = invalid
            with pytest.raises(ValueError, match=field):
                call(SimpleNamespace(engine="A", is_aniso=False), None, G=1.0, **args)


def test_indexed_linear_quantile_matches_interp_and_both_derivatives():
    from progenax.cluster._radial_cdf import component_linear_quantile

    u = jnp.array([-0.1, 0.0, 0.2, 0.3, 0.7, 1.0, 1.1])
    labels = jnp.array([0, 1, 0, 1, 0, 1, 0])

    def run(theta, indexed):
        grid = theta[0] * jnp.array([0.0, 1.0, 2.0, 3.0])
        cdf = jnp.array([[0.0, theta[1], theta[1], 1.0], [0.0, 0.3, 0.7, 1.0]])
        if indexed:
            return component_linear_quantile(u, labels, grid, cdf)
        return jax.vmap(lambda x, c: jnp.interp(x, cdf[c], grid))(u, labels)

    theta = jnp.array([2.0, 0.4])
    a = lambda t: run(t, True)
    b = lambda t: run(t, False)
    np.testing.assert_allclose(jax.jit(a)(theta), b(theta), rtol=1e-12)
    for ad in [jax.jacfwd, jax.jacrev]:
        np.testing.assert_allclose(ad(a)(theta), ad(b)(theta), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("anisotropic", [False, True])
def test_engine_b_population_moments_against_polynomial_df(anisotropic):
    # Unit oracle: uniform radial probability, constant potential, and constant
    # or linear energy DFs. This is an integration test fixture, not an equilibrium.
    from types import SimpleNamespace

    def evaluate(theta):
        probability, radius, gravity = theta
        r = jnp.linspace(0.0, 1.0, 65)
        energy = jnp.linspace(0.0, 2.0, 17)
        model = SimpleNamespace(
            engine="B",
            m_j=jnp.array([0.5, 1.5]),
            N_frac_j=jnp.array([probability, 1 - probability]),
            _r_grid=r,
            _cdf_j=jnp.stack((r, r)),
            engine_b=SimpleNamespace(
                E_grid=energy,
                f_j_grid=jnp.stack((jnp.ones_like(energy), energy)),
                Psi_grid=jnp.full_like(r, 0.7),
                r_a_j=jnp.array([radius if anisotropic else jnp.inf, jnp.inf]),
                mu=1 / (4 * jnp.pi),
            ),
        )
        result = MultiComponentCluster.estimate_population_moments(
            model,
            jax.random.PRNGKey(0),
            1000,
            gravity,
            n_realizations=1,
            draws_per_component=1,
        )
        return jnp.array(
            [
                result["mean_mass"],
                result["mean_radius_squared"],
                result["mean_speed_squared"],
            ]
        )

    def reference(theta):
        probability, radius, gravity = theta
        pi = jnp.array([probability, 1 - probability])
        mass = jnp.array([0.5, 1.5])
        mean_mass = pi @ mass
        angular = (
            1 / 3 + 2 / 3 * radius * jnp.arctan(1 / radius) if anisotropic else 1.0
        )
        speed = jnp.array([0.7 * 6 / 5 * angular, 0.7 * 6 / 7])
        return jnp.array(
            [mean_mass, 1 / 3, gravity * jnp.sum(pi * speed * (mass + 999 * mean_mass))]
        )

    theta = jnp.array([0.3, 3.0, 1.0])
    np.testing.assert_allclose(
        jax.jit(evaluate)(theta), reference(theta), rtol=1e-12, atol=1e-12
    )
    for transform in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(
            transform(evaluate)(theta),
            transform(reference)(theta),
            rtol=1e-12,
            atol=1e-12,
        )

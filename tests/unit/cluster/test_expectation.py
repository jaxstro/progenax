"""First-order mixture gradients against complete discrete enumeration."""

import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progenax.cluster.multicomponent import MultiComponentCluster
from progenax.cluster.sampling import (
    _expectation_with_score,
    _sample_cluster_arrays_given_components,
)

LABELS = jnp.array(list(itertools.product(range(2), repeat=2)))
PAIRS = jnp.array(list(itertools.product(range(4), repeat=2)))


def _enumerated_score(theta, values_fn):
    values, log_p = values_fn(theta)
    # Freeze only the outer enumeration weights: they represent the sampling
    # distribution under which the estimator's derivative is averaged.
    pair_p = jax.lax.stop_gradient(jnp.exp(jnp.sum(log_p[PAIRS], axis=1)))
    estimates = jax.vmap(_expectation_with_score)(values[PAIRS], log_p[PAIRS])
    return jnp.sum(pair_p * estimates)


@pytest.mark.parametrize("mode", [jax.jacfwd, jax.jacrev])
def test_score_and_pathwise_identity_with_leave_one_out(mode):
    def values_fn(theta):
        pi = jax.nn.softmax(jnp.array([theta[0], 0.0]))
        masses = jnp.array([1.0, jnp.exp(theta[1])])
        total = jnp.sum(masses[LABELS], axis=1)
        # A nonlinear whole-cluster observable makes assignments interact.
        return total**2, jnp.sum(jnp.log(pi)[LABELS], axis=1)

    def exact(theta):
        values, log_p = values_fn(theta)
        return jnp.sum(jnp.exp(log_p) * values)

    theta = jnp.array([0.3, 0.7])
    expected = mode(exact)(theta)
    actual = mode(lambda t: _enumerated_score(t, values_fn))(theta)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    # Omitting the score loses the nonzero probability derivative.
    assert abs(float(expected[0])) > 1.0
    np.testing.assert_allclose(_enumerated_score(theta, values_fn), exact(theta))


@pytest.mark.parametrize("mass_weighted", [False, True])
def test_real_conditional_sampler_expectation_matches_enumeration(mass_weighted):
    model = MultiComponentCluster.from_components(
        W0=5.0,
        g=1.0,
        alpha_j=jnp.array([0.5, 0.5]),
        m_j=jnp.array([1.0, 2.0]),
        w_j=jnp.array([1.0, 0.9]),
        n_ode_points=1500,
        n_grid=600,
        xi_max=800.0,
    )
    key = jax.random.PRNGKey(13)

    def values_fn(theta):
        pi = jax.nn.softmax(jnp.array([theta[0], 0.0]))
        masses = jnp.array([1.0, jnp.exp(theta[1])])
        current = eqx.tree_at(lambda m: (m.N_frac_j, m.m_j), model, (pi, masses))

        def value(labels):
            _, vel, _, _, _ = _sample_cluster_arrays_given_components(
                current, key, labels, 1.0
            )
            weight = current.m_j[labels] if mass_weighted else 1.0
            return jnp.mean(weight * jnp.sum(vel**2, axis=1))

        return jax.vmap(value)(LABELS), jnp.sum(jnp.log(pi + 1e-30)[LABELS], axis=1)

    def exact(theta):
        values, log_p = values_fn(theta)
        return jnp.sum(jnp.exp(log_p) * values)

    theta = jnp.array([0.3, 0.7])
    forward = jax.jit(jax.jacfwd(lambda t: _enumerated_score(t, values_fn)))(theta)
    reverse = jax.jit(jax.jacrev(lambda t: _enumerated_score(t, values_fn)))(theta)
    f = jax.jit(exact)
    fd = jnp.array([(f(theta + d) - f(theta - d)) / 2e-5 for d in jnp.eye(2) * 1e-5])
    np.testing.assert_allclose(forward, reverse, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(forward, fd, rtol=0.003, atol=0.0)

    observable = lambda ic: jnp.mean(ic.masses)
    keys = jax.random.split(key, 4)
    ordinary = jnp.mean(
        jax.vmap(lambda k: observable(model.sample_cluster(k, 2, G=1.0)))(keys)
    )
    estimate = model.estimate_expectation(observable, key, 2, G=1.0, n_realizations=4)
    np.testing.assert_array_equal(estimate, ordinary)

    def public(t):
        current = eqx.tree_at(
            lambda m: (m.N_frac_j, m.m_j),
            model,
            (jax.nn.softmax(jnp.array([t[0], 0.0])), jnp.array([1.0, jnp.exp(t[1])])),
        )
        return current.estimate_expectation(observable, key, 2, G=1.0, n_realizations=4)

    pi = jax.nn.softmax(jnp.array([theta[0], 0.0]))
    counts = jax.vmap(
        lambda k: jnp.sum(
            jax.random.categorical(
                jax.random.split(k, 5)[0], jnp.log(pi + 1e-30), shape=(2,)
            )
        )
    )(keys)
    # The mass/logit score reduces to the unbiased sample variance of heavy
    # counts. The mass-scale derivative is the ordinary pathwise contribution.
    expected = jnp.array(
        [
            -(jnp.exp(theta[1]) - 1.0) * jnp.var(counts, ddof=1) / 2.0,
            jnp.exp(theta[1]) * jnp.mean(counts) / 2.0,
        ]
    )
    np.testing.assert_allclose(
        jax.jacfwd(public)(theta), expected, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        jax.jacrev(public)(theta), expected, rtol=1e-12, atol=1e-12
    )
    with pytest.raises(ValueError, match="at least 2"):
        model.estimate_expectation(observable, key, 2, G=1.0, n_realizations=1)
    with pytest.raises(ValueError, match="scalar"):
        model.estimate_expectation(
            lambda ic: ic.masses, key, 2, G=1.0, n_realizations=2
        )


@pytest.mark.parametrize("n_stars", [1, 2, 4])
def test_mass_squared_and_kinetic_references_against_all_assignments(n_stars):
    # This algebra fixture is not a population/recovery experiment. Full catalogue
    # qualification uses the retained 100000-star model in the linked script.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    from jaxstro.registry import load_source
    from jaxstro.registry.symbolic import build_equation, lambdify_jax
    from qualify_general_observables import mixture_references

    from progenax.registry import registry_root

    source = load_source(registry_root(), "progenaxderivations")
    mass_oracle = lambdify_jax(build_equation(source, "EQ-CATALOGUE-MASS-SQUARED"))
    kinetic_oracle = lambdify_jax(
        build_equation(source, "EQ-CATALOGUE-KINETIC-PER-STAR")
    )

    assignments = jnp.array(list(itertools.product(range(2), repeat=n_stars)))

    def exact(theta):
        pi = jax.nn.softmax(jnp.array([theta[0], 0.0]))
        masses = jnp.array([1.0, jnp.exp(theta[1])])
        q = jnp.array([0.7, jnp.exp(theta[2])])
        sampled_mass = masses[assignments]
        total = sampled_mass.sum(axis=1)
        weights = jnp.prod(pi[assignments], axis=1)
        # Conditional velocities have squared speed = scale * total * q[label].
        kinetic = 0.3 * total * jnp.mean(sampled_mass * q[assignments], axis=1)
        return jnp.array([weights @ total**2, weights @ kinetic])

    def reference(theta):
        pi = jax.nn.softmax(jnp.array([theta[0], 0.0]))
        masses = jnp.array([1.0, jnp.exp(theta[1])])
        q = jnp.array([0.7, jnp.exp(theta[2])])
        return mixture_references(pi, masses, q, 0.3, n_stars)

    def registered(theta):
        pi = jax.nn.softmax(jnp.array([theta[0], 0.0]))
        masses = jnp.array([1.0, jnp.exp(theta[1])])
        q = jnp.array([0.7, jnp.exp(theta[2])])
        mean = pi @ masses
        return jnp.array(
            [
                mass_oracle(N_pop=n_stars, mu1_pop=mean, mu2_pop=pi @ masses**2),
                kinetic_oracle(
                    N_pop=n_stars,
                    mu1_pop=mean,
                    K1_pop=pi @ (masses * q),
                    K2_pop=pi @ (masses**2 * q),
                    A0_pop=0.3,
                ),
            ]
        )

    theta = jnp.array([0.3, 0.7, 0.4])
    for oracle in (reference, registered):
        np.testing.assert_allclose(oracle(theta), exact(theta), rtol=1e-12)
        for mode in (jax.jacfwd, jax.jacrev):
            np.testing.assert_allclose(
                mode(oracle)(theta), mode(exact)(theta), rtol=1e-12, atol=1e-12
            )
    # Removing the mass-label covariance changes B at finite N.
    pi = jax.nn.softmax(jnp.array([theta[0], 0.0]))
    masses = jnp.array([1.0, jnp.exp(theta[1])])
    q = jnp.array([0.7, jnp.exp(theta[2])])
    independent_mass = 0.3 * n_stars * (pi @ masses) * (pi @ (masses * q))
    assert abs(float(reference(theta)[1] - independent_mass)) > 0.01


def test_named_mass_kinetic_contract_and_single_component_homology():
    from types import SimpleNamespace

    call = MultiComponentCluster.estimate_mass_squared_and_kinetic
    with pytest.raises(ValueError, match="Engine A only"):
        call(
            SimpleNamespace(engine="B"),
            None,
            10,
            G=1.0,
            n_realizations=2,
            draws_per_component=10,
        )
    for name in ("n_stars", "n_realizations", "draws_per_component"):
        for invalid in (0, -1, 1.5, True):
            args = dict(n_stars=10, n_realizations=2, draws_per_component=10)
            args[name] = invalid
            with pytest.raises(ValueError, match=name):
                call(SimpleNamespace(engine="A"), None, G=1.0, **args)

    model = MultiComponentCluster.from_components(
        W0=5.0,
        g=1.0,
        alpha_j=jnp.array([1.0]),
        m_j=jnp.array([1.7]),
        w_j=jnp.array([1.0]),
        n_ode_points=1500,
        n_grid=600,
        xi_max=800.0,
    )
    key = jax.random.PRNGKey(31)

    def values(mass):
        current = eqx.tree_at(lambda m: m.m_j, model, jnp.array([mass]))
        result = current.estimate_mass_squared_and_kinetic(
            key,
            1200,
            G=1.0,
            n_realizations=2,
            draws_per_component=200,
        )
        return jnp.array(
            [result["total_mass_squared"], result["twice_kinetic_energy_per_star"]]
        )

    measured = values(1.7)
    baseline = model.estimate_population_moments(
        key,
        1200,
        G=1.0,
        n_realizations=2,
        draws_per_component=200,
    )
    np.testing.assert_allclose(measured[0], (1200 * 1.7) ** 2, rtol=1e-12)
    np.testing.assert_allclose(
        measured[1], 1.7 * baseline["mean_speed_squared"], rtol=1e-12
    )
    # At fixed dimensionless equilibrium, M^2 and 2T/N both scale as mass^2.
    for mode in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(mode(values)(1.7), 2 * measured / 1.7, rtol=1e-12)


@pytest.mark.parametrize("target_stars", [1, 2])
def test_named_kinetic_finite_count_correction_against_label_enumeration(target_stars):
    # Tiny label enumeration verifies the finite-N algebra only. The scientific
    # exposure is the separate 100000-star catalogue/integration campaign.
    from progenax.cluster.sampling import _sample_cluster_arrays_given_components

    model = MultiComponentCluster.from_components(
        W0=5.0,
        g=1.0,
        alpha_j=jnp.array([0.5, 0.5]),
        m_j=jnp.array([1.0, 2.0]),
        w_j=jnp.array([1.0, 0.9]),
        n_ode_points=1500,
        n_grid=600,
        xi_max=800.0,
    )
    key = jax.random.PRNGKey(15)
    draws = 64
    labels = jnp.repeat(jnp.arange(2), draws)
    _, velocity, _, _, _ = _sample_cluster_arrays_given_components(
        model,
        jax.random.split(key, 1)[0],
        labels,
        1.0,
    )
    # Conditional squared speeds per unit sampled total mass; measured from the
    # actual sampler. Enumerating target labels then gives an independent oracle
    # for the weight/covariance reduction, not for continuous sampler accuracy.
    conditional = jnp.mean(jnp.sum(velocity**2, axis=1).reshape(2, draws), axis=1)
    per_total_mass = conditional / (draws * jnp.sum(model.m_j))
    assignments = jnp.array(list(itertools.product(range(2), repeat=target_stars)))
    pi = jax.nn.softmax(jnp.log(model.N_frac_j + 1e-30))
    weights = jnp.prod(pi[assignments], axis=1)
    masses = model.m_j[assignments]
    total = jnp.sum(masses, axis=1)
    expected = weights @ (
        total * jnp.mean(masses * per_total_mass[assignments], axis=1)
    )
    result = model.estimate_mass_squared_and_kinetic(
        key,
        target_stars,
        G=1.0,
        n_realizations=1,
        draws_per_component=draws,
    )
    np.testing.assert_allclose(
        result["total_mass_squared"], weights @ total**2, rtol=1e-12
    )
    np.testing.assert_allclose(
        result["twice_kinetic_energy_per_star"], expected, rtol=1e-12
    )
    independent_mass = (
        target_stars * (pi @ model.m_j) * (pi @ (model.m_j * per_total_mass))
    )
    assert abs(float(expected - independent_mass)) > 1e-5

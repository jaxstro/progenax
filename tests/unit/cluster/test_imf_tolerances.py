"""One tolerance pair must cover the IMF fit, implicit AD and final solve."""

import inspect

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import progenax.cluster.multicomponent as cluster
import progenax.profiles.limepy_multimass as solver
from progenax.imf import Maschberger


def _model(alpha, *, r_a=None, adjoint=None, **kwargs):
    return cluster.MultiComponentCluster.from_imf(
        Maschberger(alpha=alpha, m_min=0.1, m_max=20.0),
        n_comp=2,
        W0=5.0,
        g=1.0,
        delta=0.4,
        m_range=(0.1, 20.0),
        r_a=r_a,
        n_iter=60,
        xi_max=80.0,
        n_ode_points=500,
        n_grid=100,
        adjoint=adjoint,
        **kwargs,
    )


def test_tolerances_are_keyword_only_with_existing_defaults():
    for function in (
        cluster.MultiComponentCluster.from_imf,
        solver.find_alpha_for_masses,
        solver.solve_multimass_limepy,
    ):
        parameters = inspect.signature(function).parameters
        for name, value in (("rtol", 1e-8), ("atol", 1e-10)):
            assert name in parameters
            assert parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
            assert parameters[name].default == value


def test_omitted_and_explicit_defaults_are_identical():
    implicit = _model(jnp.array(2.3))
    explicit = _model(jnp.array(2.3), rtol=1e-8, atol=1e-10)
    for actual, expected in zip(jax.tree.leaves(implicit), jax.tree.leaves(explicit)):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("r_a", [None, 10.0], ids=["isotropic", "anisotropic"])
@pytest.mark.parametrize("mode", ["jit", "forward", "reverse"])
def test_tighter_tolerances_reach_every_poisson_solve(monkeypatch, r_a, mode):
    """Trace every ODE call, including the implicit residual Jacobian."""
    calls = []
    original = solver.solve_multicomponent_limepy

    def recorded(*args, **kwargs):
        calls.append((kwargs.get("rtol"), kwargs.get("atol")))
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, "solve_multicomponent_limepy", recorded)
    monkeypatch.setattr(cluster, "solve_multicomponent_limepy", recorded)
    adjoint = diffrax.ForwardMode() if mode == "forward" else None

    def evaluate(alpha):
        model = _model(
            alpha, r_a=r_a, adjoint=adjoint, fit_tol=1e-9, rtol=1e-10, atol=1e-12
        )
        return jnp.sum(model.alpha_j * model.m_j) + model.mu_tot

    transform = {"jit": lambda f: f, "forward": jax.jacfwd, "reverse": jax.jacrev}[mode]
    value = jax.block_until_ready(jax.jit(transform(evaluate))(jnp.array(2.3)))
    assert jnp.isfinite(value)
    assert calls and set(calls) == {(1e-10, 1e-12)}

"""Finished Engine A models must reject an unresolved tidal boundary.

The two public constructor paths share one physical configuration and two static
domains. Module-level compiled readers are reused across eager/JIT/JVP/VJP checks;
the undersized domain changes no model parameter, tolerance or default.
There are 13 explicit JIT entry-point specializations: six readers for each of
two constructors and one raw diagnostic. Internal solver programs are separate.
"""

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progenax.cluster.multicomponent import MultiComponentCluster
from progenax.imf import Maschberger
from progenax.profiles.limepy_multimass import solve_multicomponent_limepy


def _model(w0, r_c, *, factory, xi_max, adjoint):
    settings = dict(
        W0=w0,
        g=1.0,
        r_c=r_c,
        xi_max=xi_max,
        n_ode_points=500,
        n_grid=100,
        adjoint=adjoint,
    )
    if factory == "components":
        return MultiComponentCluster.from_components(
            alpha_j=jnp.array([0.5, 0.5]),
            w_j=jnp.ones(2),
            m_j=jnp.array([1.0, 2.0]),
            **settings,
        )
    # At delta=0 all component density laws coincide, so the initial target
    # fractions already satisfy the fit. Its residual must not mask this guard.
    return MultiComponentCluster.from_imf(
        Maschberger(alpha=2.3, m_min=0.1, m_max=20.0),
        n_comp=2,
        delta=0.0,
        n_iter=0,
        **settings,
    )


def _unrelated_field(w0, *, factory, adjoint):
    model = _model(w0, 1.0, factory=factory, xi_max=1.0, adjoint=adjoint)
    return model.m_j if factory == "components" else model.r_c


_unrelated_jit = eqx.filter_jit(_unrelated_field)
_unrelated_forward = eqx.filter_jit(eqx.filter_jacfwd(_unrelated_field))
_unrelated_reverse = eqx.filter_jit(eqx.filter_jacrev(_unrelated_field))


@pytest.mark.parametrize("factory", ["components", "imf"])
@pytest.mark.parametrize("mode", ["eager", "jit", "forward", "reverse"])
def test_unresolved_model_cannot_return_unrelated_fields(factory, mode):
    """Catch traced constructors returning masses/scales from an untruncated solve."""
    evaluate = {
        "eager": _unrelated_field,
        "jit": _unrelated_jit,
        "forward": _unrelated_forward,
        "reverse": _unrelated_reverse,
    }[mode]
    adjoint = diffrax.ForwardMode() if mode == "forward" else None
    with pytest.raises((ValueError, RuntimeError), match="truncat|tidal"):
        jax.block_until_ready(
            evaluate(jnp.array(5.0), factory=factory, adjoint=adjoint)
        )


def _resolved_radius(r_c, *, factory, adjoint):
    return _model(
        jnp.array(5.0), r_c, factory=factory, xi_max=30.0, adjoint=adjoint
    ).r_t


_radius_jit = eqx.filter_jit(_resolved_radius)
_radius_forward = eqx.filter_jit(eqx.filter_jacfwd(_resolved_radius))
_radius_reverse = eqx.filter_jit(eqx.filter_jacrev(_resolved_radius))


@pytest.mark.parametrize("factory", ["components", "imf"])
def test_resolved_model_retains_radius_homology_and_both_derivatives(factory):
    """Catch rejecting a resolved model or perturbing r_t=r_c*xi_t and its AD."""
    unit_radius = _radius_jit(jnp.array(1.0), factory=factory, adjoint=None)
    scaled = _radius_jit(jnp.array(1.5), factory=factory, adjoint=None)
    forward = _radius_forward(
        jnp.array(1.5), factory=factory, adjoint=diffrax.ForwardMode()
    )
    reverse = _radius_reverse(jnp.array(1.5), factory=factory, adjoint=None)
    # With fixed dimensionless parameters xi_t is independent of the length
    # scale. These are algebraic homology checks at roundoff, not domain budgets.
    np.testing.assert_allclose(scaled, 1.5 * unit_radius, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(forward, unit_radius, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(reverse, unit_radius, rtol=1e-12, atol=0.0)


@eqx.filter_jit
def _partial_raw_potential(w0):
    return solve_multicomponent_limepy(
        jnp.array([0.5, 0.5]),
        jnp.ones(2),
        w0,
        1.0,
        xi_max=1.0,
        n_points=500,
    )[2]


def test_partial_raw_solver_diagnostic_remains_available():
    """Catch moving the finished-model rejection into raw traced diagnostic solves."""
    raw = _partial_raw_potential(jnp.array(5.0))
    assert bool(jnp.all(jnp.isfinite(raw)))
    assert float(raw[0]) > 0.0
    assert float(raw[-1]) > 0.0

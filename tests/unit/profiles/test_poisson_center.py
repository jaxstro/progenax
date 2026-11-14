"""Regular spherical center regressions, identified by Saloni.

Two compiled programs: pointwise RHS/AD checks and a shared central-solve batch.
No population qualification or higher-order autodifferentiability is asserted.
"""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progenax.profiles._spherical_poisson import (
    poisson_second_derivative,
    regular_center_state,
)
from progenax.profiles.king import _king_poisson_rhs, solve_king_profile
from progenax.profiles.limepy import (
    _limepy_poisson_rhs_aniso,
    _limepy_poisson_rhs_iso,
    solve_limepy_profile,
)
from progenax.profiles.limepy_multimass import solve_multicomponent_limepy
from progenax.profiles.michie import _michie_poisson_rhs, solve_michie_profile


def _rhs_values(w):
    radii = jnp.array([0.0, 1e-8, 1e-6, 1e-4])

    def at_radius(x):
        y = regular_center_state(w, x)
        return jnp.stack(
            (
                _king_poisson_rhs(x, y, (w,))[1],
                _michie_poisson_rhs(x, y, (w, 2.0))[1],
                _limepy_poisson_rhs_iso(x, y, (w, 1.0))[1],
                _limepy_poisson_rhs_aniso(x, y, (w, 1.0, 2.0))[1],
            )
        )

    return jax.vmap(at_radius)(radii)


_rhs_and_derivatives = jax.jit(
    lambda w: (_rhs_values(w), jax.jacfwd(_rhs_values)(w), jax.jacrev(_rhs_values)(w))
)


def test_regular_curvature_and_zero_radius_parameter_derivatives():
    """Catch -9 center guards and inactive divisions that poison JVP/VJP at zero."""
    value, forward, reverse = _rhs_and_derivatives(7.0)
    # At 1e-4 the density correction is O(xi^2), with a coefficient of order 10.
    np.testing.assert_allclose(value, -3.0, atol=3e-7, rtol=0)
    np.testing.assert_allclose(forward[0], 0.0, atol=2e-14, rtol=0)
    # Michie's reverse density ratio accumulates 256 quadrature terms. Use
    # 16*N_u*eps per term in the four-term normalized derivative cancellation.
    # The earlier 2e-14 and single-term roundoff premises failed at 2.41e-13
    # and 1.05e-12. This bound is 3.5 times the measured worst discrepancy.
    roundoff = 64 * 256 * np.finfo(float).eps
    np.testing.assert_allclose(reverse[0], 0.0, atol=roundoff, rtol=0)
    np.testing.assert_allclose(forward, reverse, atol=roundoff, rtol=0)


def test_regular_state_matches_integrated_constant_source():
    """Catch inconsistent W=W0, W'=0 initialization at a nonzero start."""
    x = 0.01
    # Integrating (xi^2 W')'=-9 f0 xi^2 gives this exact constant-source solution.
    np.testing.assert_allclose(
        regular_center_state(7.0, x, 0.8),
        [7.0 - 1.2 * x**2, -2.4 * x],
        atol=2e-15,
        rtol=0,
    )
    assert float(poisson_second_derivative(0.0, 0.0, 0.8)) == pytest.approx(-2.4)


def test_solver_wiring_passes_both_regular_initial_components(monkeypatch):
    """Catch a solver zeroing the initial slope while keeping the correct W value."""
    starts = []

    class CapturedInitialState(Exception):
        pass

    def capture(*args, **kwargs):
        starts.append((float(kwargs["t0"]), np.asarray(kwargs["y0"])))
        raise CapturedInitialState

    # Stop before any numerical integration: the independent regular series
    # fixes both components. W-only integration tests cannot see a 3e-12 slope-
    # initialization bias beneath their O(xi^4) remainder at the saved radius.
    monkeypatch.setattr(diffrax, "diffeqsolve", capture)
    solvers = [
        lambda: solve_king_profile(7.0),
        lambda: solve_michie_profile(7.0, 2.0),
        lambda: solve_limepy_profile(7.0, 1.0),
        lambda: solve_limepy_profile(7.0, 1.0, 2.0),
        lambda: solve_multicomponent_limepy(
            jnp.array([0.3, 0.7]), jnp.ones(2), 7.0, 1.0
        ),
    ]
    for solve in solvers:
        with pytest.raises(CapturedInitialState):
            solve()
    for start, state in starts:
        assert start == 1e-6
        np.testing.assert_allclose(state[0], 7.0 - 1.5 * start**2, atol=8e-15, rtol=0)
        np.testing.assert_allclose(state[1], -3.0 * start, atol=1e-20, rtol=0)


def _central_solves(w):
    kwargs = dict(xi_max=0.01, n_points=6)
    return jnp.stack(
        (
            solve_king_profile(w, **kwargs)[2],
            solve_michie_profile(w, 100.0, **kwargs)[2],
            solve_limepy_profile(w, 1.0, **kwargs)[2],
            solve_limepy_profile(w, 1.0, 100.0, **kwargs)[2],
            solve_multicomponent_limepy(
                jnp.array([0.3, 0.7]), jnp.ones(2), w, 1.0, **kwargs
            )[2],
        )
    )


_solve_centers = jax.jit(jax.vmap(_central_solves))


@pytest.fixture(scope="module")
def central_profiles():
    return np.asarray(_solve_centers(jnp.array([3.0, 7.0, 9.0])))


@pytest.mark.slow(
    reason="Compiles five distinct Poisson solver families, including velocity quadrature."
)
def test_every_solver_starts_on_the_regular_series(central_profiles):
    """Catch a solver retaining [W0,0] even if its pointwise center guard is fixed."""
    expected = np.array([3.0, 7.0, 9.0])[:, None] - 1.5e-12
    np.testing.assert_allclose(
        central_profiles[:, :, 0], np.broadcast_to(expected, (3, 5)), atol=8e-15, rtol=0
    )


@pytest.mark.slow(
    reason="Shares the five-family central-solve compilation with the initialization check."
)
def test_solved_central_quadratic_coefficient(central_profiles):
    """Catch incorrect regular curvature propagated through the actual ODE solves."""
    # At xi=.01 the first omitted coefficient is 27 rho'(W0)/(40 rho(W0));
    # in W0=3..9 its magnitude is below 0.9, so the xi^4 remainder is <9e-9.
    expected = np.array([3.0, 7.0, 9.0])[:, None] - 1.5e-4
    np.testing.assert_allclose(
        central_profiles[:, :, -1],
        np.broadcast_to(expected, (3, 5)),
        atol=1.2e-8,
        rtol=0,
    )

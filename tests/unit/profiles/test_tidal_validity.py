"""Finite-profile factory guards, separate from raw partial-solve diagnostics.

The mocked linear potential has the independent root xi=W0. Five transformed
programs per family share traced endpoint values. Three physical ODE programs
per family cover values and forward/reverse AD on resolved and short domains.
"""

from functools import partial

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progenax.profiles import king, limepy, michie

_FAMILIES = ("king", "michie", "limepy")


def _linear_solve(W0, *args, xi_max, n_points, **kwargs):
    del args, kwargs
    xi = jnp.linspace(0.0, xi_max, n_points)
    raw = W0 - xi
    return xi, jnp.maximum(raw, 0.0), raw


def _make_profile(w, endpoint, *, family, r_c=1.0):
    kwargs = dict(xi_max=endpoint, n_ode_points=5, n_grid=8)
    if family == "king":
        return king.KingProfile.from_W0_rc(w, r_c, **kwargs)
    if family == "michie":
        return michie.MichieProfile.from_W0_rc(w, r_c, 100.0 * r_c, **kwargs)
    return limepy.LIMEPYProfile.from_W0_rc(w, 1.0, r_c, **kwargs)


def _unrelated_field(w, endpoint, *, family):
    # W0 carries no root/CDF dependency. Its constant derivative can discard
    # the entire primal profile, so a value-dependent error wrapper is insufficient.
    return _make_profile(w, endpoint, family=family).W0


_compiled_value = jax.jit(_unrelated_field, static_argnames="family")
_compiled_forward = jax.jit(jax.jacfwd(_unrelated_field), static_argnames="family")
_compiled_reverse = jax.jit(jax.grad(_unrelated_field), static_argnames="family")


@partial(jax.jit, static_argnames="family")
def _compiled_batch(w, endpoints, *, family):
    return jax.vmap(lambda endpoint: _unrelated_field(w, endpoint, family=family))(
        endpoints
    )


@pytest.fixture
def linear_solvers(monkeypatch):
    for module, name in (
        (king, "solve_king_profile"),
        (michie, "solve_michie_profile"),
        (limepy, "solve_limepy_profile"),
    ):
        monkeypatch.setattr(module, name, _linear_solve)


@pytest.mark.parametrize("family", _FAMILIES)
def test_unresolved_factory_rejected_in_eager_jit_and_ad(linear_solvers, family):
    """Catch invalid factories returning when only an unrelated field is consumed."""
    # Same W0=3 model: endpoint 2 has no zero, endpoint 4 brackets xi_t=3.
    functions = (
        _unrelated_field,
        _compiled_value,
        _compiled_forward,
        _compiled_reverse,
    )
    for function, expected in zip(functions, (3.0, 3.0, 1.0, 1.0), strict=True):
        assert float(function(3.0, 4.0, family=family)) == expected
        with pytest.raises(Exception, match="tidal crossing|xi_max"):
            jax.block_until_ready(function(3.0, 2.0, family=family))


@partial(jax.jit, static_argnames="family")
def _root_and_scale_derivatives(w, endpoint, *, family):
    def radius(r_c):
        return _make_profile(w, endpoint, family=family, r_c=r_c).r_t

    return radius(2.0), jax.jacfwd(radius)(2.0), jax.grad(radius)(2.0)


@pytest.mark.parametrize("family", _FAMILIES)
def test_valid_factory_preserves_root_and_unit_homology(linear_solvers, family):
    """Catch a guard altering a valid root or its analytic length-scale derivative."""
    # The supplied potential W0-xi has the exact simple root W0=3. Thus
    # r_t=3*r_c and d(r_t)/d(r_c)=3, independently of profile density arithmetic.
    value, forward, reverse = _root_and_scale_derivatives(3.0, 4.0, family=family)
    np.testing.assert_allclose(
        [value, forward, reverse], [6.0, 3.0, 3.0], atol=2e-14, rtol=0
    )


@pytest.mark.parametrize("family", _FAMILIES)
def test_batched_factory_distinguishes_valid_and_invalid_members(
    linear_solvers, family
):
    """Catch unconditional callback refusal when vmap executes both cond branches."""
    valid = _compiled_batch(3.0, jnp.array([4.0, 5.0]), family=family)
    np.testing.assert_array_equal(valid, [3.0, 3.0])
    with pytest.raises(Exception, match="tidal crossing"):
        jax.block_until_ready(
            _compiled_batch(3.0, jnp.array([4.0, 2.0]), family=family)
        )


_crossing_batch = jax.jit(jax.vmap(king._has_tidal_crossing))


def test_raw_crossing_requires_finite_positive_start_and_adjacent_sign_bracket():
    """Catch nonfinite or initially nonpositive false tidal boundaries."""
    raw = jnp.array(
        [
            [3.0, 1.0, 0.0, -1.0],
            [3.0, 1.0, -0.5, -1.0],
            [3.0, 2.0, 1.0, 0.1],
            [3.0, 1.0, jnp.nan, -1.0],
            [3.0, 1.0, jnp.inf, -1.0],
            [0.0, -1.0, -2.0, -3.0],
            [-1.0, 1.0, 0.0, -1.0],
        ]
    )
    np.testing.assert_array_equal(
        _crossing_batch(raw), [True, True, False, False, False, False, False]
    )


def _make_physical_profile(w, endpoint, *, family, forward=False):
    kwargs = dict(
        xi_max=endpoint,
        n_ode_points=128,
        n_grid=32,
        adjoint=diffrax.ForwardMode() if forward else None,
    )
    if family == "king":
        return king.KingProfile.from_W0_rc(w, 1.0, **kwargs)
    if family == "michie":
        return michie.MichieProfile.from_W0_rc(w, 1.0, 100.0, **kwargs)
    return limepy.LIMEPYProfile.from_W0_rc(w, 1.0, 1.0, **kwargs)


_physical_profile = jax.jit(_make_physical_profile, static_argnames="family")


@partial(jax.jit, static_argnames=("family", "forward"))
def _physical_derivative(w, endpoint, *, family, forward):
    def radius(value):
        return _make_physical_profile(
            value, endpoint, family=family, forward=forward
        ).r_t

    return jax.jacfwd(radius)(w) if forward else jax.grad(radius)(w)


@pytest.mark.parametrize(
    "family,baseline_rt",
    zip(
        _FAMILIES,
        (4.699407361698638, 4.699697460269684, 4.699407361698630),
        strict=True,
    ),
)
def test_physical_factory_retains_valid_solution_and_rejects_short_domain(
    family, baseline_rt
):
    """Catch a real partial ODE solve being exposed as a finite self-consistent model."""
    # Immutable pre-guard values: W0=3, r_c=1, g=1, r_a=100 (Michie), xi_max=20,
    # 128 saved ODE points and 32 CDF points. This is a change regression, not a
    # convergence criterion. Only a few ulps of arithmetic reordering are allowed.
    profile = _physical_profile(3.0, 20.0, family=family)
    np.testing.assert_allclose(
        profile.r_t, baseline_rt, rtol=16 * np.finfo(float).eps, atol=0
    )
    assert 0.0 < float(profile.r_t) < 20.0
    # The regular central expansion is W(0.1) = 3 - 0.015 + O(0.1^4)>0;
    # this domain cannot contain the retained root near xi=4.7.
    with pytest.raises(Exception, match="tidal crossing"):
        jax.block_until_ready(_physical_profile(3.0, 0.1, family=family))


@pytest.mark.parametrize(
    "family,baseline_derivatives",
    zip(
        _FAMILIES,
        (
            (1.7688824714912263, 1.7688824714912288),
            (1.7692594531027983, 1.7692594531027210),
            (1.7688824714911964, 1.7688824714911966),
        ),
        strict=True,
    ),
)
def test_physical_factory_preserves_ad_and_rejects_short_domain_derivatives(
    family, baseline_derivatives
):
    """Catch changed valid ODE gradients or AD admitting a partial-domain model."""
    # Values retained from the pre-guard source at the same physical/grid config
    # above; these fix numerical regression, not derivative convergence accuracy.
    for forward, expected in zip((True, False), baseline_derivatives, strict=True):
        actual = _physical_derivative(3.0, 20.0, family=family, forward=forward)
        np.testing.assert_allclose(
            actual, expected, rtol=16 * np.finfo(float).eps, atol=0
        )
        with pytest.raises(Exception, match="tidal crossing"):
            jax.block_until_ready(
                _physical_derivative(3.0, 0.1, family=family, forward=forward)
            )

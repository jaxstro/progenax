"""Standalone lowered-DF consumers must reject unresolved raw ODE domains.

Four cheap mocked programs per family cover transformed guard retention and
batching. Three physical programs per family cover scale values and both AD
modes, with one shared W0/grid configuration and no velocity draws or tables.
"""

from functools import partial

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progenax.kinematics import king_df, limepy_df, michie_df

_FAMILIES = ("king", "michie", "limepy")


def _linear_solve(W0, *args, xi_max, n_points, **kwargs):
    del args, kwargs
    xi = jnp.linspace(0.0, xi_max, n_points)
    raw = W0 - xi
    return xi, jnp.maximum(raw, 0.0), raw


def _make_df(w, endpoint, *, family, n_points=5, forward=False):
    kwargs = dict(
        xi_max=endpoint,
        n_ode_points=n_points,
        speed_method="quadrature",
        adjoint=diffrax.ForwardMode() if forward else None,
    )
    if family == "king":
        return king_df.KingVelocityDF(w, 1.0, **kwargs)
    if family == "michie":
        return michie_df.MichieVelocityDF(w, 1.0, 100.0, **kwargs)
    return limepy_df.LIMEPYVelocityDF(w, 1.0, 1.0, **kwargs)


def _unrelated_field(w, endpoint, *, family):
    return _make_df(w, endpoint, family=family).W0


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
        (king_df, "solve_king_profile"),
        (michie_df, "solve_michie_profile"),
        (limepy_df, "solve_limepy_profile"),
    ):
        monkeypatch.setattr(module, name, _linear_solve)


@pytest.mark.parametrize("family", _FAMILIES)
def test_standalone_df_rejects_partial_domain_in_eager_jit_and_ad(
    linear_solvers, family
):
    """Catch standalone DF normalization bypassing the profile-factory guard."""
    # The independent linear potential W0-xi has root 3: endpoint 4 brackets
    # it, endpoint 2 cannot. W0's derivative is constant, so all primals may die.
    for function, expected in zip(
        (_unrelated_field, _compiled_value, _compiled_forward, _compiled_reverse),
        (3.0, 3.0, 1.0, 1.0),
        strict=True,
    ):
        assert float(function(3.0, 4.0, family=family)) == expected
        with pytest.raises(Exception, match="tidal crossing"):
            jax.block_until_ready(function(3.0, 2.0, family=family))


@pytest.mark.parametrize("family", _FAMILIES)
def test_standalone_df_batch_rejects_only_invalid_members(linear_solvers, family):
    """Catch a vmapped conditional callback refusing even an all-valid batch."""
    np.testing.assert_array_equal(
        _compiled_batch(3.0, jnp.array([4.0, 5.0]), family=family), [3.0, 3.0]
    )
    with pytest.raises(Exception, match="tidal crossing"):
        jax.block_until_ready(
            _compiled_batch(3.0, jnp.array([4.0, 2.0]), family=family)
        )


def _scale(w, endpoint, *, family, forward=False):
    df = _make_df(w, endpoint, family=family, n_points=128, forward=forward)
    if family == "king":
        return df._sigma(1.0, 1.0)
    if family == "michie":
        return jnp.sqrt(1.0 / (9.0 * df.r_c * df.mu))
    return df._s(1.0, 1.0)


_physical_scale = jax.jit(_scale, static_argnames="family")


@partial(jax.jit, static_argnames=("family", "forward"))
def _physical_scale_derivative(w, endpoint, *, family, forward):
    function = lambda x: _scale(x, endpoint, family=family, forward=forward)
    return jax.jacfwd(function)(w) if forward else jax.grad(function)(w)


@pytest.mark.parametrize(
    "family,baseline",
    zip(
        _FAMILIES,
        (
            (0.5189253539901412, -0.13156725853132323, -0.13156725853132334),
            (0.5189256914091708, -0.13156672906247896, -0.13156672906247566),
            (0.5189253539901401, -0.13156725853132203, -0.13156725853132162),
        ),
        strict=True,
    ),
)
def test_physical_df_preserves_scale_and_ad_and_rejects_short_domain(family, baseline):
    """Catch changed resolved velocity scales or incomplete-domain normalization."""
    # Immutable pre-guard source values: W0=3, r_c=1, Michie r_a=100,
    # LIMEPY g=1 isotropic, xi_max=20, 128 ODE nodes, G=M_total=1.
    # A 16-eps regression bound permits compiler roundoff, not a new scientific
    # accuracy criterion. No velocity populations or speed tables are built.
    value = _physical_scale(3.0, 20.0, family=family)
    np.testing.assert_allclose(
        value, baseline[0], rtol=16 * np.finfo(float).eps, atol=0
    )
    with pytest.raises(Exception, match="tidal crossing"):
        jax.block_until_ready(_physical_scale(3.0, 0.1, family=family))
    for forward, expected in zip((True, False), baseline[1:], strict=True):
        actual = _physical_scale_derivative(3.0, 20.0, family=family, forward=forward)
        np.testing.assert_allclose(
            actual, expected, rtol=16 * np.finfo(float).eps, atol=0
        )
        # The retained root is xi~4.7, while W(.1)=3-.015+O(.1^4)>0.
        with pytest.raises(Exception, match="tidal crossing"):
            jax.block_until_ready(
                _physical_scale_derivative(3.0, 0.1, family=family, forward=forward)
            )

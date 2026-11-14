"""Independent polynomial-DF moments, including an interior clipping boundary."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progenax.kinematics.eddington import _df_speed_second_moment


@pytest.mark.parametrize("linear,cutoff", [(False, 0.0), (True, 0.0), (True, 0.2)])
def test_speed_moment_and_derivatives(linear, cutoff):
    energy = jnp.linspace(0.0, 2.0, 17)

    def evaluate(theta):
        psi, zero = theta
        df = energy - zero if linear else jnp.ones_like(energy)
        return _df_speed_second_moment(psi, energy, df)

    theta = jnp.array([0.7, cutoff])
    coefficient = 6 / 7 if linear else 6 / 5
    expected = coefficient * (theta[0] - cutoff)
    derivative = jnp.array([coefficient, -coefficient if linear else 0.0])
    np.testing.assert_allclose(jax.jit(evaluate)(theta), expected, rtol=1e-12)
    for transform in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(
            transform(evaluate)(theta), derivative, rtol=1e-12, atol=1e-12
        )


def test_registry_power_df_moment_matches_beta_integral():
    import sympy as sp
    from jaxstro.registry import load_source
    from jaxstro.registry.symbolic import build_equation

    from progenax.registry import registry_root

    record = build_equation(
        load_source(registry_root(), "progenaxderivations"), "EQ-POWER-DF-SPEED-MOMENT"
    )
    psi, cutoff, power = (record.symbols[x] for x in ("Psi", "E_cut", "p_df"))
    # y=w^2/[2(Psi-E_cut)] has weight y^(1/2) (1-y)^power.
    ratio = (
        2
        * (psi - cutoff)
        * sp.beta(sp.Rational(5, 2), power + 1)
        / sp.beta(sp.Rational(3, 2), power + 1)
    )
    # Apply Gamma(z+1)=z Gamma(z); SymPy leaves this half-integer shift unevaluated.
    expanded = sp.expand_func(ratio).subs(
        sp.gamma(power + sp.Rational(7, 2)),
        (power + sp.Rational(5, 2)) * sp.gamma(power + sp.Rational(5, 2)),
    )
    assert sp.simplify(expanded - record.expression) == 0

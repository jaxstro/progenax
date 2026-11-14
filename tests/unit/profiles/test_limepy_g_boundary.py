"""One-sided Woolley-limit derivatives from the defining gamma integral."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
from scipy.special import dawsn, exp1

from progenax.profiles.limepy import _aniso_density_scalar, lowered_exponential


def test_lowered_exponential_one_sided_index_derivative():
    W = jnp.array([1e-8, 0.01, 1.0, 10.0])
    da = jax.jit(jax.jacfwd(lambda a: lowered_exponential(a, W)))(0.0)
    expected = -np.exp(np.array(W)) * exp1(np.array(W))
    np.testing.assert_allclose(da, expected, rtol=1e-12)
    dw = jax.jit(jax.jacrev(lambda w: lowered_exponential(0.0, w)))(W)
    np.testing.assert_allclose(jnp.diag(dw), jnp.exp(W), rtol=1e-12)
    assert float(jax.grad(lambda a: lowered_exponential(a, jnp.array(0.0)))(0.0)) == 0.0


def test_density_one_sided_index_derivative_against_adaptive_integral():
    for W, p in [(1e-3, 0.0), (1.0, 1.0), (10.0, 100.0)]:
        b = p * np.sqrt(W)

        def integrand(t):
            z = W * (1 - t * t)
            a = b * t
            angle = 2 * dawsn(a) / a if a else 2.0
            return (2 * W) ** 1.5 * t * t * (-np.exp(z) * exp1(z)) * angle

        ref, _ = quad(
            integrand,
            0,
            1,
            points=[a / b for a in [1, 4, 16, 64] if b > a],
            epsabs=1e-12,
            epsrel=1e-10,
            limit=300,
        )
        ad = jax.jit(
            jax.grad(lambda g: _aniso_density_scalar(jnp.array(W), jnp.array(p), g))
        )(0.0)
        # Existing derivative budget, unchanged.
        np.testing.assert_allclose(ad, ref, rtol=0.003, atol=0.0)


def test_speed_table_one_sided_g_derivative_at_escape_limit():
    from scipy.special import betaincinv

    from progenax.profiles.limepy_tables import AnisoSpeedCDFTable

    W, q, h = 1e-8, 0.8, 1e-5

    def draw(g):
        return AnisoSpeedCDFTable.build(11.0, 200.0, g).inverse(W, 0.0, q)

    value, derivative = jax.jit(jax.value_and_grad(draw))(0.0)
    # Independent beta-limit derivative, normalized by sqrt(2W).
    expected = (
        np.sqrt(betaincinv(1.5, 1 + h, q)) - np.sqrt(betaincinv(1.5, 1.0, q))
    ) / h
    np.testing.assert_allclose(
        derivative / np.sqrt(2 * W), expected, rtol=0.003, atol=0.0
    )
    finite_difference = (jax.jit(draw)(h) - value) / h
    np.testing.assert_allclose(derivative, finite_difference, rtol=0.003, atol=0.0)


def test_density_table_index_derivative_at_deep_potential_edge():
    from progenax.profiles.limepy_tables import AnisoDensityTable

    W, p = 7.0 * (4.0 / 2.2) ** 0.6, 100.0

    def evaluate(g):
        return AnisoDensityTable.build(W, 200.0, g).evaluate(W, p)

    ad = jax.jit(jax.grad(evaluate))(0.0)
    b = p * np.sqrt(W)

    def integrand(t):
        z = W * (1 - t * t)
        a = b * t
        angle = 2 * dawsn(a) / a if a else 2.0
        return -((2 * W) ** 1.5) * t * t * np.exp(z) * exp1(z) * angle

    expected, _ = quad(
        integrand,
        0,
        1,
        points=[a / b for a in [1, 4, 16, 64] if b > a],
        epsabs=1e-20,
        epsrel=1e-11,
        limit=300,
    )
    np.testing.assert_allclose(ad, expected, rtol=0.003, atol=0.0)


def test_registered_boundary_derivative_identity():
    import sympy as sp
    from jaxstro.registry import load_source
    from jaxstro.registry.symbolic import build_equation

    from progenax.registry import registry_root

    record = build_equation(
        load_source(registry_root(), "progenaxderivations"),
        "EQ-LIMEPY-G0-RIGHT-DERIVATIVE",
    )
    W = record.symbols["W_limepy"]
    # Fundamental theorem of calculus, independently of the production E1 helper.
    assert sp.simplify(sp.diff(record.expression, W) - record.expression - 1 / W) == 0
    for value in [0.01, 1.0, 10.0]:
        expected = float(record.expression.subs(W, value).evalf())
        actual = float(
            jax.grad(lambda g: lowered_exponential(g, jnp.array(value)))(0.0)
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

"""Algebraic eccentricity and period boundaries; four compiled configurations."""

import jax
import jax.numpy as jnp
import numpy as np

from progenax.binaries import MoeEccentricity, SanaOBPeriod, ThermalEccentricity


def _period_law(power, uniform, query):
    law = SanaOBPeriod(power=power)
    return jnp.array([law.ppf(uniform), law.cdf(query), law.pdf(query)])


_period_values = jax.jit(jax.vmap(_period_law, in_axes=(0, None, None)))
_period_derivatives = jax.jit(
    lambda power, uniform, query: (
        jax.jacfwd(_period_law)(power, uniform, query),
        jax.jacrev(_period_law)(power, uniform, query),
    )
)
_thermal_cdf = jax.jit(jax.vmap(jax.value_and_grad(ThermalEccentricity().cdf)))
_moe_pdf = jax.jit(
    jax.value_and_grad(
        lambda period: jnp.sum(
            MoeEccentricity().pdf(jnp.array([0.1]), period[None], jnp.array([10.0]))
        )
    )
)


def test_thermal_cdf_has_zero_probability_and_slope_below_support():
    """Catch squaring a negative eccentricity into a positive cumulative probability."""
    eccentricity = jnp.array([-1.0, -0.1, 0.5, 2.0])
    value, derivative = _thermal_cdf(eccentricity)
    cap = 0.99
    np.testing.assert_allclose(
        value, [0.0, 0.0, 0.25 / cap**2, 1.0], rtol=2e-15, atol=0
    )
    np.testing.assert_allclose(
        derivative, [0.0, 0.0, 1 / cap**2, 0.0], rtol=2e-15, atol=0
    )


def test_sana_logarithmic_limit_retains_power_derivatives():
    """Catch power-independent limit branches; check the declared x=log10(P) law independently."""
    uniform, query_x = 0.37, 0.7
    query = 10**query_x
    a, b = 0.15, 3.5
    length = np.log(b / a)
    x = a * np.exp(uniform * length)
    period = 10**x
    cdf = np.log(query_x / a) / length
    pdf = 1 / (query_x * length * query * np.log(10.0))
    expected = np.array([period, cdf, pdf])
    expected_derivative = np.array(
        [
            np.log(10.0) * period * x * uniform * (1 - uniform) * length**2 / 2,
            -length * cdf * (1 - cdf) / 2,
            pdf * (np.log(query_x) - (np.log(a) + np.log(b)) / 2),
        ]
    )
    power = jnp.array([-1.0, -1.0 - 1e-8, -1.0 + 1e-8, -0.55, 0.0])
    values = np.asarray(_period_values(power, uniform, query))
    forward, reverse = _period_derivatives(-1.0, uniform, query)
    np.testing.assert_allclose(values[0], expected, rtol=3e-14, atol=0)
    np.testing.assert_allclose(forward, expected_derivative, rtol=3e-13, atol=0)
    np.testing.assert_allclose(reverse, expected_derivative, rtol=3e-13, atol=0)
    # An independent Gaussian integral in x tests the law near its log limit
    # and at the empirical default, rather than comparing two API methods.
    nodes, weights = np.polynomial.legendre.leggauss(128)
    for row, exponent in zip(values, np.asarray(power)):

        def integral(upper):
            xx = a + (upper - a) * (nodes + 1) / 2
            return (upper - a) / 2 * np.sum(weights * xx**exponent)

        norm = integral(b)
        np.testing.assert_allclose(row[1], integral(query_x) / norm, rtol=3e-14, atol=0)
        np.testing.assert_allclose(
            integral(np.log10(row[0])) / norm, uniform, rtol=3e-14, atol=0
        )
        np.testing.assert_allclose(
            row[2], query_x**exponent / (norm * query * np.log(10)), rtol=3e-14, atol=0
        )


def test_circular_moe_density_has_zero_regular_parameter_derivative():
    """Catch inactive zero-cap powers/divisions poisoning AD of the zero continuous density."""
    value, derivative = _moe_pdf(jnp.asarray(1.0))
    # At one day the physical prescription has an atom at e=0. Its
    # continuous density at e=.1 is identically zero in a neighborhood.
    assert float(value) == 0.0
    assert float(derivative) == 0.0

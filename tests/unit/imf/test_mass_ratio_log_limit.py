"""Power-law logarithmic-limit regression; three shared compiled programs.

Gamma sweeps are traced inputs of one vmapped value/forward/reverse configuration.
"""

import jax
import jax.numpy as jnp
import numpy as np

from progenax.imf.binary import PowerLawMassRatio, TwinPeakedMassRatio


def _law(gamma, u, q, lower):
    law = PowerLawMassRatio(gamma=gamma, q_min=lower)
    twin = TwinPeakedMassRatio(gamma=gamma, q_min=lower, f_twin=0.2)
    return jnp.array([law._norm(), law.cdf(q), law.ppf(u), law.pdf(q), twin.cdf(q)])


_values = jax.jit(jax.vmap(_law, in_axes=(0, None, None, None)))
_forward = jax.jit(jax.vmap(jax.jacfwd(_law), in_axes=(0, None, None, None)))
_reverse = jax.jit(jax.vmap(jax.jacrev(_law), in_axes=(0, None, None, None)))


def test_logarithmic_powerlaw_values_and_parameter_derivatives():
    """Catch gamma-independent log branches and cancellation near gamma=-1, including twins."""
    gammas = jnp.array([-1.0, -1.0 - 1e-8, -1.0 + 1e-8, -0.1, 1.0])
    u, query, lower = 0.37, 0.4, 0.1
    values = _values(gammas, u, query, lower)
    forward = _forward(gammas, u, query, lower)
    reverse = _reverse(gammas, u, query, lower)
    length = jnp.log(1 / lower)
    probability = jnp.log(query / lower) / length
    quantile = lower * jnp.exp(u * length)
    density = 1 / (query * length)
    expected = jnp.array([length, probability, quantile, density])
    cdf_slope = -0.5 * length * probability * (1 - probability)
    expected_slope = jnp.array(
        [
            -0.5 * length**2,
            cdf_slope,
            0.5 * quantile * u * (1 - u) * length**2,
            density * (jnp.log(query) + length / 2),
            0.8 * cdf_slope,
        ]
    )
    # Closed logarithmic-limit arithmetic is accurate to a few float64 ulps.
    assert jnp.allclose(values[0, :4], expected, rtol=3e-14, atol=0.0)
    assert jnp.allclose(forward[0], expected_slope, rtol=3e-13, atol=0.0)
    assert jnp.allclose(reverse[0], expected_slope, rtol=3e-13, atol=0.0)

    # Independent Gaussian quadrature checks the forward law at all gamma
    # values. Integrating to each output quantile must recover the supplied u.
    nodes, weights = np.polynomial.legendre.leggauss(128)
    for i, gamma in enumerate(np.asarray(gammas)):

        def integral(upper):
            q_nodes = lower + (upper - lower) * (nodes + 1) / 2
            return (upper - lower) / 2 * np.sum(weights * q_nodes**gamma)

        norm = integral(1.0)
        assert np.isclose(values[i, 0], norm, rtol=3e-14, atol=0.0)
        assert np.isclose(values[i, 1], integral(query) / norm, rtol=3e-14, atol=0.0)
        assert np.isclose(integral(float(values[i, 2])) / norm, u, rtol=3e-14, atol=0.0)
        assert np.isclose(values[i, 3], query**gamma / norm, rtol=3e-14, atol=0.0)
    assert jnp.allclose(forward, reverse, rtol=3e-13, atol=0.0)


def test_normalizable_zero_cutoff_powerlaw_is_preserved():
    """Catch a logarithm of zero introduced while repairing the positive-cutoff limit."""
    gammas = jnp.array([-0.5, 0.0, 1.0, 2.0, 3.0])
    u, query = 0.37, 0.4
    values = _values(gammas, u, query, 0.0)
    a = gammas + 1
    expected = jnp.stack([1 / a, query**a, u ** (1 / a), a * query**gammas], axis=1)
    expected_derivative = jnp.stack(
        [
            -1 / a**2,
            query**a * jnp.log(query),
            -(u ** (1 / a)) * jnp.log(u) / a**2,
            query**gammas * (1 + a * jnp.log(query)),
        ],
        axis=1,
    )
    assert jnp.allclose(values[:, :4], expected, rtol=3e-14, atol=0.0)
    assert jnp.allclose(
        _forward(gammas, u, query, 0.0)[:, :4],
        expected_derivative,
        rtol=3e-13,
        atol=0.0,
    )
    assert jnp.allclose(
        _reverse(gammas, u, query, 0.0)[:, :4],
        expected_derivative,
        rtol=3e-13,
        atol=0.0,
    )

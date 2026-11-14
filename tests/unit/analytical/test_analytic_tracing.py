"""Analytic homology regressions for traced initial-condition metadata.

Three compiled programs cover the value, forward Jacobian, and reverse Jacobian.
"""

import jax
import jax.numpy as jnp

from progenax.analytical import (
    harmonic_oscillator,
    three_body_figure_eight,
    two_body_kepler,
)


def _metadata(theta):
    a, scale, omega = theta
    binary = two_body_kepler(1.0, 0.5, a, 1.0)
    figure = three_body_figure_eight(scale=scale, mass=2.0, G=3.0)
    harmonic = harmonic_oscillator(amplitude=2.0, omega=omega, mass=3.0)
    return jnp.array(
        [binary.period, binary.energy, figure.period, harmonic.period, harmonic.energy]
    )


_metadata_jit = jax.jit(_metadata)
_metadata_forward = jax.jit(jax.jacfwd(_metadata))
_metadata_reverse = jax.jit(jax.jacrev(_metadata))


def test_analytic_metadata_retains_traced_periods_and_energies():
    """Catch float coercions that block JIT/AD; check Newtonian homology derivatives."""
    theta = jnp.array([1.7, 1.2, 0.8])
    values = _metadata_jit(theta)
    # Kepler T scales as a^(3/2), E=-G m1 m2/(2a); the figure-eight
    # Newtonian homology has T proportional to scale^(3/2). Harmonic
    # motion has T=2 pi/omega and E=m omega^2 A^2/2.
    a, scale, omega = theta
    expected_values = jnp.array(
        [
            2 * jnp.pi * jnp.sqrt(a**3 / 1.5),
            -0.25 / a,
            6.32591398 * jnp.sqrt(scale**3 / 6),
            2 * jnp.pi / omega,
            6 * omega**2,
        ]
    )
    expected_jacobian = jnp.zeros((5, 3))
    expected_jacobian = expected_jacobian.at[0, 0].set(1.5 * values[0] / a)
    expected_jacobian = expected_jacobian.at[1, 0].set(0.25 / a**2)
    expected_jacobian = expected_jacobian.at[2, 1].set(1.5 * values[2] / scale)
    expected_jacobian = expected_jacobian.at[3, 2].set(-2 * jnp.pi / omega**2)
    expected_jacobian = expected_jacobian.at[4, 2].set(12 * omega)
    # Closed-form arithmetic has errors of a few ulps at these O(1) scales.
    assert jnp.allclose(values, expected_values, rtol=2e-14, atol=0.0)
    assert jnp.allclose(
        _metadata_forward(theta), expected_jacobian, rtol=2e-14, atol=0.0
    )
    assert jnp.allclose(
        _metadata_reverse(theta), expected_jacobian, rtol=2e-14, atol=0.0
    )

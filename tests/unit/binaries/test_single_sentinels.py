"""A single-star slot must ignore every unused orbital element.

One compiled value-and-gradient program covers all six invalid orbit sentinels.
"""

import jax
import jax.numpy as jnp

from progenax.binaries import resolve_binary_components


def _single_observable(com, orbit):
    rb = resolve_binary_components(
        com[None, :3],
        com[None, 3:],
        jnp.array([2.0]),
        jnp.array([0.0]),
        jnp.array([False]),
        *(orbit[i : i + 1] for i in range(6)),
        G=1.0,
    )
    return jnp.sum(rb.positions[0]) + jnp.sum(rb.velocities[0])


_single_value_grad = jax.jit(jax.value_and_grad(_single_observable, argnums=(0, 1)))


def test_single_ignores_nan_orbits_in_values_and_derivatives():
    """Catch unused NaN angles/anomalies contaminating a real single via zero times NaN."""
    com = jnp.array([1.0, -2.0, 3.0, 0.3, -0.2, 0.1])
    value, (d_com, d_orbit) = _single_value_grad(com, jnp.full(6, jnp.nan))
    # A single's primary is exactly its supplied COM, with no orbit dependence.
    assert jnp.allclose(value, jnp.sum(com), rtol=2e-15, atol=0.0)
    assert jnp.array_equal(d_com, jnp.ones(6))
    assert jnp.array_equal(d_orbit, jnp.zeros(6))

"""Per-channel scalar reductions for the params->IC audit direction (design D4).

Each reduction maps an initial-conditions array to a single scalar so the audit
engine can take ``jax.grad`` of (entry point -> reduction) along one physical
channel. The ``+ 1e-30`` guards inside the ``sqrt`` keep both the value and its
gradient finite when the input is exactly zero (a bare ``sqrt(0)`` has a NaN
derivative).
"""

import jax.numpy as jnp


def mean_radius(positions):  # (N,3) -> ()
    return jnp.mean(jnp.sqrt(jnp.sum(positions**2, axis=-1) + 1e-30))


def mean_speed(velocities):  # (N,3) -> ()
    return jnp.mean(jnp.sqrt(jnp.sum(velocities**2, axis=-1) + 1e-30))


def mean_mass(masses):  # (N,) -> ()
    return jnp.mean(masses)


def identity_sum(x):  # params->summary: reduce a vector statistic
    return jnp.sum(x)

"""Isotropic orbital-orientation sampling for binary populations.

References:
    Binney & Tremaine (2008) "Galactic Dynamics" §3.1 — isotropic orientation.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


def sample_isotropic_orientations(
    key: PRNGKeyArray,
    n: int,
) -> Tuple[Float[Array, "n"], Float[Array, "n"], Float[Array, "n"], Float[Array, "n"]]:
    """Sample isotropic orbital orientations.

    For randomly oriented orbits in 3D space:
    - cos(i) ~ U(-1, 1)  =>  i = arccos(u) where u ~ U(-1, 1)
    - Ω ~ U(0, 2π)  (longitude of ascending node)
    - ω ~ U(0, 2π)  (argument of periapsis)
    - M₀ ~ U(0, 2π)  (mean anomaly at epoch)

    Args:
        key: JAX random key
        n: Number of orientations to sample

    Returns:
        Tuple of (inclination, Omega, omega, M_anom) arrays, each shape (n,)
        - inclination: [0, π] radians
        - Omega: [0, 2π) radians
        - omega: [0, 2π) radians
        - M_anom: [0, 2π) radians

    Reference:
        Binney & Tremaine (2008) "Galactic Dynamics" Section 3.1
    """
    key1, key2, key3, key4 = jax.random.split(key, 4)

    # Inclination: cos(i) ~ U(-1, 1) for isotropic
    cos_i = jax.random.uniform(key1, (n,), minval=-1.0, maxval=1.0)
    inclination = jnp.arccos(cos_i)

    # Other angles: uniform on [0, 2π)
    Omega = jax.random.uniform(key2, (n,), minval=0.0, maxval=2.0 * jnp.pi)
    omega = jax.random.uniform(key3, (n,), minval=0.0, maxval=2.0 * jnp.pi)
    M_anom = jax.random.uniform(key4, (n,), minval=0.0, maxval=2.0 * jnp.pi)

    return inclination, Omega, omega, M_anom


__all__ = ["sample_isotropic_orientations"]

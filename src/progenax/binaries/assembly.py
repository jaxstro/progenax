"""Binary -> spatial-IC connector: resolve binary COMs into two components.

`resolve_binary_components` takes N *system* COMs (positions+velocities) plus their
masses, binary flags, and sampled orbital elements, and emits a **masked, fixed-shape
2N representation**: each system maps to two interleaved slots (slot 2i = primary /
single, slot 2i+1 = secondary or a zero-mass ghost), with an `is_real[2N]` mask. This
is jit/vmap/grad-safe (no data-dependent shapes). The orchestrator
(`builders.build_binary_cluster`) eagerly compacts it to the real-particle `ICResult`.

Each binary's COM is preserved exactly: with the barycentric split
m1·δr1 + m2·δr2 = 0 (from `KeplerElements.to_binary_state`), placing
r = X_com + δr leaves m1·r1 + m2·r2 = (m1+m2)·X_com — the cluster phase space is
untouched, only internal structure is resolved.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from .kepler import KeplerElements


class ResolvedBinaries(NamedTuple):
    """Masked fixed-shape (2N) particle set from resolving N systems.

    Attributes:
        positions: (2N, 3) component positions [length units].
        velocities: (2N, 3) component velocities [velocity units].
        masses: (2N,) component masses [M_sun] (ghosts = 0).
        is_real: (2N,) bool — True for real particles (all primaries + binary
            secondaries); False for single-star ghost secondaries.
        primordial_system_id: (2N,) int — slots 2i, 2i+1 both = i.
        is_primordial_secondary: (2N,) bool — True on the odd (secondary) slots.
    """

    positions: Float[Array, "M 3"]
    velocities: Float[Array, "M 3"]
    masses: Float[Array, "M"]
    is_real: Bool[Array, "M"]
    primordial_system_id: Int[Array, "M"]
    is_primordial_secondary: Bool[Array, "M"]


def _interleave(primary: Array, secondary: Array) -> Array:
    """Interleave per-system primary/secondary arrays into 2N slots [p0,s0,p1,s1,…]."""
    return jnp.stack([primary, secondary], axis=1).reshape((-1,) + primary.shape[1:])


def resolve_binary_components(
    com_pos: Float[Array, "N 3"],
    com_vel: Float[Array, "N 3"],
    m1: Float[Array, "N"],
    m2: Float[Array, "N"],
    is_binary: Bool[Array, "N"],
    a: Float[Array, "N"],
    e: Float[Array, "N"],
    inc: Float[Array, "N"],
    Omega: Float[Array, "N"],
    omega: Float[Array, "N"],
    M_anom: Float[Array, "N"],
    *,
    G: float,
) -> ResolvedBinaries:
    """Resolve N system COMs into a masked 2N component set (see module docstring).

    Args:
        com_pos, com_vel: (N, 3) system center-of-mass positions/velocities.
        m1, m2: (N,) primary / secondary masses (m2 = 0 for singles).
        is_binary: (N,) bool — True for binary systems.
        a, e, inc, Omega, omega, M_anom: (N,) Keplerian elements of the relative
            orbit (ignored for singles; sanitized internally so grads stay finite).
        G: gravitational constant (REQUIRED).

    Returns:
        ResolvedBinaries (2N slots + is_real mask + primordial provenance).
    """
    N = m1.shape[0]

    # Sanitize single-star slots so to_binary_state never sees garbage elements
    # (both jnp.where branches are traced — unsanitized NaN/0 a would poison grads).
    a_safe = jnp.where(is_binary, a, 1.0)
    e_safe = jnp.where(is_binary, e, 0.0)
    m2_safe = jnp.where(is_binary, m2, 0.0)  # singles: m2=0 -> δr1=0 -> primary at COM

    def _one(a_i, e_i, inc_i, Om_i, om_i, M_i, m1_i, m2_i):
        elements = KeplerElements(a=a_i, e=e_i, i=inc_i, Omega=Om_i, omega=om_i, M0=M_i)
        bs = elements.to_binary_state(m1=m1_i, m2=m2_i, G=G)
        return bs.r1, bs.v1, bs.r2, bs.v2

    dr1, dv1, dr2, dv2 = jax.vmap(_one)(
        a_safe, e_safe, inc, Omega, omega, M_anom, m1, m2_safe
    )

    # Place components around the COM (singles: δr1=δv1=0 since m2_safe=0).
    prim_pos = com_pos + dr1
    prim_vel = com_vel + dv1
    sec_pos = com_pos + dr2
    sec_vel = com_vel + dv2

    positions = _interleave(prim_pos, sec_pos)
    velocities = _interleave(prim_vel, sec_vel)
    masses = _interleave(m1, m2_safe)

    is_real = _interleave(jnp.ones(N, dtype=bool), is_binary)
    primordial_system_id = jnp.repeat(jnp.arange(N), 2)
    is_primordial_secondary = (jnp.arange(2 * N) % 2) == 1

    return ResolvedBinaries(
        positions=positions,
        velocities=velocities,
        masses=masses,
        is_real=is_real,
        primordial_system_id=primordial_system_id,
        is_primordial_secondary=is_primordial_secondary,
    )


__all__ = ["ResolvedBinaries", "resolve_binary_components"]

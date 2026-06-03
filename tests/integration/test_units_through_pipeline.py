"""Unit-system propagation through build_spatial_ic (audit finding C1).

C1: build_spatial_ic dropped ``G`` when sampling velocities
(``builders.py`` called ``velocity_df.sample_velocities(positions, masses, key_vel)``
with no ``G``), so velocities silently used ``DEFAULT_UNITS.G`` (STELLAR) regardless
of the ``G`` the caller threaded in. The global virial rescale (``Q`` set) masked the
error because it only fixes the *total* energy ratio; with ``Q=None`` the bug is fully
exposed: the realized virial ratio collapses to ``0.5 * G_STELLAR / G_used``.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxstro.units import PLANETARY, STELLAR
from progenax import (
    PlummerProfile,
    PlummerVelocityDF,
    build_spatial_ic,
    compute_kinetic_energy,
    compute_potential_energy,
)


def test_build_spatial_ic_respects_G_without_virial_rescale():
    """With Q=None (no rescale), realized Q=T/|V| must be ~0.5 for the G actually used.

    Before the C1 fix this returned Q ~= 0.5 * G_STELLAR/G_PLANETARY ~= 5.7e-5,
    because the velocities were sampled with STELLAR G while the potential used
    PLANETARY G.
    """
    m = jnp.ones(800)
    key = jax.random.PRNGKey(0)

    ic = build_spatial_ic(
        PlummerProfile(1.0), m, PlummerVelocityDF(1.0),
        key=key, G=PLANETARY.G, Q=None,
    )

    T = compute_kinetic_energy(ic.velocities, m)
    V = compute_potential_energy(ic.positions, m, G=PLANETARY.G)
    Q_realized = float(T / jnp.abs(V))

    assert abs(Q_realized - 0.5) < 0.05, (
        f"realized Q={Q_realized:.6g} (expected ~0.5); a value near "
        f"{0.5 * STELLAR.G / PLANETARY.G:.3g} indicates the dropped-G bug (C1)"
    )


def test_virial_rescale_masks_dropped_G():
    """Regression guard / documentation: with Q=0.5 the global rescale yields Q~0.5
    in any unit system. This path passed both before and after the C1 fix and is
    why the bug went unnoticed."""
    m = jnp.ones(800)
    key = jax.random.PRNGKey(0)

    ic = build_spatial_ic(
        PlummerProfile(1.0), m, PlummerVelocityDF(1.0),
        key=key, G=PLANETARY.G, Q=0.5,
    )

    T = compute_kinetic_energy(ic.velocities, m)
    V = compute_potential_energy(ic.positions, m, G=PLANETARY.G)
    Q_realized = float(T / jnp.abs(V))

    assert abs(Q_realized - 0.5) < 0.05, f"rescaled Q={Q_realized:.6g} (expected ~0.5)"

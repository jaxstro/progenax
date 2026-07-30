"""System-level birth provenance for cataloged primordial-binary ICs.

The catalog records how a population was generated.  It does not describe the
current bound population after dynamical evolution and does not select a
downstream numerical integration method.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


class PrimordialSystemCatalog(eqx.Module):
    """Immutable birth record for sampled single and binary systems.

    Orbital fields are finite zeroes for single systems.  Component particle
    IDs refer to stable logical birth IDs, never array-row positions; the
    inactive secondary of a single has ID ``-1``.
    """

    system_ids: Int[Array, "S"]
    is_binary: Bool[Array, "S"]
    component_particle_ids: Int[Array, "S 2"]
    component_active: Bool[Array, "S 2"]
    semimajor_axes: Float[Array, "S"]
    eccentricities: Float[Array, "S"]
    inclinations: Float[Array, "S"]
    longitudes_ascending_node: Float[Array, "S"]
    arguments_periapsis: Float[Array, "S"]
    mean_anomalies: Float[Array, "S"]
    periapsis_contact_margins: Float[Array, "S"]


class CatalogedBinaryClusterIC(eqx.Module):
    """Particle state plus immutable primordial-system provenance.

    ``compact=True`` products contain real particles only and have an all-true
    ``is_real`` mask.  ``compact=False`` products retain the interleaved ghost
    secondary of every single system for fixed-shape differentiation.

    Stellar radii remain in solar radii, matching :class:`ICResult`.  Contact
    margins in ``primordial_systems`` are stored in the position unit used to
    construct the cluster.
    """

    positions: Float[Array, "M 3"]
    velocities: Float[Array, "M 3"]
    masses: Float[Array, "M"]
    stellar_radii: Float[Array, "M"]
    ids: Int[Array, "M"]
    is_real: Bool[Array, "M"]
    primordial_system_id: Int[Array, "M"]
    is_primordial_secondary: Bool[Array, "M"]
    primordial_systems: PrimordialSystemCatalog


def periapsis_contact_margin(
    semimajor_axis: Float[Array, "S"],
    eccentricity: Float[Array, "S"],
    primary_radius: Float[Array, "S"],
    secondary_radius: Float[Array, "S"],
) -> Float[Array, "S"]:
    """Return ``a(1-e) - (R1+R2)`` in one common length unit."""

    return semimajor_axis * (1.0 - eccentricity) - (primary_radius + secondary_radius)


def validate_primordial_system_catalog(
    catalog: PrimordialSystemCatalog,
    particle_ids: Int[Array, "M"],
    is_real: Bool[Array, "M"],
) -> None:
    """Validate identity and finite-value invariants at an eager API boundary."""

    system_count = catalog.system_ids.shape[0]
    one_dimensional = (
        catalog.is_binary,
        catalog.semimajor_axes,
        catalog.eccentricities,
        catalog.inclinations,
        catalog.longitudes_ascending_node,
        catalog.arguments_periapsis,
        catalog.mean_anomalies,
        catalog.periapsis_contact_margins,
    )
    if catalog.system_ids.ndim != 1 or any(
        value.shape != (system_count,) for value in one_dimensional
    ):
        raise ValueError("catalog system fields must all have shape (S,).")
    if catalog.component_particle_ids.shape != (system_count, 2):
        raise ValueError("component_particle_ids must have shape (S, 2).")
    if catalog.component_active.shape != (system_count, 2):
        raise ValueError("component_active must have shape (S, 2).")
    if particle_ids.ndim != 1 or is_real.shape != particle_ids.shape:
        raise ValueError(
            "particle_ids and is_real must be matching one-dimensional arrays."
        )
    if not jnp.issubdtype(catalog.system_ids.dtype, jnp.integer):
        raise ValueError("system IDs must have integer dtype.")
    if not jnp.issubdtype(catalog.component_particle_ids.dtype, jnp.integer):
        raise ValueError("component particle IDs must have integer dtype.")
    if not jnp.issubdtype(particle_ids.dtype, jnp.integer):
        raise ValueError("particle IDs must have integer dtype.")
    if not bool(jnp.all(catalog.component_active[:, 0])):
        raise ValueError("every primordial system must have an active primary.")
    if not bool(jnp.array_equal(catalog.component_active[:, 1], catalog.is_binary)):
        raise ValueError("secondary activity must agree with is_binary.")
    if not bool(
        jnp.all(
            jnp.where(
                catalog.is_binary,
                catalog.component_particle_ids[:, 1] >= 0,
                catalog.component_particle_ids[:, 1] == -1,
            )
        )
    ):
        raise ValueError(
            "secondary IDs must be active for binaries and -1 for singles."
        )

    active_component_ids = catalog.component_particle_ids[catalog.component_active]
    real_particle_ids = particle_ids[is_real]
    if int(jnp.unique(active_component_ids).shape[0]) != active_component_ids.shape[0]:
        raise ValueError("active catalog component IDs must be unique.")
    if int(jnp.unique(real_particle_ids).shape[0]) != real_particle_ids.shape[0]:
        raise ValueError("real particle IDs must be unique.")
    if not bool(jnp.all(jnp.isin(active_component_ids, real_particle_ids))):
        raise ValueError(
            "every active catalog component must reference a real particle ID."
        )
    if active_component_ids.shape[0] != real_particle_ids.shape[0]:
        raise ValueError("every real particle must appear exactly once in the catalog.")

    finite_fields = one_dimensional[1:]
    if not all(bool(jnp.all(jnp.isfinite(value))) for value in finite_fields):
        raise ValueError("catalog orbital and contact fields must be finite.")
    single = ~catalog.is_binary
    if not all(bool(jnp.all(value[single] == 0.0)) for value in finite_fields):
        raise ValueError("single systems must use finite zero orbital/contact fields.")


__all__ = [
    "CatalogedBinaryClusterIC",
    "PrimordialSystemCatalog",
    "periapsis_contact_margin",
    "validate_primordial_system_catalog",
]

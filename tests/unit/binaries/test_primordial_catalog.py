"""System-level birth-provenance contracts for cataloged binary ICs."""

import jax
import jax.numpy as jnp
import pytest

from progenax.binaries import (
    CatalogedBinaryClusterIC,
    PrimordialSystemCatalog,
    periapsis_contact_margin,
    validate_primordial_system_catalog,
)


def _catalog(*, contact_margins=jnp.array([0.0, -0.1])):
    return PrimordialSystemCatalog(
        system_ids=jnp.array([0, 1], dtype=jnp.int64),
        is_binary=jnp.array([False, True]),
        component_particle_ids=jnp.array([[0, -1], [2, 3]], dtype=jnp.int64),
        component_active=jnp.array([[True, False], [True, True]]),
        semimajor_axes=jnp.array([0.0, 2.0]),
        eccentricities=jnp.array([0.0, 0.25]),
        inclinations=jnp.array([0.0, 0.4]),
        longitudes_ascending_node=jnp.array([0.0, 0.5]),
        arguments_periapsis=jnp.array([0.0, 0.6]),
        mean_anomalies=jnp.array([0.0, 0.7]),
        periapsis_contact_margins=contact_margins,
    )


def test_contact_margin_uses_periapsis_and_both_radii():
    margin = periapsis_contact_margin(
        jnp.array([2.0]), jnp.array([0.25]), jnp.array([0.2]), jnp.array([0.1])
    )
    assert jnp.array_equal(margin, jnp.array([1.2]))


def test_single_uses_finite_zero_orbit_storage_and_absent_secondary():
    catalog = _catalog()
    assert catalog.component_particle_ids.tolist()[0] == [0, -1]
    assert catalog.component_active.tolist()[0] == [True, False]
    for field in (
        catalog.semimajor_axes,
        catalog.eccentricities,
        catalog.inclinations,
        catalog.longitudes_ascending_node,
        catalog.arguments_periapsis,
        catalog.mean_anomalies,
        catalog.periapsis_contact_margins,
    ):
        assert jnp.isfinite(field[0])
        assert float(field[0]) == 0.0


def test_validator_accepts_negative_binary_contact_margin():
    validate_primordial_system_catalog(
        _catalog(),
        jnp.array([0, 1, 2, 3], dtype=jnp.int64),
        jnp.array([True, False, True, True]),
    )


@pytest.mark.parametrize(
    ("catalog", "match"),
    [
        (
            _catalog().__class__(
                **{
                    **_catalog().__dict__,
                    "component_particle_ids": jnp.array(
                        [[0, -1], [2, 2]], dtype=jnp.int64
                    ),
                }
            ),
            "unique",
        ),
        (
            _catalog().__class__(
                **{
                    **_catalog().__dict__,
                    "component_active": jnp.array([[True, True], [True, True]]),
                }
            ),
            "activity",
        ),
    ],
)
def test_validator_rejects_identity_or_activity_inconsistency(catalog, match):
    with pytest.raises(ValueError, match=match):
        validate_primordial_system_catalog(
            catalog,
            jnp.array([0, 1, 2, 3], dtype=jnp.int64),
            jnp.array([True, False, True, True]),
        )


def test_cataloged_result_is_a_jax_pytree():
    catalog = _catalog(contact_margins=jnp.array([0.0, 1.2]))
    result = CatalogedBinaryClusterIC(
        positions=jnp.zeros((3, 3)),
        velocities=jnp.zeros((3, 3)),
        masses=jnp.ones(3),
        stellar_radii=jnp.ones(3),
        ids=jnp.array([0, 2, 3], dtype=jnp.int64),
        is_real=jnp.ones(3, dtype=bool),
        primordial_system_id=jnp.array([0, 1, 1], dtype=jnp.int64),
        is_primordial_secondary=jnp.array([False, False, True]),
        primordial_systems=catalog,
    )
    leaves, treedef = jax.tree.flatten(result)
    assert leaves
    rebuilt = jax.tree.unflatten(treedef, leaves)
    assert jnp.array_equal(rebuilt.ids, result.ids)

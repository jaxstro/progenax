"""Binary-builder unit boundaries; deterministic contracts, not population gates."""

import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxstro.constants import RSUN_CM
from jaxstro.units import CGS, PLANETARY, STELLAR

from progenax import (
    IndependentCompanions,
    LogUniformPeriod,
    MoeCompanions,
    PlummerProfile,
    PlummerVelocityDF,
    PowerLawIMF,
    Stars,
    Systems,
    ThermalEccentricity,
    TotalMass,
    build_binary_cluster,
    build_cataloged_binary_cluster,
)
from progenax.binaries import compute_period, validate_primordial_system_catalog
from progenax.imf.binary import ConstantBinaryFraction, FlatMassRatio


def _build(
    units, *, cataloged, compact=True, Q=None, target=None, moe=False, alpha=2.3
):
    length = STELLAR.convert_length(1.0, to=units)
    companion = (
        MoeCompanions()
        if moe
        else IndependentCompanions(
            binary_fraction=ConstantBinaryFraction(0.5),
            q_distribution=FlatMassRatio(q_min=0.2),
            period_distribution=LogUniformPeriod(log_P_min=2.0, log_P_max=4.0),
            eccentricity_distribution=ThermalEccentricity(),
        )
    )
    builder = build_cataloged_binary_cluster if cataloged else build_binary_cluster
    return builder(
        PlummerProfile(r_h=length),
        PlummerVelocityDF(r_h=length),
        PowerLawIMF(exponents=[-alpha], breakpoints=[], m_min=0.5, m_max=4.0),
        companion,
        Systems(16) if target is None else target,
        jax.random.PRNGKey(0),
        units=units,
        compact=compact,
        Q=Q,
    )


def _assert_same_physical_state(actual, reference, units, tolerances):
    for field, convert in (
        ("masses", units.convert_mass),
        ("positions", units.convert_length),
        ("velocities", units.convert_velocity),
    ):
        # Ghost Cartesian slots are finite sanitization values, not particles;
        # their arbitrary unit-sized internal orbit has no physical covariance.
        mask = (
            actual.is_real
            if hasattr(actual, "is_real") and field != "masses"
            else slice(None)
        )
        np.testing.assert_allclose(
            convert(getattr(actual, field)[mask], to=STELLAR),
            getattr(reference, field)[mask],
            rtol=tolerances.EXACT,
            atol=0.0,
        )
    for field in ("primordial_system_id", "is_primordial_secondary", "is_real", "ids"):
        if hasattr(actual, field):
            np.testing.assert_array_equal(
                getattr(actual, field), getattr(reference, field)
            )
    if hasattr(actual, "stellar_radii"):
        assert bool(jnp.all(jnp.isfinite(actual.stellar_radii)))
        np.testing.assert_allclose(
            actual.stellar_radii,
            reference.stellar_radii,
            rtol=tolerances.EXACT,
            atol=0.0,
        )
    if hasattr(actual, "primordial_systems"):
        a = actual.primordial_systems
        b = reference.primordial_systems
        for field in a.__dataclass_fields__:
            value = getattr(a, field)
            if field in ("semimajor_axes", "periapsis_contact_margins"):
                value = units.convert_length(value, to=STELLAR)
                np.testing.assert_allclose(
                    value, getattr(b, field), rtol=tolerances.EXACT, atol=0.0
                )
            else:
                np.testing.assert_array_equal(value, getattr(b, field))
        validate_primordial_system_catalog(a, actual.ids, actual.is_real)


@pytest.mark.parametrize("units", [PLANETARY, CGS])
@pytest.mark.parametrize("cataloged", [False, True])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("Q", [None, 0.5])
def test_binary_same_physical_inputs_across_units(
    units, cataloged, compact, Q, tolerances
):
    kwargs = dict(cataloged=cataloged, compact=compact, Q=Q)
    _assert_same_physical_state(
        _build(units, **kwargs), _build(STELLAR, **kwargs), units, tolerances
    )


@pytest.mark.parametrize("cataloged", [False, True])
@pytest.mark.parametrize("target", [Stars(16), TotalMass(20.0)])
def test_binary_eager_budgets_stay_in_solar_masses(cataloged, target, tolerances):
    kwargs = dict(cataloged=cataloged, target=target)
    _assert_same_physical_state(
        _build(CGS, **kwargs), _build(STELLAR, **kwargs), CGS, tolerances
    )


@pytest.mark.parametrize("cataloged", [False, True])
def test_moe_empirical_mass_inputs_remain_solar(cataloged, tolerances):
    kwargs = dict(cataloged=cataloged, compact=False, moe=True)
    _assert_same_physical_state(
        _build(CGS, **kwargs), _build(STELLAR, **kwargs), CGS, tolerances
    )


def test_cataloged_cgs_kepler_period_and_contact_use_native_arrays(tolerances):
    result = _build(CGS, cataloged=True, compact=False)
    reference = _build(STELLAR, cataloged=True, compact=False)
    catalog = result.primordial_systems
    binary = catalog.is_binary
    masses = result.masses.reshape((-1, 2)).sum(axis=1)
    periods = compute_period(catalog.semimajor_axes[binary], masses[binary], G=CGS.G)
    ref_catalog = reference.primordial_systems
    ref_mass = reference.masses.reshape((-1, 2)).sum(axis=1)
    ref_periods = compute_period(
        ref_catalog.semimajor_axes[binary], ref_mass[binary], G=STELLAR.G
    )
    np.testing.assert_allclose(
        CGS.convert_time(periods, to=STELLAR),
        ref_periods,
        rtol=tolerances.EXACT,
        atol=0.0,
    )
    _assert_same_physical_state(result, reference, CGS, tolerances)
    radius_native = (
        result.stellar_radii.reshape((-1, 2)) * RSUN_CM / CGS.length_scale_cgs
    )
    expected = catalog.semimajor_axes * (
        1.0 - catalog.eccentricities
    ) - radius_native.sum(axis=1)
    np.testing.assert_allclose(
        catalog.periapsis_contact_margins[binary],
        expected[binary],
        rtol=tolerances.EXACT,
        atol=0.0,
    )


@pytest.mark.parametrize("units", [STELLAR, CGS])
def test_binary_imf_derivatives_preserve_mass_units(units, tolerances):
    def summary(alpha, unit):
        result = _build(unit, cataloged=True, compact=False, alpha=alpha)
        return jnp.array(
            [
                jnp.sum(unit.convert_mass(result.masses, to=STELLAR)),
                jnp.sum(result.stellar_radii),
                jnp.sum(
                    unit.convert_length(
                        result.primordial_systems.semimajor_axes, to=STELLAR
                    )
                ),
            ]
        )

    reference = lambda alpha: summary(alpha, STELLAR)
    actual = lambda alpha: summary(alpha, units)
    np.testing.assert_allclose(
        jax.jit(actual)(2.3), reference(2.3), rtol=tolerances.EXACT, atol=0.0
    )
    np.testing.assert_allclose(
        jax.jacfwd(actual)(2.3),
        jax.jacrev(actual)(2.3),
        rtol=tolerances.EXACT,
        atol=0.0,
    )
    for derivative in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(
            derivative(actual)(2.3),
            derivative(reference)(2.3),
            rtol=tolerances.EXACT,
            atol=0.0,
        )


def test_binary_recipe_primary_imf_matches_declared_slope(tolerances):
    """The public alpha=2.3 recipe must decrease as m^-2.3; no cluster draw."""
    recipe = (
        Path(__file__).resolve().parents[3]
        / "docs/website/40-howto/add-binary-population.md"
    )
    code = recipe.read_text().split("```python\n", 1)[1].split("```", 1)[0]
    assignment = next(
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "primary_imf"
            for target in node.targets
        )
    )
    imf = eval(
        compile(ast.Expression(assignment.value), str(recipe), "eval"),
        {"PowerLawIMF": PowerLawIMF},
    )
    alpha = 2.3  # Declared recipe contract; PDF density is per unit mass.
    np.testing.assert_allclose(
        imf.logpdf(10.0) - imf.logpdf(1.0),
        -alpha * np.log(10.0),
        rtol=tolerances.EXACT,
        atol=0.0,
    )
    integral = lambda e: (imf.m_max ** (e + 1) - imf.m_min ** (e + 1)) / (e + 1)
    np.testing.assert_allclose(
        imf.mean_mass(),
        integral(1 - alpha) / integral(-alpha),
        rtol=tolerances.EXACT,
        atol=0.0,
    )

"""Single-population unit conversions, separate from equilibrium qualification."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxstro.units import CGS, PLANETARY, STELLAR

from progenax import (
    EFFProfile,
    KingProfile,
    LIMEPYProfile,
    MichieProfile,
    PlummerProfile,
    PlummerVelocityDF,
    build_cluster,
    build_spatial_ic,
    zams_radius,
)
from progenax.imf import PowerLawIMF

_KEY = jax.random.PRNGKey(0)
_N = 16
_MASSES = jnp.linspace(0.5, 4.0, _N)


def _assert_same_physical_ic(actual, reference, units, tolerances):
    # Reuse the suite's exact-formula budget for unit round trips, not a new
    # stochastic or scientific-precision acceptance threshold.
    for field, convert in (
        ("positions", units.convert_length),
        ("velocities", units.convert_velocity),
        ("masses", units.convert_mass),
    ):
        np.testing.assert_allclose(
            convert(getattr(actual, field), to=STELLAR),
            getattr(reference, field),
            rtol=tolerances.EXACT,
            atol=0.0,
        )
    assert bool(jnp.all(jnp.isfinite(actual.stellar_radii)))
    np.testing.assert_allclose(
        actual.stellar_radii,
        reference.stellar_radii,
        rtol=tolerances.EXACT,
        atol=0.0,
    )


@pytest.mark.parametrize("units", [STELLAR, PLANETARY, CGS])
@pytest.mark.parametrize("source", ["supplied", "equal", "imf"])
@pytest.mark.parametrize("Q", [None, 0.5])
def test_cluster_mass_sources_preserve_physical_units(units, source, Q, tolerances):
    kwargs = {"key": _KEY, "Q": Q}
    if source == "supplied":
        kwargs["masses"] = _MASSES
    else:
        kwargs["n"] = _N
        if source == "imf":
            kwargs["imf"] = PowerLawIMF.salpeter(m_min=0.5, m_max=4.0)
    reference = build_cluster(PlummerProfile(r_h=1.0), **kwargs)
    if source == "supplied":
        kwargs["masses"] = STELLAR.convert_mass(_MASSES, to=units)
    actual = build_cluster(
        PlummerProfile(r_h=STELLAR.convert_length(1.0, to=units)), units=units, **kwargs
    )
    if source == "supplied":
        np.testing.assert_array_equal(actual.masses, kwargs["masses"])
    _assert_same_physical_ic(actual, reference, units, tolerances)


@pytest.mark.parametrize("modifier", ["tidal", "rotation", "both"])
def test_cgs_modifier_labels_use_solar_mass(modifier, tolerances):
    kwargs = {"key": _KEY, "masses": _MASSES, "Q": 0.5}
    if modifier in ("tidal", "both"):
        kwargs.update(tidal_radius=1.5, revirialize=True)
    if modifier in ("rotation", "both"):
        kwargs["rotation"] = 0.2
    reference = build_cluster(PlummerProfile(r_h=1.0), **kwargs)
    kwargs["masses"] = STELLAR.convert_mass(_MASSES, to=CGS)
    if "tidal_radius" in kwargs:
        kwargs["tidal_radius"] = STELLAR.convert_length(kwargs["tidal_radius"], to=CGS)
    if "rotation" in kwargs:
        kwargs["rotation"] *= CGS.time_scale_cgs / STELLAR.time_scale_cgs
    actual = build_cluster(
        PlummerProfile(r_h=STELLAR.convert_length(1.0, to=CGS)), units=CGS, **kwargs
    )
    _assert_same_physical_ic(actual, reference, CGS, tolerances)
    if modifier in ("tidal", "both"):
        assert bool(jnp.any(actual.masses == 0.0))
        np.testing.assert_array_equal(actual.stellar_radii[actual.masses == 0.0], 0.0)


def test_low_level_cgs_builder_accepts_explicit_mass_scale(tolerances):
    r_h = STELLAR.convert_length(1.0, to=CGS)
    actual = build_spatial_ic(
        PlummerProfile(r_h=r_h),
        STELLAR.convert_mass(_MASSES, to=CGS),
        PlummerVelocityDF(r_h=r_h),
        _KEY,
        G=CGS.G,
        Q=None,
        mass_scale_to_msun=CGS.mass_scale_cgs / STELLAR.mass_scale_cgs,
    )
    reference = build_cluster(PlummerProfile(r_h=1.0), masses=_MASSES, key=_KEY, Q=None)
    _assert_same_physical_ic(actual, reference, CGS, tolerances)


@pytest.mark.parametrize("units", [STELLAR, PLANETARY, CGS])
def test_mass_conversion_jit_and_first_derivatives(units, tolerances):
    def observables(scale):
        ic = build_cluster(
            PlummerProfile(r_h=STELLAR.convert_length(1.0, to=units)),
            masses=STELLAR.convert_mass(scale * _MASSES, to=units),
            units=units,
            key=_KEY,
            Q=None,
        )
        v = units.convert_velocity(ic.velocities, to=STELLAR)
        return jnp.array([jnp.sum(ic.stellar_radii), jnp.sum(v**2)])

    scale = 1.25
    value = jax.jit(observables)(scale)
    # Plummer positions are independent of mass, so v^2 scales linearly when
    # every stellar mass is multiplied by the same factor (Q=None).
    expected = jnp.array(
        [jax.grad(lambda s: jnp.sum(zams_radius(s * _MASSES)))(scale), value[1] / scale]
    )
    assert bool(jnp.all(jnp.isfinite(value)))
    for derivative in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(
            jax.jit(derivative(observables))(scale),
            expected,
            rtol=tolerances.EXACT,
            atol=0.0,
        )


@pytest.mark.parametrize("family", ["king", "eff", "michie", "limepy"])
def test_other_profile_families_preserve_cgs_units(family, tolerances):
    def profile(units):
        length = STELLAR.convert_length(1.0, to=units)
        if family == "king":
            return KingProfile.from_W0_rc(W0=5.0, r_c=length)
        if family == "eff":
            return EFFProfile(a=length, gamma=3.0, r_t=10.0 * length)
        if family == "michie":
            return MichieProfile.from_W0_rc(W0=5.0, r_c=length, r_a=8.0 * length)
        return LIMEPYProfile.from_W0_rc(W0=5.0, g=1.0, r_c=length)

    reference = build_cluster(profile(STELLAR), masses=_MASSES, key=_KEY, Q=None)
    actual = build_cluster(
        profile(CGS),
        masses=STELLAR.convert_mass(_MASSES, to=CGS),
        units=CGS,
        key=_KEY,
        Q=None,
    )
    _assert_same_physical_ic(actual, reference, CGS, tolerances)


def test_eff_cgs_stellar_labels_and_native_masses(tolerances):
    # Check the stellar-label conversion independently of velocity covariance.
    length = STELLAR.convert_length(1.0, to=CGS)
    native_masses = STELLAR.convert_mass(_MASSES, to=CGS)
    actual = build_cluster(
        EFFProfile(a=length, gamma=3.0, r_t=10.0 * length),
        masses=native_masses,
        units=CGS,
        key=_KEY,
        Q=None,
    )
    np.testing.assert_array_equal(actual.masses, native_masses)
    assert bool(jnp.all(jnp.isfinite(actual.stellar_radii)))
    np.testing.assert_allclose(
        actual.stellar_radii, zams_radius(_MASSES), rtol=tolerances.EXACT, atol=0.0
    )


def test_generated_imf_conversion_preserves_first_derivatives(tolerances):
    def radii(alpha, units):
        imf = PowerLawIMF([alpha], [], m_min=0.5, m_max=4.0)
        ic = build_cluster(
            PlummerProfile(r_h=STELLAR.convert_length(1.0, to=units)),
            n=_N,
            imf=imf,
            units=units,
            key=_KEY,
            Q=None,
        )
        return jnp.sum(ic.stellar_radii)

    key_imf, _ = jax.random.split(_KEY)

    def direct_labels(alpha):
        return jnp.sum(
            zams_radius(
                PowerLawIMF([alpha], [], m_min=0.5, m_max=4.0).sample(key_imf, _N)
            )
        )

    alpha = 2.35
    expected_value = direct_labels(alpha)
    for units in (STELLAR, CGS):
        actual = lambda a: radii(a, units)
        np.testing.assert_allclose(
            jax.jit(actual)(alpha), expected_value, rtol=tolerances.EXACT, atol=0.0
        )
        for derivative in (jax.jacfwd, jax.jacrev):
            np.testing.assert_allclose(
                jax.jit(derivative(actual))(alpha),
                derivative(direct_labels)(alpha),
                rtol=tolerances.EXACT,
                atol=0.0,
            )

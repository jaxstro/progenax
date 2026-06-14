import jax
import jax.numpy as jnp
import pytest
from jaxstro.units import STELLAR
from progenax import (
    PlummerProfile, EFFProfile, KingProfile, MichieProfile, LIMEPYProfile,
    PlummerVelocityDF, EFFVelocityDF, KingVelocityDF, MichieVelocityDF, LIMEPYVelocityDF,
)
from progenax.builders_cluster import matched_velocity_df


def test_matched_plummer_scale_matched():
    p = PlummerProfile(r_h=2.3)
    df = matched_velocity_df(p)
    assert isinstance(df, PlummerVelocityDF)
    assert float(df.r_h) == float(p.r_h)        # scale never desyncs


def test_matched_eff_scale_matched():
    p = EFFProfile(a=1.4, gamma=3.2, r_t=12.0)
    df = matched_velocity_df(p)
    assert isinstance(df, EFFVelocityDF)
    assert float(df.a) == float(p.a)
    assert float(df.gamma) == float(p.gamma)
    assert float(df.r_t) == float(p.r_t)


def test_matched_king_scale_matched():
    p = KingProfile.from_W0_rc(W0=7.0, r_c=1.3)
    df = matched_velocity_df(p)
    assert isinstance(df, KingVelocityDF)
    assert float(df.W0) == float(p.W0)
    assert float(df.r_c) == float(p.r_c)


def test_matched_michie_scale_matched():
    p = MichieProfile.from_W0_rc(W0=7.0, r_c=1.0, r_a=8.0)
    df = matched_velocity_df(p)
    assert isinstance(df, MichieVelocityDF)
    assert float(df.W0) == float(p.W0)
    assert float(df.r_a) == float(p.r_a)


def test_matched_limepy_isotropic_passes_none_r_a():
    p = LIMEPYProfile.from_W0_rc(W0=5.0, g=1.5, r_c=1.0)   # isotropic -> profile.r_a = inf
    df = matched_velocity_df(p)
    assert isinstance(df, LIMEPYVelocityDF)
    assert not bool(jnp.isfinite(df.r_a))  # isotropic DF stores r_a = inf


def test_matched_limepy_anisotropic_threads_r_a():
    p = LIMEPYProfile.from_W0_rc(W0=5.0, g=1.0, r_c=1.0, r_a=6.0, xi_max=800.0)
    df = matched_velocity_df(p)
    assert isinstance(df, LIMEPYVelocityDF)
    assert float(df.r_a) == pytest.approx(6.0)


def test_anisotropy_radius_valid_for_plummer_eff():
    df = matched_velocity_df(PlummerProfile(r_h=1.0), anisotropy_radius=0.9)
    assert df.anisotropy_radius is not None
    df2 = matched_velocity_df(EFFProfile(a=1.0, gamma=5.0, r_t=10.0), anisotropy_radius=3.0)
    assert df2.anisotropy_radius is not None


@pytest.mark.parametrize("profile", [
    KingProfile.from_W0_rc(W0=7.0, r_c=1.0),
    MichieProfile.from_W0_rc(W0=7.0, r_c=1.0, r_a=8.0),
    LIMEPYProfile.from_W0_rc(W0=5.0, g=1.0, r_c=1.0),
])
def test_anisotropy_radius_errors_for_non_om_models(profile):
    with pytest.raises(ValueError, match="anisotropy_radius"):
        matched_velocity_df(profile, anisotropy_radius=2.0)


def test_unknown_profile_type_errors():
    class Bogus:  # not a known profile
        pass
    with pytest.raises(TypeError, match="matched_velocity_df"):
        matched_velocity_df(Bogus())


# ===========================================================================
# Batch 2: build_cluster base — mass-spec resolution + bit-identical purity
# ===========================================================================
from progenax import build_spatial_ic, PowerLawIMF, DEFAULT_UNITS
from progenax.builders_cluster import build_cluster

_K = jax.random.PRNGKey(0)
_M = jnp.ones(200)


def _assert_ic_equal(a, b):
    for field in ("positions", "velocities", "masses", "stellar_radii"):
        assert bool(jnp.all(getattr(a, field) == getattr(b, field))), f"{field} differs"


def test_build_cluster_is_bit_identical_to_manual_base_case():
    # The linchpin: build_cluster(profile, masses, key) MUST equal the manual
    # build_spatial_ic composition exactly (pure sugar, no physics drift).
    p = PlummerProfile(r_h=1.0)
    ic = build_cluster(p, masses=_M, key=_K)                       # units=None -> STELLAR
    df = matched_velocity_df(p)
    manual = build_spatial_ic(p, _M, df, _K, G=STELLAR.G, Q=0.5)
    _assert_ic_equal(ic, manual)


def test_units_none_resolves_to_default_stellar():
    p = PlummerProfile(r_h=1.0)
    ic_none = build_cluster(p, masses=_M, key=_K, units=None)
    ic_stellar = build_cluster(p, masses=_M, key=_K, units=STELLAR)
    _assert_ic_equal(ic_none, ic_stellar)
    assert DEFAULT_UNITS is STELLAR  # documents the resolution target


def test_mass_spec_n_only_is_equal_one_msun():
    ic = build_cluster(PlummerProfile(r_h=1.0), n=128, key=_K)
    assert ic.masses.shape == (128,)
    assert bool(jnp.allclose(ic.masses, 1.0))


def test_mass_spec_n_plus_imf_samples():
    imf = PowerLawIMF.kroupa()
    ic = build_cluster(PlummerProfile(r_h=1.0), n=256, imf=imf, key=_K)
    assert ic.masses.shape == (256,)
    assert float(jnp.std(ic.masses)) > 0.0          # not all equal -> IMF actually sampled


def test_mass_spec_masses_array_used_verbatim():
    m = jnp.linspace(0.5, 3.0, 64)
    ic = build_cluster(PlummerProfile(r_h=1.0), masses=m, key=_K)
    assert bool(jnp.all(ic.masses == m))


def test_mass_spec_error_both_masses_and_n():
    with pytest.raises(ValueError, match="masses.*or.*n|exactly one"):
        build_cluster(PlummerProfile(r_h=1.0), masses=_M, n=10, key=_K)


def test_mass_spec_error_neither_masses_nor_n():
    with pytest.raises(ValueError, match="masses.*or.*n|exactly one"):
        build_cluster(PlummerProfile(r_h=1.0), key=_K)


def test_mass_spec_error_imf_without_n():
    with pytest.raises(ValueError, match="imf.*requires.*n|n.*imf"):
        build_cluster(PlummerProfile(r_h=1.0), masses=_M, imf=PowerLawIMF.kroupa(), key=_K)

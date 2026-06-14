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

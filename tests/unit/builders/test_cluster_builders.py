import jax
import jax.numpy as jnp
import pytest
from jaxstro.units import STELLAR

from progenax import (
    EFFProfile,
    EFFVelocityDF,
    KingProfile,
    KingVelocityDF,
    LIMEPYProfile,
    LIMEPYVelocityDF,
    MichieProfile,
    MichieVelocityDF,
    PlummerProfile,
    PlummerVelocityDF,
)
from progenax.builders_cluster import matched_velocity_df


def test_matched_plummer_scale_matched():
    p = PlummerProfile(r_h=2.3)
    df = matched_velocity_df(p)
    assert isinstance(df, PlummerVelocityDF)
    assert float(df.r_h) == float(p.r_h)  # scale never desyncs


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
    p = LIMEPYProfile.from_W0_rc(
        W0=5.0, g=1.5, r_c=1.0
    )  # isotropic -> profile.r_a = inf
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
    df2 = matched_velocity_df(
        EFFProfile(a=1.0, gamma=5.0, r_t=10.0), anisotropy_radius=3.0
    )
    assert df2.anisotropy_radius is not None


@pytest.mark.parametrize(
    "profile",
    [
        KingProfile.from_W0_rc(W0=7.0, r_c=1.0),
        MichieProfile.from_W0_rc(W0=7.0, r_c=1.0, r_a=8.0),
        LIMEPYProfile.from_W0_rc(W0=5.0, g=1.0, r_c=1.0),
    ],
)
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
from progenax import DEFAULT_UNITS, PowerLawIMF, build_spatial_ic
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
    ic = build_cluster(p, masses=_M, key=_K)  # units=None -> STELLAR
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
    assert float(jnp.std(ic.masses)) > 0.0  # not all equal -> IMF actually sampled


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
        build_cluster(
            PlummerProfile(r_h=1.0), masses=_M, imf=PowerLawIMF.kroupa(), key=_K
        )


# ===========================================================================
# Batch 3: Modifiers — RotationSpec, anisotropy/tidal/rotation, revirialize,
#          and the Q=None faithful-unscaled-equilibrium path (Anna-ratified A/B).
# ===========================================================================
from progenax.builders_cluster import RotationSpec


def test_anisotropy_threads_into_df_radial_bias():
    # OM anisotropy -> radial velocity bias (beta(r) > 0). Compare radial vs tangential
    # velocity variance at large radius for an anisotropic vs isotropic Plummer.
    p = PlummerProfile(r_h=1.0)
    iso = build_cluster(p, masses=_M, key=_K)
    ani = build_cluster(p, masses=_M, key=_K, anisotropy_radius=0.7)
    # Anisotropic build must differ from isotropic (threading actually happened).
    assert not bool(jnp.allclose(iso.velocities, ani.velocities))


def test_anisotropy_unsupported_model_errors():
    with pytest.raises(ValueError, match="anisotropy_radius"):
        build_cluster(
            KingProfile.from_W0_rc(W0=7.0, r_c=1.0),
            masses=_M,
            key=_K,
            anisotropy_radius=2.0,
        )


def test_tidal_zeroes_outer_masses():
    ic = build_cluster(PlummerProfile(r_h=1.0), masses=_M, key=_K, tidal_radius=1.5)
    radii = jnp.linalg.norm(ic.positions, axis=1)
    assert bool(jnp.all(ic.masses[radii > 1.5] == 0.0))  # ghosts
    assert bool(jnp.any(ic.masses[radii <= 1.5] > 0.0))  # survivors kept


def test_tidal_double_truncation_errors_for_king():
    with pytest.raises(ValueError, match="already truncated|double"):
        build_cluster(
            KingProfile.from_W0_rc(W0=7.0, r_c=1.0), masses=_M, key=_K, tidal_radius=5.0
        )


def test_tidal_double_truncation_errors_for_limepy():
    with pytest.raises(ValueError, match="already truncated|double"):
        build_cluster(
            LIMEPYProfile.from_W0_rc(W0=5.0, g=1.0, r_c=1.0),
            masses=_M,
            key=_K,
            tidal_radius=5.0,
        )


def test_tidal_double_truncation_errors_for_michie():
    # Michie (the anisotropic King) carries a native derived r_t -> tidal_radius double-truncates.
    with pytest.raises(ValueError, match="already truncated|double"):
        build_cluster(
            MichieProfile.from_W0_rc(W0=7.0, r_c=1.0, r_a=8.0),
            masses=_M,
            key=_K,
            tidal_radius=5.0,
        )


def test_tidal_double_truncation_errors_for_eff():
    # EFF carries a native (prescribed) r_t -> tidal_radius double-truncates / no-ops.
    with pytest.raises(ValueError, match="already truncated|double"):
        build_cluster(
            EFFProfile(a=1.0, gamma=3.0, r_t=10.0), masses=_M, key=_K, tidal_radius=5.0
        )


def test_tidal_allowed_for_plummer_only():
    # Plummer is the one untruncated profile -> tidal_radius is valid (no raise).
    ic = build_cluster(PlummerProfile(r_h=1.0), masses=_M, key=_K, tidal_radius=2.0)
    radii = jnp.linalg.norm(ic.positions, axis=1)
    assert bool(jnp.all(ic.masses[radii > 2.0] == 0.0))


def _Lz(ic):
    x, y = ic.positions[:, 0], ic.positions[:, 1]
    vx, vy = ic.velocities[:, 0], ic.velocities[:, 1]
    return float(jnp.sum(ic.masses * (x * vy - y * vx)))


def test_rotation_float_injects_positive_Lz():
    ic = build_cluster(PlummerProfile(r_h=1.0), masses=_M, key=_K, rotation=0.3)
    assert _Lz(ic) > 0.0


def test_rotation_spec_solid_matches_float():
    ic_f = build_cluster(PlummerProfile(r_h=1.0), masses=_M, key=_K, rotation=0.3)
    ic_s = build_cluster(
        PlummerProfile(r_h=1.0), masses=_M, key=_K, rotation=RotationSpec(omega=0.3)
    )
    _assert_ic_equal(ic_f, ic_s)


def test_rotation_spec_differential_injects_Lz():
    spec = RotationSpec(kind="differential", v_peak=2.0, R_peak=1.0)
    ic = build_cluster(PlummerProfile(r_h=1.0), masses=_M, key=_K, rotation=spec)
    assert _Lz(ic) > 0.0


def test_revirialize_rescales_survivors_to_Q():
    # After a tidal cut, survivors are super-virial (S4); revirialize=True restores Q≈0.5.
    from progenax import compute_kinetic_energy, compute_potential_energy

    ic = build_cluster(
        PlummerProfile(r_h=1.0),
        masses=jnp.ones(2000),
        key=_K,
        tidal_radius=2.0,
        revirialize=True,
    )
    keep = ic.masses > 0
    pos, vel, m = ic.positions[keep], ic.velocities[keep], ic.masses[keep]
    T = compute_kinetic_energy(vel, m)
    V = compute_potential_energy(pos, m, G=STELLAR.G)
    Q = float(T / jnp.abs(V))
    assert Q == pytest.approx(0.5, abs=0.05)


# --- Addition A: Q=None faithful unscaled equilibrium ----------------------
def test_Q_none_king_unscaled_is_true_df_equilibrium():
    # Q=None disables the virial rescale; the King true-DF is sampled in detailed
    # equilibrium so the measured Q = T/|V| still lands near 0.5 (no rescale).
    from progenax import compute_kinetic_energy, compute_potential_energy

    p = KingProfile.from_W0_rc(W0=7.0, r_c=1.0)
    m = jnp.ones(3000)
    ic_unscaled = build_cluster(p, masses=m, key=_K, Q=None)
    T = compute_kinetic_energy(ic_unscaled.velocities, ic_unscaled.masses)
    V = compute_potential_energy(ic_unscaled.positions, ic_unscaled.masses, G=STELLAR.G)
    Q = float(T / jnp.abs(V))
    assert Q == pytest.approx(0.5, abs=0.05)
    # ...and the unscaled velocities differ from the Q=0.5-rescaled build.
    ic_scaled = build_cluster(p, masses=m, key=_K, Q=0.5)
    assert not bool(jnp.allclose(ic_unscaled.velocities, ic_scaled.velocities))


def test_Q_none_skips_scaling_for_plummer():
    # The unscaled Plummer DF is also near-virial; Q=None must skip the rescale,
    # leaving velocities that differ from the explicitly-rescaled Q=0.5 build.
    p = PlummerProfile(r_h=1.0)
    ic_none = build_cluster(p, masses=_M, key=_K, Q=None)
    ic_half = build_cluster(p, masses=_M, key=_K, Q=0.5)
    assert not bool(jnp.allclose(ic_none.velocities, ic_half.velocities))


# --- Addition B: revirialize + Q=None is an explicit error -----------------
def test_revirialize_with_Q_none_errors():
    with pytest.raises(ValueError, match="revirialize.*Q|numeric Q"):
        build_cluster(
            PlummerProfile(r_h=1.0),
            masses=_M,
            key=_K,
            tidal_radius=1.5,
            revirialize=True,
            Q=None,
        )


# ===========================================================================
# Batch 4: Aliases + ClusterParams + build_cluster_from_params + exports
# ===========================================================================
from progenax.builders_cluster import (
    ClusterParams,
    build_cluster_from_params,
    build_eff_cluster,
    build_king_cluster,
    build_limepy_cluster,
    build_michie_cluster,
    build_plummer_cluster,
)


def test_plummer_alias_identical():
    ic_a = build_plummer_cluster(masses=_M, r_h=1.7, key=_K)
    ic_b = build_cluster(PlummerProfile(r_h=1.7), masses=_M, key=_K)
    _assert_ic_equal(ic_a, ic_b)


def test_plummer_alias_n_path():
    ic = build_plummer_cluster(n=100, r_h=1.0, key=_K)
    assert ic.masses.shape == (100,)


def test_king_alias_identical():
    ic_a = build_king_cluster(masses=_M, W0=7.0, r_c=1.2, key=_K)
    ic_b = build_cluster(KingProfile.from_W0_rc(W0=7.0, r_c=1.2), masses=_M, key=_K)
    _assert_ic_equal(ic_a, ic_b)


def test_eff_alias_identical():
    ic_a = build_eff_cluster(masses=_M, a=1.0, gamma=3.0, r_t=10.0, key=_K)
    ic_b = build_cluster(EFFProfile(a=1.0, gamma=3.0, r_t=10.0), masses=_M, key=_K)
    _assert_ic_equal(ic_a, ic_b)


def test_michie_alias_identical():
    ic_a = build_michie_cluster(masses=_M, W0=7.0, r_c=1.0, r_a=8.0, key=_K)
    ic_b = build_cluster(
        MichieProfile.from_W0_rc(W0=7.0, r_c=1.0, r_a=8.0), masses=_M, key=_K
    )
    _assert_ic_equal(ic_a, ic_b)


def test_limepy_alias_identical():
    ic_a = build_limepy_cluster(masses=_M, W0=5.0, g=1.0, r_c=1.0, key=_K)
    ic_b = build_cluster(
        LIMEPYProfile.from_W0_rc(W0=5.0, g=1.0, r_c=1.0), masses=_M, key=_K
    )
    _assert_ic_equal(ic_a, ic_b)


def test_cluster_params_wrapper_identical_to_build_cluster():
    params = ClusterParams(
        profile=PlummerProfile(r_h=1.3), tidal_radius=3.0, rotation=0.2
    )
    ic_w = build_cluster_from_params(params, masses=_M, key=_K)
    ic_d = build_cluster(
        PlummerProfile(r_h=1.3), masses=_M, key=_K, tidal_radius=3.0, rotation=0.2
    )
    _assert_ic_equal(ic_w, ic_d)


def test_cluster_params_defaults_base_case():
    params = ClusterParams(profile=PlummerProfile(r_h=1.0))
    ic_w = build_cluster_from_params(params, masses=_M, key=_K)
    ic_d = build_cluster(PlummerProfile(r_h=1.0), masses=_M, key=_K)
    _assert_ic_equal(ic_w, ic_d)


def test_all_new_symbols_exported_from_progenax():
    import progenax

    for sym in (
        "build_cluster",
        "build_plummer_cluster",
        "build_king_cluster",
        "build_eff_cluster",
        "build_michie_cluster",
        "build_limepy_cluster",
        "matched_velocity_df",
        "RotationSpec",
        "ClusterParams",
        "build_cluster_from_params",
    ):
        assert sym in progenax.__all__, f"{sym} missing from progenax.__all__"
        assert hasattr(progenax, sym), f"progenax.{sym} not importable"

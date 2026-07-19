"""W1: MagneticSpec + build_cluster_ic(magnetic=...) scaffold and coupling guards (ADR-0060).

magnetic=None ⇒ byte-identical legacy path (enforced by the existing byte-identity pins in the
full suite). When set, μ_Φ-primary magnetism is a gas-cloud property: it requires
VelocitySpec(mode='physical') + a GasSpec (so M_cl = M_star/sfe and c_s exist) — decision (a).
"""

import jax
import jax.numpy as jnp
import pytest

from gravoturb.cluster import build_cluster_ic
from gravoturb.specs import (
    CloudSpec,
    CompositionSpec,
    GasSpec,
    GeometrySpec,
    MagneticSpec,
    VelocitySpec,
)
from jaxstro.units import STELLAR

from progenax import PlummerProfile

pytestmark = [pytest.mark.experimental, pytest.mark.unit]


# --------------------------------------------------------------------------- #
# MagneticSpec constructor validation
# --------------------------------------------------------------------------- #
def test_magnetic_spec_valid_and_pytree():
    m = MagneticSpec(mu_phi=2.0)
    assert float(m.mu_phi) == 2.0
    assert m.realize == "field"            # default: full RMHD-ready vector field
    assert m.mean_field_axis == 2          # default z (Extension-A line of sight)
    assert m.anisotropy == "theory"        # default: sourced Hu & Lazarian closure


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(mu_phi=0.0), "mu_phi"),
        (dict(mu_phi=-1.0), "mu_phi"),
        (dict(mu_phi=1.0, realize="bogus"), "realize"),
        (dict(mu_phi=1.0, mean_field_axis=3), "mean_field_axis"),
        (dict(mu_phi=1.0, field_slope=0.0), "field_slope"),
        (dict(mu_phi=1.0, anisotropy="bogus"), "anisotropy"),
        (dict(mu_phi=1.0, anisotropy="fixed"), "anisotropy_value"),   # fixed needs a value
        (dict(mu_phi=1.0, anisotropy="theory", anisotropy_value=3.0), "anisotropy_value"),
    ],
)
def test_magnetic_spec_rejects_bad_physics(kwargs, match):
    with pytest.raises((ValueError, TypeError), match=match):
        MagneticSpec(**kwargs)


def test_magnetic_spec_fixed_anisotropy_ok():
    m = MagneticSpec(mu_phi=1.0, anisotropy="fixed", anisotropy_value=2.5)
    assert float(m.anisotropy_value) == 2.5


# --------------------------------------------------------------------------- #
# (a) coupling guard at the builder boundary
# --------------------------------------------------------------------------- #
def _masses():
    return jnp.linspace(0.5, 5.0, 40)


def _physical_gas_kwargs():
    return dict(
        cloud=CloudSpec(mach=8.0, b=0.5, alpha=1.8, beta=3.0),
        geometry=GeometrySpec(profile=PlummerProfile(r_h=1.0), box_size=4.0, shape=(16, 16, 16)),
        velocity=VelocitySpec(beta_v=4.0, mode="physical", c_s=0.2),
        composition=CompositionSpec(),
        gas=GasSpec(sfe=0.3),
        G=STELLAR.G, units=STELLAR, key=jax.random.PRNGKey(0),
    )


def test_magnetic_requires_physical_velocity_mode():
    kw = _physical_gas_kwargs()
    kw["velocity"] = VelocitySpec(beta_v=4.0, Q_target=0.5)  # virial_target
    kw["gas"] = None
    with pytest.raises(ValueError, match="physical"):
        build_cluster_ic(_masses(), magnetic=MagneticSpec(mu_phi=2.0), **kw)


def test_magnetic_requires_gas():
    kw = _physical_gas_kwargs()
    kw["gas"] = None
    with pytest.raises(ValueError, match="gas"):
        build_cluster_ic(_masses(), magnetic=MagneticSpec(mu_phi=2.0), **kw)


def test_magnetic_none_builds_and_matches_absent():
    # magnetic=None must run the untouched legacy path (byte-identity backstop; the pinned
    # hashes in the full suite are the exhaustive gate).
    kw = _physical_gas_kwargs()
    a = build_cluster_ic(_masses(), **kw)
    b = build_cluster_ic(_masses(), magnetic=None, **kw)
    for la, lb in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        if isinstance(la, str):          # Physics carries string leaves (coupling, mode, ...)
            assert la == lb
        else:
            assert jnp.array_equal(la, lb)


# --------------------------------------------------------------------------- #
# W2: L2 velocity anisotropy wiring (realize >= 'anisotropic'). Fixed r_A keeps the
# mechanism gates deterministic; theory-closure-through-the-builder is validated in W4.
# --------------------------------------------------------------------------- #
# Moderate-field regime (c_s=1 km/s, low realistic sfe): a mu_phi range [2,20] builds with the
# s_crit channel active, spanning trans-Alfvenic (M_A~1.5) to weak (M_A~16). Strongly sub-critical
# mu_phi (<~1) drives the collapse-eligible fraction below sfe -> the gas solver refuses (SF ceases).
_MAG_MASSES = jnp.linspace(0.3, 8.0, 300)


def _mag_kwargs():
    return dict(
        cloud=CloudSpec(mach=8.0, b=0.5, alpha=1.8, beta=3.0),
        geometry=GeometrySpec(profile=PlummerProfile(r_h=1.0), box_size=4.0, shape=(24, 24, 24)),
        velocity=VelocitySpec(beta_v=4.0, mode="physical", c_s=1.0),
        composition=CompositionSpec(),
        gas=GasSpec(sfe=0.02),
        G=STELLAR.G, units=STELLAR, key=jax.random.PRNGKey(0),
    )


def _build_hydro():
    return build_cluster_ic(_MAG_MASSES, **_mag_kwargs())


def _build_mag(realize, mu_phi=5.0, axis=2, **mag):
    return build_cluster_ic(
        _MAG_MASSES, **_mag_kwargs(),
        magnetic=MagneticSpec(mu_phi=mu_phi, realize=realize, mean_field_axis=axis, **mag),
    )


def _perp_par_ratio(ic, axis=2):
    var = jnp.var(ic.stars.velocities, axis=0)
    perp = [var[i] for i in range(3) if i != axis]
    return 0.5 * (perp[0] + perp[1]) / var[axis]


def test_anisotropic_realize_increases_perp_dominance_vs_scalar():
    # Comparison-based (cancels baseline GRF scatter): only 'anisotropic' touches the velocity
    # field, so its perp/par ratio must clearly exceed the 'scalar' baseline for the same params.
    aniso = _build_mag("anisotropic", anisotropy="fixed", anisotropy_value=6.0)
    scal = _build_mag("scalar", anisotropy="fixed", anisotropy_value=6.0)
    assert _perp_par_ratio(aniso) > 1.5 * _perp_par_ratio(scal)


def test_mean_field_axis_0_suppresses_that_component_vs_scalar():
    aniso = _build_mag("anisotropic", axis=0, anisotropy="fixed", anisotropy_value=6.0)
    scal = _build_mag("scalar", axis=0, anisotropy="fixed", anisotropy_value=6.0)
    frac0 = lambda ic: jnp.var(ic.stars.velocities, axis=0)[0] / jnp.sum(jnp.var(ic.stars.velocities, axis=0))
    assert frac0(aniso) < frac0(scal)  # axis-0 (parallel) power fraction is suppressed


# --------------------------------------------------------------------------- #
# W3: L3 vector B-field attachment (realize='field'). Scalars logged at any realize;
# the divergence-free B grid materialises only at realize='field'.
# --------------------------------------------------------------------------- #
def _spectral_div_over_scale(B):
    n = B.shape[1]
    kk = jnp.fft.fftfreq(n) * n
    KX, KY, KZ = jnp.meshgrid(kk, kk, kk, indexing="ij")
    Bk = jnp.fft.fftn(B, axes=(1, 2, 3))
    div_k = KX * Bk[0] + KY * Bk[1] + KZ * Bk[2]
    return jnp.max(jnp.abs(div_k)) / (jnp.max(jnp.abs(Bk)) + 1e-30)


def test_realize_field_attaches_divergence_free_B_grid():
    ic = _build_mag("field", anisotropy="fixed", anisotropy_value=2.0)
    assert ic.magnetic is not None
    B = ic.magnetic.B_field
    assert B.shape == (3, 24, 24, 24)
    assert _spectral_div_over_scale(B) < 1e-10           # ∇·B = 0 to machine precision
    means = jnp.mean(B, axis=(1, 2, 3))
    assert jnp.abs(means[2] - ic.magnetic.B0) < 1e-6     # uniform mean = B0 along z
    assert jnp.abs(means[0]) < 1e-6 and jnp.abs(means[1]) < 1e-6


def test_realize_scalar_logs_quantities_but_no_B_grid():
    ic = _build_mag("scalar")
    assert ic.magnetic is not None          # scalars are always logged when magnetism is on
    assert ic.magnetic.B_field is None      # the grid only materialises at realize='field'
    assert float(ic.magnetic.beta0) > 0.0
    assert float(ic.magnetic.mach_alfven) > 0.0


def test_magnetic_none_has_no_magnetic_state():
    ic = build_cluster_ic(_masses(), **_physical_gas_kwargs())
    assert ic.magnetic is None


# --------------------------------------------------------------------------- #
# W4: L1 magnetized sigma_s^2 in the density build (b_eff = b*sqrt(beta0/(beta0+1))).
# Applies at every realize level; magnetic support narrows the PDF -> fewer dense cells.
# --------------------------------------------------------------------------- #
def test_magnetic_narrows_density_field_variance():
    # The physically-defensible width-only L1 effect: magnetic cushioning narrows sigma_s^2, so
    # the realized log-density field is LESS variable. (The BM19 f_dense *diagnostic* moves
    # counterintuitively under width-only because s_t=(alpha-1/2)sigma_s^2 drops with it — the
    # physical dense-fraction reduction needs the deferred s_crit channel, ADR-0058.)
    hydro = _build_hydro()
    mag = _build_mag("scalar", mu_phi=3.0)   # L1 applies even at realize='scalar'
    assert jnp.var(mag.fields.s_turb.s) < jnp.var(hydro.fields.s_turb.s)


def test_weak_field_recovers_hydro_field_variance():
    hydro = _build_hydro()
    mag = _build_mag("scalar", mu_phi=1.0e4)  # beta0 -> inf, sigma_s^2 -> hydro
    rel = abs(jnp.var(mag.fields.s_turb.s) - jnp.var(hydro.fields.s_turb.s)) / jnp.var(hydro.fields.s_turb.s)
    assert float(rel) < 1e-3


# --------------------------------------------------------------------------- #
# W5: ideal s_crit collapse-threshold channel (magnetothermal Jeans, F&K12 Eq.21:
# raise collapse threshold by ln(1+1/beta0)). Magnetic support -> fewer collapse-eligible
# cells; strong field -> SF ceases. W6: ambipolar flux-loss softening.
# --------------------------------------------------------------------------- #
def _f_elig(ic):
    return float(ic.ledger.collapse_eligible_fraction)


def test_magnetic_ideal_reduces_collapse_eligible_fraction():
    hydro = _build_hydro()
    mag = _build_mag("scalar", mu_phi=3.0)  # s_crit channel applies at any realize
    assert _f_elig(mag) < _f_elig(hydro)     # magnetic support -> less collapsing gas


def test_stronger_field_suppresses_collapse_more():
    weak = _build_mag("scalar", mu_phi=5.0)   # M_A~4
    strong = _build_mag("scalar", mu_phi=2.0)  # M_A~1.5, trans-Alfvenic
    assert _f_elig(strong) < _f_elig(weak)     # stronger field -> less collapsing gas


def test_too_much_flux_ceases_star_formation():
    # Strongly sub-critical (small mu_phi) with a demanding sfe: the s_crit shift drives the
    # collapse-eligible ceiling below sfe, so the requested SFE is unreachable and the gas solver
    # refuses — "too much flux -> SF ceases". (c_s=0.2, sfe=0.3, mu_phi=0.3: ceiling ~0.016 << 0.3.)
    with pytest.raises((RuntimeError, ValueError), match="SFE|converge|eligible|achievable"):
        build_cluster_ic(
            _MAG_MASSES,
            cloud=CloudSpec(mach=8.0, b=0.5, alpha=1.8, beta=3.0),
            geometry=GeometrySpec(profile=PlummerProfile(r_h=1.0), box_size=4.0, shape=(24, 24, 24)),
            velocity=VelocitySpec(beta_v=4.0, mode="physical", c_s=0.2),
            composition=CompositionSpec(), gas=GasSpec(sfe=0.3),
            G=STELLAR.G, units=STELLAR, key=jax.random.PRNGKey(0),
            magnetic=MagneticSpec(mu_phi=0.3, realize="scalar"),
        )


def test_ambipolar_recovers_collapse_vs_ideal():
    ideal = _build_mag("scalar", mu_phi=3.0, collapse_threshold="ideal")
    ambi = _build_mag(
        "scalar", mu_phi=3.0, collapse_threshold="ambipolar",
        flux_loss_density=1.0, flux_loss_sharpness=4.0,
    )
    # ambipolar flux loss at high density lets sub-critical dense gas collapse -> MORE eligible
    assert _f_elig(ambi) > _f_elig(ideal)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(mu_phi=1.0, collapse_threshold="bogus"), "collapse_threshold"),
        (dict(mu_phi=1.0, collapse_threshold="ambipolar"), "flux_loss_density"),
        (dict(mu_phi=1.0, collapse_threshold="ideal", flux_loss_density=2.0), "flux_loss_density"),
    ],
)
def test_magnetic_spec_collapse_threshold_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        MagneticSpec(**kwargs)


def test_magnetic_sigma_s_matches_analytic_via_beta0():
    # The realized transition density s_t must equal the magnetic BM19 value from the logged beta0,
    # confirming b_eff threaded through the whole PDF (not just f_dense).
    mag = _build_mag("scalar", mu_phi=3.0)
    beta0 = float(mag.magnetic.beta0)
    b, mach, alpha = 0.5, 8.0, 1.8
    sigma_s2 = jnp.log(1.0 + (b * mach) ** 2 * beta0 / (beta0 + 1.0))
    s_t_expected = (alpha - 0.5) * sigma_s2
    assert abs(float(mag.fields.s_turb.s_t) - float(s_t_expected)) < 1e-9

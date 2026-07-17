"""End-to-end FDF cluster IC builder (Build 4 — the forward generative tool).

``build_cluster_ic`` composes the verified pieces into a complete N-body IC:
  turbulent BM19 field (β,ℳ,α)  →  spherical envelope (progenax SpatialProfile)
  →  star positions (placement ∝ ρ_total, tail mask on s_turb)
  →  coherent turbulent velocities (β_v)  →  virial-scaled to a chosen Q.

Returns positions/velocities/masses (physical units via explicit G) plus the realized
field (for diagnostics) and the realized virial ratio. Spherical + centrally concentrated
(envelope), substructured (turbulence), and dynamically prepared (chosen Q).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.experimental

SHAPE = (16, 16, 16)
BOX = 4.0  # pc


def _profile():
    from progenax import PlummerProfile

    return PlummerProfile(r_h=BOX / 8)  # r_h = 0.5 pc, tapers well inside the box


def _ic(n=600, Q_target=0.5, beta=3.5, key=0):
    from gravoturb.cluster import build_cluster_ic

    masses = jnp.ones(n)
    from jaxstro.units import STELLAR

    from gravoturb.specs import CloudSpec, CompositionSpec, GeometrySpec, VelocitySpec

    return build_cluster_ic(
        masses,
        cloud=CloudSpec(mach=8.0, b=0.5, alpha=1.8, beta=beta),
        geometry=GeometrySpec(profile=_profile(), box_size=BOX, shape=SHAPE),
        velocity=VelocitySpec(beta_v=4.0, Q_target=Q_target),
        composition=CompositionSpec(placement="two_population", f_sub=0.3),
        G=STELLAR.G,
        key=jax.random.PRNGKey(key),
    )


def test_build_cluster_ic_shapes_and_finite():
    """Returns positions/velocities (n,3) and masses (n,), all finite; n = len(masses)."""
    ic = _ic(n=600)
    assert ic.positions.shape == (600, 3)
    assert ic.velocities.shape == (600, 3)
    assert ic.masses.shape == (600,)
    assert np.all(np.isfinite(np.asarray(ic.positions)))
    assert np.all(np.isfinite(np.asarray(ic.velocities)))


def test_build_cluster_ic_is_spherical_and_concentrated():
    """Envelope makes the cluster centrally concentrated vs a uniform box, and COM-centered.

    Uses the MEDIAN radius (robust to the BM19 density tail, which physically broadens the
    cluster by scattering stars into clumps at larger radii — the pure-envelope median is ≈ r_h,
    turbulence broadens it but the cluster stays decisively more concentrated than uniform).
    """
    ic = _ic(n=3000)
    pos = np.asarray(ic.positions)
    com = pos.mean(axis=0)
    assert np.allclose(com, 0.0, atol=0.15)  # COM-centered
    med_r = float(np.median(np.linalg.norm(pos - com, axis=1)))
    # uniform-box baseline (same centered box) for an honest comparison
    u = (
        np.asarray(jax.random.uniform(jax.random.PRNGKey(99), (20000, 3))) * BOX
        - BOX / 2
    )
    med_r_uniform = float(np.median(np.linalg.norm(u, axis=1)))
    assert med_r < 0.78 * med_r_uniform  # clearly concentrated (worst seed ≈0.62)


def test_build_cluster_ic_achieves_target_Q():
    """Realized virial ratio Q = T/|V| matches the target (velocities + virial scaling wired)."""
    from jaxstro.units import STELLAR

    from progenax import compute_kinetic_energy, compute_potential_energy

    ic = _ic(n=600, Q_target=0.5)
    T = compute_kinetic_energy(ic.velocities, ic.masses)
    V = compute_potential_energy(ic.positions, ic.masses, G=STELLAR.G)
    Q = float(T / jnp.abs(V))
    assert abs(Q - 0.5) < 1e-2
    assert abs(float(ic.Q_virial) - 0.5) < 1e-2  # reported Q matches


MACH = 8.0
C_S_KMS = 0.2  # cold-GMC sound speed [km/s] (released C_S_DEFAULT)


def _ic_physical(n=600, eta_v=1.0, c_s=C_S_KMS, key=0):
    from gravoturb.cluster import build_cluster_ic
    from gravoturb.specs import CloudSpec, CompositionSpec, GeometrySpec, VelocitySpec
    from jaxstro.units import STELLAR

    return build_cluster_ic(
        jnp.ones(n),
        cloud=CloudSpec(mach=MACH, b=0.5, alpha=1.8, beta=3.5),
        geometry=GeometrySpec(profile=_profile(), box_size=BOX, shape=SHAPE),
        velocity=VelocitySpec(beta_v=4.0, mode="physical", c_s=c_s, eta_v=eta_v),
        composition=CompositionSpec(placement="two_population", f_sub=0.3),
        G=STELLAR.G,
        units=STELLAR,
        key=jax.random.PRNGKey(key),
    )


def _sigma_3d(ic):
    """Mass-weighted 3-D velocity dispersion of the IC (COM frame) [pc/Myr]."""
    return float(jnp.sqrt(
        jnp.sum(ic.masses * jnp.sum(ic.velocities**2, axis=1)) / jnp.sum(ic.masses)))


def test_physical_mode_dispersion_round_trip():
    """sigma_star = eta_v * mach * c_s, with c_s [km/s] converted to pc/Myr (STELLAR).

    Scaling happens after COM removal, so the round trip is exact (gate bound <1%)."""
    from jaxstro.units import STELLAR

    ic = _ic_physical(n=2000)
    target = MACH * C_S_KMS / STELLAR.velocity_scale_km_s  # pc/Myr
    assert _sigma_3d(ic) == pytest.approx(target, rel=1e-10)


def test_physical_mode_units_pin():
    """The km/s -> pc/Myr conversion pin: 1 pc/Myr = 0.9778 km/s (design gate AC-IC8c)."""
    from jaxstro.units import STELLAR

    assert STELLAR.velocity_scale_km_s == pytest.approx(0.9778, abs=2e-4)
    # c_s numerically equal to the scale => internal sound speed exactly 1 pc/Myr
    ic = _ic_physical(n=500, c_s=float(STELLAR.velocity_scale_km_s))
    assert _sigma_3d(ic) == pytest.approx(MACH * 1.0, rel=1e-10)


def test_physical_mode_Q_emergent_and_eta_v_squared_scaling():
    """Q_virial is an OUTPUT; same key => same positions, so Q scales exactly as eta_v^2."""
    ic1 = _ic_physical(n=600, eta_v=1.0, key=3)
    ic5 = _ic_physical(n=600, eta_v=0.5, key=3)
    assert float(ic1.Q_virial) > 0.0 and np.isfinite(float(ic1.Q_virial))
    assert float(ic5.Q_virial) / float(ic1.Q_virial) == pytest.approx(0.25, rel=1e-10)


def test_physical_mode_reports_alpha_vir():
    """alpha_vir = 5 sigma_1D^2 r_h / (G M) — BM92/Heyer LITERATURE convention (1-D
    dispersion, sigma_1D = sigma_3D/sqrt(3)), measured on the realized cluster; the
    consistency diagnostic, reported in BOTH modes; scales as eta_v^2 with frozen
    positions. (Review 2026-07-16: the 3-D form inflated the diagnostic ~3x vs the
    GMC literature scale where alpha_vir ~ 1 means virial.)"""
    from jaxstro.units import STELLAR

    ic = _ic_physical(n=600, key=3)
    sigma_1d_sq = _sigma_3d(ic) ** 2 / 3.0
    r = np.sort(np.linalg.norm(np.asarray(ic.positions), axis=1))
    r_h = r[np.searchsorted(np.cumsum(np.ones_like(r)), r.size / 2.0)]  # equal masses
    expected = 5.0 * sigma_1d_sq * r_h / (STELLAR.G * float(jnp.sum(ic.masses)))
    assert float(ic.alpha_vir) == pytest.approx(expected, rel=1e-6)
    ic5 = _ic_physical(n=600, eta_v=0.5, key=3)
    assert float(ic5.alpha_vir) / float(ic.alpha_vir) == pytest.approx(0.25, rel=1e-10)
    # virial_target mode reports it too (same measured definition)
    assert np.isfinite(float(_ic(n=400).alpha_vir)) and float(_ic(n=400).alpha_vir) > 0


def test_physical_mode_requires_consistent_units():
    """Physical mode needs a UnitSystem for the km/s conversion; G/units must agree."""
    from gravoturb.cluster import build_cluster_ic
    from gravoturb.specs import CloudSpec, CompositionSpec, GeometrySpec, VelocitySpec
    from jaxstro.units import PLANETARY, STELLAR

    kwargs = dict(
        cloud=CloudSpec(mach=MACH, b=0.5, alpha=1.8, beta=3.5),
        geometry=GeometrySpec(profile=_profile(), box_size=BOX, shape=SHAPE),
        velocity=VelocitySpec(beta_v=4.0, mode="physical", c_s=C_S_KMS),
        composition=CompositionSpec(placement="two_population", f_sub=0.3),
        key=jax.random.PRNGKey(0),
    )
    with pytest.raises(ValueError, match="units"):
        build_cluster_ic(jnp.ones(100), G=STELLAR.G, **kwargs)
    with pytest.raises(ValueError, match="disagree"):
        build_cluster_ic(jnp.ones(100), G=STELLAR.G, units=PLANETARY, **kwargs)


def test_units_consistency_checked_in_any_mode():
    """A provided units must agree with G even in virial_target mode (review: no
    silent precedence — a units/G mismatch is never ignored)."""
    from gravoturb.cluster import build_cluster_ic
    from gravoturb.specs import CloudSpec, CompositionSpec, GeometrySpec, VelocitySpec
    from jaxstro.units import PLANETARY, STELLAR

    with pytest.raises(ValueError, match="disagree"):
        build_cluster_ic(
            jnp.ones(100),
            cloud=CloudSpec(mach=MACH, b=0.5, alpha=1.8, beta=3.5),
            geometry=GeometrySpec(profile=_profile(), box_size=BOX, shape=SHAPE),
            velocity=VelocitySpec(beta_v=4.0, Q_target=0.5),
            composition=CompositionSpec(placement="two_population", f_sub=0.3),
            G=STELLAR.G, units=PLANETARY, key=jax.random.PRNGKey(0),
        )


@pytest.mark.parametrize("mode", ["virial_target", "physical"])
def test_frame_transform_reconstructs_grid_frame(mode):
    """ClusterIC.frame records the exact affine star↔grid map (review: the star-only COM
    shift was silently unrecorded, leaving stars and the field grid in different frames).

    Contract: positions + frame.origin reproduces the box-frame draw, and
    velocities = frame.velocity_scale · (v_field(sampled at box positions) − frame.bulk_velocity),
    both to machine roundoff (fp: (x−c)+c is not bitwise x)."""
    from gravoturb.realization.envelope import apply_spherical_envelope
    from gravoturb.realization.pipeline import build_turbulent_field
    from gravoturb.realization.placement import sample_positions
    from gravoturb.realization.turbulent_velocity import (
        sample_turbulent_velocities,
        turbulent_velocity_field,
    )

    n, seed = 600, 0
    ic = _ic(n=n, key=seed) if mode == "virial_target" else _ic_physical(n=n, key=seed)

    # replicate the builder's deterministic pipeline from the same key
    k_field, k_vfield, k_pos = jax.random.split(jax.random.PRNGKey(seed), 3)
    field = build_turbulent_field(8.0, 0.5, 1.8, 3.5, SHAPE, k_field)
    s_tot = apply_spherical_envelope(field.s, _profile(), BOX)
    pos_box = sample_positions(field.s, field.s_t, 8.0, 0.3, n, k_pos,
                               box_size=BOX, s_density=s_tot)
    np.testing.assert_allclose(
        np.asarray(ic.positions) + np.asarray(ic.frame.origin),
        np.asarray(pos_box), rtol=0.0, atol=1e-12)

    v_field = turbulent_velocity_field(SHAPE, 4.0, k_vfield)
    v_raw = np.asarray(sample_turbulent_velocities(pos_box, v_field, box_size=BOX))
    expected = float(ic.frame.velocity_scale) * (v_raw - np.asarray(ic.frame.bulk_velocity))
    np.testing.assert_allclose(np.asarray(ic.velocities), expected, rtol=1e-10, atol=1e-13)


def test_frame_velocity_scale_semantics():
    """velocity_scale is the global post-COM rescale: in physical mode it maps the raw
    COM-frame dispersion onto σ_⋆ exactly (σ_after/σ_before); Q_target=0 gives scale 0."""
    ic = _ic_physical(n=800, key=2)
    from jaxstro.units import STELLAR

    sigma_after = _sigma_3d(ic)
    target = MACH * C_S_KMS / STELLAR.velocity_scale_km_s
    assert sigma_after == pytest.approx(target, rel=1e-10)
    assert float(ic.frame.velocity_scale) > 0.0
    ic_cold = _ic(n=300, Q_target=0.0, key=1)
    assert float(ic_cold.frame.velocity_scale) == pytest.approx(0.0, abs=1e-15)


def test_build_cluster_ic_carries_field_for_diagnostics():
    """The realized TurbulentField is returned (BM19 provenance: f_dense_realized defined on s_turb)."""
    ic = _ic(n=400)
    assert hasattr(ic.field, "f_dense_realized")
    assert np.isfinite(float(ic.field.f_dense_realized))
    assert float(ic.field.f_dense_realized) > 0.0

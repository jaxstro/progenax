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
    from gravoturb.specs import CloudSpec, CompositionSpec, GeometrySpec, VelocitySpec
    from jaxstro.units import STELLAR

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
    assert ic.stars.positions.shape == (600, 3)
    assert ic.stars.velocities.shape == (600, 3)
    assert ic.stars.masses.shape == (600,)
    assert np.all(np.isfinite(np.asarray(ic.stars.positions)))
    assert np.all(np.isfinite(np.asarray(ic.stars.velocities)))


def test_build_cluster_ic_is_spherical_and_concentrated():
    """Envelope makes the cluster centrally concentrated vs a uniform box, and COM-centered.

    Uses the MEDIAN radius (robust to the BM19 density tail, which physically broadens the
    cluster by scattering stars into clumps at larger radii — the pure-envelope median is ≈ r_h,
    turbulence broadens it but the cluster stays decisively more concentrated than uniform).
    """
    ic = _ic(n=3000)
    pos = np.asarray(ic.stars.positions)
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
    T = compute_kinetic_energy(ic.stars.velocities, ic.stars.masses)
    V = compute_potential_energy(ic.stars.positions, ic.stars.masses, G=STELLAR.G)
    Q = float(T / jnp.abs(V))
    assert abs(Q - 0.5) < 1e-2
    assert abs(float(ic.ledger.Q_virial) - 0.5) < 1e-2  # reported Q matches


MACH = 8.0
C_S_KMS = 0.2  # cold-GMC sound speed [km/s] (released C_S_DEFAULT)


def _ic_physical(n=600, eta_v=1.0, c_s=C_S_KMS, key=0, gas=None):
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
        gas=gas,
    )


def _sigma_3d(ic):
    """Mass-weighted 3-D velocity dispersion of the IC (COM frame) [pc/Myr]."""
    return float(jnp.sqrt(
        jnp.sum(ic.stars.masses * jnp.sum(ic.stars.velocities**2, axis=1)) / jnp.sum(ic.stars.masses)))


def test_physical_mode_emergent_stellar_dispersion_band():
    """FIELD-FIRST semantics (Phase 4a, ratified): the GAS GRID carries σ_g = ℳ·c_s
    exactly; the stellar COM-frame dispersion is EMERGENT — systematically BELOW
    η_v·ℳ·c_s because COM removal strips the coherent box-scale bulk the stars share
    (β_v=4 is large-scale dominated). Measured characterization (16 seeds, 16³+32³
    fiducial): σ_⋆/(η_v ℳ c_s) = 0.75–0.79 ± 0.12–0.14, min 0.52 / max 0.97 — the
    band below brackets that with margin; it is a characterization gate, not a
    round-trip identity (that identity moved to the gas grid)."""
    from jaxstro.units import STELLAR

    ic = _ic_physical(n=2000)
    target = MACH * C_S_KMS / STELLAR.velocity_scale_km_s  # pc/Myr
    ratio = _sigma_3d(ic) / target
    assert 0.4 < ratio < 1.1


def test_physical_mode_units_pin_and_gas_grid_exactness():
    """The km/s -> pc/Myr conversion pin (AC-IC8c), plus the EXACT half of the
    ratified re-scope: the (unshifted) gas velocity grid's volume-weighted rms is
    σ_g = ℳ·c_s to machine precision."""
    from gravoturb.specs import GasSpec
    from jaxstro.units import STELLAR

    assert STELLAR.velocity_scale_km_s == pytest.approx(0.9778, abs=2e-4)
    ic = _ic_physical(n=500, gas=GasSpec(sfe=0.2))
    frame = ic.ledger.frame
    bulk_physical = np.asarray(frame.bulk_velocity) * float(frame.velocity_scale)
    v_unshifted = np.asarray(ic.gas.velocity) + bulk_physical
    rms = float(np.sqrt(np.mean(np.sum(v_unshifted**2, axis=-1))))
    sigma_g = MACH * C_S_KMS / STELLAR.velocity_scale_km_s
    assert rms == pytest.approx(sigma_g, rel=1e-10)


def test_physical_mode_Q_emergent_and_eta_v_squared_scaling():
    """Q_virial is an OUTPUT; same key => same positions, so Q scales exactly as eta_v^2."""
    ic1 = _ic_physical(n=600, eta_v=1.0, key=3)
    ic5 = _ic_physical(n=600, eta_v=0.5, key=3)
    assert float(ic1.ledger.Q_virial) > 0.0 and np.isfinite(float(ic1.ledger.Q_virial))
    assert float(ic5.ledger.Q_virial) / float(ic1.ledger.Q_virial) == pytest.approx(0.25, rel=1e-10)


def test_physical_mode_reports_alpha_vir():
    """alpha_vir = 5 sigma_1D^2 r_h / (G M) — BM92/Heyer LITERATURE convention (1-D
    dispersion, sigma_1D = sigma_3D/sqrt(3)), measured on the realized cluster; the
    consistency diagnostic, reported in BOTH modes; scales as eta_v^2 with frozen
    positions. (Review 2026-07-16: the 3-D form inflated the diagnostic ~3x vs the
    GMC literature scale where alpha_vir ~ 1 means virial.)"""
    from jaxstro.units import STELLAR

    ic = _ic_physical(n=600, key=3)
    sigma_1d_sq = _sigma_3d(ic) ** 2 / 3.0
    r = np.sort(np.linalg.norm(np.asarray(ic.stars.positions), axis=1))
    r_h = r[np.searchsorted(np.cumsum(np.ones_like(r)), r.size / 2.0)]  # equal masses
    expected = 5.0 * sigma_1d_sq * r_h / (STELLAR.G * float(jnp.sum(ic.stars.masses)))
    assert float(ic.ledger.alpha_vir) == pytest.approx(expected, rel=1e-6)
    ic5 = _ic_physical(n=600, eta_v=0.5, key=3)
    assert float(ic5.ledger.alpha_vir) / float(ic.ledger.alpha_vir) == pytest.approx(0.25, rel=1e-10)
    # virial_target mode reports it too (same measured definition)
    assert np.isfinite(float(_ic(n=400).ledger.alpha_vir)) and float(_ic(n=400).ledger.alpha_vir) > 0


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
    """TurbulentCloudIC.ledger.frame records the exact affine star↔grid map (review: the star-only COM
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
        np.asarray(ic.stars.positions) + np.asarray(ic.ledger.frame.origin),
        np.asarray(pos_box), rtol=0.0, atol=1e-12)

    v_field = turbulent_velocity_field(SHAPE, 4.0, k_vfield)
    v_raw = np.asarray(sample_turbulent_velocities(pos_box, v_field, box_size=BOX))
    expected = float(ic.ledger.frame.velocity_scale) * (v_raw - np.asarray(ic.ledger.frame.bulk_velocity))
    np.testing.assert_allclose(np.asarray(ic.stars.velocities), expected, rtol=1e-10, atol=1e-13)


def test_frame_velocity_scale_semantics():
    """velocity_scale is the ONE global amplitude factor: under field-first physical
    mode it is η_v·(σ_g/rms_grid) — exactly linear in η_v (same key freezes the grid);
    Q_target=0 gives scale 0 in the legacy mode."""
    ic1 = _ic_physical(n=800, eta_v=1.0, key=2)
    ic5 = _ic_physical(n=800, eta_v=0.5, key=2)
    s1 = float(ic1.ledger.frame.velocity_scale)
    s5 = float(ic5.ledger.frame.velocity_scale)
    assert s1 > 0.0
    assert s5 / s1 == pytest.approx(0.5, rel=1e-12)
    ic_cold = _ic(n=300, Q_target=0.0, key=1)
    assert float(ic_cold.ledger.frame.velocity_scale) == pytest.approx(0.0, abs=1e-15)


def test_build_cluster_ic_carries_field_for_diagnostics():
    """The realized TurbulentField is returned (BM19 provenance: f_dense_realized defined on s_turb)."""
    ic = _ic(n=400)
    assert hasattr(ic.fields.s_turb, "f_dense_realized")
    assert np.isfinite(float(ic.fields.s_turb.f_dense_realized))
    assert float(ic.fields.s_turb.f_dense_realized) > 0.0


def test_build_cluster_ic_helmholtz_coupling():
    """coupling='helmholtz': one white field drives BOTH the density carrier and the
    stellar velocities (β derived = β_v − 2, χ = chi_f10(b) default); the realized
    log-density spectrum carries the derived slope, and stars near dense clumps share
    the converging-flow kinematics (the strong version is gate AC-IC9)."""
    from gravoturb.cluster import build_cluster_ic
    from gravoturb.specs import CloudSpec, CompositionSpec, GeometrySpec, VelocitySpec
    from jaxstro.units import STELLAR

    ic = build_cluster_ic(
        jnp.ones(600),
        cloud=CloudSpec(mach=8.0, b=0.5, alpha=1.8, beta=None, coupling="helmholtz"),
        geometry=GeometrySpec(profile=_profile(), box_size=BOX, shape=(32,) * 3),
        velocity=VelocitySpec(beta_v=4.0, Q_target=0.5),
        composition=CompositionSpec(placement="two_population", f_sub=0.3),
        G=STELLAR.G, key=jax.random.PRNGKey(0),
    )
    assert ic.stars.positions.shape == (600, 3)
    assert np.all(np.isfinite(np.asarray(ic.stars.velocities)))
    assert abs(float(ic.ledger.Q_virial) - 0.5) < 1e-2
    # the coupled carrier imposes the DERIVED slope beta_v − 2 = 2 on the log-density
    s = np.asarray(ic.fields.s_turb.s)
    n = s.shape[0]
    pk = np.abs(np.fft.fftn(s - s.mean())) ** 2
    k1 = np.fft.fftfreq(n) * n
    KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2).ravel()
    keep = (kmag > 2.0) & (kmag < 8.0)
    kb = np.round(kmag[keep]).astype(int)
    pf = pk.ravel()[keep]
    ks = np.unique(kb)
    means = np.array([pf[kb == k].mean() for k in ks])
    coef, *_ = np.linalg.lstsq(
        np.vstack([np.log10(ks.astype(float)), np.ones(len(ks))]).T,
        np.log10(means), rcond=None)
    assert -coef[0] == pytest.approx(2.0, abs=0.5)  # single-realization scatter


def test_gas_build_vertical_slice():
    """Phase 4a vertical slice: normalization → partition → residual gas → ledger.

    Mass closure M_cl = Σmᵢ + ∫ρ_g dV exact (AC-G1 bound); pointwise positivity and
    ρ⋆+ρ_g=ρ_cl by the partition tests; cold isothermal pressure P = ρ_g·c_s²; ONE
    joint frame (total stars+gas momentum ~0 in the adopted frame); gas_included
    labels the product loudly."""
    from gravoturb.specs import GasSpec
    from jaxstro.units import STELLAR

    ic = _ic_physical(n=1500, gas=GasSpec(sfe=0.2))
    led = ic.ledger
    assert led.gas_included and ic.gas is not None
    # mass contract + exact closure
    assert float(led.M_cl) == pytest.approx(1500.0 / 0.2, rel=1e-12)
    m_gas_int = float(jnp.sum(ic.gas.rho_residual) * ic.gas.cell_volume)
    assert m_gas_int == pytest.approx(float(led.M_gas), rel=1e-12)
    assert abs(float(led.mass_closure_residual)) < 1e-8 * float(led.M_cl)
    # positivity + parent-cloud normalization closure
    assert bool(jnp.all(ic.gas.rho_residual >= 0))
    assert float(jnp.sum(ic.gas.rho_cloud) * ic.gas.cell_volume) == pytest.approx(
        float(led.M_cl), rel=1e-10)
    # cold isothermal pressure convention
    c_s_int = C_S_KMS / STELLAR.velocity_scale_km_s
    np.testing.assert_allclose(np.asarray(ic.gas.pressure),
                               np.asarray(ic.gas.rho_residual) * c_s_int**2, rtol=1e-12)
    # ONE joint frame: total (stars+gas) momentum vanishes in the adopted frame
    p = np.asarray(led.total_momentum)
    scale = float(led.M_cl) * MACH * c_s_int
    assert np.all(np.abs(p) < 1e-8 * scale)
    # physics block carries the partition provenance
    assert float(ic.physics.sfe_global) == 0.2
    assert float(ic.physics.tau_star) > 0.0


def test_gas_build_star_only_and_refusals():
    """gas=None stays first-class (gas_included=False, no grids); virial_target + gas
    is refused loudly; the uniform-SFE ablation partitions exactly."""
    from gravoturb.cluster import build_cluster_ic
    from gravoturb.specs import (
        CloudSpec,
        CompositionSpec,
        GasSpec,
        GeometrySpec,
        VelocitySpec,
    )
    from jaxstro.units import STELLAR

    ic = _ic_physical(n=400)
    assert ic.gas is None and not ic.ledger.gas_included
    with pytest.raises(ValueError, match="physical"):
        build_cluster_ic(
            jnp.ones(100),
            cloud=CloudSpec(mach=MACH, b=0.5, alpha=1.8, beta=3.5),
            geometry=GeometrySpec(profile=_profile(), box_size=BOX, shape=SHAPE),
            velocity=VelocitySpec(beta_v=4.0, Q_target=0.5),
            composition=CompositionSpec(placement="two_population", f_sub=0.3),
            G=STELLAR.G, units=STELLAR, key=jax.random.PRNGKey(0),
            gas=GasSpec(sfe=0.2),
        )
    ic_u = _ic_physical(n=400, gas=GasSpec(sfe=0.3, partition="uniform"))
    np.testing.assert_allclose(
        np.asarray(ic_u.gas.rho_residual),
        0.7 * np.asarray(ic_u.gas.rho_cloud), rtol=1e-12)
    assert ic_u.physics.tau_star is None

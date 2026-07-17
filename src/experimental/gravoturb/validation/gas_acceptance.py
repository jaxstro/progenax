"""AC-G1–G8 acceptance for the Phase-4a stars+gas handoff (TurbulentCloudIC).

Each check PRINTS an expected-vs-measured table with a PASS/FAIL verdict and returns
``{"passed": bool, ...}`` (same discipline as the other acceptance scripts). numpy is
permitted (validation side). Design: the 2026-07-16 Phase-4a addendum (ratified model
checkpoint) + the Aim 2 brain note's initial-condition gates.

Run:
    PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync \
        python -m gravoturb.validation.gas_acceptance
"""

import jax
import jax.numpy as jnp
import numpy as np
from jaxstro.units import STELLAR

from gravoturb.cluster import build_cluster_ic
from gravoturb.realization.envelope import apply_spherical_envelope
from gravoturb.realization.gas import (
    local_freefall_time,
    normalized_cloud_density,
    partition_star_gas,
    solve_tau_star,
)
from gravoturb.realization.pipeline import build_turbulent_field
from gravoturb.realization.placement import collapse_weights
from gravoturb.specs import (
    CloudSpec,
    CompositionSpec,
    GasSpec,
    GeometrySpec,
    VelocitySpec,
)
from progenax import PlummerProfile

BOX = 4.0
G = STELLAR.G
C_S = 0.2  # km/s


def _ic(n=1000, sfe=0.2, mach=8.0, seed=0, shape=(16,) * 3, partition="local_freefall"):
    return build_cluster_ic(
        jnp.ones(n),
        cloud=CloudSpec(mach=mach, b=0.5, alpha=1.8, beta=3.0),
        geometry=GeometrySpec(profile=PlummerProfile(r_h=0.5), box_size=BOX,
                              shape=shape),
        velocity=VelocitySpec(beta_v=4.0, mode="physical", c_s=C_S),
        composition=CompositionSpec(placement="two_population", f_sub=0.3),
        G=G, units=STELLAR, key=jax.random.PRNGKey(seed),
        gas=GasSpec(sfe=sfe, partition=partition),
    )


def _grid_fields(mach=8.0, seed=0, shape=(16,) * 3, builder_stream=False):
    """Standalone fields, or (builder_stream=True) the EXACT fields the builder
    realizes for PRNGKey(seed) — the builder splits its key 3-ways and uses the
    first stream for the density field (gate-oracle key discipline)."""
    key = (jax.random.split(jax.random.PRNGKey(seed), 3)[0]
           if builder_stream else jax.random.PRNGKey(seed))
    fld = build_turbulent_field(mach, 0.5, 1.8, 3.0, shape, key)
    s_total = apply_spherical_envelope(fld.s, PlummerProfile(r_h=0.5), BOX)
    w = collapse_weights(fld.s, fld.s_t, 8.0)
    return s_total, w


def ac_g1_mass_closure(seeds=(0, 1, 2)):
    print("\n=== AC-G1 — mass closure: M_cl = Σmᵢ + ∫ρ_g dV (exact) ===")
    ok = True
    for sfe in [0.05, 0.2, 0.3]:
        res = []
        for sd in seeds:
            ic = _ic(sfe=sfe, seed=sd)
            res.append(abs(float(ic.ledger.mass_closure_residual))
                       / float(ic.ledger.M_cl))
        worst = max(res)
        ok = ok and worst < 1e-10
        print(f"  sfe={sfe:.2f}: worst |residual|/M_cl = {worst:.2e} (<1e-10)")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok}


def ac_g2_positivity_and_pointwise_conservation(seed=0):
    print("\n=== AC-G2 — pointwise positivity + ρ⋆+ρ_g = ρ_cl ===")
    s_total, w = _grid_fields(seed=seed)
    rho, dv = normalized_cloud_density(s_total, BOX, 5000.0)
    t_ff = local_freefall_time(rho, G=G)
    tau = solve_tau_star(w, t_ff, rho, dv, sfe_global=0.2)
    rs, rg = partition_star_gas(rho, w, t_ff, tau)
    dev = float(jnp.max(jnp.abs(rs + rg - rho) / rho))
    pos = bool(jnp.all(rs >= 0)) and bool(jnp.all(rg >= 0))
    ok = dev < 1e-12 and pos
    print(f"  max |ρ⋆+ρ_g−ρ_cl|/ρ_cl = {dev:.2e} (<1e-12); all nonnegative = {pos}  "
          f"{'PASS' if ok else 'FAIL'}")
    return {"passed": ok, "dev": dev}


def ac_g3_global_sfe_across_parameters(seeds=(0, 1)):
    print("\n=== AC-G3 — requested global SFE reproduced across (ℳ, sfe) ===")
    ok = True
    # Reachability is PHYSICAL: the freefall partition caps the SFE at the
    # collapse-eligible mass share, which falls with Mach (higher s_t). The sweep
    # stays within the ceiling and the ceiling itself is printed (characterization);
    # over-ceiling requests are the AC-G8 refusal check.
    for mach in [4.0, 8.0, 12.0]:
        for sfe in [0.1, 0.2]:
            # per-row reachability ceiling (depends on M_cl = n/sfe through t_ff)
            s_total0, w0 = _grid_fields(mach=mach, seed=0, builder_stream=True)
            rho0, _dv0 = normalized_cloud_density(s_total0, BOX, 1000.0 / sfe)
            t_ff0 = local_freefall_time(rho0, G=G)
            eps_max = 1.0 - jnp.exp(-jnp.asarray(2.0**40) * w0 / t_ff0)
            ceiling = float(jnp.sum(eps_max * rho0) / jnp.sum(rho0))
            errs = []
            for sd in seeds:
                ic = _ic(sfe=sfe, mach=mach, seed=sd)
                achieved = float(ic.ledger.M_star / ic.ledger.M_cl)
                # the DISCRETE stellar mass satisfies the contract identically;
                # the continuous-partition SFE is what tau_star solves for:
                s_total, w = _grid_fields(mach=mach, seed=sd, builder_stream=True)
                rho, dv = normalized_cloud_density(
                    s_total, BOX, float(ic.ledger.M_cl))
                t_ff = local_freefall_time(rho, G=G)
                tau = float(ic.physics.tau_star)
                rs, _ = partition_star_gas(rho, w, t_ff, tau)
                cont = float(jnp.sum(rs) / jnp.sum(rho))
                errs.append(abs(cont - sfe))
            worst = max(errs)
            ok = ok and worst < 1e-6 and achieved == sfe
            print(f"  ℳ={mach:>4.1f} sfe={sfe:.1f}: continuous-partition SFE err "
                  f"{worst:.2e} (<1e-6); discrete M⋆/M_cl = {achieved:.3f}; "
                  f"reachability ceiling {ceiling:.3f}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok}


def ac_g4_low_efficiency_limit(seed=0):
    print("\n=== AC-G4 — low-efficiency limit: ρ⋆ ∝ w·ρ_cl^{3/2} (the AC-IC7 law) ===")
    s_total, w = _grid_fields(seed=seed)
    rho, dv = normalized_cloud_density(s_total, BOX, 5000.0)
    t_ff = local_freefall_time(rho, G=G)
    tau = solve_tau_star(w, t_ff, rho, dv, sfe_global=1e-4)
    rs, _ = partition_star_gas(rho, w, t_ff, tau)
    law = np.asarray(w * rho**1.5)
    mask = law > law.max() * 1e-6
    ratio = np.asarray(rs)[mask] / law[mask]
    cv = float(np.std(ratio) / np.mean(ratio))
    ok = cv < 1e-3
    print(f"  ratio-field CV at sfe=1e-4: {cv:.2e} (<1e-3 → same law)  "
          f"{'PASS' if ok else 'FAIL'}")
    return {"passed": ok, "cv": cv}


def ac_g5_root_convergence_and_derivative(seed=0):
    print("\n=== AC-G5 — τ⋆ root: convergence, IFT derivative vs FD, loud failure ===")
    s_total, w = _grid_fields(seed=seed)
    rho, dv = normalized_cloud_density(s_total, BOX, 5000.0)
    t_ff = local_freefall_time(rho, G=G)
    errs = []
    for sfe in [0.01, 0.2, 0.55]:  # within this field's reachability ceiling (~0.62)
        tau = solve_tau_star(w, t_ff, rho, dv, sfe_global=sfe)
        rs, _ = partition_star_gas(rho, w, t_ff, tau)
        errs.append(abs(float(jnp.sum(rs) / jnp.sum(rho)) - sfe))
    conv = max(errs)
    # the loud non-convergence failure: an over-ceiling request must RAISE, not clip
    try:
        solve_tau_star(w, t_ff, rho, dv, sfe_global=0.8)
        loud_fail = False
    except RuntimeError:
        loud_fail = True

    def tau_of(sfe):
        return solve_tau_star(w, t_ff, rho, dv, sfe_global=sfe)

    g = float(jax.grad(tau_of)(0.2))
    eps = 1e-6
    fd = (float(tau_of(0.2 + eps)) - float(tau_of(0.2 - eps))) / (2 * eps)
    ad_fd = abs(g / fd - 1.0)
    try:
        solve_tau_star(jnp.zeros_like(w), t_ff, rho, dv, sfe_global=0.2)
        refused = False
    except ValueError:
        refused = True
    ok = conv < 1e-8 and ad_fd < 1e-5 and refused and loud_fail
    print(f"  achieved-SFE err ≤ {conv:.2e} (<1e-8); AD/FD−1 = {ad_fd:.2e} (<1e-5); "
          f"empty-support refused = {refused}; over-ceiling RAISES = {loud_fail}  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok, "conv": conv, "ad_fd": ad_fd}


def ac_g6_normalization_invariance(seed=0):
    print("\n=== AC-G6 — parent-cloud normalization: resolution + envelope invariance ===")
    ok = True
    for shape in [(16,) * 3, (32,) * 3]:
        fld = build_turbulent_field(8.0, 0.5, 1.8, 3.0, shape, jax.random.PRNGKey(seed))
        s_total = apply_spherical_envelope(fld.s, PlummerProfile(r_h=0.5), BOX)
        rho, dv = normalized_cloud_density(s_total, BOX, 5000.0)
        closure = abs(float(jnp.sum(rho) * dv) - 5000.0) / 5000.0
        rho_off, _ = normalized_cloud_density(s_total + 2.5, BOX, 5000.0)
        env_inv = float(jnp.max(jnp.abs(rho_off - rho) / rho))
        ok = ok and closure < 1e-12 and env_inv < 1e-11
        print(f"  {shape[0]:>3}³: |∫ρdV−M_cl|/M_cl = {closure:.2e} (<1e-12); "
              f"envelope-offset invariance = {env_inv:.2e} (<1e-11)")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok}


def ac_g7_gas_velocity_and_stellar_inheritance(seeds=(0, 1, 2)):
    print("\n=== AC-G7 — gas grid σ_g exact; stellar inheritance band; gas moments ===")
    pin = float(STELLAR.velocity_scale_km_s)
    sigma_g = 8.0 * C_S / pin
    ok = True
    ratios = []
    for sd in seeds:
        ic = _ic(n=2000, seed=sd)
        fr = ic.ledger.frame
        v_un = (np.asarray(ic.gas.velocity)
                + np.asarray(fr.bulk_velocity) * float(fr.velocity_scale))
        rms = float(np.sqrt(np.mean(np.sum(v_un**2, axis=-1))))
        ok = ok and abs(rms / sigma_g - 1.0) < 1e-10
        m, v = ic.stars.masses, ic.stars.velocities
        sig = float(jnp.sqrt(jnp.sum(m * jnp.sum(v**2, axis=1)) / jnp.sum(m)))
        ratios.append(sig / sigma_g)
    band = all(0.4 < r < 1.1 for r in ratios)
    ok = ok and band
    print(f"  gas grid rms/σ_g − 1 < 1e-10 across seeds = {ok and band == band}; "
          f"stellar σ_⋆/σ_g = {np.mean(ratios):.3f} ± {np.std(ratios):.3f} "
          f"(characterized band (0.4, 1.1)) {'PASS' if band else 'FAIL'}")
    # gas-density moments across resolution (report — spectrum studied at AC-IC6/9)
    for shape in [(16,) * 3, (32,) * 3]:
        ic = _ic(n=500, seed=0, shape=shape)
        rg = np.asarray(ic.gas.rho_residual)
        consumed = int(np.sum(rg == 0.0))  # eps->1 cells: gas fully consumed (physics)
        lr = np.log(rg[rg > 0])
        print(f"  {shape[0]:>3}³ residual-gas ln ρ_g (ρ>0): mean {lr.mean():+.3f}, "
              f"std {lr.std():.3f}; fully-consumed cells (ε⋆→1): {consumed}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok, "ratios": ratios}


def ac_g8_frame_determinism_refusals(seed=0):
    print("\n=== AC-G8 — joint frame closure, determinism, loud refusals ===")
    ic = _ic(n=1500, seed=seed)
    c_s_int = C_S / float(STELLAR.velocity_scale_km_s)
    p = np.asarray(ic.ledger.total_momentum)
    p_ok = bool(np.all(np.abs(p) < 1e-8 * float(ic.ledger.M_cl) * 8.0 * c_s_int))
    # joint COM position: stars + gas about the recorded origin
    grid_ax = [(np.arange(16) + 0.5) * (BOX / 16)] * 3
    X, Y, Z = np.meshgrid(*grid_ax, indexing="ij")
    grid = np.stack([X, Y, Z], axis=-1) - np.asarray(ic.ledger.frame.origin)
    m_cells = np.asarray(ic.gas.rho_residual) * float(ic.gas.cell_volume)
    com = (np.sum(np.asarray(ic.stars.positions)
                  * np.asarray(ic.stars.masses)[:, None], axis=0)
           + np.sum(grid * m_cells[..., None], axis=(0, 1, 2))) / float(ic.ledger.M_cl)
    com_ok = bool(np.all(np.abs(com) < 1e-10 * BOX))
    ic2 = _ic(n=1500, seed=seed)
    det = bool(np.array_equal(np.asarray(ic.stars.positions),
                              np.asarray(ic2.stars.positions))
               and np.array_equal(np.asarray(ic.gas.rho_residual),
                                  np.asarray(ic2.gas.rho_residual)))
    try:
        build_cluster_ic(
            jnp.ones(100),
            cloud=CloudSpec(mach=8.0, b=0.5, alpha=1.8, beta=3.0),
            geometry=GeometrySpec(profile=PlummerProfile(r_h=0.5), box_size=BOX,
                                  shape=(16,) * 3),
            velocity=VelocitySpec(beta_v=4.0, Q_target=0.5),
            composition=CompositionSpec(placement="two_population", f_sub=0.3),
            G=G, units=STELLAR, key=jax.random.PRNGKey(0), gas=GasSpec(sfe=0.2),
        )
        refuse_vt = False
    except ValueError:
        refuse_vt = True
    # physically unreachable SFE (well above the ~0.4–0.8 field-dependent
    # reachability ceilings; sfe=0.5 is reachable at some seeds — the ceiling is
    # seed-dependent, so the deterministic probe sits far above it): must refuse
    # loudly, never clip (AC-G8/design)
    try:
        _ic(n=1000, sfe=0.9, seed=0)
        refuse_sfe = False
    except (RuntimeError, ValueError):
        refuse_sfe = True
    ic_so = build_cluster_ic(
        jnp.ones(100),
        cloud=CloudSpec(mach=8.0, b=0.5, alpha=1.8, beta=3.0),
        geometry=GeometrySpec(profile=PlummerProfile(r_h=0.5), box_size=BOX,
                              shape=(16,) * 3),
        velocity=VelocitySpec(beta_v=4.0, mode="physical", c_s=C_S),
        composition=CompositionSpec(placement="two_population", f_sub=0.3),
        G=G, units=STELLAR, key=jax.random.PRNGKey(0),
    )
    label_ok = (ic_so.gas is None) and (not ic_so.ledger.gas_included) \
        and bool(ic.ledger.gas_included)
    ok = p_ok and com_ok and det and refuse_vt and refuse_sfe and label_ok
    print(f"  joint momentum ~0 = {p_ok}; joint COM ~0 = {com_ok}; deterministic "
          f"(fixed seed) = {det}; virial_target+gas refused = {refuse_vt}; "
          f"unreachable-SFE refused = {refuse_sfe}; gas_included labels = {label_ok}  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok}


def main():
    print("=" * 78)
    print("GRAVOTURB STARS+GAS HANDOFF ACCEPTANCE (AC-G1–G8)  |  TurbulentCloudIC")
    print("=" * 78)
    results = {
        "AC-G1 mass closure": ac_g1_mass_closure(),
        "AC-G2 positivity + pointwise": ac_g2_positivity_and_pointwise_conservation(),
        "AC-G3 global SFE across params": ac_g3_global_sfe_across_parameters(),
        "AC-G4 low-efficiency w·ρ^1.5": ac_g4_low_efficiency_limit(),
        "AC-G5 root + IFT derivative": ac_g5_root_convergence_and_derivative(),
        "AC-G6 normalization invariance": ac_g6_normalization_invariance(),
        "AC-G7 gas σ_g + inheritance": ac_g7_gas_velocity_and_stellar_inheritance(),
        "AC-G8 frame/determinism/refusals": ac_g8_frame_determinism_refusals(),
    }
    print("\n" + "=" * 78)
    print("SUMMARY")
    for name, r in results.items():
        print(f"  {name:<36} {'PASS' if r['passed'] else 'FAIL'}")
    n_pass = sum(r["passed"] for r in results.values())
    print(f"  {n_pass}/{len(results)} acceptance checks passed")
    print("=" * 78)
    return results


if __name__ == "__main__":
    main()

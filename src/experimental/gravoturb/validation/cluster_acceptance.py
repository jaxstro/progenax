"""Acceptance suite + figure gallery for the gravoturbulent *cluster* IC (Build 4 forward tool).

``build_cluster_ic`` turns natal-turbulence parameters into a complete N-body IC:
turbulent BM19 field (β,ℳ,α) → spherical envelope (progenax SpatialProfile) → star
positions (placement ∝ ρ_total, dense-tail mask on s_turb) → coherent turbulent
velocities (β_v) → COM-centred + virial-scaled to a chosen Q.

Each ``ac_*`` PRINTS an expected-vs-measured table with a PASS/FAIL verdict and returns
``{"passed": bool, ...}``; the suite also writes the figure gallery to ``plots/``.
"Validated" = a number one of these committed functions just printed — no prose claims
without a fresh artifact. numpy/matplotlib are permitted here (validation/analysis side).

Run:
    PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync \
        python -m gravoturb.validation.cluster_acceptance
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
from jaxstro.units import STELLAR

from gravoturb.cluster import build_cluster_ic
from gravoturb.diagnostics.q import q_components
from gravoturb.realization.envelope import apply_spherical_envelope, radius_grid
from gravoturb.realization.pipeline import build_turbulent_field
from gravoturb.realization.placement import FREEFALL_EXPONENT, sample_positions
from gravoturb.specs import CloudSpec, CompositionSpec, GeometrySpec, VelocitySpec
from gravoturb.validation.oracles import ks_two_sample, rho_weighted_reference_positions
from progenax import (
    PlummerProfile,
    compute_kinetic_energy,
    compute_potential_energy,
)

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "plots")
os.makedirs(PLOTS, exist_ok=True)

# fiducial cluster config
BOX = 4.0           # pc
SHAPE = (32, 32, 32)
MACH, B, ALPHA, BETA_V = 8.0, 0.5, 1.8, 4.0
G = STELLAR.G


def _ic(n=2000, beta=3.0, r_h=0.5, Q_target=0.5, seed=0,
        placement="two_population", f_sub=0.3, shape=None):
    # NB multi-freefall at the fiducial ℳ=8 needs ≥64³ (AC-IC0/AC-IC4 caveat: at 32³ the
    # PMF concentrates ~90% of stars into ~8 cells, n_eff≈few, and COM-frame velocity
    # coherence is erased). Position-sensitive multi-freefall gates pass shape=(64,)*3.
    composition = (CompositionSpec()  # multi_freefall default; f_sub is DERIVED
                   if placement == "multi_freefall"
                   else CompositionSpec(placement="two_population", f_sub=f_sub))
    return build_cluster_ic(
        jnp.ones(n),
        cloud=CloudSpec(mach=MACH, b=B, alpha=ALPHA, beta=beta),
        geometry=GeometrySpec(profile=PlummerProfile(r_h=r_h), box_size=BOX,
                              shape=shape or SHAPE),
        velocity=VelocitySpec(beta_v=BETA_V, Q_target=Q_target),
        composition=composition,
        G=G, key=jax.random.PRNGKey(seed),
    )


def _qms(beta, r_h, seeds, n=2000):
    """Per-seed (Q, m̄, s̄) → (mean[3], std[3]) for the CW04 plane (expected value ± scatter)."""
    vals = np.array([q_components(np.asarray(_ic(n=n, beta=beta, r_h=r_h, seed=sd).positions))
                     for sd in seeds])
    return vals.mean(axis=0), vals.std(axis=0)


# ── power-spectrum tools (β recovery) ──
def radial_power_spectrum(field3d, n_bins=24):
    """Isotropic P(k) of a 3D field via FFT periodogram, radially binned in integer |k|.

    |k| convention: ``fftfreq(n)·n`` integer wavenumbers — the SAME convention as
    ``inference/covariance.measured_bandpowers`` and ``theory/projection.kmag_grid``.
    Kept as an independent numpy re-derivation on purpose (oracle); do not refactor
    to import those JAX implementations.
    """
    # unit consistency: k in dimensionless grid wavenumbers (cycles per box), matching
    # the inference/theory modules above.
    f = np.asarray(field3d, dtype=float)
    f = f - f.mean()
    pk = np.abs(np.fft.fftn(f)) ** 2 / f.size
    n = f.shape[0]
    k1 = np.fft.fftfreq(n) * n
    KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    kf, pf = kmag.ravel(), pk.ravel()
    keep = (kf > 0) & (kf <= n // 2)
    kf, pf = kf[keep], pf[keep]
    edges = np.logspace(0, np.log10(n // 2), n_bins + 1)
    idx = np.clip(np.digitize(kf, edges) - 1, 0, n_bins - 1)
    kc = np.full(n_bins, np.nan); pc = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = idx == i
        if m.sum() > 2:
            kc[i] = kf[m].mean(); pc[i] = pf[m].mean()
    good = ~np.isnan(kc)
    return kc[good], pc[good]


def fit_slope(k, p):
    """Fit log10(P) = a − slope·log10(k); return slope (P ∝ k^−slope)."""
    lk, lp = np.log10(k), np.log10(p)
    coef, *_ = np.linalg.lstsq(np.vstack([lk, np.ones_like(lk)]).T, lp, rcond=None)
    return -coef[0]


# ── AC-IC1: spherical envelope sets the cluster scale ──
def ac_ic1_envelope(seeds=(0, 1, 2), placement="two_population"):
    print("\n=== AC-IC1 — spherical envelope: median radius scales with r_h "
          f"(placement={placement}) ===")
    r_hs = [0.3, 0.5, 0.8]
    med = []
    for r_h in r_hs:
        rs = []
        for sd in seeds:
            shape = (64,) * 3 if placement == "multi_freefall" else None  # ≥64³ caveat
            pos = np.asarray(_ic(n=3000, r_h=r_h, seed=sd, placement=placement,
                                 shape=shape).positions)
            rs.append(np.median(np.linalg.norm(pos, axis=1)))
        med.append(float(np.mean(rs)))
        print(f"  r_h={r_h:.2f} pc  ->  median cluster radius = {med[-1]:.3f} pc")
    monotonic = all(med[i] < med[i + 1] for i in range(len(med) - 1))
    # concentrated vs uniform-box median (~0.74*L for a centred cube of side L)
    u = np.asarray(jax.random.uniform(jax.random.PRNGKey(99), (20000, 3))) * BOX - BOX / 2
    med_uniform = float(np.median(np.linalg.norm(u, axis=1)))
    concentrated = med[-1] < 0.85 * med_uniform
    ok = monotonic and concentrated
    print(f"  monotonic(median↑ with r_h)={monotonic}; "
          f"largest r_h median {med[-1]:.3f} < 0.85·uniform {0.85*med_uniform:.3f} = {concentrated}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok, "median_radii": med, "r_hs": r_hs}


# ── AC-IC2: velocities virial-scale to the chosen Q ──
def ac_ic2_virial(seeds=(0, 1, 2)):
    print("\n=== AC-IC2 — coherent velocities scale to the chosen virial ratio Q=T/|V| ===")
    ok = True
    targets = [0.3, 0.5, 0.75]
    realized = []
    for Qt in targets:
        qs = []
        for sd in seeds:
            ic = _ic(n=1500, Q_target=Qt, seed=sd)
            T = float(compute_kinetic_energy(ic.velocities, ic.masses))
            V = float(compute_potential_energy(ic.positions, ic.masses, G=G))
            qs.append(T / abs(V))
        q = float(np.mean(qs))
        realized.append(q)
        err = abs(q - Qt)
        ok = ok and err < 1e-2
        print(f"  Q_target={Qt:.2f}  ->  realized Q={q:.4f}  |err|={err:.2e}  "
              f"{'PASS' if err < 1e-2 else 'FAIL'}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok, "targets": targets, "realized": realized}


# ── AC-IC3: substructure diagnostic — (m̄,s̄) plane separates β from concentration ──
def ac_ic3_substructure(seeds=(0, 1, 2)):
    print("\n=== AC-IC3 — CW04 (m̄,s̄) plane: β-substructure separable from concentration ===")
    betas = [2.0, 2.5, 3.0, 3.5, 4.0]
    r_hs = [0.3, 0.5, 0.8, 1.2]

    print("  β-sweep (envelope r_h=0.5 fixed):   β       Q              m̄             s̄  (mean±σ)")
    beta_rows = []
    for beta in betas:
        (Q, m, s), (dQ, dm, ds) = _qms(beta, 0.5, seeds)
        beta_rows.append((beta, Q, m, s))
        print(f"                                   {beta:>4} {Q:>6.3f}±{dQ:.3f} {m:>6.3f}±{dm:.3f} {s:>6.3f}±{ds:.3f}")
    print("  concentration-sweep (β=3.0 fixed): r_h       Q              m̄             s̄  (mean±σ)")
    conc_rows = []
    for r_h in r_hs:
        (Q, m, s), (dQ, dm, ds) = _qms(3.0, r_h, seeds)
        conc_rows.append((r_h, Q, m, s))
        print(f"                                   {r_h:>4} {Q:>6.3f}±{dQ:.3f} {m:>6.3f}±{dm:.3f} {s:>6.3f}±{ds:.3f}")

    Qb = np.array([r[1] for r in beta_rows]); mb = np.array([r[2] for r in beta_rows])
    mc = np.array([r[2] for r in conc_rows])

    # (1) Q monotonic in β (Q is a substructure indicator at fixed concentration)
    q_mono_beta = bool(np.all(np.diff(Qb) < 0))
    # (2) m̄ monotonic in concentration (m̄ is a concentration indicator)
    m_mono_conc = bool(np.all(np.diff(mc) > 0))
    # (3) separability: m̄ is far less β-sensitive than concentration-sensitive
    m_swing_beta = (mb.max() - mb.min()) / mb.mean()
    m_swing_conc = (mc.max() - mc.min()) / mc.mean()
    separable = bool(m_swing_beta < 0.4 * m_swing_conc)
    ok = q_mono_beta and m_mono_conc and separable
    print(f"  Q monotonic↓ in β = {q_mono_beta}; m̄ monotonic↑ in concentration = {m_mono_conc}")
    print(f"  m̄ swing: β {m_swing_beta:.2f} vs concentration {m_swing_conc:.2f} "
          f"(β-swing < 0.4·conc-swing = {separable})")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok, "beta_rows": beta_rows, "conc_rows": conc_rows,
            "m_swing_beta": m_swing_beta, "m_swing_conc": m_swing_conc}


# ── AC-IC4: turbulent velocities are spatially coherent ──
def ac_ic4_velocity_coherence(seed=0, placement="two_population"):
    # multi-freefall runs at 64³ (the recorded ℳ≥8 caveat): at 32³ the star sample
    # concentrates into ~8 cells (90% of stars) and COM subtraction erases coherence
    # (measured near-alignment +0.013 at 32³ vs +0.631 at 64³) — the 2026-07-16 AC-IC4
    # diagnosis. Resolution fix per the recorded caveat, NOT a weakened threshold.
    print("\n=== AC-IC4 — turbulent velocities are spatially coherent "
          f"(nearby stars move together) (placement={placement}) ===")
    shape = (64,) * 3 if placement == "multi_freefall" else None  # ≥64³ caveat
    ic = _ic(n=2500, seed=seed, placement=placement, shape=shape)
    if ic.placement_n_eff is not None:
        print(f"  placement_n_eff = {float(ic.placement_n_eff):.1f} cells "
              "(resolution-monitoring diagnostic)")
    pos = np.asarray(ic.positions); vel = np.asarray(ic.velocities)
    # cosine alignment of velocity vectors vs pair separation
    rng = np.random.default_rng(0)
    i = rng.integers(0, len(pos), 6000); j = rng.integers(0, len(pos), 6000)
    keep = i != j; i, j = i[keep], j[keep]
    sep = np.linalg.norm(pos[i] - pos[j], axis=1)
    vi, vj = vel[i], vel[j]
    cos = np.sum(vi * vj, axis=1) / (np.linalg.norm(vi, axis=1) * np.linalg.norm(vj, axis=1) + 1e-30)
    near = cos[sep < 0.3]; far = cos[sep > 1.5]
    align_near, align_far = float(np.mean(near)), float(np.mean(far))
    coherent = align_near > 0.3 and align_near > align_far + 0.15
    print(f"  mean velocity alignment (cosθ):  near (<0.3pc) {align_near:+.3f}  "
          f"far (>1.5pc) {align_far:+.3f}")
    print(f"  coherent (near>0.3 and near>far+0.15) = {coherent}  {'PASS' if coherent else 'FAIL'}")
    return {"passed": coherent, "align_near": align_near, "align_far": align_far,
            "sep": sep, "cos": cos}


# ── AC-IC5: the density construction is differentiable ──
def ac_ic5_gradient():
    print("\n=== AC-IC5 — density construction differentiable (jax.grad through envelope+field) ===")

    def total_mass_in_core(r_h):
        # envelope-modulated density on the grid; smooth scalar of the construction
        fld = build_turbulent_field(MACH, B, ALPHA, 3.0, SHAPE, jax.random.PRNGKey(0))
        s_tot = apply_spherical_envelope(fld.s, PlummerProfile(r_h=r_h), BOX)
        r = radius_grid(SHAPE, BOX)
        return jnp.sum(jnp.where(r < 1.0, jnp.exp(s_tot), 0.0))

    g = float(jax.grad(total_mass_in_core)(0.5))
    ok = np.isfinite(g) and g != 0.0
    print(f"  d(core mass)/d(r_h) at r_h=0.5 = {g:+.4g}  finite&nonzero = {ok}  "
          f"{'PASS' if ok else 'FAIL'}")
    return {"passed": ok, "grad": g}


# ── AC-IC6: input β is recovered from the log-density power spectrum ──
def ac_ic6_beta_recovery(seeds=(0, 1, 2), n_grid=64):
    """The realized field's power-spectrum slope vs the INPUT β (the expected/ground-truth value).

    The mass-conserving copula is monotone, so it preserves the LOG-DENSITY spectral slope:
    measured slope of s_turb ≈ input β (sub-% recovery). The DENSITY field exp(s) has a
    compressed, nonlinear slope (the heavy BM19 tail flattens P(k)) — reported for honesty,
    and the reason β-inference uses Gaussianized/log observables (Phase-0). Fit range k∈[2, N/3].
    """
    print("\n=== AC-IC6 — β recovery: input β vs realized P(k) slope (expected value = input β) ===")
    betas = [2.0, 2.5, 3.0, 3.5, 4.0]
    fit_lo, fit_hi = 2.0, n_grid // 3
    rows = []
    print(f"  (64³, {len(seeds)} seeds, fit k∈[{fit_lo:.0f},{fit_hi}])")
    print(f"  {'input β':>8} {'slope(log-dens s)':>20} {'|err|':>7} {'slope(dens e^s)':>16}")
    for beta in betas:
        ss, se = [], []
        for sd in seeds:
            fld = build_turbulent_field(MACH, B, ALPHA, beta, (n_grid,) * 3, jax.random.PRNGKey(sd))
            s = np.asarray(fld.s)
            for arr, acc in [(s, ss), (np.exp(s), se)]:
                k, p = radial_power_spectrum(arr)
                sel = (k > fit_lo) & (k < fit_hi)
                acc.append(fit_slope(k[sel], p[sel]))
        ms, dms = float(np.mean(ss)), float(np.std(ss))
        me, dme = float(np.mean(se)), float(np.std(se))
        err = abs(ms - beta)
        rows.append((beta, ms, dms, me, dme, err))
        print(f"  {beta:>8.1f} {ms:>13.3f}±{dms:.3f} {err:>7.3f} {me:>11.3f}±{dme:.3f}")
    # expected value = input β: log-density slope must track 1:1 within tolerance
    max_err = max(r[5] for r in rows)
    inp = np.array([r[0] for r in rows]); meas = np.array([r[1] for r in rows])
    A = np.vstack([inp, np.ones_like(inp)]).T
    fit_slope_1to1 = float(np.linalg.lstsq(A, meas, rcond=None)[0][0])
    ok = max_err < 0.15 and abs(fit_slope_1to1 - 1.0) < 0.05
    print(f"  max |measured−input| = {max_err:.3f} (<0.15); recovery line slope = "
          f"{fit_slope_1to1:.3f} (≈1.00)  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok, "rows": rows, "max_err": max_err, "recovery_slope": fit_slope_1to1}


# ── AC-IC0: envelope-fidelity map — requested r_h vs realized concentration ──
def ac_ic0_envelope_fidelity(seeds=(0, 1, 2), n=3000, r_h=0.5, placement="two_population"):
    """Characterization map (audit C9 / design amendment A4): how far the realized
    half-mass radius sits from the requested envelope r_h, vs Mach and resolution.

    The turbulence-OFF rows isolate the grid/box/sampling bias; the ON−OFF gap is the
    turbulent mass relocation. PASS = map fully printed + OFF-bias small (<15%); the ON
    rows are *documentation* (r_h is an envelope-shape parameter, not the realized r_h).
    NOTE (A4): this map is placement-mode-dependent — re-run at Phase 1 close under the
    multi-freefall placement law.
    """
    from gravoturb.realization.placement import sample_positions_multi_freefall

    print("\n=== AC-IC0 — envelope fidelity: realized r_half vs requested r_h "
          f"(r_h={r_h} pc, Plummer, placement={placement}) ===")
    rows = []
    for shape in [(32,) * 3, (64,) * 3]:
        for mach in [None, 4.0, 8.0, 12.0]:  # None → turbulence OFF (pure envelope)
            r50 = []
            for sd in seeds:
                if placement == "multi_freefall" and mach is None:
                    # turbulence-OFF multi_freefall never reads the field (sentinel
                    # s_t=-1e3 below) — skip the dead build_turbulent_field call
                    fld = None
                    s_turb = jnp.zeros(shape)
                else:
                    fld = build_turbulent_field(mach or MACH, B, ALPHA, 3.0, shape,
                                          jax.random.PRNGKey(sd))
                    s_turb = jnp.zeros(shape) if mach is None else fld.s
                s_tot = apply_spherical_envelope(s_turb, PlummerProfile(r_h=r_h), BOX)
                if placement == "multi_freefall":
                    pos = np.asarray(sample_positions_multi_freefall(
                        s_turb, fld.s_t if mach is not None else -1e3, 8.0, n,
                        jax.random.PRNGKey(sd + 77), box_size=BOX,
                        s_density=s_tot)) - BOX / 2
                else:
                    pos = np.asarray(sample_positions(
                        s_turb, fld.s_t, 8.0, 0.3, n, jax.random.PRNGKey(sd + 77),
                        box_size=BOX, s_density=s_tot)) - BOX / 2
                r50.append(np.median(np.linalg.norm(pos, axis=1)))
            m, s = float(np.mean(r50)), float(np.std(r50))
            rows.append((shape[0], mach, m, s))
            lbl = "OFF " if mach is None else f"{mach:4.1f}"
            print(f"  grid {shape[0]:>3}³  ℳ={lbl}  realized r_half = {m:.3f}±{s:.3f} pc"
                  f"   (ratio {m / r_h:.2f}× requested)")
    # Placement-consistent OFF reference: the ρ^p-weighted median radius of the SAME
    # truncated grid (p=1 two-population smooth stars, p=3/2 multi-freefall). Judging
    # the ρ^{3/2} law against r_h itself would mislabel correct extra concentration
    # as a sampling bias (amendment A4).
    p_exp = FREEFALL_EXPONENT if placement == "multi_freefall" else 1.0
    ref = {}
    for shape in [(32,) * 3, (64,) * 3]:
        rg = np.asarray(radius_grid(shape, BOX)).ravel()
        wgt = np.asarray(PlummerProfile(r_h=r_h).density(jnp.asarray(rg))) ** p_exp
        order = np.argsort(rg)
        cw = np.cumsum(wgt[order])
        ref[shape[0]] = float(rg[order][np.searchsorted(cw, 0.5 * cw[-1])])
    off_bias = max(abs(r[2] / ref[r[0]] - 1.0) for r in rows if r[1] is None)
    ok = off_bias < 0.15 and len(rows) == 8
    print(f"  OFF reference (ρ^{p_exp}-weighted median radius, truncated grid): "
          + ", ".join(f"{k}³ → {v:.3f} pc" for k, v in ref.items()))
    print(f"  turbulence-OFF |bias vs reference| = {off_bias:.3f} (<0.15 → "
          f"grid/box/sampling honest); ON rows document turbulent relocation "
          f"(r_h = SHAPE parameter).")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok, "rows": rows, "off_bias": off_bias, "off_reference": ref}


# ── AC-IC7: FK12 multi-freefall placement (Phase 1 gate) ──
def ac_ic7_multi_freefall(seeds=(0, 1, 2)):
    """(a) turbulence-OFF control: multi-freefall placement = ρ_env^{3/2}-weighted
    envelope (independent numpy oracle, two-sample KS); (b) collapse_eligible_fraction vs (α, ℳ)
    printed table (α-direction asserted — PDF-grounded; ℳ characterized); (c) Q(β)
    re-baselined under multi_freefall; (d) paired legacy-vs-new Q at matched tail
    fraction. Design 2026-07-16 Phase 1 + amendment A4 (AC-IC0 re-run appended)."""
    from gravoturb.realization.placement import (
        collapse_eligible_fraction,
        sample_positions_multi_freefall,
    )

    print("\n=== AC-IC7 — FK12 multi-freefall placement (p ∝ w·ρ_total^{3/2}) ===")

    # (a) envelope control vs the independent numpy oracle (validation/oracles.py;
    # documented constants: 48³ grid, N=40000 stars, oracle rng seed 2026)
    n_grid, n_star = 48, 40000
    prof = PlummerProfile(r_h=0.5)
    s0 = jnp.zeros((n_grid,) * 3)
    s_tot = apply_spherical_envelope(s0, prof, BOX)
    pos = np.asarray(sample_positions_multi_freefall(
        s0, -1e3, 8.0, n_star, jax.random.PRNGKey(11), box_size=BOX,
        s_density=s_tot)) - BOX / 2
    r_star = np.linalg.norm(pos, axis=1)
    ref = rho_weighted_reference_positions(
        prof, (n_grid,) * 3, BOX, FREEFALL_EXPONENT, n_star,
        np.random.default_rng(2026))
    r_ref = np.linalg.norm(ref, axis=1)
    ks = ks_two_sample(r_star, r_ref)
    ok_a = ks < 0.015
    print(f"  (a) envelope control: two-sample KS vs numpy ρ^{FREEFALL_EXPONENT} "
          f"oracle = {ks:.4f} (<0.015) {'PASS' if ok_a else 'FAIL'}")

    # (b) collapse_eligible_fraction response table (the smooth analytic diagnostic)
    print("  (b) collapse_eligible_fraction(α, ℳ)  [envelope-free box, 32³, seed-averaged]:")
    grid_a = [1.5, 1.8, 2.2, 2.6]
    grid_m = [4.0, 8.0, 12.0]
    fsd = {}
    for al in grid_a:
        for m in grid_m:
            vals = []
            for sd in seeds:
                fld = build_turbulent_field(m, B, al, 3.0, (32,) * 3,
                                            jax.random.PRNGKey(sd))
                vals.append(float(collapse_eligible_fraction(fld.s, fld.s_t, 8.0)))
            fsd[(al, m)] = float(np.mean(vals))
    header = "      α\\ℳ " + "".join(f"{m:>9.1f}" for m in grid_m)
    print(header)
    for al in grid_a:
        print(f"      {al:>4.1f} " + "".join(f"{fsd[(al, m)]:>9.4f}" for m in grid_m))
    ok_b = all(fsd[(grid_a[i], m)] > fsd[(grid_a[i + 1], m)]
               for i in range(len(grid_a) - 1) for m in grid_m)
    print(f"      monotone ↓ in α at every ℳ = {ok_b} {'PASS' if ok_b else 'FAIL'} "
          "(ℳ-direction characterized, not asserted)")

    # (c) Q(β) under multi_freefall (the Phase-1 re-baseline; cf. AC-IC3 legacy values)
    print("  (c) CW04 Q(β) re-baseline, placement=multi_freefall (r_h=0.5, n=2000):")
    q_rows = []
    for beta in [2.0, 3.0, 4.0]:
        qs = []
        for sd in seeds:
            ic = _ic(n=2000, beta=beta, seed=sd, placement="multi_freefall")
            qs.append(q_components(np.asarray(ic.positions))[0])
        q_rows.append((beta, float(np.mean(qs)), float(np.std(qs))))
        print(f"      β={beta:.1f}  Q = {q_rows[-1][1]:.3f} ± {q_rows[-1][2]:.3f}")
    ok_c = q_rows[0][1] > q_rows[-1][1]  # rough→smooth ordering preserved
    print(f"      Q(β=2) > Q(β=4) = {ok_c} {'PASS' if ok_c else 'FAIL'}")

    # (d) paired legacy-vs-new at matched TAIL-STAR fraction (the actual placement-PMF
    # fraction — review fix: the ungated eligible fraction was a >2x mismatched knob)
    ic_new = _ic(n=2000, beta=3.0, seed=0, placement="multi_freefall")
    f_match = float(ic_new.tail_star_fraction)
    f_elig = float(ic_new.collapse_eligible_fraction)
    ic_old = _ic(n=2000, beta=3.0, f_sub=min(max(f_match, 0.0), 1.0), seed=0)
    q_new = q_components(np.asarray(ic_new.positions))[0]
    q_old = q_components(np.asarray(ic_old.positions))[0]
    print(f"  (d) matched-fraction comparison: tail_star_fraction={f_match:.3f} "
          f"(eligible fraction {f_elig:.3f}); "
          f"Q_new={q_new:.3f} vs Q_legacy={q_old:.3f} (documented, not gated)")

    ok = ok_a and ok_b and ok_c
    print(f"  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok, "ks": ks, "fsd": fsd, "q_rows": q_rows,
            "f_match": f_match, "q_new": q_new, "q_legacy": q_old}


# ── AC-IC8: physical velocity mode (Phase 2 gate) ──
def ac_ic8_physical_velocity(seeds=(0, 1, 2)):
    """(a) round-trip σ_⋆ = η_v·ℳ·c_s to <1% after COM removal; (b) emergent Q_virial
    across a (ℳ, r_h) grid with seed bands + exact Q ∝ η_v² sanity; (c) units pin
    (1 pc/Myr = 0.9778 km/s); (d) the physical-mode gravax seam re-run lives in
    tests/experimental/integration/test_gravax_seam.py (needs gravax installed).
    Design 2026-07-16 Phase 2."""
    c_s = 0.2  # cold-GMC sound speed [km/s] (released C_S_DEFAULT)

    def _ic_phys(mach, r_h=0.5, eta_v=1.0, seed=0, n=2000):
        return build_cluster_ic(
            jnp.ones(n),
            cloud=CloudSpec(mach=mach, b=B, alpha=ALPHA, beta=3.0),
            geometry=GeometrySpec(profile=PlummerProfile(r_h=r_h), box_size=BOX,
                                  shape=SHAPE),
            velocity=VelocitySpec(beta_v=BETA_V, mode="physical", c_s=c_s, eta_v=eta_v),
            composition=CompositionSpec(placement="two_population", f_sub=0.3),
            G=G, units=STELLAR, key=jax.random.PRNGKey(seed),
        )

    def _sigma_3d(ic):
        return float(jnp.sqrt(jnp.sum(ic.masses * jnp.sum(ic.velocities**2, axis=1))
                              / jnp.sum(ic.masses)))

    print("\n=== AC-IC8 — physical velocity mode: σ_⋆ = η_v·ℳ·c_s, Q emergent ===")

    # (c) units pin — the km/s ↔ pc/Myr conversion the mode depends on
    pin = float(STELLAR.velocity_scale_km_s)
    ok_c = abs(pin - 0.9778) < 2e-4
    print(f"  (c) units pin: 1 pc/Myr = {pin:.5f} km/s (expected 0.9778, |err|<2e-4) "
          f"{'PASS' if ok_c else 'FAIL'}")

    # (a) dispersion round trip (COM removed by the builder before scaling)
    print(f"  (a) σ_⋆ round trip (c_s={c_s} km/s):"
          f"  {'ℳ':>4} {'η_v':>5} {'target σ_⋆ [pc/Myr]':>20} {'measured':>10} {'rel err':>9}")
    ok_a = True
    for mach in [4.0, 8.0, 12.0]:
        for eta_v in [0.5, 1.0]:
            ic = _ic_phys(mach, eta_v=eta_v)
            target = eta_v * mach * c_s / pin
            meas = _sigma_3d(ic)
            rel = abs(meas / target - 1.0)
            ok_a = ok_a and rel < 0.01
            print(f"      {mach:>8.1f} {eta_v:>5.1f} {target:>20.4f} {meas:>10.4f} "
                  f"{rel:>9.2e}")
    print(f"      all <1% {'PASS' if ok_a else 'FAIL'} "
          "(exact by construction; the bound is the design gate)")

    # (b) emergent Q_virial over (ℳ, r_h) with seed bands
    print(f"  (b) emergent Q_virial = T/|V| (output, not imposed) ± seed σ, "
          f"n=2000, {len(seeds)} seeds:")
    machs, r_hs = [4.0, 8.0, 12.0], [0.3, 0.5, 0.8]
    qgrid = {}
    print("      ℳ\\r_h " + "".join(f"{r:>15.1f}" for r in r_hs))
    for mach in machs:
        row = []
        for r_h in r_hs:
            qs = [float(_ic_phys(mach, r_h=r_h, seed=sd).Q_virial) for sd in seeds]
            qgrid[(mach, r_h)] = (float(np.mean(qs)), float(np.std(qs)))
            row.append(f"{qgrid[(mach, r_h)][0]:>9.3f}±{qgrid[(mach, r_h)][1]:.3f}")
        print(f"      {mach:>5.1f} " + "".join(f"{c:>15}" for c in row))
    # physics direction: T ∝ ℳ² at fixed positions-statistics, |V| shrinks with r_h,
    # so Q must rise along BOTH axes of the grid
    ok_b_mono = all(qgrid[(machs[i], r)][0] < qgrid[(machs[i + 1], r)][0]
                    for i in range(len(machs) - 1) for r in r_hs) and \
                all(qgrid[(m, r_hs[j])][0] < qgrid[(m, r_hs[j + 1])][0]
                    for j in range(len(r_hs) - 1) for m in machs)
    # α_vir consistency diagnostic at the fiducial point
    ic_fid = _ic_phys(8.0)
    print(f"      fiducial (ℳ=8, r_h=0.5): Q = {float(ic_fid.Q_virial):.3f}, "
          f"α_vir = {float(ic_fid.alpha_vir):.3f} (BM92 form on the realized cluster)")
    # exact η_v² scaling at frozen key (same positions)
    q1 = float(_ic_phys(8.0, eta_v=1.0, seed=0).Q_virial)
    q5 = float(_ic_phys(8.0, eta_v=0.5, seed=0).Q_virial)
    ok_b_eta = abs(q5 / q1 - 0.25) < 1e-9
    print(f"      Q(η_v=0.5)/Q(η_v=1) = {q5 / q1:.12f} (exact 0.25) "
          f"{'PASS' if ok_b_eta else 'FAIL'};  grid monotone in ℳ and r_h = {ok_b_mono} "
          f"{'PASS' if ok_b_mono else 'FAIL'}")

    ok = ok_a and ok_b_mono and ok_b_eta and ok_c
    print("  (d) physical-mode seam: tests/experimental/integration/test_gravax_seam.py")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok, "qgrid": qgrid, "units_pin": pin,
            "eta_ratio": q5 / q1, "Q_fiducial": float(ic_fid.Q_virial),
            "alpha_vir_fiducial": float(ic_fid.alpha_vir)}


# ── figure gallery (moved) ──
# _fig_scatter / _fig_radial_profile / _fig_beta_recovery / _fig_substructure_plane /
# _fig_velocity now live in gravoturb.validation.cluster_figures (imported in main()).


def main():
    print("=" * 78)
    print(f"GRAVOTURB CLUSTER IC ACCEPTANCE  |  ℳ={MACH}, b={B}, α={ALPHA}, box={BOX}pc, shape={SHAPE}")
    print("=" * 78)
    r0 = ac_ic0_envelope_fidelity()
    r0m = ac_ic0_envelope_fidelity(placement="multi_freefall")  # A4 re-run (Phase 1)
    r7 = ac_ic7_multi_freefall()
    r1 = ac_ic1_envelope()
    r1m = ac_ic1_envelope(placement="multi_freefall")   # shipped default, gated too
    r2 = ac_ic2_virial()
    r3 = ac_ic3_substructure()
    r4 = ac_ic4_velocity_coherence()
    r4m = ac_ic4_velocity_coherence(placement="multi_freefall")
    r5 = ac_ic5_gradient()
    r6 = ac_ic6_beta_recovery()
    r8 = ac_ic8_physical_velocity()

    print("\n[gallery] writing figures ...")
    from gravoturb.validation.cluster_figures import (  # deferred: avoids import cycle
        _fig_beta_recovery,
        _fig_radial_profile,
        _fig_scatter,
        _fig_substructure_plane,
        _fig_velocity,
    )
    _fig_scatter()
    _fig_radial_profile()
    _fig_substructure_plane(r3)
    _fig_velocity(r4)
    _fig_beta_recovery(r6)

    results = {"AC-IC0 envelope fidelity": r0,
               "AC-IC0 (multi_freefall, A4)": r0m,
               "AC-IC7 multi-freefall": r7,
               "AC-IC1 envelope (legacy)": r1,
               "AC-IC1 envelope (multi_freefall)": r1m,
               "AC-IC2 virial": r2, "AC-IC3 substructure": r3,
               "AC-IC4 coherence (legacy)": r4,
               "AC-IC4 coherence (multi_freefall)": r4m,
               "AC-IC5 gradient": r5, "AC-IC6 β-recovery": r6,
               "AC-IC8 physical velocity": r8}
    print("\n" + "=" * 78)
    print("SUMMARY")
    for name, r in results.items():
        print(f"  {name:<34} {'PASS' if r['passed'] else 'FAIL'}")
    n_pass = sum(r["passed"] for r in results.values())
    print(f"  {n_pass}/{len(results)} acceptance checks passed")
    print("  figures:")
    for fn in ["cluster_scatter.png", "cluster_radial_profile.png",
               "cluster_substructure_plane.png", "cluster_velocity_coherence.png",
               "cluster_beta_recovery.png"]:
        print("   ", os.path.join(PLOTS, fn))
    print("=" * 78)
    return results


if __name__ == "__main__":
    main()

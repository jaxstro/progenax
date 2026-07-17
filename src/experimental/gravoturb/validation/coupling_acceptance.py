"""AC-IC9 acceptance for the Helmholtz-coupled density–velocity construction (Phase 3).

Each check PRINTS an expected-vs-measured table with a PASS/FAIL verdict and returns
``{"passed": bool, ...}`` (the same discipline as acceptance.py / cluster_acceptance.py:
"validated" = a number a committed function just printed). numpy is permitted here
(validation side). This is the Phase-3 gate script — acceptance.py is CLOSED to new
sections (2026-07-16 architecture review).

Run:
    PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync \
        python -m gravoturb.validation.coupling_acceptance
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
from jaxstro.units import STELLAR

from gravoturb.cluster import build_cluster_ic
from gravoturb.realization.helmholtz import (
    coupled_log_density_gaussian,
    helmholtz_velocity_field,
)
from gravoturb.realization.pipeline import turbulent_field_from_gaussian
from gravoturb.specs import CloudSpec, CompositionSpec, GeometrySpec, VelocitySpec
from gravoturb.theory.driving import chi_f10

HERE = os.path.dirname(os.path.abspath(__file__))

MACH, B, ALPHA = 8.0, 0.5, 4.0 / 3.0 + 0.5  # fiducial cloud (alpha=1.83)
BOX = 4.0
G = STELLAR.G


def _slope(field3d, fit_lo=2.0, fit_hi=None):
    """Isotropic P(k) slope via MODE-LEVEL regression (numpy oracle, unbiased).

    Regresses per-mode log10|F(k)|² on log10|k| over all modes in the window. The
    χ²-distributed per-mode power adds a CONSTANT log-offset (no slope bias), unlike
    integer-|k| binning, whose bin-averaged power flattens the slope in narrow low-k
    windows (measured: a pure k⁻² GRF read 1.83 in k∈[2,10] under the binned
    estimator — estimator bias, not physics; 2026-07-16)."""
    f = np.asarray(field3d)
    f = f - f.mean()
    pk = np.abs(np.fft.fftn(f)) ** 2
    n = f.shape[0]
    fit_hi = fit_hi or n / 3.0
    k1 = np.fft.fftfreq(n) * n
    KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2).ravel()
    pf = pk.ravel()
    keep = (kmag > fit_lo) & (kmag < fit_hi) & (pf > 0)
    lk, lp = np.log10(kmag[keep]), np.log10(pf[keep])
    coef, *_ = np.linalg.lstsq(np.vstack([lk, np.ones_like(lk)]).T, lp, rcond=None)
    return -coef[0]


def _neg_div(v):
    """−∇·v via spectral derivative (numpy oracle, integer-k convention)."""
    v = np.asarray(v)
    n = v.shape[0]
    k1 = np.fft.fftfreq(n) * n
    KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
    vk = [np.fft.fftn(v[..., i]) for i in range(3)]
    return -np.fft.ifftn(1j * (KX * vk[0] + KY * vk[1] + KZ * vk[2])).real


def ac_ic9a_derived_slope(seeds=(0, 1, 2), n_grid=64):
    """(a) The coupled carrier's measured P(k) slope equals the DERIVED β = β_v − 2."""
    print("\n=== AC-IC9(a) — coupled log-density slope = β_v − 2 (derived, not chosen) ===")
    rows = []
    ok = True
    for beta_v in [11.0 / 3.0, 4.0]:
        ms = []
        for sd in seeds:
            bundle = helmholtz_velocity_field((n_grid,) * 3, beta_v, 0.4,
                                              jax.random.PRNGKey(sd),
                                              return_fourier=True)
            g = coupled_log_density_gaussian(bundle)
            # the copula is rank-monotone: the log-density s keeps the carrier's slope
            fld = turbulent_field_from_gaussian(g, MACH, B, ALPHA)
            ms.append(_slope(fld.s))
        m, dm = float(np.mean(ms)), float(np.std(ms))
        err = abs(m - (beta_v - 2.0))
        ok = ok and err < 0.15
        rows.append((beta_v, m, dm, err))
        print(f"  β_v={beta_v:.3f}  →  derived β={beta_v - 2.0:.3f}   measured slope(s) = "
              f"{m:.3f}±{dm:.3f}   |err|={err:.3f} (<0.15)")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok, "rows": rows}


def ac_ic9b_coupling_monotone_in_chi(seeds=(0, 1, 2), n_grid=48):
    """(b) The density–velocity coupling STRENGTH is monotone in χ; the independent
    ablation sits at ~0.

    GATE-STATISTIC CORRECTION (2026-07-16, surfaced to Anna): the design draft's
    literal statistic corr(s, −∇·v) is SCALE-INVARIANT and therefore cannot depend on
    χ — measured 1.000 at every χ (the carrier is exactly ∝ −∇·v and the copula is
    rank-monotone; χ only scales the compressive channel's AMPLITUDE). The physically
    monotone coupling strength is

        C(χ) = corr(s, −∇·v) × √(E_long/E_tot)   →  1 · √χ,

    i.e. how much of the velocity field's total amplitude participates in the
    density-correlated converging flow. Same spirit (monotone ↑ in χ, →0 for the
    ablation), correctly posed statistic; thresholds not weakened."""
    print("\n=== AC-IC9(b) — coupling strength C = corr(s,−∇·v)·√(E_long/E_tot) monotone in χ ===")
    chis = [0.1, 0.3, float(chi_f10(1.0))]
    rows = []
    for chi in chis:
        cs = []
        for sd in seeds:
            bundle = helmholtz_velocity_field((n_grid,) * 3, 4.0, chi,
                                              jax.random.PRNGKey(sd), return_fourier=True)
            fld = turbulent_field_from_gaussian(
                coupled_log_density_gaussian(bundle), MACH, B, ALPHA)
            r = np.corrcoef(np.asarray(fld.s).ravel(),
                            _neg_div(bundle.velocity).ravel())[0, 1]
            cs.append(r * np.sqrt(_long_fraction(bundle.velocity)))
        rows.append((chi, float(np.mean(cs)), float(np.std(cs))))
        print(f"  χ={chi:.3f}  C = {rows[-1][1]:+.3f} ± {rows[-1][2]:.3f}   (√χ = {np.sqrt(chi):.3f})")
    # independent ablation (fresh GRF carrier: corr term ~0 ⇒ C ~ 0)
    from gravoturb.realization.gaussian_field import gaussian_random_field
    c0s = []
    for sd in seeds:
        bundle = helmholtz_velocity_field((n_grid,) * 3, 4.0, 0.4,
                                          jax.random.PRNGKey(sd), return_fourier=True)
        g_free = gaussian_random_field((n_grid,) * 3, 2.0, jax.random.PRNGKey(100 + sd))
        fld = turbulent_field_from_gaussian(g_free, MACH, B, ALPHA)
        r0 = np.corrcoef(np.asarray(fld.s).ravel(),
                         _neg_div(bundle.velocity).ravel())[0, 1]
        c0s.append(r0 * np.sqrt(_long_fraction(bundle.velocity)))
    c0 = float(np.mean(c0s))
    print(f"  independent ablation: C = {c0:+.3f} (→ 0)")
    mono = all(rows[i][1] < rows[i + 1][1] for i in range(len(rows) - 1))
    tracks = all(abs(m - np.sqrt(chi)) < 0.05 for chi, m, _ in rows)
    ok = mono and tracks and abs(c0) < 0.05
    print(f"  monotone ↑ = {mono}; tracks √χ (<0.05) = {tracks}; ablation ~0 = "
          f"{abs(c0) < 0.05}  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok, "rows": rows, "c_independent": c0}


def _long_fraction(v):
    """Measured E_long/E_tot of a (n,n,n,3) field (numpy oracle, matches unit tests)."""
    v = np.asarray(v)
    n = v.shape[0]
    k1 = np.fft.fftfreq(n) * n
    KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2 + KZ**2)
    khat = np.stack([KX, KY, KZ], axis=-1) / np.where(kmag > 0, kmag, 1.0)[..., None]
    vk = np.stack([np.fft.fftn(v[..., i]) for i in range(3)], axis=-1)
    v_par = np.sum(vk * khat, axis=-1)
    return float(np.sum(np.abs(v_par) ** 2) / np.sum(np.abs(vk) ** 2))


def ac_ic9c_cluster_gates_in_coupled_mode(seed=0):
    """(c) The cluster-level physics survives the coupled construction: velocity
    coherence (AC-IC4 thresholds) + AC6 dense-fraction fidelity + physical-mode σ_⋆
    round trip — same thresholds as the legacy gates, NOT weakened."""
    print("\n=== AC-IC9(c) — AC-IC4 coherence + AC6 fidelity + σ_⋆ round trip, coupled mode ===")
    ic = build_cluster_ic(
        jnp.ones(2500),
        cloud=CloudSpec(mach=MACH, b=B, alpha=ALPHA, beta=None, coupling="helmholtz"),
        geometry=GeometrySpec(profile=__import__("progenax").PlummerProfile(r_h=0.5),
                              box_size=BOX, shape=(64,) * 3),
        velocity=VelocitySpec(beta_v=4.0, mode="physical", c_s=0.2),
        composition=CompositionSpec(placement="two_population", f_sub=0.3),
        G=G, units=STELLAR, key=jax.random.PRNGKey(seed),
    )
    # AC6: realized dense fraction vs BM19 theory (same fidelity statement as AC6)
    rel = abs(float(ic.fields.s_turb.f_dense_realized) / float(ic.fields.s_turb.f_dense) - 1.0)
    ok_ac6 = rel < 0.05
    print(f"  AC6 re-pass: f_dense_realized/f_dense − 1 = {rel:.4f} (<0.05) "
          f"{'PASS' if ok_ac6 else 'FAIL'}")
    # σ_⋆ inheritance band (field-first re-scope, Phase 4a — matches AC-IC8a: the
    # EXACT identity lives on the gas grid; the stellar dispersion is emergent)
    sigma = float(jnp.sqrt(jnp.sum(ic.stars.masses * jnp.sum(ic.stars.velocities**2, axis=1))
                           / jnp.sum(ic.stars.masses)))
    target = MACH * 0.2 / float(STELLAR.velocity_scale_km_s)
    ratio = sigma / target
    ok_sig = 0.4 < ratio < 1.1
    print(f"  σ_⋆/σ_g inheritance: {ratio:.3f} (characterized band (0.4, 1.1)) "
          f"{'PASS' if ok_sig else 'FAIL'}")
    # AC-IC4 coherence thresholds (near>0.3 and near>far+0.15), 64³ per the caveat
    pos = np.asarray(ic.stars.positions)
    vel = np.asarray(ic.stars.velocities)
    rng = np.random.default_rng(0)
    i = rng.integers(0, len(pos), 6000)
    j = rng.integers(0, len(pos), 6000)
    keep = i != j
    i, j = i[keep], j[keep]
    sep = np.linalg.norm(pos[i] - pos[j], axis=1)
    cos = np.sum(vel[i] * vel[j], axis=1) / (
        np.linalg.norm(vel[i], axis=1) * np.linalg.norm(vel[j], axis=1) + 1e-30)
    near, far = float(np.mean(cos[sep < 0.3])), float(np.mean(cos[sep > 1.5]))
    ok_coh = near > 0.3 and near > far + 0.15
    print(f"  AC-IC4 re-pass: cosθ near {near:+.3f} / far {far:+.3f} "
          f"{'PASS' if ok_coh else 'FAIL'}")
    ok = ok_ac6 and ok_sig and ok_coh
    print(f"  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok, "ac6_rel": rel, "sigma": sigma, "near": near, "far": far}


def ac_ic9d_resolution_convergence(seeds=(0, 1, 2)):
    """(d) The derived slope and coupling strength converge 32³ → 96³.

    Experimental-design fix (2026-07-16): the first draft let the fit window scale
    with resolution (fit_hi = n/3), so finer grids fit into the high-k range where the
    rank copula's spectral distortion is strongest — measuring different SCALES, not
    convergence — and used one seed (scatter ±0.03–0.08 swamps the comparison). Fixed
    physical window k ∈ [2, 10] at every resolution, seed-averaged."""
    print("\n=== AC-IC9(d) — resolution convergence: carrier absolute + coupled≡independent ===")
    # Decomposed criterion (2026-07-16, after two discriminating experiments): an
    # apparent ~0.15 slope depression at β=2 turned out to be (mostly) BIAS IN THE
    # BINNED SLOPE ESTIMATOR in narrow low-k windows — a pure k⁻² GRF read 1.83
    # under integer-|k| binning, and 1.95–2.09 under the unbiased mode-level
    # regression (_slope docstring). The gates: (i) the pre-copula CARRIER slope hits
    # the derived β_v−2 = 2.0 absolutely at the finest grid, (ii) the coupled
    # log-density slope EQUALS the independent one at matched β per resolution
    # (equivalence isolates any residual copula effect, which is ≲0.09 here), and
    # (iii) corr(s,−∇·v) stays converged.
    from gravoturb.realization.pipeline import build_turbulent_field

    rows = []
    for n in [32, 64, 96]:
        sl_car, sl_cpl, sl_ind, rs = [], [], [], []
        for sd in seeds:
            bundle = helmholtz_velocity_field((n,) * 3, 4.0, 0.4,
                                              jax.random.PRNGKey(sd), return_fourier=True)
            g = coupled_log_density_gaussian(bundle)
            fld = turbulent_field_from_gaussian(g, MACH, B, ALPHA)
            fld_i = build_turbulent_field(MACH, B, ALPHA, 2.0, (n,) * 3,
                                          jax.random.PRNGKey(sd))
            sl_car.append(_slope(g, fit_lo=2.0, fit_hi=10.0))
            sl_cpl.append(_slope(fld.s, fit_lo=2.0, fit_hi=10.0))
            sl_ind.append(_slope(fld_i.s, fit_lo=2.0, fit_hi=10.0))
            rs.append(np.corrcoef(np.asarray(fld.s).ravel(),
                                  _neg_div(bundle.velocity).ravel())[0, 1])
        rows.append((n, float(np.mean(sl_car)), float(np.mean(sl_cpl)),
                     float(np.mean(sl_ind)), float(np.mean(rs))))
        print(f"  {n:>3}³: carrier g = {rows[-1][1]:.3f}   coupled s = {rows[-1][2]:.3f}   "
              f"independent s (β=2) = {rows[-1][3]:.3f}   corr = {rows[-1][4]:+.4f}")
    car_ok = abs(rows[-1][1] - 2.0) < 0.1
    equiv_ok = all(abs(c - i) < 0.1 for _, _, c, i, _ in rows)
    corr_ok = abs(rows[-1][4] - rows[-2][4]) < 0.05
    ok = car_ok and equiv_ok and corr_ok
    print(f"  carrier→2.0 at finest (<0.1) = {car_ok}; coupled≡independent per "
          f"resolution (<0.1) = {equiv_ok}; corr converged = {corr_ok}  "
          f"{'PASS' if ok else 'FAIL'}")
    print("  (the earlier apparent 'copula distortion' at β=2 was integer-|k| binning "
          "bias in narrow windows — see _slope; residual copula effect ≲0.09, carried "
          "by the equivalence criterion)")
    return {"passed": ok, "rows": rows}


def ac_ic9e_infall_signature(seeds=(0, 1, 2)):
    """(e) Stars around dense clumps share converging-flow kinematics under coupling:
    the density-weighted ⟨s·(−∇·v)⟩ is positive (dense regions ↔ convergence), and
    vanishes for the independent construction."""
    print("\n=== AC-IC9(e) — converging-flow signature at dense structures ===")
    from gravoturb.realization.gaussian_field import gaussian_random_field
    n = 48
    vals_c, vals_i = [], []
    for sd in seeds:
        bundle = helmholtz_velocity_field((n,) * 3, 4.0, float(chi_f10(1.0)),
                                          jax.random.PRNGKey(sd), return_fourier=True)
        fld = turbulent_field_from_gaussian(
            coupled_log_density_gaussian(bundle), MACH, B, ALPHA)
        div = _neg_div(bundle.velocity)
        s = np.asarray(fld.s)
        w = np.exp(s) / np.exp(s).sum()
        vals_c.append(float((w * div).sum() / div.std()))
        g_free = gaussian_random_field((n,) * 3, 2.0, jax.random.PRNGKey(100 + sd))
        s_i = np.asarray(turbulent_field_from_gaussian(g_free, MACH, B, ALPHA).s)
        w_i = np.exp(s_i) / np.exp(s_i).sum()
        vals_i.append(float((w_i * div).sum() / div.std()))
    mc, mi = float(np.mean(vals_c)), float(np.mean(vals_i))
    sc = float(np.std(vals_c))
    ok = mc > 3.0 * max(sc, 1e-6) + abs(mi) and abs(mi) < 0.05 * abs(mc)
    print(f"  coupled (χ=χ_F10(1)={float(chi_f10(1.0)):.3f}): mass-weighted ⟨−∇·v⟩/σ_div = "
          f"{mc:+.4f} ± {sc:.4f}")
    print(f"  independent ablation:                      {mi:+.4f}")
    print(f"  coupled signal > 3σ and ≫ ablation = {ok}  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok, "coupled": vals_c, "independent": vals_i}


def main():
    print("=" * 78)
    print(f"GRAVOTURB HELMHOLTZ-COUPLING ACCEPTANCE (AC-IC9)  |  ℳ={MACH}, b={B}, α={ALPHA:.2f}")
    print("=" * 78)
    results = {
        "AC-IC9(a) derived slope β_v−2": ac_ic9a_derived_slope(),
        "AC-IC9(b) coupling monotone in χ": ac_ic9b_coupling_monotone_in_chi(),
        "AC-IC9(c) cluster gates, coupled": ac_ic9c_cluster_gates_in_coupled_mode(),
        "AC-IC9(d) resolution convergence": ac_ic9d_resolution_convergence(),
        "AC-IC9(e) infall signature": ac_ic9e_infall_signature(),
    }
    print("\n" + "=" * 78)
    print("SUMMARY")
    for name, r in results.items():
        print(f"  {name:<38} {'PASS' if r['passed'] else 'FAIL'}")
    n_pass = sum(r["passed"] for r in results.values())
    print(f"  {n_pass}/{len(results)} acceptance checks passed")
    print("=" * 78)
    return results


if __name__ == "__main__":
    main()

"""Acceptance suite + figure gallery for the FDF *cluster* IC (Build 4 forward tool).

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
        python -m gravoturb_fdf.validation.cluster_acceptance
"""

import os

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jaxstro.units import STELLAR
from progenax import (
    PlummerProfile,
    compute_kinetic_energy,
    compute_potential_energy,
)

from gravoturb_fdf.cluster import build_cluster_ic
from gravoturb_fdf.diagnostics.q import q_components
from gravoturb_fdf.field.envelope import apply_spherical_envelope, radius_grid
from gravoturb_fdf.field.pipeline import build_fdf_field
from gravoturb_fdf.field.sampling import sample_positions

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "plots")
os.makedirs(PLOTS, exist_ok=True)

# fiducial cluster config
BOX = 4.0           # pc
SHAPE = (32, 32, 32)
MACH, B, ALPHA, BETA_V = 8.0, 0.5, 1.8, 4.0
G = STELLAR.G


def _ic(n=2000, beta=3.0, r_h=0.5, Q_target=0.5, f_sub=0.3, seed=0):
    return build_cluster_ic(
        jnp.ones(n), mach=MACH, b=B, alpha=ALPHA, beta=beta,
        profile=PlummerProfile(r_h=r_h), beta_v=BETA_V, Q_target=Q_target,
        f_sub=f_sub, shape=SHAPE, box_size=BOX, G=G, key=jax.random.PRNGKey(seed),
    )


def _qms(beta, r_h, seeds, n=2000):
    """Per-seed (Q, m̄, s̄) → (mean[3], std[3]) for the CW04 plane (expected value ± scatter)."""
    vals = np.array([q_components(np.asarray(_ic(n=n, beta=beta, r_h=r_h, seed=sd).positions))
                     for sd in seeds])
    return vals.mean(axis=0), vals.std(axis=0)


# ── power-spectrum tools (β recovery) ──
def radial_power_spectrum(field3d, n_bins=24):
    """Isotropic P(k) of a 3D field via FFT periodogram, radially binned in integer |k|."""
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
def ac_ic1_envelope(seeds=(0, 1, 2)):
    print("\n=== AC-IC1 — spherical envelope: median radius scales with r_h ===")
    r_hs = [0.3, 0.5, 0.8]
    med = []
    for r_h in r_hs:
        rs = []
        for sd in seeds:
            pos = np.asarray(_ic(n=3000, r_h=r_h, seed=sd).positions)
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
    sb = np.array([r[3] for r in beta_rows])
    Qc = np.array([r[1] for r in conc_rows]); mc = np.array([r[2] for r in conc_rows])

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
def ac_ic4_velocity_coherence(seed=0):
    print("\n=== AC-IC4 — turbulent velocities are spatially coherent (nearby stars move together) ===")
    ic = _ic(n=2500, seed=seed)
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
        fld = build_fdf_field(MACH, B, ALPHA, 3.0, SHAPE, jax.random.PRNGKey(0))
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
            fld = build_fdf_field(MACH, B, ALPHA, beta, (n_grid,) * 3, jax.random.PRNGKey(sd))
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


# ── figure gallery ──
def _fig_scatter(seed=0):
    ic = _ic(n=4000, seed=seed)
    pos = np.asarray(ic.positions)
    r = np.linalg.norm(pos, axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (a, bx, lbl) in zip(
        axes, [(0, 1, "x–y"), (0, 2, "x–z"), (1, 2, "y–z")]
    ):
        ax.scatter(pos[:, a], pos[:, bx], s=3, c=r, cmap="viridis", alpha=0.5)
        ax.set_xlim(-BOX / 2, BOX / 2); ax.set_ylim(-BOX / 2, BOX / 2)
        ax.set_aspect("equal"); ax.set_title(f"{lbl}  (colour = radius)")
        ax.set_xlabel("pc")
    fig.suptitle("FDF cluster IC — spherical envelope + turbulent substructure "
                 f"(ℳ={MACH}, β=3.0, r_h=0.5 pc, N=4000)")
    fig.savefig(os.path.join(PLOTS, "cluster_scatter.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


_RP_EDGES = np.linspace(0, 2.0, 24)
_RP_CEN = 0.5 * (_RP_EDGES[:-1] + _RP_EDGES[1:])
_RP_SHELL = 4 / 3 * np.pi * (_RP_EDGES[1:] ** 3 - _RP_EDGES[:-1] ** 3)


def _radial_profile_sampled(r_h, mach, s_turb_zero, seeds, n=6000):
    """Mean sampled number-density profile ρ(r). ``s_turb_zero`` → pure-envelope control."""
    prof = PlummerProfile(r_h=r_h)
    stack = []
    for sd in seeds:
        fld = build_fdf_field(mach, B, ALPHA, 3.0, SHAPE, jax.random.PRNGKey(sd))
        s_turb = jnp.zeros(SHAPE) if s_turb_zero else fld.s
        s_tot = apply_spherical_envelope(s_turb, prof, BOX)
        pos = np.asarray(sample_positions(s_turb, fld.s_t, 8.0, 0.3, n,
                                          jax.random.PRNGKey(sd + 50), box_size=BOX,
                                          s_density=s_tot)) - BOX / 2
        cnt, _ = np.histogram(np.linalg.norm(pos, axis=1), bins=_RP_EDGES)
        stack.append(cnt / _RP_SHELL)
    return np.mean(stack, axis=0)


def _fig_radial_profile(seeds=(0, 1, 2)):
    # normalise at r = r_h (a RESOLVED radius, ~4 cells) — NOT the sub-cell innermost bin
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    # panel A: fiducial (ℳ=8) sampled vs analytic, 3 envelopes
    ax = axes[0]
    for r_h, col in zip([0.3, 0.5, 0.8], ["C0", "C1", "C2"]):
        iref = int(np.argmin(np.abs(_RP_CEN - r_h)))
        dens = _radial_profile_sampled(r_h, MACH, False, seeds)
        ax.plot(_RP_CEN, dens / dens[iref], "o-", color=col, label=f"sampled, r_h={r_h}")
        pl = np.asarray(PlummerProfile(r_h=r_h).density(jnp.asarray(_RP_CEN)))
        ax.plot(_RP_CEN, pl / pl[iref], "--", color=col, alpha=0.7)
    ax.set_yscale("log"); ax.set_xlabel("r (pc)"); ax.set_ylabel("ρ(r)/ρ(r_h)")
    ax.set_title("Fiducial (ℳ=8) sampled (points) vs analytic Plummer (dashed)\n"
                 "normalised at r=r_h; turbulent BM19 tail broadens the wings")
    ax.legend()
    # panel B: control — turbulence on/off at r_h=0.5 proves envelope fidelity
    ax = axes[1]
    iref = int(np.argmin(np.abs(_RP_CEN - 0.5)))
    pl = np.asarray(PlummerProfile(r_h=0.5).density(jnp.asarray(_RP_CEN)))
    pe = _radial_profile_sampled(0.5, MACH, True, seeds)    # pure envelope (turbulence off)
    m8 = _radial_profile_sampled(0.5, MACH, False, seeds)   # fiducial (turbulence on)
    ax.plot(_RP_CEN, pl / pl[iref], "k--", lw=2, label="analytic Plummer")
    ax.plot(_RP_CEN, pe / pe[iref], "s-", color="C2", label="sampled, turbulence OFF")
    ax.plot(_RP_CEN, m8 / m8[iref], "o-", color="C3", label="sampled, ℳ=8 (turbulence ON)")
    ax.set_yscale("log"); ax.set_xlabel("r (pc)"); ax.set_ylabel("ρ(r)/ρ(r_h)")
    ax.set_title("Control (r_h=0.5): envelope sampling = analytic (few %);\n"
                 "turbulence is the large-r EXCESS; central cusp grid-under-resolved")
    ax.legend()
    fig.savefig(os.path.join(PLOTS, "cluster_radial_profile.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def _fig_beta_recovery(rec):
    rows = np.array(rec["rows"])  # (β, slope_s, σ_s, slope_dens, σ_dens, err)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    lo, hi = rows[:, 0].min() - 0.2, rows[:, 0].max() + 0.2
    ax.plot([lo, hi], [lo, hi], "k:", label="1:1 (perfect recovery)")
    ax.errorbar(rows[:, 0], rows[:, 1], yerr=rows[:, 2], fmt="o-", color="C0", capsize=3,
                label="log-density slope (≈ input β)")
    ax.errorbar(rows[:, 0], rows[:, 3], yerr=rows[:, 4], fmt="s-", color="C1", capsize=3,
                label="density e^s slope (compressed)")
    ax.set_xlabel("input β"); ax.set_ylabel("measured P(k) slope")
    ax.set_title(f"β recovery: log-density slope tracks input β to "
                 f"max |err|={rec['max_err']:.3f}\n(density slope is compressed by the BM19 tail)")
    ax.legend()
    fig.savefig(os.path.join(PLOTS, "cluster_beta_recovery.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def _fig_substructure_plane(sub):
    beta_rows = np.array(sub["beta_rows"]); conc_rows = np.array(sub["conc_rows"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # (m̄, s̄) plane with the two trajectories
    ax = axes[0]
    ax.plot(beta_rows[:, 2], beta_rows[:, 3], "o-", color="C3", label="β-sweep (r_h=0.5)")
    for b, _, m, s in beta_rows:
        ax.annotate(f"{b:.1f}", (m, s), fontsize=7, color="C3")
    ax.plot(conc_rows[:, 2], conc_rows[:, 3], "s-", color="C0", label="concentration (β=3.0)")
    for rh, _, m, s in conc_rows:
        ax.annotate(f"{rh:.1f}", (m, s), fontsize=7, color="C0")
    ax.set_xlabel("m̄  (normalised MST edge — concentration axis)")
    ax.set_ylabel("s̄  (normalised mean separation)")
    ax.set_title("CW04 (m̄, s̄) plane\nβ and concentration trace independent directions")
    ax.legend()
    # Q vs β and Q vs r_h (the conflation Q alone can't resolve)
    axes[1].plot(beta_rows[:, 0], beta_rows[:, 1], "o-", color="C3")
    axes[1].set_xlabel("β"); axes[1].set_ylabel("CW04 Q")
    axes[1].set_title("Q ↓ with β (substructure) — at fixed envelope")
    axes[2].plot(conc_rows[:, 0], conc_rows[:, 1], "s-", color="C0")
    axes[2].set_xlabel("envelope r_h (pc)"); axes[2].set_ylabel("CW04 Q")
    axes[2].set_title("Q ↓ with r_h (less concentrated) — at fixed β\n→ Q alone conflates the two")
    fig.savefig(os.path.join(PLOTS, "cluster_substructure_plane.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def _fig_velocity(vc, seed=0):
    ic = _ic(n=1200, seed=seed)
    pos = np.asarray(ic.positions); vel = np.asarray(ic.velocities)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sp = np.linalg.norm(vel, axis=1)
    axes[0].quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1], sp,
                   cmap="coolwarm", scale_units="xy", angles="xy", width=0.003)
    axes[0].set_aspect("equal"); axes[0].set_xlabel("x (pc)"); axes[0].set_ylabel("y (pc)")
    axes[0].set_title("Stellar velocity field (x–y)\ncoherent — nearby stars move together")
    # alignment vs separation
    sep, cos = vc["sep"], vc["cos"]
    edges = np.linspace(0, sep.max(), 18); cen = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(sep, edges) - 1, 0, len(cen) - 1)
    prof = np.array([cos[idx == i].mean() if np.any(idx == i) else np.nan for i in range(len(cen))])
    axes[1].plot(cen, prof, "o-")
    axes[1].axhline(0, color="k", lw=0.8, ls=":")
    axes[1].set_xlabel("pair separation (pc)"); axes[1].set_ylabel("mean velocity alignment cosθ")
    axes[1].set_title(f"Velocity coherence decays with separation\n"
                      f"near {vc['align_near']:+.2f} → far {vc['align_far']:+.2f}")
    fig.savefig(os.path.join(PLOTS, "cluster_velocity_coherence.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 78)
    print(f"FDF CLUSTER IC ACCEPTANCE  |  ℳ={MACH}, b={B}, α={ALPHA}, box={BOX}pc, shape={SHAPE}")
    print("=" * 78)
    r1 = ac_ic1_envelope()
    r2 = ac_ic2_virial()
    r3 = ac_ic3_substructure()
    r4 = ac_ic4_velocity_coherence()
    r5 = ac_ic5_gradient()
    r6 = ac_ic6_beta_recovery()

    print("\n[gallery] writing figures ...")
    _fig_scatter()
    _fig_radial_profile()
    _fig_substructure_plane(r3)
    _fig_velocity(r4)
    _fig_beta_recovery(r6)

    results = {"AC-IC1 envelope": r1, "AC-IC2 virial": r2, "AC-IC3 substructure": r3,
               "AC-IC4 coherence": r4, "AC-IC5 gradient": r5, "AC-IC6 β-recovery": r6}
    print("\n" + "=" * 78)
    print("SUMMARY")
    for name, r in results.items():
        print(f"  {name:<26} {'PASS' if r['passed'] else 'FAIL'}")
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

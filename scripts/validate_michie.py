#!/usr/bin/env python
"""
Michie-King anisotropic model validation figures.

Michie's (1963) radial-anisotropy term on King's (1966) lowered cutoff. Five
publication-quality figures, each anchored to passing tests in
``tests/validation/test_michie_physics.py`` and printing expected-vs-measured.

Key physics (verified): the DF f ~ exp(-s^2 u_t^2/2)[exp(W - u^2/2) - 1] is NOT a
pure function of Q = E + J^2/2 r_a^2 (the lowering term depends on J alone), so the
anisotropy beta(r) sits *below* the pure Osipkov-Merritt ceiling r^2/(r^2+r_a^2) --
the King energy cutoff suppresses radial anisotropy, increasingly so outward. The
figures compare the sampler to the model's *own* DF beta (2nd-moment oracle), with
the OM ceiling shown for reference.

Figures (-> anchoring tests):
  1. michie_density_king_limit.png   density vs r_a; large r_a overlays King
  2. michie_beta_profile.png         sampled beta(r) vs DF oracle, vs OM ceiling
  3. michie_velocity_equilibrium.png sigma_r & sigma_t; v<=v_esc; virial Q
  4. michie_king_limit_sweep.png     max|d rho| Michie vs King -> 0 as r_a -> inf
  5. michie_gradient_validation.png  AD vs central-FD d(obs)/d(W0, r_c, M)

References:
    Michie (1963), MNRAS 125, 127; King (1966), AJ 71, 64; Binney & Tremaine (2008).

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_michie.py
"""

import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from progenax.builders import compute_kinetic_energy, compute_potential_energy
from progenax.kinematics import MichieVelocityDF
from progenax.profiles import KingProfile, MichieProfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plotstyle import OI, apply_pub_style, panel_label, save_fig

apply_pub_style()

OUTPUT_DIR = "validation/plots"
W0, RC = 7.0, 1.0
RA_ANISO = 8.0  # clear anisotropy that truncates at W0=7 (r_t/r_c ~ 56)
RA_ISO = 1.0e4  # isotropic King limit
G = 1.0
N_SAMPLES = 60_000
N_VIRIAL = 5_000
SEED = 42


def _beta_oracle(W, s, n=400):
    """Analytic beta(W,s) from 2nd moments of the exact sampled density."""
    if W <= 0:
        return 0.0
    umax = np.sqrt(2.0 * W)
    UR, UT = np.meshgrid(np.linspace(-umax, umax, n), np.linspace(0.0, umax, n))
    bound = UR**2 + UT**2 < 2.0 * W
    w = UT * np.exp(-(s**2) * UT**2 / 2.0) * (np.exp(W - (UR**2 + UT**2) / 2.0) - 1.0)
    w = np.where(bound, np.maximum(w, 0.0), 0.0)
    norm = w.sum()
    return 1.0 - (w * UT**2).sum() / norm / (2.0 * (w * UR**2).sum() / norm)


def _radial_tangential(pos, vel):
    radii = jnp.linalg.norm(pos, axis=1)
    r_hat = pos / (radii[:, None] + 1e-30)
    v_r = jnp.sum(vel * r_hat, axis=1)
    v_t = jnp.linalg.norm(vel - v_r[:, None] * r_hat, axis=1)
    return np.asarray(radii), np.asarray(v_r), np.asarray(v_t)


# ============================================================================
# Figure 1 -- density profile + isotropic King limit
# ============================================================================
def fig_density_king_limit(output_dir):
    print("\n" + "=" * 60)
    print("FIG 1: density profile + isotropic King limit")
    print("=" * 60)

    king = KingProfile.from_W0_rc(W0, RC)
    r = jnp.linspace(0.05, 0.95 * float(king.r_t), 60)
    rk = np.asarray(king.density(r))
    rk = rk / rk[0]

    fig, ax = plt.subplots(figsize=(3.9, 3.5))
    ax.semilogy(
        np.asarray(r) / RC,
        rk,
        "-",
        color=OI["black"],
        lw=2.2,
        label="King (isotropic)",
        zorder=1,
    )

    max_rel = 0.0
    for ra, col in [
        (RA_ISO, OI["sky"]),
        (20.0, OI["green"]),
        (RA_ANISO, OI["vermilion"]),
    ]:
        prof = MichieProfile.from_W0_rc(W0, RC, ra)
        rm = np.asarray(prof.density(r))
        rm = rm / rm[0]
        lab = r"$r_a\to\infty$" if ra > 1e3 else rf"$r_a={ra:.0f}\,r_c$"
        ax.semilogy(np.asarray(r) / RC, rm, "--", color=col, lw=1.5, label=lab)
        if ra > 1e3:
            max_rel = float(np.max(np.abs(rm - rk)))
    iso_pass = max_rel < 1e-2
    print(
        f"  Michie(r_a->inf) vs King: max rel = {max_rel:.2e} (tol 1e-2)  "
        f"-> {'PASS' if iso_pass else 'FAIL'}"
    )
    rt_a = float(MichieProfile.from_W0_rc(W0, RC, RA_ANISO).r_t)
    rt_i = float(king.r_t)
    ext_pass = rt_a > rt_i
    print(
        f"  r_t: aniso(r_a=8)={rt_a:.1f} > iso(King)={rt_i:.1f}  "
        f"-> {'PASS' if ext_pass else 'FAIL'}"
    )

    ax.set_xlabel(r"$r / r_c$")
    ax.set_ylabel(r"$\rho(r) / \rho_0$")
    ax.set_xlim(0, float(king.r_t) / RC)
    ax.set_ylim(1e-4, 2)
    ax.legend(
        loc="upper right", title=r"$r_a\to\infty$ recovers King", title_fontsize=7.5
    )
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "michie_density_king_limit")
    print("  saved michie_density_king_limit.{png,pdf}")
    return iso_pass and ext_pass


# ============================================================================
# Figure 2 -- the anisotropy signature beta(r)
# ============================================================================
def fig_beta_profile(output_dir):
    print("\n" + "=" * 60)
    print("FIG 2: anisotropy beta(r) vs DF oracle vs Osipkov-Merritt ceiling")
    print("=" * 60)

    prof = MichieProfile.from_W0_rc(W0, RC, RA_ANISO)
    df = MichieVelocityDF(W0=W0, r_c=RC, r_a=RA_ANISO)
    m = jnp.ones(N_SAMPLES)
    kp, kv = jax.random.split(jax.random.PRNGKey(SEED))
    pos = prof.sample_positions(m, kp)
    vel = df.sample_velocities(pos, m, kv, G=G)
    radii, v_r, v_t = _radial_tangential(pos, vel)

    edges = [(1, 3), (3, 5), (5, 8), (8, 12), (12, 18), (18, 28), (28, 42)]
    mids, beta_s, oracle = [], [], []
    for lo, hi in edges:
        msk = (radii >= lo) & (radii < hi)
        if int(msk.sum()) < 200:
            continue
        rmid = float(np.mean(radii[msk]))
        sr2 = float(np.mean(v_r[msk] ** 2))
        st2 = float(np.mean(v_t[msk] ** 2))
        mids.append(rmid)
        beta_s.append(1.0 - st2 / (2.0 * sr2))
        W_mid = float(
            jnp.interp(rmid / RC, df.xi_grid, df.psi_grid, left=df.W0, right=0.0)
        )
        oracle.append(_beta_oracle(W_mid, rmid / RA_ANISO))
    mids, beta_s, oracle = np.array(mids), np.array(beta_s), np.array(oracle)
    rr = np.linspace(0.1, float(prof.r_t), 200)
    Wc = np.asarray(jnp.interp(rr / RC, df.xi_grid, df.psi_grid, left=df.W0, right=0.0))
    oracle_curve = np.array(
        [_beta_oracle(float(w), float(r) / RA_ANISO) for w, r in zip(Wc, rr)]
    )
    om_curve = rr**2 / (rr**2 + RA_ANISO**2)

    max_dev = float(np.max(np.abs(beta_s - oracle)))
    below_om = bool(np.all(beta_s <= mids**2 / (mids**2 + RA_ANISO**2) + 1e-9))
    passed = max_dev < 0.05 and below_om
    print(
        f"  sampled beta vs DF oracle: max dev = {max_dev:.3f} (tol 0.05)  "
        f"-> {'PASS' if max_dev < 0.05 else 'FAIL'}"
    )
    print(
        f"  beta below Osipkov-Merritt ceiling: {below_om}  "
        f"-> {'PASS' if below_om else 'FAIL'}"
    )

    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    ax.plot(
        rr / RA_ANISO,
        om_curve,
        ":",
        color="0.5",
        lw=1.5,
        label=r"Osipkov-Merritt $\frac{r^2}{r^2+r_a^2}$ (ceiling)",
    )
    ax.plot(
        rr / RA_ANISO,
        oracle_curve,
        "-",
        color=OI["black"],
        lw=2.0,
        label="Michie-King DF oracle",
    )
    ax.plot(
        mids / RA_ANISO,
        beta_s,
        "o",
        color=OI["vermilion"],
        ms=5,
        mec="white",
        mew=0.5,
        label=rf"sampled ($N={N_SAMPLES:,}$)",
    )
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.set_xlabel(r"$r / r_a$")
    ax.set_ylabel(r"anisotropy $\beta(r) = 1 - \sigma_t^2 / 2\sigma_r^2$")
    ax.set_xlim(0, float(prof.r_t) / RA_ANISO)
    ax.set_ylim(-0.05, 1.0)
    ax.legend(loc="lower right")
    ax.text(
        0.04,
        0.95,
        "isotropic core\n→ radial outward",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
    )
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "michie_beta_profile")
    print("  saved michie_beta_profile.{png,pdf}")
    return passed


# ============================================================================
# Figure 3 -- velocity-space equilibrium (anisotropic dispersions)
# ============================================================================
def fig_velocity_equilibrium(output_dir):
    print("\n" + "=" * 60)
    print("FIG 3: velocity-space equilibrium (anisotropic dispersions)")
    print("=" * 60)

    prof = MichieProfile.from_W0_rc(W0, RC, RA_ANISO)
    df = MichieVelocityDF(W0=W0, r_c=RC, r_a=RA_ANISO)
    m = jnp.ones(N_SAMPLES)
    kp, kv = jax.random.split(jax.random.PRNGKey(SEED))
    pos = prof.sample_positions(m, kp)
    vel = df.sample_velocities(pos, m, kv, G=G)
    radii, v_r, v_t = _radial_tangential(pos, vel)

    edges = [(1, 3), (3, 6), (6, 10), (10, 16), (16, 26), (26, 42)]
    mids, sr, st, sr_e = [], [], [], []
    for lo, hi in edges:
        msk = (radii >= lo) & (radii < hi)
        n = int(msk.sum())
        if n < 200:
            continue
        mids.append(float(np.mean(radii[msk])))
        sr.append(float(np.std(v_r[msk])))
        st.append(float(np.std(v_t[msk] / np.sqrt(2.0))))  # per-component tangential
        sr_e.append(sr[-1] / np.sqrt(2.0 * max(n - 1, 1)))

    # boundedness + virial (separate seed)
    mv = jnp.ones(N_VIRIAL)
    kp2, kv2 = jax.random.split(jax.random.PRNGKey(0))
    pv = prof.sample_positions(mv, kp2)
    vv = df.sample_velocities(pv, mv, kv2, G=G)
    rv = jnp.linalg.norm(pv, axis=1)
    Wv = jnp.maximum(
        jnp.interp(rv / df.r_c, df.xi_grid, df.psi_grid, left=df.W0, right=0.0), 0.0
    )
    sigma = jnp.sqrt(G * jnp.sum(mv) / (9.0 * df.r_c * df.mu))
    v_esc = sigma * jnp.sqrt(2.0 * Wv)
    ratio = np.asarray(jnp.linalg.norm(vv, axis=1) / (v_esc + 1e-30))
    bound_frac = float(jnp.mean(jnp.linalg.norm(vv, axis=1) <= v_esc + 1e-9))
    Q = float(
        compute_kinetic_energy(vv, mv) / jnp.abs(compute_potential_energy(pv, mv, G=G))
    )
    bound_pass = bound_frac == 1.0
    q_pass = abs(Q - 0.5) < 0.05
    passed = bound_pass and q_pass
    print(
        f"  bound fraction: {bound_frac * 100:.2f}%  -> {'PASS' if bound_pass else 'FAIL'}"
    )
    print(
        f"  virial Q=T/|V|: {Q:.3f} (expect 0.5+-0.05)  -> {'PASS' if q_pass else 'FAIL'}"
    )

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.2, 2.7))

    axA.errorbar(
        mids,
        sr,
        yerr=sr_e,
        fmt="o",
        color=OI["vermilion"],
        ms=5,
        capsize=2,
        lw=1.0,
        label=r"$\sigma_r$ (radial)",
    )
    axA.plot(mids, st, "s", color=OI["blue"], ms=5, label=r"$\sigma_t$ (per comp.)")
    axA.set_xlabel(r"$r$ [$r_c$]")
    axA.set_ylabel(r"$\sigma$ [code units, $G{=}1$]")
    axA.set_xscale("log")
    axA.legend(loc="upper right")
    axA.text(
        0.04, 0.1, r"$\sigma_r>\sigma_t$ outward", transform=axA.transAxes, fontsize=8
    )
    panel_label(axA, "(a)", loc="upper left")

    axB.hist(ratio, bins=40, color=OI["sky"], edgecolor="white", linewidth=0.3)
    axB.axvline(1.0, color=OI["vermilion"], ls="--", lw=1.4)
    axB.text(
        0.5,
        0.92,
        rf"{bound_frac * 100:.1f}% bound",
        transform=axB.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
    )
    axB.set_xlabel(r"$v / v_{\rm esc}(r)$")
    axB.set_ylabel("count")
    axB.set_xlim(0, 1.05)
    panel_label(axB, "(b)", loc="upper left")

    axC.axhspan(0.45, 0.55, color=OI["green"], alpha=0.18, label=r"equilib. $\pm0.05$")
    axC.axhline(0.5, color=OI["black"], ls="--", label=r"$Q=0.5$")
    axC.plot([0], [Q], "o", color=OI["vermilion"], ms=8, zorder=5)
    axC.text(
        0.5,
        0.84,
        rf"$Q={Q:.3f}$",
        transform=axC.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.6),
    )
    axC.set_xticks([])
    axC.set_xlim(-0.6, 0.6)
    axC.set_ylim(0.40, 0.60)
    axC.set_xlabel("unscaled IC")
    axC.set_ylabel(r"virial ratio $Q = T/|V|$")
    axC.legend(loc="lower center", fontsize=7)
    panel_label(axC, "(c)", loc="upper left")

    fig.tight_layout(pad=0.4, w_pad=0.8)
    save_fig(fig, output_dir, "michie_velocity_equilibrium")
    print("  saved michie_velocity_equilibrium.{png,pdf}")
    return passed


# ============================================================================
# Figure 4 -- isotropic King limit convergence sweep
# ============================================================================
def fig_king_limit_sweep(output_dir):
    print("\n" + "=" * 60)
    print("FIG 4: isotropic King-limit convergence vs r_a/r_c")
    print("=" * 60)

    king = KingProfile.from_W0_rc(W0, RC)
    r = jnp.linspace(0.05, 0.9 * float(king.r_t), 50)
    rk = np.asarray(king.density(r))
    rk = rk / rk[0]

    ras = [5.0, 6.0, 8.0, 12.0, 20.0, 40.0, 100.0, 1000.0]
    max_rel = []
    for ra in ras:
        rm = np.asarray(MichieProfile.from_W0_rc(W0, RC, ra).density(r))
        rm = rm / rm[0]
        max_rel.append(float(np.max(np.abs(rm - rk))))
    max_rel = np.array(max_rel)
    monotone = bool(np.all(np.diff(max_rel) < 1e-9))
    converges = max_rel[-1] < 1e-2
    passed = converges
    print(f"  max|d rho| at r_a/r_c=[5..1000]: {[f'{x:.1e}' for x in max_rel]}")
    print(
        f"  converges (r_a=1000): {max_rel[-1]:.2e} < 1e-2  "
        f"-> {'PASS' if converges else 'FAIL'};  monotone decreasing: {monotone}"
    )

    fig, ax = plt.subplots(figsize=(3.9, 3.3))
    ax.loglog(ras, max_rel, "o-", color=OI["blue"], ms=6, mec="white", mew=0.5)
    ax.axhline(1e-2, color=OI["vermilion"], ls="--", lw=1.2, label=r"$10^{-2}$ tol")
    ax.set_xlabel(r"$r_a / r_c$")
    ax.set_ylabel(r"max$_r\,|\rho_{\rm Michie}-\rho_{\rm King}| / \rho_0$")
    ax.legend(loc="upper right")
    ax.text(
        0.5,
        0.06,
        r"$r_a\to\infty$: Michie $\to$ King",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "michie_king_limit_sweep")
    print("  saved michie_king_limit_sweep.{png,pdf}")
    return passed


# ============================================================================
# Figure 5 -- gradient validation
# ============================================================================
def _grad_sweep(loss, xs, h):
    ad = np.array([float(jax.grad(loss)(float(x))) for x in xs])
    fd = np.array(
        [float((loss(float(x) + h) - loss(float(x) - h)) / (2 * h)) for x in xs]
    )
    rel = np.abs(ad - fd) / (np.abs(ad) + np.abs(fd) + 1e-30)
    return ad, fd, rel


def fig_gradient_validation(output_dir):
    print("\n" + "=" * 60)
    print("FIG 5: gradient validation (autodiff vs finite difference)")
    print("=" * 60)

    def log_density_at(r_phys, W0_, r_c_):
        p = MichieProfile.from_W0_rc(W0_, r_c_, RA_ANISO)
        return jnp.log(p.density(jnp.array([r_phys]))[0] + 1e-30)

    def vel_scale(M_total):
        df = MichieVelocityDF(W0=W0, r_c=RC, r_a=RA_ANISO)
        return jnp.sqrt(G * M_total / (9.0 * df.r_c * df.mu))

    specs = [
        # W0 range kept where r_a=8 still truncates (higher W0 needs larger r_a).
        (
            "W0",
            r"$W_0$",
            r"$\partial\,\log\rho(2\,r_c) / \partial W_0$",
            lambda w: log_density_at(2.0, w, 1.0),
            np.linspace(4.0, 7.0, 9),
            1e-3,
        ),
        (
            "r_c",
            r"$r_c$ [pc]",
            r"$\partial\,\log\rho(2\,{\rm pc}) / \partial r_c$",
            lambda rc: log_density_at(2.0, 7.0, rc),
            np.linspace(0.7, 1.6, 9),
            1e-4,
        ),
        (
            "M",
            r"$M_{\rm tot}$ [$M_\odot$]",
            r"$\partial\,\sigma_0 / \partial M_{\rm tot}$",
            lambda M: vel_scale(M),
            np.linspace(300.0, 2000.0, 9),
            1.0,
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    worst = 0.0
    for ax, (key, xlab, ylab, loss, xs, h), tag in zip(axes, specs, "abc"):
        ad, fd, rel = _grad_sweep(loss, xs, h)
        worst = max(worst, float(np.max(rel)))
        ax.plot(xs, ad, "-", color=OI["blue"], lw=1.8, label="autodiff", zorder=2)
        ax.plot(
            xs,
            fd,
            "o",
            color=OI["vermilion"],
            ms=4.5,
            mfc="none",
            mew=1.2,
            label="finite diff",
            zorder=3,
        )
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.legend(loc="best")
        ax.text(
            0.5,
            0.05,
            rf"max rel err $={np.max(rel):.0e}$",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.5),
        )
        panel_label(ax, f"({tag})", loc="upper left")
        print(
            f"  d(loss)/d{key:3}: max rel err {np.max(rel):.2e}  "
            f"-> {'DIFFERENTIABLE' if np.max(rel) < 1e-3 else 'CHECK'}"
        )

    passed = worst < 1e-3
    print(f"  overall worst rel err {worst:.2e}  -> {'PASS' if passed else 'FAIL'}")
    fig.tight_layout(pad=0.4, w_pad=0.8)
    save_fig(fig, output_dir, "michie_gradient_validation")
    print("  saved michie_gradient_validation.{png,pdf}")
    return passed


def main():
    print("\n" + "=" * 70)
    print("PROGENAX MICHIE-KING ANISOTROPIC MODEL VALIDATION FIGURES")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {
        "Fig 1  density + King limit": fig_density_king_limit(OUTPUT_DIR),
        "Fig 2  anisotropy beta(r)": fig_beta_profile(OUTPUT_DIR),
        "Fig 3  velocity-space equilibrium": fig_velocity_equilibrium(OUTPUT_DIR),
        "Fig 4  King-limit convergence sweep": fig_king_limit_sweep(OUTPUT_DIR),
        "Fig 5  gradient validation (AD vs FD)": fig_gradient_validation(OUTPUT_DIR),
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    all_ok = all(results.values())
    print("=" * 70)
    print(
        "  ALL MICHIE VALIDATION FIGURES PASS"
        if all_ok
        else "  SOME MICHIE VALIDATION FIGURES FAILED"
    )
    print("=" * 70)
    print(f"\nFigures written to {OUTPUT_DIR}/michie_*.png")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

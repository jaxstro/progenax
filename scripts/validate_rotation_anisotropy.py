#!/usr/bin/env python
"""
Rotation transforms + Osipkov-Merritt anisotropy validation figures.

Five publication-quality figures anchored to passing tests in
``tests/validation/test_rotation_anisotropy_physics.py``, each printing
expected-vs-measured.

Two distinct kinematic structures:
  - ROTATION: an *additive* streaming transform v -> v + v_rot. Solid body gives
    v_phi(R) = Omega R; the differential curve is v_phi(R)=v_peak(R/R_peak)e^{1-R/R_peak}.
    We validate the added field v_rot = v_after - v_before (exact, scatter-free).
  - OSIPKOV-MERRITT ANISOTROPY: the `anisotropy_radius` r_a realizes
    beta(r) = r^2/(r^2+r_a^2) *exactly* via a velocity-direction stretch
    (Merritt 1985, Eq 15) -- in contrast to the self-consistent Michie-King DF,
    whose beta is suppressed below this OM ceiling.

Figures (-> anchoring tests):
  1. rotation_solid_body.png       added v_phi(R)=Omega R; L_z budget
  2. rotation_differential.png     added v_phi(R) peaked curve
  3. om_anisotropy_beta.png        Plummer & EFF beta(r) vs exact OM target
  4. rotation_gradient.png         AD vs central-FD d(KE)/d(Omega, v_peak); d beta/d r_a
  5. rotation_velocity_field.png   face-on (x,y): isotropic vs solid-body streaming

References:
    Binney & Tremaine (2008) Sec 4.8; Merritt (1985), AJ 90, 1027; Lynden-Bell (1960).

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_rotation_anisotropy.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from progenax.profiles import PlummerProfile, EFFProfile
from progenax.kinematics import PlummerVelocityDF, EFFVelocityDF
from progenax.kinematics.rotation import (
    apply_solid_body_rotation,
    apply_differential_rotation,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plotstyle import OI, apply_pub_style, panel_label, save_fig

apply_pub_style()

OUTPUT_DIR = "validation/plots"
G = 1.0
ZAXIS = jnp.array([0.0, 0.0, 1.0])
N_SAMPLES = 60_000
SEED = 42


def _cyl(pos, vel):
    x, y = np.asarray(pos[:, 0]), np.asarray(pos[:, 1])
    R = np.sqrt(x ** 2 + y ** 2)
    v_phi = (x * np.asarray(vel[:, 1]) - y * np.asarray(vel[:, 0])) / (R + 1e-30)
    return R, v_phi


def _isotropic_ic(n=N_SAMPLES, seed=0):
    prof = PlummerProfile(r_h=1.0)
    df = PlummerVelocityDF(r_h=1.0)
    m = jnp.ones(n)
    kp, kv = jax.random.split(jax.random.PRNGKey(seed))
    pos = prof.sample_positions(m, kp)
    vel = df.sample_velocities(pos, m, kv, G=G)
    return m, pos, vel


def _beta_binned(pos, vel, edges):
    radii = np.linalg.norm(np.asarray(pos), axis=1)
    r_hat = np.asarray(pos) / (radii[:, None] + 1e-30)
    v_r = np.sum(np.asarray(vel) * r_hat, axis=1)
    v_t = np.linalg.norm(np.asarray(vel) - v_r[:, None] * r_hat, axis=1)
    mids, beta = [], []
    for lo, hi in edges:
        mm = (radii >= lo) & (radii < hi)
        if mm.sum() < 200:
            continue
        mids.append(0.5 * (lo + hi))
        beta.append(1.0 - np.mean(v_t[mm] ** 2) / (2.0 * np.mean(v_r[mm] ** 2)))
    return np.array(mids), np.array(beta)


# ============================================================================
# Figure 1 -- solid-body rotation
# ============================================================================
def fig_solid_body(output_dir):
    print("\n" + "=" * 60)
    print("FIG 1: solid-body rotation -- added v_phi(R) = Omega R")
    print("=" * 60)

    omega = 0.3
    m, pos, vel = _isotropic_ic()
    dvel = apply_solid_body_rotation(vel, pos, omega, ZAXIS) - vel
    R, dphi = _cyl(pos, dvel)
    sel = R < 3.0
    slope, intercept = np.polyfit(R[sel], dphi[sel], 1)
    resid = float(np.max(np.abs(dphi[sel] - omega * R[sel])))
    Rcyl = np.sqrt(np.asarray(pos[:, 0]) ** 2 + np.asarray(pos[:, 1]) ** 2)
    dLz = float(jnp.sum(m * (pos[:, 0] * dvel[:, 1] - pos[:, 1] * dvel[:, 0])))
    Lz_exp = omega * float(np.sum(np.asarray(m) * Rcyl ** 2))
    slope_pass = abs(slope - omega) < 1e-6
    lz_pass = abs(dLz - Lz_exp) / Lz_exp < 1e-6
    passed = slope_pass and lz_pass
    print(f"  fitted slope = {slope:.6f} (Omega={omega})  max resid={resid:.1e}  "
          f"-> {'PASS' if slope_pass else 'FAIL'}")
    print(f"  added L_z = {dLz:.1f} vs Omega*sum(m R^2) = {Lz_exp:.1f}  "
          f"-> {'PASS' if lz_pass else 'FAIL'}")

    fig, ax = plt.subplots(figsize=(3.9, 3.4))
    idx = np.argsort(R[sel])[::200]
    ax.plot(R[sel][idx], dphi[sel][idx], "o", color=OI["sky"], ms=3, mec="none",
            alpha=0.5, label="per particle")
    rr = np.linspace(0, 3, 50)
    ax.plot(rr, omega * rr, "-", color=OI["vermilion"], lw=2.0,
            label=rf"$\Omega R$ ($\Omega={omega}$)")
    ax.set_xlabel(r"cylindrical $R$ [pc]")
    ax.set_ylabel(r"added $v_\phi$ [code units]")
    ax.set_xlim(0, 3)
    ax.legend(loc="upper left")
    ax.text(0.96, 0.06, rf"slope $={slope:.4f}$" + "\n" + rf"$L_z$ err $<10^{{-6}}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.5))
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "rotation_solid_body")
    print("  saved rotation_solid_body.{png,pdf}")
    return passed


# ============================================================================
# Figure 2 -- differential rotation
# ============================================================================
def fig_differential(output_dir):
    print("\n" + "=" * 60)
    print("FIG 2: differential rotation -- added v_phi(R) peaked curve")
    print("=" * 60)

    v_peak, R_peak = 2.0, 1.0
    m, pos, vel = _isotropic_ic()
    dvel = apply_differential_rotation(vel, pos, v_peak, R_peak, ZAXIS) - vel
    R, dphi = _cyl(pos, dvel)
    expected = v_peak * (R / R_peak) * np.exp(1 - R / R_peak)
    resid = float(np.max(np.abs(dphi - expected)))
    passed = resid < 1e-6
    print(f"  max |v_phi - curve| = {resid:.1e}  -> {'PASS' if passed else 'FAIL'}")

    fig, ax = plt.subplots(figsize=(3.9, 3.4))
    sel = R < 5.0
    idx = np.argsort(R[sel])[::200]
    ax.plot(R[sel][idx], dphi[sel][idx], "o", color=OI["sky"], ms=3, mec="none",
            alpha=0.5, label="per particle")
    rr = np.linspace(0, 5, 200)
    ax.plot(rr, v_peak * (rr / R_peak) * np.exp(1 - rr / R_peak), "-",
            color=OI["vermilion"], lw=2.0,
            label=r"$v_{\rm peak}\frac{R}{R_p}e^{1-R/R_p}$")
    ax.axvline(R_peak, color="0.5", ls=":", lw=1.0)
    ax.axhline(v_peak, color="0.5", ls=":", lw=1.0)
    ax.text(R_peak + 0.1, 0.3, r"$R_{\rm peak}$", color="0.4", fontsize=8.5)
    ax.set_xlabel(r"cylindrical $R$ [pc]")
    ax.set_ylabel(r"added $v_\phi$ [code units]")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, v_peak * 1.15)
    ax.legend(loc="upper right")
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "rotation_differential")
    print("  saved rotation_differential.{png,pdf}")
    return passed


# ============================================================================
# Figure 3 -- Osipkov-Merritt anisotropy beta(r)
# ============================================================================
def fig_om_anisotropy(output_dir):
    print("\n" + "=" * 60)
    print("FIG 3: Osipkov-Merritt anisotropy beta(r) = r^2/(r^2+r_a^2)")
    print("=" * 60)

    r_a = 1.5
    edges = [(0.3, 0.6), (0.6, 0.9), (0.9, 1.3), (1.3, 1.8), (1.8, 2.5), (2.5, 3.5)]
    m = jnp.ones(N_SAMPLES)

    prof_p = PlummerProfile(r_h=1.0)
    df_p = PlummerVelocityDF(r_h=1.0, anisotropy_radius=r_a)
    kp, kv = jax.random.split(jax.random.PRNGKey(SEED))
    pos_p = prof_p.sample_positions(m, kp)
    vel_p = df_p.sample_velocities(pos_p, m, kv, G=G)
    mp, bp = _beta_binned(pos_p, vel_p, edges)

    prof_e = EFFProfile(a=1.0, gamma=4.0, r_t=15.0)
    df_e = EFFVelocityDF(a=1.0, gamma=4.0, r_t=15.0, anisotropy_radius=r_a)
    kp2, kv2 = jax.random.split(jax.random.PRNGKey(SEED + 1))
    pos_e = prof_e.sample_positions(m, kp2)
    vel_e = df_e.sample_velocities(pos_e, m, kv2, G=G)
    me, be = _beta_binned(pos_e, vel_e, edges)

    tp = mp ** 2 / (mp ** 2 + r_a ** 2)
    te = me ** 2 / (me ** 2 + r_a ** 2)
    dev_p = float(np.max(np.abs(bp - tp)))
    dev_e = float(np.max(np.abs(be - te)))
    passed = dev_p < 0.04 and dev_e < 0.05
    print(f"  Plummer OM beta vs target: max dev = {dev_p:.3f} (tol 0.04)  "
          f"-> {'PASS' if dev_p < 0.04 else 'FAIL'}")
    print(f"  EFF OM beta vs target:     max dev = {dev_e:.3f} (tol 0.05)  "
          f"-> {'PASS' if dev_e < 0.05 else 'FAIL'}")

    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    rr = np.linspace(0.1, 3.5, 200)
    ax.plot(rr, rr ** 2 / (rr ** 2 + r_a ** 2), "-", color=OI["black"], lw=2.0,
            label=r"exact OM $\frac{r^2}{r^2+r_a^2}$")
    ax.plot(mp, bp, "o", color=OI["blue"], ms=5, mec="white", mew=0.5, label="Plummer (sampled)")
    ax.plot(me, be, "s", color=OI["vermilion"], ms=5, mec="white", mew=0.5, label="EFF (sampled)")
    ax.axvline(r_a, color="0.5", ls=":", lw=1.0)
    ax.text(r_a + 0.05, 0.05, r"$r_a$", color="0.4", fontsize=8.5)
    ax.set_xlabel(r"$r$ [pc]")
    ax.set_ylabel(r"anisotropy $\beta(r)$")
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right")
    ax.text(0.04, 0.96, "direction-stretch\n→ exact OM\n(cf. Michie: suppressed)",
            transform=ax.transAxes, fontsize=7.5, va="top")
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "om_anisotropy_beta")
    print("  saved om_anisotropy_beta.{png,pdf}")
    return passed


# ============================================================================
# Figure 4 -- gradient validation
# ============================================================================
def _grad_sweep(loss, xs, h):
    ad = np.array([float(jax.grad(loss)(float(x))) for x in xs])
    fd = np.array([float((loss(float(x) + h) - loss(float(x) - h)) / (2 * h)) for x in xs])
    rel = np.abs(ad - fd) / (np.abs(ad) + np.abs(fd) + 1e-30)
    return ad, fd, rel


def fig_gradient(output_dir):
    print("\n" + "=" * 60)
    print("FIG 4: gradient validation (autodiff vs finite difference)")
    print("=" * 60)
    m, pos, vel = _isotropic_ic(n=3000)

    def rot_ke_omega(omega):
        v = apply_solid_body_rotation(vel, pos, omega, ZAXIS)
        return jnp.sum(m * jnp.sum(v ** 2, axis=1))

    def rot_ke_vpeak(v_peak):
        v = apply_differential_rotation(vel, pos, v_peak, 1.0, ZAXIS)
        return jnp.sum(m * jnp.sum(v ** 2, axis=1))

    # r_a anisotropy: with a fixed PRNG key the OM stretch makes <v_r^2> a smooth,
    # deterministic function of r_a (the random draws are fixed, then r_a-scaled).
    prof_p = PlummerProfile(r_h=1.0)
    kfix = jax.random.PRNGKey(7)
    pos_fix = prof_p.sample_positions(jnp.ones(3000), jax.random.PRNGKey(8))

    def mean_vr2(r_a):
        df = PlummerVelocityDF(r_h=1.0, anisotropy_radius=r_a)
        v = df.sample_velocities(pos_fix, jnp.ones(3000), kfix, G=G)
        radii = jnp.linalg.norm(pos_fix, axis=1)
        r_hat = pos_fix / (radii[:, None] + 1e-30)
        v_r = jnp.sum(v * r_hat, axis=1)
        return jnp.mean(v_r ** 2)

    specs = [
        ("Omega", r"$\Omega$ [rad/Myr]", r"$\partial\,T_{\rm rot} / \partial \Omega$",
         rot_ke_omega, np.linspace(0.1, 0.6, 9), 1e-5),
        ("v_peak", r"$v_{\rm peak}$", r"$\partial\,T_{\rm rot} / \partial v_{\rm peak}$",
         rot_ke_vpeak, np.linspace(1.0, 3.0, 9), 1e-5),
        ("r_a", r"$r_a$ [pc]", r"$\partial\,\langle v_r^2\rangle / \partial r_a$",
         mean_vr2, np.linspace(1.0, 3.0, 9), 1e-4),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    worst = 0.0
    for ax, (key, xlab, ylab, loss, xs, h), tag in zip(axes, specs, "abc"):
        ad, fd, rel = _grad_sweep(loss, xs, h)
        worst = max(worst, float(np.max(rel)))
        ax.plot(xs, ad, "-", color=OI["blue"], lw=1.8, label="autodiff", zorder=2)
        ax.plot(xs, fd, "o", color=OI["vermilion"], ms=4.5, mfc="none", mew=1.2,
                label="finite diff", zorder=3)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.legend(loc="best")
        ax.text(0.5, 0.05, rf"max rel err $={np.max(rel):.0e}$", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.5))
        panel_label(ax, f"({tag})", loc="upper left")
        print(f"  d(loss)/d{key:6}: max rel err {np.max(rel):.2e}  "
              f"-> {'DIFFERENTIABLE' if np.max(rel) < 1e-3 else 'CHECK'}")

    passed = worst < 1e-3
    print(f"  overall worst rel err {worst:.2e}  -> {'PASS' if passed else 'FAIL'}")
    fig.tight_layout(pad=0.4, w_pad=0.8)
    save_fig(fig, output_dir, "rotation_gradient")
    print("  saved rotation_gradient.{png,pdf}")
    return passed


# ============================================================================
# Figure 5 -- face-on velocity field: isotropic vs solid-body
# ============================================================================
def fig_velocity_field(output_dir):
    print("\n" + "=" * 60)
    print("FIG 5: face-on velocity field (isotropic vs solid-body)")
    print("=" * 60)

    omega = 0.3
    m, pos, vel = _isotropic_ic(n=20_000, seed=3)
    x, y, z = np.asarray(pos[:, 0]), np.asarray(pos[:, 1]), np.asarray(pos[:, 2])
    R = np.sqrt(x ** 2 + y ** 2)
    slab = (np.abs(z) < 0.4) & (R < 2.5) & (R > 0.15)
    idx = np.where(slab)[0]
    rng = np.random.default_rng(0)
    if len(idx) > 280:
        idx = rng.choice(idx, 280, replace=False)

    vx, vy = np.asarray(vel[:, 0]), np.asarray(vel[:, 1])
    # isotropic in-plane directions (unit), and solid-body streaming directions
    norm_i = np.hypot(vx[idx], vy[idx]) + 1e-30
    vrot = apply_solid_body_rotation(vel, pos, omega, ZAXIS) - vel
    sx, sy = np.asarray(vrot[:, 0]), np.asarray(vrot[:, 1])
    norm_s = np.hypot(sx[idx], sy[idx]) + 1e-30
    vphi = (x[idx] * sy[idx] - y[idx] * sx[idx]) / (R[idx] + 1e-30)
    mean_vphi = float(np.mean(vphi))
    coherent = bool(np.all(vphi > 0))  # solid-body: all co-rotating
    print(f"  solid-body streaming: all co-rotating (v_phi>0): {coherent}; "
          f"<v_phi>={mean_vphi:.3f}  -> {'PASS' if coherent else 'FAIL'}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.8, 3.4), sharex=True, sharey=True)
    axA.quiver(x[idx], y[idx], vx[idx] / norm_i, vy[idx] / norm_i, color=OI["sky"],
               scale=22, width=0.005, alpha=0.85)
    axA.set_title("isotropic IC", fontsize=9)
    axA.set_xlabel(r"$x$ [pc]"); axA.set_ylabel(r"$y$ [pc]")
    panel_label(axA, "(a)", loc="upper left")

    axB.quiver(x[idx], y[idx], sx[idx] / norm_s, sy[idx] / norm_s, vphi,
               cmap="viridis", scale=22, width=0.005)
    axB.set_title(rf"$+$ solid body ($\Omega={omega}$)", fontsize=9)
    axB.set_xlabel(r"$x$ [pc]")
    panel_label(axB, "(b)", loc="upper left")
    for ax in (axA, axB):
        ax.set_aspect("equal"); ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6)

    fig.tight_layout(pad=0.4, w_pad=0.6)
    save_fig(fig, output_dir, "rotation_velocity_field")
    print("  saved rotation_velocity_field.{png,pdf}")
    return coherent


def main():
    print("\n" + "=" * 70)
    print("PROGENAX ROTATION + OSIPKOV-MERRITT ANISOTROPY VALIDATION FIGURES")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {
        "Fig 1  solid-body rotation": fig_solid_body(OUTPUT_DIR),
        "Fig 2  differential rotation": fig_differential(OUTPUT_DIR),
        "Fig 3  Osipkov-Merritt anisotropy": fig_om_anisotropy(OUTPUT_DIR),
        "Fig 4  gradient validation (AD vs FD)": fig_gradient(OUTPUT_DIR),
        "Fig 5  face-on velocity field": fig_velocity_field(OUTPUT_DIR),
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    all_ok = all(results.values())
    print("=" * 70)
    print("  ALL ROTATION/ANISOTROPY FIGURES PASS" if all_ok
          else "  SOME FIGURES FAILED")
    print("=" * 70)
    print(f"\nFigures written to {OUTPUT_DIR}/")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

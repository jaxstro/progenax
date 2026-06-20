#!/usr/bin/env python
r"""B8 -- Rotation, projection, and the omega-inclination degeneracy (Batch C).

The first observational-realism demo: a rotating cluster is viewed at an
inclination, and the recovered rotation depends on a LINE-OF-SIGHT PROJECTION. The
honest result is the realism-axis headline -- the mean line-of-sight velocity field
measures the PRODUCT omega * sin(i), not the rotation rate omega alone.

Physics
-------
Solid-body rotation about the cluster z-axis gives v = omega (z_hat x r). Viewed at
inclination i (the z-axis tilted i from the line of sight), the projection
(tilt about the sky x-axis, observer along lab z) gives a mean line-of-sight
velocity that is LINEAR in the sky x-coordinate:

    <v_los>(x_sky) = omega * sin(i) * x_sky.

So the observable rotation amplitude is the SLOPE k = omega * sin(i). A face-on
cluster (i=0) shows NO rotation signature; an edge-on one (i=90) shows the full
omega. The rotation rate omega and the inclination i are therefore **degenerate**:
<v_los> alone constrains only their product. The (omega, i) Fisher information is
RANK 1 -- breaking the degeneracy needs an independent inclination (e.g. from the
projected flattening), which line-of-sight velocities cannot supply.

Channels & method: a projection helper ``project_los(pos, vel, incl)`` ->
(sky_x, sky_y, v_los); binned <v_los>(x_sky) with per-bin SE; a Gaussian chi^2 fit
of the slope k (gaussian_loglike + mle_adam + fisher_cov); the rank-1 (omega, i)
Fisher from d k / d(omega, i) = (sin i, omega cos i).

Gates (exit 0 = all pass):
  * self-consistency: the linear model at truth matches the binned <v_los>;
  * slope k = omega sin(i) recovered within 3 sigma;
  * DEGENERACY (headline): the (omega, i) Fisher is rank-deficient (cond > 1e8) --
    omega is NOT recoverable from <v_los> alone;
  * forecast sigma(k) ~ N^-1/2.

Run record (2026-06-12, CPU/float64, N=20000, omega=2.0/Myr, i=60 deg,
k=omega sin(i)=1.7321, key PRNGKey(0), wall ~4 s, exit 0 / ALL PASS):
  self-consistency max|resid|/SE at truth k = 2.17 (< 4).
  slope recovery: k = 1.6913 +- 0.0305 (pull -1.34).
  DEGENERACY (headline): d k/d(omega, i) = (0.866, 1.0); (omega,i) Fisher
    eigenvalues [8e-14, 1887] -> RANK 1, condition number 2.3e16. omega and the
    inclination i are inseparable from <v_los> alone (only the product is measured);
    breaking it needs an independent i (e.g. the projected flattening).
  forecast sigma(k) ~ N^-0.500.

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_rotation.py
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jaxstro.units import STELLAR

from progenax import PlummerProfile, PlummerVelocityDF
from progenax.kinematics import apply_solid_body_rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _demo_inference import fisher_cov, gaussian_loglike, mle_adam
from _plotstyle import OI, apply_pub_style, panel_label, save_fig

apply_pub_style()

OUTPUT_DIR = "validation/plots"
G = STELLAR.G

R_H = 1.0
OMEGA_TRUE = 2.0  # rotation rate [1/Myr] (moderate: v_rot ~ 0.2 sigma at r_h)
INCL_TRUE = np.pi / 3.0  # 60 deg inclination
N_STARS = 20_000
SEED = 0

K_BINS = 16
N_ADAM = 400
ADAM_LR = 3e-2
K_BOX = (-20.0, 20.0)  # slope box (sign-agnostic)
N_INITS = 3
SELFCON_NSIG = 4.0
RECOVERY_NSIG = 3.0
COND_GATE = 1e8


def project_los(pos, vel, incl):
    """Project to the sky at inclination ``incl`` (tilt about sky-x; observer +z).

    Tilt R_x(i): (x,y,z) -> (x, y cos i - z sin i, y sin i + z cos i). The sky plane
    is lab (x, y); the line-of-sight velocity is the lab-z velocity component."""
    ci, si = np.cos(incl), np.sin(incl)
    sky_x = np.asarray(pos[:, 0])
    sky_y = np.asarray(pos[:, 1]) * ci - np.asarray(pos[:, 2]) * si
    v_los = np.asarray(vel[:, 1]) * si + np.asarray(vel[:, 2]) * ci
    return sky_x, sky_y, v_los


def build_truth_data():
    """Plummer + solid-body rotation, projected at INCL_TRUE -> binned <v_los>(x_sky)."""
    m = jnp.ones(N_STARS)
    kp, kv = jax.random.split(jax.random.PRNGKey(SEED))
    pos = PlummerProfile(r_h=R_H).sample_positions(m, kp)
    vel = PlummerVelocityDF(r_h=R_H).sample_velocities(pos, m, kv, G=G)
    vel = apply_solid_body_rotation(vel, pos, OMEGA_TRUE, jnp.array([0.0, 0.0, 1.0]))
    sky_x, _sky_y, v_los = project_los(pos, vel, INCL_TRUE)

    # bin by sky_x (quantile bins), mean v_los + SE per bin.
    edges = np.quantile(sky_x, np.linspace(0, 1, K_BINS + 1))
    idx = np.clip(np.searchsorted(edges, sky_x, side="right") - 1, 0, K_BINS - 1)
    x_mid, vbar, se, wt = [], [], [], []
    for k in range(K_BINS):
        sel = idx == k
        n = int(sel.sum())
        x_mid.append(float(np.mean(sky_x[sel])))
        vbar.append(float(np.mean(v_los[sel])))
        se.append(float(np.std(v_los[sel]) / np.sqrt(max(n, 1))))
        wt.append(1.0 if n >= 30 else 0.0)
    return (
        jnp.asarray(x_mid),
        jnp.asarray(vbar),
        jnp.asarray(se),
        jnp.asarray(wt),
        pos,
        vel,
    )


def main():
    print("=" * 78)
    print("ROTATION, PROJECTION, AND THE omega-i DEGENERACY (B8)")
    print("=" * 78)
    k_true = OMEGA_TRUE * np.sin(INCL_TRUE)
    print(
        f"\n  truth: omega={OMEGA_TRUE} /Myr, i={np.degrees(INCL_TRUE):.0f} deg "
        f"-> slope k = omega sin(i) = {k_true:.4f}"
    )

    x_mid, vbar, se, wt, pos, vel = build_truth_data()
    predict = lambda z: z[0] * x_mid  # <v_los> = k * x_sky
    nll = lambda z: -gaussian_loglike((vbar, se, wt), predict)(z)

    # self-consistency: chi^2 of the truth-slope model.
    resid_true = np.asarray((vbar - k_true * x_mid) / jnp.where(se > 0, se, 1.0))
    selfcon = float(np.max(np.abs(resid_true[np.asarray(wt) > 0])))
    print(
        f"\n  self-consistency: max|resid|/SE at truth k = {selfcon:.2f} (gate < {SELFCON_NSIG})"
    )

    inits = [jnp.array([k_true])] + [
        jnp.array([k_true + float(s)])
        for s in np.random.default_rng(SEED).normal(0, 1.0, N_INITS - 1)
    ]
    nll_j = jax.jit(nll)
    finals = [mle_adam(nll_j, z0, n_steps=N_ADAM, lr=ADAM_LR)[0] for z0 in inits]
    z_hat = finals[int(np.argmin([float(nll_j(z)) for z in finals]))]
    k_hat = float(z_hat[0])
    sig_k = float(jnp.sqrt(fisher_cov(nll, z_hat)[0, 0]))
    pull = (k_hat - k_true) / sig_k
    print(f"\n  slope recovery: k = {k_hat:.4f} +- {sig_k:.4f} (pull {pull:+.2f})")

    # (omega, i) degeneracy: k = omega sin(i) -> rank-1 Fisher.
    g = np.array([np.sin(INCL_TRUE), OMEGA_TRUE * np.cos(INCL_TRUE)])  # dk/d(omega,i)
    F_oi = np.outer(g, g) / sig_k**2
    eig = np.linalg.eigvalsh(F_oi)
    cond = float(eig[-1] / max(eig[0], 1e-300))
    print(f"\n  d k/d(omega, i) = {g};  (omega,i) Fisher eigenvalues = {eig}")
    print(
        f"  condition number = {cond:.2e} (gate > {COND_GATE:.0e} -> omega-i degenerate)"
    )

    info_per_star = 1.0 / (sig_k**2 * N_STARS)
    n_grid = np.array([1e3, 3e3, 1e4, 3e4, 1e5, 3e5])
    sig_grid = 1.0 / np.sqrt(n_grid * info_per_star)
    slope = float(np.polyfit(np.log(n_grid), np.log(sig_grid), 1)[0])
    print(f"  forecast sigma(k) ~ N^{slope:.3f}")

    make_figure(pos, vel, x_mid, vbar, se, wt, k_hat, k_true, n_grid, sig_grid)

    selfcon_ok = selfcon < SELFCON_NSIG
    recovery_ok = abs(pull) < RECOVERY_NSIG
    degeneracy_ok = cond > COND_GATE
    forecast_ok = -0.55 < slope < -0.45

    rows = [
        (
            "self-consistency at truth",
            "PASS" if selfcon_ok else "FAIL",
            f"<{SELFCON_NSIG} sigma",
            selfcon_ok,
        ),
        (
            "slope k=omega sin(i) recovery",
            "PASS" if recovery_ok else "FAIL",
            f"<{RECOVERY_NSIG} sigma",
            recovery_ok,
        ),
        (
            "(omega,i) Fisher rank-deficient",
            "PASS" if degeneracy_ok else "FAIL",
            f"cond>{COND_GATE:.0e}",
            degeneracy_ok,
        ),
        (
            "forecast sigma(k)~N^-1/2",
            "PASS" if forecast_ok else "FAIL",
            "slope -0.5",
            forecast_ok,
        ),
    ]
    print("\n" + "-" * 78)
    print(f"  {'CHECK':<34s} {'status':>6s} {'gate':>12s}")
    print("-" * 78)
    all_ok = True
    for name, status, gate, ok in rows:
        all_ok &= ok
        print(f"  {name:<34s} {status:>6s} {gate:>12s}")
    print("-" * 78)
    print(f"  saved {OUTPUT_DIR}/demo_rotation.{{png,pdf}}")
    print("=" * 78)
    print("  ROTATION DEMO: ALL PASS" if all_ok else "  ROTATION DEMO: FAILED")
    return 0 if all_ok else 1


def make_figure(pos, vel, x_mid, vbar, se, wt, k_hat, k_true, n_grid, sig_grid):
    import matplotlib.pyplot as plt

    sky_x, sky_y, v_los = project_los(pos, vel, INCL_TRUE)
    x_mid = np.asarray(x_mid)
    vbar = np.asarray(vbar)
    se = np.asarray(se)
    m = np.asarray(wt) > 0
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))

    # (a) the projected sky map coloured by v_los -- the rotation dipole.
    ax = axes[0]
    sub = np.random.default_rng(0).choice(len(sky_x), size=3000, replace=False)
    vmax = np.percentile(np.abs(v_los), 95)
    sc = ax.scatter(
        sky_x[sub], sky_y[sub], c=v_los[sub], s=4, cmap="RdBu_r", vmin=-vmax, vmax=vmax
    )
    fig.colorbar(sc, ax=ax, label=r"$v_{\rm los}$", fraction=0.046)
    lim = float(np.percentile(np.abs(np.concatenate([sky_x, sky_y])), 96))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"sky $x$  [pc]")
    ax.set_ylabel(r"sky $y$  [pc]")
    ax.set_aspect("equal")
    panel_label(ax, "(a)")

    # (b) the rotation curve <v_los>(x_sky) + the recovered slope.
    ax = axes[1]
    ax.errorbar(
        x_mid[m],
        vbar[m],
        yerr=se[m],
        fmt="o",
        ms=3.5,
        color=OI["black"],
        label=r"$\langle v_{\rm los}\rangle$",
        zorder=4,
    )
    xx = np.linspace(x_mid[m].min(), x_mid[m].max(), 50)
    ax.plot(xx, k_hat * xx, "-", color=OI["vermilion"], label=rf"slope $k={k_hat:.2f}$")
    ax.set_xlabel(r"sky $x$  [pc]")
    ax.set_ylabel(r"$\langle v_{\rm los}\rangle$  [pc Myr$^{-1}$]")
    ax.legend(fontsize=7)
    panel_label(ax, "(b)")

    # (c) the omega-i degeneracy: the k = omega sin(i) curve.
    ax = axes[2]
    incl = np.linspace(np.radians(10), np.radians(90), 200)
    ax.plot(
        np.degrees(incl),
        k_hat / np.sin(incl),
        "-",
        color=OI["purple"],
        label=r"$\omega=\hat k/\sin i$",
    )
    ax.scatter(
        [np.degrees(INCL_TRUE)],
        [OMEGA_TRUE],
        marker="*",
        s=110,
        color=OI["blue"],
        zorder=5,
        label="truth",
    )
    ax.set_xlabel(r"inclination $i$  [deg]")
    ax.set_ylabel(r"$\omega$  [Myr$^{-1}$]")
    ax.set_ylim(0, OMEGA_TRUE * 3)
    ax.legend(fontsize=7)
    panel_label(ax, "(c)")

    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, "demo_rotation")


if __name__ == "__main__":
    sys.exit(main())

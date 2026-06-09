#!/usr/bin/env python
"""
EFF (Elson, Fall & Freeman 1987) profile + Eddington velocity-DF validation.

Five publication-quality figures, each anchored to *passing* tests in
``tests/validation/test_eff_physics.py``. The script recomputes the same
quantities those tests assert and prints expected-vs-measured PASS/FAIL tables,
so the figures are a faithful visualization layer over already-verified physics.

Note on conventions (verified against src/progenax/profiles/eff.py):
  - ``gamma`` is the **3-D density slope**: rho(r) = rho_0 (1 + r^2/a^2)^{-gamma/2},
    rho -> r^{-gamma} for r >> a. It is offset by ~1 from EFF87's *surface*-
    brightness slope (Abel projection of r^{-gamma} gives a surface slope
    gamma-1). gamma = 5 reduces EFF exactly to the Plummer profile.
  - The model is truncated at r_t. Mild truncation (gamma=5, r_t >> a) gives a
    near-virial Eddington IC; the steep gamma=3 default is mildly sub-virial
    under sharp truncation (an intrinsic property of truncating an empirical,
    non-DF profile -- see test docstring).

Figures (-> anchoring tests):
  1. eff_density_slope.png       rho(r)/rho0 + CDF; rho(a)=2^{-gamma/2}; slope -gamma; r<=r_t
  2. eff_gamma_family.png        gamma=2,3,4,5 concentration sweep + gamma=5 -> Plummer
  3. eff_velocity_equilibrium.png isotropy/sigma(r); v<=v_esc; virial Q (gamma=5 vs 3)
  4. eff_gradient_validation.png  AD vs central-FD d(observable)/d(a, gamma, M)
  5. eff_eddington_f.png          Eddington f(E) >= 0 and increasing with energy

References:
    Elson, Fall & Freeman (1987), ApJ 323, 54; Binney & Tremaine (2008).

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_eff.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from progenax.profiles import PlummerProfile
from progenax.profiles.eff import EFFProfile
from progenax.kinematics.eff_df import EFFVelocityDF
from progenax.builders import compute_kinetic_energy, compute_potential_energy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plotstyle import OI, apply_pub_style, panel_label, save_fig

apply_pub_style()

OUTPUT_DIR = "validation/plots"
N_SAMPLES = 50_000
N_DISPERSION = 40_000
N_VIRIAL = 6_000
SEED = 42


# ============================================================================
# Figure 1 -- density profile, asymptotic slope, truncation, CDF
# ============================================================================
def fig_density_slope(output_dir):
    print("\n" + "=" * 60)
    print("FIG 1: density rho(r)/rho0, slope -gamma, truncation, CDF")
    print("=" * 60)

    a, gamma, r_t = 1.0, 4.0, 30.0
    prof = EFFProfile(a=a, gamma=gamma, r_t=r_t)
    key = jax.random.PRNGKey(SEED)
    radii = jnp.linalg.norm(prof.sample_positions(jnp.ones(N_SAMPLES), key), axis=1)
    radii_np = np.asarray(radii)

    # --- metrics ---
    rho_a = float(prof.density(jnp.array([a]))[0])
    rho_a_exp = 1.0 / 2.0 ** (gamma / 2.0)
    rho_a_pass = abs(rho_a - rho_a_exp) < 1e-10

    r1, r2 = 8.0, 16.0   # r >> a, < r_t
    slope = float(jnp.log(prof.density(jnp.array([r2]))[0] / prof.density(jnp.array([r1]))[0])
                  / jnp.log(r2 / r1))
    slope_pass = abs(slope - (-gamma)) / gamma < 0.01

    max_r = float(jnp.max(radii))
    trunc_frac = float(jnp.mean(radii <= r_t))
    trunc_pass = (max_r <= r_t + 0.01) and trunc_frac == 1.0

    sorted_r = np.sort(radii_np)
    ecdf = np.arange(1, N_SAMPLES + 1) / N_SAMPLES
    cdf_theory = np.asarray(jnp.interp(jnp.asarray(sorted_r), prof._r_grid, prof._cdf_grid))
    max_cdf_dev = float(np.max(np.abs(ecdf - cdf_theory)))
    cdf_pass = max_cdf_dev < 0.02
    passed = rho_a_pass and slope_pass and trunc_pass and cdf_pass

    print(f"  {'quantity':<28}{'expected':>12}{'measured':>12}{'pass':>8}")
    print(f"  {'rho(a)':<28}{rho_a_exp:>12.4f}{rho_a:>12.4f}"
          f"{'PASS' if rho_a_pass else 'FAIL':>8}")
    print(f"  {'outer slope d ln rho/d ln r':<28}{-gamma:>12.3f}{slope:>12.3f}"
          f"{'PASS' if slope_pass else 'FAIL':>8}")
    print(f"  {'frac r<=r_t':<28}{1.0:>12.3f}{trunc_frac:>12.3f}"
          f"{'PASS' if trunc_pass else 'FAIL':>8}")
    print(f"  {'max |ECDF-CDF|':<28}{0.02:>12.3f}{max_cdf_dev:>12.4f}"
          f"{'PASS' if cdf_pass else 'FAIL':>8}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.8, 3.3))

    r = jnp.logspace(np.log10(0.05 * a), np.log10(0.999 * r_t), 300)
    rho = np.asarray(prof.density(r))
    xr = np.asarray(r / a)
    axA.loglog(xr, (1.0 + (np.asarray(r) / a) ** 2) ** (-gamma / 2.0), "-",
               color=OI["black"], lw=2.0, label=r"analytic $(1+r^2/a^2)^{-\gamma/2}$")
    axA.loglog(xr, rho, "--", color=OI["blue"], label=r"progenax $\rho(r)$")

    bins = np.logspace(np.log10(0.05 * a), np.log10(r_t), 40)
    centers = np.sqrt(bins[1:] * bins[:-1])
    hist, _ = np.histogram(radii_np, bins=bins)
    shell_v = 4.0 / 3.0 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    rho_h = hist / (shell_v + 1e-30)
    valid = rho_h > 0
    rho_an_c = (1.0 + (centers / a) ** 2) ** (-gamma / 2.0)
    scale = np.median(rho_an_c[valid] / rho_h[valid])
    axA.loglog(centers[valid] / a, rho_h[valid] * scale, "o", color=OI["green"],
               ms=3.5, alpha=0.7, mec="none", label=rf"sampled ($N={N_SAMPLES:,}$)")
    # slope guide
    xg = np.array([2.0, 20.0])
    axA.loglog(xg, rho_an_c[0] * (centers[0] / a) ** gamma * xg ** (-gamma) * 0.4, ":",
               color=OI["vermilion"], lw=1.3, label=rf"$r^{{-\gamma}},\ \gamma={gamma:.0f}$")
    axA.axvline(r_t / a, color="0.5", ls=":", lw=1.0)
    axA.text(r_t / a * 0.5, 3e-4, r"$r_t$", color="0.4", fontsize=8.5)
    axA.set_xlabel(r"$r / a$")
    axA.set_ylabel(r"$\rho(r) / \rho_0$")
    axA.set_ylim(1e-4, 2)
    axA.legend(loc="lower left")
    panel_label(axA, "(a)", loc="upper right")

    axB.plot(np.asarray(prof._r_grid) / a, np.asarray(prof._cdf_grid), "-",
             color=OI["black"], lw=2.0, label="analytic CDF")
    axB.plot(sorted_r[::200] / a, ecdf[::200], "o", color=OI["blue"], ms=2.6,
             mec="none", alpha=0.6, label="empirical CDF")
    axB.set_xlabel(r"$r / a$")
    axB.set_ylabel(r"$M(<r) / M_{\rm tot}$")
    axB.set_xlim(0, r_t / a)
    axB.set_ylim(0, 1.02)
    axB.legend(loc="lower right")
    axB.text(0.04, 0.94, rf"max $|\Delta\mathrm{{CDF}}|={max_cdf_dev:.1e}$",
             transform=axB.transAxes, fontsize=8, va="top",
             bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))
    panel_label(axB, "(b)", loc="upper left")

    fig.tight_layout(pad=0.4, w_pad=0.8)
    save_fig(fig, output_dir, "eff_density_slope")
    print("  saved eff_density_slope.{png,pdf}")
    return passed


# ============================================================================
# Figure 2 -- gamma family + gamma=5 -> Plummer limit
# ============================================================================
def fig_gamma_family(output_dir):
    print("\n" + "=" * 60)
    print("FIG 2: gamma family + gamma=5 -> Plummer limit")
    print("=" * 60)

    a, r_t = 1.0, 30.0
    gammas = [2.0, 3.0, 4.0, 5.0]
    colors = [OI["sky"], OI["green"], OI["orange"], OI["vermilion"]]

    # gamma=5 vs Plummer (Plummer rho ~ (1+r^2/a^2)^{-5/2})
    r = jnp.linspace(0.01, 10.0, 400)
    eff5 = np.asarray(EFFProfile(a=a, gamma=5.0, r_t=r_t).density(r))
    plummer = np.asarray(PlummerProfile(r_h=1.0).density(r))   # uses its own a
    # compare shapes at matched scale radius a: build Plummer with a=1 via r_h
    a_pl = 1.0
    plummer_a1 = (1.0 + (np.asarray(r) / a_pl) ** 2) ** -2.5
    max_rel_pl = float(np.max(np.abs(eff5 / eff5[0] - plummer_a1 / plummer_a1[0])))
    pl_pass = max_rel_pl < 1e-10

    print(f"  gamma=5 vs Plummer (a=1): max rel = {max_rel_pl:.2e}  "
          f"-> {'PASS' if pl_pass else 'FAIL'}")

    # concentration ordering: median radius decreasing with gamma
    med = []
    for g in gammas:
        rad = jnp.linalg.norm(
            EFFProfile(a=a, gamma=g, r_t=r_t).sample_positions(
                jnp.ones(20_000), jax.random.PRNGKey(7)), axis=1)
        med.append(float(jnp.median(rad)))
    conc_pass = med[0] > med[1] > med[2] > med[3]
    print(f"  median radius vs gamma=[2,3,4,5]: {[round(m,2) for m in med]}  "
          f"-> {'PASS' if conc_pass else 'FAIL'}")
    passed = pl_pass and conc_pass

    fig, ax = plt.subplots(figsize=(3.9, 3.5))
    for g, col in zip(gammas, colors):
        rho = np.asarray(EFFProfile(a=a, gamma=g, r_t=r_t).density(r))
        ax.semilogy(np.asarray(r) / a, rho / rho[0], "-", color=col, lw=1.6,
                    label=rf"$\gamma={g:.0f}$")
    ax.semilogy(np.asarray(r) / a, plummer_a1 / plummer_a1[0], "k:", lw=1.4,
                label="Plummer", zorder=5)
    ax.set_xlabel(r"$r / a$")
    ax.set_ylabel(r"$\rho(r) / \rho_0$")
    ax.set_xlim(0, 10)
    ax.set_ylim(1e-4, 2)
    ax.legend(loc="upper right", title=r"$\gamma=5$ overlies Plummer", title_fontsize=7.5)
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "eff_gamma_family")
    print("  saved eff_gamma_family.{png,pdf}")
    return passed


# ============================================================================
# Figure 3 -- velocity-space equilibrium (Eddington DF)
# ============================================================================
def _virial_Q(gamma, r_t, n=N_VIRIAL, seed=0):
    a, G = 1.0, 1.0
    prof = EFFProfile(a=a, gamma=gamma, r_t=r_t)
    df = EFFVelocityDF(a=a, gamma=gamma, r_t=r_t)
    m = jnp.ones(n)
    kp, kv = jax.random.split(jax.random.PRNGKey(seed))
    pos = prof.sample_positions(m, kp)
    vel = df.sample_velocities(pos, m, kv, G=G)
    Q = float(compute_kinetic_energy(vel, m) / jnp.abs(compute_potential_energy(pos, m, G=G)))
    return prof, df, pos, vel, Q


def fig_velocity_equilibrium(output_dir):
    print("\n" + "=" * 60)
    print("FIG 3: velocity-space equilibrium (Eddington DF)")
    print("=" * 60)
    G = 1.0

    # mild truncation gamma=5 -> near-virial; sharp gamma=3 -> sub-virial caveat
    prof5, df5, pos5, vel5, Q5 = _virial_Q(5.0, 15.0)
    _, _, _, _, Q3 = _virial_Q(3.0, 10.0)
    q5_pass = abs(Q5 - 0.5) < 0.05

    # dispersion + isotropy from the gamma=5 mild-truncation IC
    m_d = jnp.ones(N_DISPERSION)
    kp, kv = jax.random.split(jax.random.PRNGKey(SEED))
    pos_d = prof5.sample_positions(m_d, kp)
    vel_d = df5.sample_velocities(pos_d, m_d, kv, G=G)
    r_d = jnp.linalg.norm(pos_d, axis=1)
    v2c = np.asarray(jnp.mean(vel_d ** 2, axis=0))
    iso_spread = float(np.max(np.abs(v2c - v2c.mean())) / v2c.mean())
    iso_pass = iso_spread < 0.10

    edges = [(0.0, 1.0), (1.0, 2.5), (2.5, 5.0), (5.0, 9.0)]
    mids, sig, err = [], [], []
    for lo, hi in edges:
        msk = (r_d >= lo) & (r_d < hi)
        n = int(jnp.sum(msk))
        mids.append(float(jnp.mean(r_d[msk])))
        sig.append(float(jnp.std(vel_d[msk, 0])))
        err.append(sig[-1] / np.sqrt(2.0 * max(n - 1, 1)))

    # boundedness (gamma=5 IC)
    kappa = G * float(jnp.sum(m_d)) / (4.0 * np.pi * float(df5.mu))
    Psi_r = jnp.interp(r_d, df5.r_grid, df5.Psi_grid, left=df5.Psi_grid[0], right=0.0)
    v_esc = jnp.sqrt(2.0 * kappa * jnp.maximum(Psi_r, 0.0))
    ratio = np.asarray(jnp.linalg.norm(vel_d, axis=1) / (v_esc + 1e-30))
    bound_frac = float(jnp.mean(jnp.linalg.norm(vel_d, axis=1) <= v_esc + 1e-9))
    bound_pass = bound_frac == 1.0
    passed = q5_pass and iso_pass and bound_pass

    print(f"  velocity isotropy: max|<v_i^2>-mean|/mean = {iso_spread:.2%} (tol 10%)  "
          f"-> {'PASS' if iso_pass else 'FAIL'}")
    print(f"  bound fraction (v<=v_esc): {bound_frac*100:.2f}%  "
          f"-> {'PASS' if bound_pass else 'FAIL'}")
    print(f"  virial Q (gamma=5, mild trunc): {Q5:.3f} (expect 0.5+-0.05)  "
          f"-> {'PASS' if q5_pass else 'FAIL'}")
    print(f"  virial Q (gamma=3, sharp trunc): {Q3:.3f}  (caveat: sub-virial, see test)")

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.2, 2.7))

    axA.errorbar(mids, sig, yerr=err, fmt="s", color=OI["blue"], ms=5, capsize=2.5,
                 lw=1.0, label=rf"sampled ($N={N_DISPERSION:,}$)")
    axA.set_xlabel(r"$r$ [$a$]")
    axA.set_ylabel(r"$\sigma_{1d}(r)$ [code units, $G{=}1$]")
    axA.set_ylim(0, None)
    axA.text(0.5, 0.9, f"isotropy = {iso_spread:.1%}", transform=axA.transAxes,
             ha="center", va="top", fontsize=8)
    axA.legend(loc="upper right")
    panel_label(axA, "(a)", loc="lower left")

    axB.hist(ratio, bins=40, color=OI["sky"], edgecolor="white", linewidth=0.3)
    axB.axvline(1.0, color=OI["vermilion"], ls="--", lw=1.4)
    axB.text(0.97, 0.5, r"$v_{\rm esc}$", transform=axB.get_xaxis_transform(),
             rotation=90, va="center", ha="right", color=OI["vermilion"], fontsize=8.5)
    axB.text(0.5, 0.92, rf"{bound_frac*100:.1f}% bound", transform=axB.transAxes,
             ha="center", va="top", fontsize=8.5)
    axB.set_xlabel(r"$v / v_{\rm esc}(r)$")
    axB.set_ylabel("count")
    axB.set_xlim(0, 1.05)
    panel_label(axB, "(b)", loc="upper left")

    axC.axhspan(0.45, 0.55, color=OI["green"], alpha=0.18, label=r"equilib. $\pm0.05$")
    axC.axhline(0.5, color=OI["black"], ls="--", label=r"$Q=0.5$")
    axC.plot([0], [Q5], "o", color=OI["vermilion"], ms=8, zorder=5,
             label=r"$\gamma=5$ (mild)")
    axC.plot([0], [Q3], "D", color=OI["orange"], ms=7, zorder=5,
             label=r"$\gamma=3$ (sharp)")
    axC.text(0.5, 0.9, rf"$Q_{{\gamma5}}={Q5:.3f}$", transform=axC.transAxes,
             ha="center", va="top", fontsize=9, fontweight="bold")
    axC.set_xticks([])
    axC.set_xlim(-0.6, 0.6)
    axC.set_ylim(0.40, 0.58)
    axC.set_xlabel("unscaled IC")
    axC.set_ylabel(r"virial ratio $Q = T/|V|$")
    axC.legend(loc="lower center", fontsize=7)
    panel_label(axC, "(c)", loc="upper left")

    fig.tight_layout(pad=0.4, w_pad=0.8)
    save_fig(fig, output_dir, "eff_velocity_equilibrium")
    print("  saved eff_velocity_equilibrium.{png,pdf}")
    return passed


# ============================================================================
# Figure 4 -- gradient validation (autodiff vs finite difference)
# ============================================================================
def _grad_sweep(loss, xs, h):
    ad = np.array([float(jax.grad(loss)(float(x))) for x in xs])
    fd = np.array([float((loss(float(x) + h) - loss(float(x) - h)) / (2 * h)) for x in xs])
    rel = np.abs(ad - fd) / (np.abs(ad) + np.abs(fd) + 1e-30)
    return ad, fd, rel


def fig_gradient_validation(output_dir):
    """AD vs central-FD gradients of EFF observables w.r.t. structural params
    (a, gamma, M) -- the basis for gradient-based / HMC inference. gamma is a
    differentiable structural parameter (the EFF analogue of King's W0)."""
    print("\n" + "=" * 60)
    print("FIG 4: gradient validation (autodiff vs finite difference)")
    print("=" * 60)
    G, r_t = 1.0, 30.0

    def log_density_at(r_phys, a, gamma):
        p = EFFProfile(a=a, gamma=gamma, r_t=r_t)
        return jnp.log10(p.density(jnp.array([r_phys]))[0] + 1e-30)

    def density_at(r_phys, a, gamma):
        # linear (not log) density: log rho is *linear* in gamma so d/dgamma would
        # be a constant; the linear density rho(2a)=5^{-gamma/2} has a gamma-varying
        # derivative, giving a meaningful AD-vs-FD curve.
        return EFFProfile(a=a, gamma=gamma, r_t=r_t).density(jnp.array([r_phys]))[0]

    def vel_scale(M_total, a=1.0, gamma=4.0):
        df = EFFVelocityDF(a=a, gamma=gamma, r_t=r_t)
        return jnp.sqrt(G * M_total / (4.0 * jnp.pi * df.mu))   # sqrt(kappa)

    specs = [
        ("a", r"$a$ [pc]", r"$\partial\,\log\rho(2a) / \partial a$",
         lambda a: log_density_at(2.0, a, 4.0), np.linspace(0.6, 2.0, 11), 1e-5),
        ("gamma", r"$\gamma$", r"$\partial\,\rho(2a) / \partial \gamma$",
         lambda g: density_at(2.0, 1.0, g), np.linspace(2.5, 5.0, 11), 1e-5),
        ("M", r"$M_{\rm tot}$ [$M_\odot$]", r"$\partial\,\sqrt{\kappa} / \partial M_{\rm tot}$",
         lambda M: vel_scale(M), np.linspace(200.0, 2000.0, 11), 1.0),
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
        print(f"  d(loss)/d{key:5}: max rel err {np.max(rel):.2e}  "
              f"-> {'DIFFERENTIABLE' if np.max(rel) < 1e-4 else 'CHECK'}")

    passed = worst < 1e-4
    print(f"  overall worst rel err {worst:.2e}  -> {'PASS' if passed else 'FAIL'}")
    fig.tight_layout(pad=0.4, w_pad=0.8)
    save_fig(fig, output_dir, "eff_gradient_validation")
    print("  saved eff_gradient_validation.{png,pdf}")
    return passed


# ============================================================================
# Figure 5 -- Eddington DF f(E) is physical
# ============================================================================
def fig_eddington_f(output_dir):
    print("\n" + "=" * 60)
    print("FIG 5: Eddington DF f(E) physical (>=0, increasing)")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(3.9, 3.3))
    all_pass = True
    for gamma, col in [(3.0, OI["green"]), (4.0, OI["orange"]), (5.0, OI["vermilion"])]:
        df = EFFVelocityDF(a=1.0, gamma=gamma, r_t=10.0)
        E = np.asarray(df.E_grid)
        f = np.asarray(df.f_grid)
        nonneg = bool(np.all(f >= 0.0))
        incr = bool(f[-1] > f[len(f) // 2] > 0.0)
        ok = nonneg and incr
        all_pass = all_pass and ok
        print(f"  gamma={gamma:.0f}: min f={f.min():.2e} (>=0:{nonneg}), "
              f"increasing:{incr}  -> {'PASS' if ok else 'FAIL'}")
        ax.plot(E, f, "-", color=col, lw=1.7, label=rf"$\gamma={gamma:.0f}$")

    ax.axhline(0.0, color="0.6", lw=0.7)
    ax.set_xlabel(r"relative energy $\mathcal{E}$")
    ax.set_ylabel(r"Eddington $f(\mathcal{E})$")
    ax.legend(loc="upper left", title=r"$f(\mathcal{E})\geq 0$, increasing",
              title_fontsize=8)
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "eff_eddington_f")
    print("  saved eff_eddington_f.{png,pdf}")
    return all_pass


def main():
    print("\n" + "=" * 70)
    print("PROGENAX EFF (1987) PROFILE + EDDINGTON-DF VALIDATION FIGURES")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {
        "Fig 1  density + slope + CDF": fig_density_slope(OUTPUT_DIR),
        "Fig 2  gamma family + Plummer limit": fig_gamma_family(OUTPUT_DIR),
        "Fig 3  velocity-space equilibrium": fig_velocity_equilibrium(OUTPUT_DIR),
        "Fig 4  gradient validation (AD vs FD)": fig_gradient_validation(OUTPUT_DIR),
        "Fig 5  Eddington f(E) physical": fig_eddington_f(OUTPUT_DIR),
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    all_ok = all(results.values())
    print("=" * 70)
    print("  ALL EFF VALIDATION FIGURES PASS" if all_ok
          else "  SOME EFF VALIDATION FIGURES FAILED")
    print("=" * 70)
    print(f"\nFigures written to {OUTPUT_DIR}/eff_*.png")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

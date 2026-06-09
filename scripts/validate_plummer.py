#!/usr/bin/env python
"""
Plummer (1911) profile + matched velocity-DF validation figures.

Five publication-quality figures, each anchored to *passing* tests in
``tests/validation/test_plummer_physics.py``. The script recomputes the same
quantities those tests assert and prints expected-vs-measured PASS/FAIL tables,
so the figures are a faithful visualization layer over already-verified physics
(not a second, drift-prone implementation).

Figures (-> anchoring tests):
  1. plummer_density.png            rho(r)/rho0 + CDF M(<r)/M, half-mass point
                                    -> TestPlummerScaleRadius, TestPlummerDensityProfile
  2. plummer_velocity_equilibrium.png  sigma_1d(r) vs GM/(6 sqrt(r^2+a^2));
                                    v<=v_esc; unscaled virial Q
                                    -> TestPlummerVelocityDispersion,
                                       TestPlummerBoundParticles, TestPlummerVirialEquilibrium
  3. plummer_beta_distribution.png  q^2=(v/v_esc)^2 vs Beta(3/2,9/2)
                                    -> TestPlummerBetaDistribution
  4. plummer_gradient_validation.png  AD vs central-FD d(observable)/d(r_h, M)
                                    -> integration/test_jax_compatibility (grad/jit)
  5. plummer_isotropy.png           cos(theta) ~ U(-1,1), phi ~ U(0,2pi)
                                    -> test_positions_isotropic, test_velocity_isotropy

References:
    Plummer (1911), MNRAS 71, 460; Aarseth+ (1974); Binney & Tremaine (2008).

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_plummer.py

Output:
    validation/plots/plummer_*.png  (curate the verified set into
    docs/website/50-validation/figures/)
"""
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.special import gammaln

jax.config.update("jax_enable_x64", True)

from jaxstro.units import STELLAR
from progenax.profiles import PlummerProfile
from progenax.kinematics import PlummerVelocityDF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plotstyle import OI, apply_pub_style, panel_label, save_fig

apply_pub_style()

OUTPUT_DIR = "validation/plots"
R_H = 1.0           # half-mass radius [pc]
G = STELLAR.G       # pc^3 Msun^-1 Myr^-2
N_SAMPLES = 50_000
N_DISPERSION = 40_000
N_VIRIAL = 5_000
SEED = 42


def _beta_pdf(x, a, b):
    """Beta(a,b) probability density (JAX-native, no scipy)."""
    log_B = gammaln(a) + gammaln(b) - gammaln(a + b)
    return jnp.exp((a - 1.0) * jnp.log(x) + (b - 1.0) * jnp.log1p(-x) - log_B)


# ============================================================================
# Figure 1 -- density profile + cumulative mass
# ============================================================================
def fig_density(output_dir):
    print("\n" + "=" * 60)
    print("FIG 1: density rho(r)/rho0 + cumulative mass M(<r)/M")
    print("=" * 60)

    prof = PlummerProfile(r_h=R_H)
    a = float(prof.a)
    key = jax.random.PRNGKey(SEED)
    masses = jnp.ones(N_SAMPLES)
    radii = jnp.linalg.norm(prof.sample_positions(masses, key), axis=1)
    radii_np = np.asarray(radii)

    # --- metrics (match the tests) ---
    a_expected = R_H * np.sqrt((1.0 - 0.5 ** (2 / 3)) / 0.5 ** (2 / 3))
    a_rel = abs(a - a_expected) / a_expected
    frac_within_rh = float(jnp.mean(radii < R_H))
    frac_within_a = float(jnp.mean(radii < a))
    m_a_expected = 1.0 / 2.0 ** 1.5  # M(<a)/M = 0.3536
    sorted_r = np.sort(radii_np)
    ecdf = np.arange(1, N_SAMPLES + 1) / N_SAMPLES
    cdf_theory = sorted_r ** 3 / (sorted_r ** 2 + a ** 2) ** 1.5
    max_cdf_dev = float(np.max(np.abs(ecdf - cdf_theory)))

    # Pass thresholds mirror the tests exactly (tests/validation/conftest.py:
    # HALF_MASS = 0.03; test_cdf_formula_accuracy uses 0.03 per radius).
    a_pass = a_rel < 1e-6
    rh_pass = abs(frac_within_rh - 0.5) < 0.03
    ma_pass = abs(frac_within_a - m_a_expected) < 0.03
    cdf_pass = max_cdf_dev < 0.03
    passed = a_pass and rh_pass and ma_pass and cdf_pass

    print(f"  {'quantity':<26}{'expected':>12}{'measured':>12}{'pass':>8}")
    print(f"  {'scale radius a [pc]':<26}{a_expected:>12.4f}{a:>12.4f}"
          f"{'PASS' if a_pass else 'FAIL':>8}")
    print(f"  {'frac r<r_h':<26}{0.5:>12.3f}{frac_within_rh:>12.3f}"
          f"{'PASS' if rh_pass else 'FAIL':>8}")
    print(f"  {'M(<a)/M':<26}{m_a_expected:>12.4f}{frac_within_a:>12.4f}"
          f"{'PASS' if ma_pass else 'FAIL':>8}")
    print(f"  {'max |ECDF - CDF|':<26}{0.03:>12.3f}{max_cdf_dev:>12.4f}"
          f"{'PASS' if cdf_pass else 'FAIL':>8}")

    # --- figure ---
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.8, 3.3))

    r = jnp.linspace(0.01 * R_H, 5.0 * R_H, 300)
    rho = np.asarray(prof.density(r))          # unnormalized: rho0 = 1 at r=0
    xr = np.asarray(r / R_H)
    axA.semilogy(xr, (1.0 + (np.asarray(r) / a) ** 2) ** -2.5, "-",
                 color=OI["black"], lw=2.0, label=r"analytic $(1+r^2/a^2)^{-5/2}$")
    axA.semilogy(xr, rho, "--", color=OI["blue"], label=r"progenax $\rho(r)$")

    bins = np.linspace(0.0, 5.0 * R_H, 50)
    centers = 0.5 * (bins[1:] + bins[:-1])
    hist, _ = np.histogram(radii_np, bins=bins)
    shell_v = 4.0 / 3.0 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    rho_h = hist / (shell_v + 1e-30)
    valid = rho_h > 0
    rho_an_c = (1.0 + (centers / a) ** 2) ** -2.5
    scale = np.median(rho_an_c[valid] / rho_h[valid])
    axA.semilogy(centers[valid] / R_H, rho_h[valid] * scale, "o", color=OI["green"],
                 ms=3.5, alpha=0.7, mec="none", label=rf"sampled ($N={N_SAMPLES:,}$)")
    axA.axvline(1.0, color=OI["vermilion"], ls=":", lw=1.0)
    axA.text(1.04, 1.3e-4, r"$r_h$", color=OI["vermilion"], fontsize=8.5)
    axA.set_xlabel(r"$r / r_h$")
    axA.set_ylabel(r"$\rho(r) / \rho_0$")
    axA.set_xlim(0, 5)
    axA.set_ylim(1e-4, 2)
    axA.legend(loc="upper right")
    panel_label(axA, "(a)", loc="lower left")

    cdf_an = np.asarray(r) ** 3 / (np.asarray(r) ** 2 + a ** 2) ** 1.5
    axB.plot(xr, cdf_an, "-", color=OI["black"], lw=2.0, label="analytic CDF")
    axB.plot(sorted_r[::200] / R_H, ecdf[::200], "o", color=OI["blue"], ms=2.6,
             mec="none", alpha=0.6, label="empirical CDF")
    axB.axhline(0.5, color="0.6", ls="--", lw=0.8)
    axB.plot([1.0], [0.5], "s", color=OI["vermilion"], ms=7, mfc="none", mew=1.4,
             label=r"half-mass $(r_h,\,0.5)$")
    axB.set_xlabel(r"$r / r_h$")
    axB.set_ylabel(r"$M(<r) / M_{\rm tot}$")
    axB.set_xlim(0, 5)
    axB.set_ylim(0, 1.02)
    axB.legend(loc="lower right")
    axB.text(0.04, 0.94, rf"max $|\Delta\mathrm{{CDF}}| = {max_cdf_dev:.1e}$",
             transform=axB.transAxes, fontsize=8, va="top",
             bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))
    panel_label(axB, "(b)", loc="upper left")

    fig.tight_layout(pad=0.4, w_pad=0.8)
    save_fig(fig, output_dir, "plummer_density")
    print("  saved plummer_density.{png,pdf}")
    return passed


# ============================================================================
# Figure 2 -- velocity-space equilibrium triptych
# ============================================================================
def fig_velocity_equilibrium(output_dir):
    print("\n" + "=" * 60)
    print("FIG 2: velocity-space equilibrium")
    print("=" * 60)

    prof = PlummerProfile(r_h=R_H)
    df = PlummerVelocityDF(r_h=R_H)
    a = float(prof.a)

    # --- dispersion profile (large N) ---
    m_d = jnp.ones(N_DISPERSION)
    M = float(jnp.sum(m_d))
    kp, kv = jax.random.split(jax.random.PRNGKey(SEED))
    pos_d = prof.sample_positions(m_d, kp)
    vel_d = df.sample_velocities(pos_d, m_d, kv, G=G)
    r_d = jnp.linalg.norm(pos_d, axis=1)

    edges = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.5)]
    mids, s_samp, s_ana, s_rel, s_err = [], [], [], [], []
    for lo, hi in edges:
        msk = (r_d >= lo) & (r_d < hi)
        n = int(jnp.sum(msk))
        r_mid = float(jnp.mean(r_d[msk]))
        ss = float(jnp.std(vel_d[msk, 0]))
        sa = float(jnp.sqrt(G * M / (6.0 * jnp.sqrt(r_mid ** 2 + a ** 2))))
        mids.append(r_mid)
        s_samp.append(ss)
        s_ana.append(sa)
        s_rel.append(abs(ss - sa) / sa)
        s_err.append(ss / np.sqrt(2.0 * max(n - 1, 1)))
    disp_pass = all(x < 0.10 for x in s_rel)   # VELOCITY_DISPERSION = 0.10
    max_disp_rel = max(s_rel)

    r_curve = np.linspace(0.0, 3.5, 120)
    sig_curve = np.sqrt(G * M / (6.0 * np.sqrt(r_curve ** 2 + a ** 2)))

    # --- boundedness + virial (separate seed) ---
    m_v = jnp.ones(N_VIRIAL)
    Mv = float(jnp.sum(m_v))
    kp2, kv2 = jax.random.split(jax.random.PRNGKey(0))
    pos_v = prof.sample_positions(m_v, kp2)
    vel_v = df.sample_velocities(pos_v, m_v, kv2, G=G)
    r_v = jnp.linalg.norm(pos_v, axis=1)
    v_v = jnp.linalg.norm(vel_v, axis=1)
    v_esc = jnp.sqrt(2.0 * G * Mv / jnp.sqrt(r_v ** 2 + a ** 2))
    ratio = np.asarray(v_v / v_esc)
    bound_frac = float(jnp.mean(v_v <= v_esc + 1e-9))
    T = 0.5 * float(jnp.sum(m_v * jnp.sum(vel_v ** 2, axis=1)))
    V = -3.0 * np.pi * G * Mv ** 2 / (32.0 * a)   # Plummer analytic PE
    Q = T / abs(V)
    bound_pass = bound_frac == 1.0
    q_pass = abs(Q - 0.5) < 0.05            # VIRIAL_RATIO = 0.05
    passed = disp_pass and bound_pass and q_pass

    print(f"  sigma_1d(r) vs GM/(6 sqrt(r^2+a^2)):  max rel = {max_disp_rel:.2%} "
          f"(tol 10%)  -> {'PASS' if disp_pass else 'FAIL'}")
    print(f"  bound fraction (v<=v_esc): {bound_frac*100:.2f}%  "
          f"-> {'PASS' if bound_pass else 'FAIL'}")
    print(f"  virial Q=T/|V|: {Q:.4f} (expect 0.5+-0.05)  "
          f"-> {'PASS' if q_pass else 'FAIL'}")

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.2, 2.7))

    axA.plot(r_curve, sig_curve, "-", color=OI["black"], lw=1.6,
             label=r"analytic $\sigma_{1d}(r)$")
    axA.errorbar(mids, s_samp, yerr=s_err, fmt="s", color=OI["blue"], ms=5,
                 capsize=2.5, lw=1.0, label=rf"sampled ($N={N_DISPERSION:,}$)")
    axA.set_xlabel(r"$r$ [pc]")
    axA.set_ylabel(r"$\sigma_{1d}(r)$ [pc Myr$^{-1}$]")
    axA.set_xlim(0, 3.5)
    axA.set_ylim(0, None)
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

    axC.axhspan(0.45, 0.55, color=OI["green"], alpha=0.18,
                label=r"equilibrium $\pm0.05$")
    axC.axhline(0.5, color=OI["black"], ls="--", label=r"$Q=0.5$")
    axC.plot([0], [Q], "o", color=OI["vermilion"], ms=8, zorder=5)
    axC.text(0.5, 0.84, rf"$Q={Q:.4f}$", transform=axC.transAxes, ha="center",
             va="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.6))
    axC.set_xticks([])
    axC.set_xlim(-0.6, 0.6)
    axC.set_ylim(0.44, 0.56)
    axC.set_xlabel("unscaled IC")
    axC.set_ylabel(r"virial ratio $Q = T/|V|$")
    axC.legend(loc="lower center")
    panel_label(axC, "(c)", loc="upper left")

    fig.tight_layout(pad=0.4, w_pad=0.8)
    save_fig(fig, output_dir, "plummer_velocity_equilibrium")
    print("  saved plummer_velocity_equilibrium.{png,pdf}")
    return passed


# ============================================================================
# Figure 3 -- Beta(3/2, 9/2) signature of q = v/v_esc
# ============================================================================
def fig_beta_distribution(output_dir):
    print("\n" + "=" * 60)
    print("FIG 3: q^2 = (v/v_esc)^2 vs Beta(3/2, 9/2)")
    print("=" * 60)

    prof = PlummerProfile(r_h=R_H)
    df = PlummerVelocityDF(r_h=R_H)
    a = float(prof.a)
    m = jnp.ones(N_DISPERSION)
    M = float(jnp.sum(m))
    kp, kv = jax.random.split(jax.random.PRNGKey(SEED))
    pos = prof.sample_positions(m, kp)
    vel = df.sample_velocities(pos, m, kv, G=G)
    r = jnp.linalg.norm(pos, axis=1)
    v = jnp.linalg.norm(vel, axis=1)
    v_esc = jnp.sqrt(2.0 * G * M / jnp.sqrt(r ** 2 + a ** 2))
    q2 = np.asarray((v / v_esc) ** 2)

    q2_mean = float(np.mean(q2))
    q2_var = float(np.var(q2))
    aB, bB = 1.5, 4.5
    mean_exp = aB / (aB + bB)                                   # 0.25
    var_exp = (aB * bB) / ((aB + bB) ** 2 * (aB + bB + 1.0))    # 0.02679
    mean_pass = abs(q2_mean - mean_exp) < 0.02
    var_pass = abs(q2_var - var_exp) / var_exp < 0.15
    passed = mean_pass and var_pass

    print(f"  <q^2>   expected {mean_exp:.4f}  measured {q2_mean:.4f}  "
          f"-> {'PASS' if mean_pass else 'FAIL'}")
    print(f"  Var(q^2) expected {var_exp:.4f}  measured {q2_var:.4f}  "
          f"-> {'PASS' if var_pass else 'FAIL'}")

    fig, ax = plt.subplots(figsize=(3.8, 3.3))
    ax.hist(q2, bins=40, range=(0, 1), density=True, color=OI["sky"],
            edgecolor="white", linewidth=0.3, label=rf"sampled ($N={N_DISPERSION:,}$)")
    xg = np.linspace(1e-4, 1 - 1e-4, 400)
    ax.plot(xg, np.asarray(_beta_pdf(jnp.asarray(xg), aB, bB)), "-",
            color=OI["vermilion"], lw=2.0, label=r"Beta$(3/2,\,9/2)$")
    ax.axvline(mean_exp, color=OI["black"], ls=":", lw=1.0)
    ax.text(mean_exp + 0.02, ax.get_ylim()[1] * 0.6,
            rf"$\langle q^2\rangle={q2_mean:.3f}$", fontsize=8.5)
    ax.set_xlabel(r"$q^2 = (v / v_{\rm esc})^2$")
    ax.set_ylabel("probability density")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper right")
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "plummer_beta_distribution")
    print("  saved plummer_beta_distribution.{png,pdf}")
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
    """AD vs central-FD gradients of IC observables w.r.t. the structural
    parameters (r_h, M) -- the basis for gradient-based / HMC inference."""
    print("\n" + "=" * 60)
    print("FIG 4: gradient validation (autodiff vs finite difference)")
    print("=" * 60)

    def log_density_at(r_phys, r_h):
        p = PlummerProfile(r_h=r_h)
        return jnp.log10(p.density(jnp.array([r_phys]))[0] + 1e-30)

    def sigma0(M_total, r_h=1.0):
        p = PlummerProfile(r_h=r_h)
        return jnp.sqrt(G * M_total / (6.0 * p.a))   # the DF's velocity scale

    specs = [
        ("r_h", r"$r_h$ [pc]", r"$\partial\,\log\rho(1\,{\rm pc}) / \partial r_h$",
         lambda rh: log_density_at(1.0, rh), np.linspace(0.6, 2.0, 11), 1e-5),
        ("M", r"$M_{\rm tot}$ [$M_\odot$]", r"$\partial\,\sigma_0 / \partial M_{\rm tot}$",
         lambda M: sigma0(M), np.linspace(200.0, 2000.0, 11), 1.0),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.7))
    worst = 0.0
    for ax, (key, xlab, ylab, loss, xs, h), tag in zip(axes, specs, "ab"):
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
        print(f"  d(loss)/d{key:3}: max rel err {np.max(rel):.2e}  "
              f"-> {'DIFFERENTIABLE' if np.max(rel) < 1e-4 else 'CHECK'}")

    passed = worst < 1e-4
    print(f"  overall worst rel err {worst:.2e}  -> {'PASS' if passed else 'FAIL'}")
    fig.tight_layout(pad=0.4, w_pad=0.8)
    save_fig(fig, output_dir, "plummer_gradient_validation")
    print("  saved plummer_gradient_validation.{png,pdf}")
    return passed


# ============================================================================
# Figure 5 -- isotropy of sampled positions
# ============================================================================
def fig_isotropy(output_dir):
    print("\n" + "=" * 60)
    print("FIG 5: isotropy of sampled positions")
    print("=" * 60)

    prof = PlummerProfile(r_h=R_H)
    df = PlummerVelocityDF(r_h=R_H)
    masses = jnp.ones(N_SAMPLES)
    kp, kv = jax.random.split(jax.random.PRNGKey(SEED))
    pos = prof.sample_positions(masses, kp)
    x, y, z = np.asarray(pos[:, 0]), np.asarray(pos[:, 1]), np.asarray(pos[:, 2])
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    cos_t = z / (r + 1e-12)
    phi = np.mod(np.arctan2(y, x), 2 * np.pi)

    def reduced_chi2(samples, lo, hi, nbins=40):
        counts, _ = np.histogram(samples, bins=nbins, range=(lo, hi))
        exp = len(samples) / nbins
        return float(np.sum((counts - exp) ** 2 / exp) / (nbins - 1))

    chi2_cos = reduced_chi2(cos_t, -1.0, 1.0)
    chi2_phi = reduced_chi2(phi, 0.0, 2 * np.pi)
    mean_cos = float(np.mean(cos_t))
    cos_pass = 0.5 < chi2_cos < 2.0 and abs(mean_cos) < 0.02
    phi_pass = 0.5 < chi2_phi < 2.0

    # velocity isotropy: <v_x^2> ~ <v_y^2> ~ <v_z^2> (test_velocity_isotropy, <5%)
    vel = df.sample_velocities(pos, masses, kv, G=G)
    v2 = np.asarray(jnp.mean(vel ** 2, axis=0))
    vel_spread = float(np.max(np.abs(v2 - v2.mean())) / v2.mean())
    vel_pass = vel_spread < 0.05
    passed = cos_pass and phi_pass and vel_pass

    print(f"  cos(theta) ~ U(-1,1): reduced chi^2 = {chi2_cos:.2f} (expect ~1), "
          f"<cos> = {mean_cos:+.4f}  -> {'PASS' if cos_pass else 'FAIL'}")
    print(f"  phi ~ U(0,2pi):       reduced chi^2 = {chi2_phi:.2f} (expect ~1)  "
          f"-> {'PASS' if phi_pass else 'FAIL'}")
    print(f"  velocity isotropy:    max |<v_i^2>-mean|/mean = {vel_spread:.2%} "
          f"(tol 5%)  -> {'PASS' if vel_pass else 'FAIL'}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.4, 2.9))

    axA.hist(cos_t, bins=40, range=(-1, 1), density=True, color=OI["sky"],
             edgecolor="white", linewidth=0.3)
    axA.axhline(0.5, color=OI["vermilion"], ls="--", lw=1.4, label=r"$U(-1,1)$")
    axA.set_xlabel(r"$\cos\theta$")
    axA.set_ylabel("probability density")
    axA.set_xlim(-1, 1)
    axA.set_ylim(0, 0.75)
    axA.legend(loc="lower center")
    axA.text(0.04, 0.93, rf"$\chi^2_\nu={chi2_cos:.2f}$", transform=axA.transAxes,
             fontsize=8.5, va="top")
    panel_label(axA, "(a)", loc="upper right")

    axB.hist(phi, bins=40, range=(0, 2 * np.pi), density=True, color=OI["orange"],
             edgecolor="white", linewidth=0.3)
    axB.axhline(1.0 / (2 * np.pi), color=OI["vermilion"], ls="--", lw=1.4,
                label=r"$U(0,2\pi)$")
    axB.set_xlabel(r"$\phi$ [rad]")
    axB.set_ylabel("probability density")
    axB.set_xlim(0, 2 * np.pi)
    axB.set_ylim(0, 0.32)
    axB.legend(loc="lower center")
    axB.text(0.04, 0.93, rf"$\chi^2_\nu={chi2_phi:.2f}$", transform=axB.transAxes,
             fontsize=8.5, va="top")
    panel_label(axB, "(b)", loc="upper right")

    fig.tight_layout(pad=0.4, w_pad=0.8)
    save_fig(fig, output_dir, "plummer_isotropy")
    print("  saved plummer_isotropy.{png,pdf}")
    return passed


def main():
    print("\n" + "=" * 70)
    print("PROGENAX PLUMMER (1911) PROFILE + VELOCITY-DF VALIDATION FIGURES")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {
        "Fig 1  density + cumulative mass": fig_density(OUTPUT_DIR),
        "Fig 2  velocity-space equilibrium": fig_velocity_equilibrium(OUTPUT_DIR),
        "Fig 3  Beta(3/2,9/2) signature": fig_beta_distribution(OUTPUT_DIR),
        "Fig 4  gradient validation (AD vs FD)": fig_gradient_validation(OUTPUT_DIR),
        "Fig 5  position isotropy": fig_isotropy(OUTPUT_DIR),
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    all_ok = all(results.values())
    print("=" * 70)
    print("  ALL PLUMMER VALIDATION FIGURES PASS" if all_ok
          else "  SOME PLUMMER VALIDATION FIGURES FAILED")
    print("=" * 70)
    print(f"\nFigures written to {OUTPUT_DIR}/plummer_*.png")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

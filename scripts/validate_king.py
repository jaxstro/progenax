#!/usr/bin/env python
"""
King (1966) profile + matched velocity-DF validation figures.

Produces four publication-quality figures, each anchored to a *passing* test in
``tests/validation/test_king_physics.py``. The script recomputes the same
quantities those tests assert and prints expected-vs-measured PASS/FAIL tables,
so the figures are a faithful visualization layer over already-verified physics
(not a second, drift-prone implementation).

Figures (-> anchoring test):
  1. king_concentration.png        c(W0)=log10(r_t/r_c) vs King 1966 Table II
                                    -> test_concentration_matches_king_table_ii
  2. king_density_oracle.png        lowered-Maxwellian density vs the independent
                                    direct-velocity-integral oracle (W0=7)
                                    -> test_density_shape_matches_direct_velocity_integral
  3. king_velocity_equilibrium.png  sigma(r) vs King moment; v<=v_esc; unscaled Q
                                    -> test_dispersion_profile_matches_king_moment,
                                       test_all_particles_bound,
                                       test_virial_ratio_is_half_unscaled
  4. king_w0_sweep.png              anchored W0=3,5,7,9 density family + Table II xi_t
                                    -> test_w0_affects_natural_tidal_radius

References:
    King (1966), AJ 71, 64 (Table II concentrations verbatim in
    docs/website/99-bibliography/per-paper/king-1966.md)
    Binney & Tremaine (2008), "Galactic Dynamics", 2nd ed.

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_king.py

Output:
    validation/plots/king_*.png   (curate the verified set into
    docs/website/50-validation/figures/)
"""
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from jaxstro.units import STELLAR
from progenax.profiles.king import KingProfile, king_lowered_maxwellian_density
from progenax.kinematics import KingVelocityDF
from progenax.builders import compute_kinetic_energy, compute_potential_energy


OUTPUT_DIR = "validation/plots"
N_SAMPLES = 50_000
N_DISPERSION = 40_000
N_VIRIAL = 5_000
SEED = 42

# King (1966) Table II, p. 73 (verbatim; see per-paper note). c = r_t/r_c.
TABLE_II = {
    2.5: 3.891, 3.0: 4.699, 4.0: 6.920, 5.0: 10.70, 6.0: 17.99,
    7.0: 33.71, 8.0: 68.15, 9.0: 131.4, 10.0: 223.7, 12.0: 548.2, 15.0: 2272.0,
}


def _ode_domain(c_ref):
    """ODE integration domain (xi_max, n_points) adequate for a King model whose
    tidal radius is c_ref = r_t/r_c. The domain must exceed xi_t or the profile
    pins to the integration boundary and under-estimates the concentration
    (verified: W0=12 needs xi_max>=800, W0=15 needs xi_max>=3000). Tiers below
    are the empirically-checked configurations.
    """
    if c_ref <= 250.0:
        return 400.0, 8000
    if c_ref <= 600.0:
        return 800.0, 12000
    return 3000.0, 24000

# Okabe-Ito colourblind-safe palette.
OI = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermilion": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "black": "#000000",
}

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
})


def _panel_label(ax, text):
    ax.text(0.02, 0.97, text, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="left")


def _direct_velocity_integral(W, n_v=100_000):
    """Independent oracle: rho(W) propto int_0^sqrt(2W) v^2 (e^{W-v^2/2}-1) dv.

    Identical to the oracle in test_density_shape_matches_direct_velocity_integral.
    """
    v = jnp.linspace(0.0, jnp.sqrt(2.0 * W), n_v)
    return float(jnp.trapezoid(v**2 * (jnp.exp(W - v**2 / 2.0) - 1.0), v))


# ============================================================================
# Figure 1 -- concentration c(W0) vs King 1966 Table II
# ============================================================================
def fig_concentration(output_dir):
    print("\n" + "=" * 60)
    print("FIG 1: concentration c(W0) vs King (1966) Table II")
    print("=" * 60)

    W0s = np.array(sorted(TABLE_II.keys()))
    c_ref = np.array([np.log10(TABLE_II[w]) for w in W0s])
    c_prog = []
    for w in W0s:
        xi_max, npts = _ode_domain(TABLE_II[w])
        prof = KingProfile.from_W0_rc(float(w), 1.0, xi_max=xi_max, n_ode_points=npts)
        c_prog.append(float(jnp.log10(prof.r_t)))
    c_prog = np.array(c_prog)
    resid = c_prog - c_ref
    tol = 0.03
    passed = bool(np.all(np.abs(resid) <= tol))

    print(f"  {'W0':>5} {'c_progenax':>11} {'c_TableII':>10} {'delta':>8} {'pass(<=0.03)':>13}")
    for w, cp, cr, d in zip(W0s, c_prog, c_ref, resid):
        print(f"  {w:>5.1f} {cp:>11.3f} {cr:>10.3f} {d:>+8.3f} "
              f"{'PASS' if abs(d) <= tol else 'FAIL':>13}")
    print(f"  max |delta| = {np.max(np.abs(resid)):.3f}  ->  "
          f"{'PASS' if passed else 'FAIL'}")

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(7, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})

    ax0.plot(W0s, c_ref, "o", color=OI["black"], ms=8, mfc="none", mew=1.8,
             label="King (1966) Table II")
    ax0.plot(W0s, c_prog, "-", color=OI["blue"], lw=2, marker=".", ms=10,
             label="progenax  $\\log_{10}(r_t/r_c)$")
    ax0.set_ylabel("concentration  $c = \\log_{10}(r_t/r_c)$")
    ax0.set_title("King concentration relation vs King (1966) Table II")
    ax0.legend(loc="upper left")
    _panel_label(ax0, "a")

    ax1.axhspan(-tol, tol, color=OI["green"], alpha=0.18,
                label=f"$\\pm{tol}$ tolerance")
    ax1.axhline(0, color=OI["black"], lw=0.8)
    ax1.plot(W0s, resid, "s", color=OI["vermilion"], ms=6)
    ax1.set_xlabel("$W_0$")
    ax1.set_ylabel("$\\Delta\\,\\log_{10} c$")
    ax1.set_ylim(-1.5 * tol, 1.5 * tol)
    ax1.legend(loc="upper right")
    _panel_label(ax1, "b")

    fig.tight_layout()
    path = f"{output_dir}/king_concentration.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")
    return passed


# ============================================================================
# Figure 2 -- lowered-Maxwellian density vs direct-velocity-integral oracle
# ============================================================================
def fig_density_oracle(output_dir):
    print("\n" + "=" * 60)
    print("FIG 2: density shape vs direct-velocity-integral oracle (W0=7)")
    print("=" * 60)

    prof = KingProfile.from_W0_rc(7.0, 1.0, xi_max=400.0, n_ode_points=8000)
    r_t = float(prof.r_t)
    r = jnp.linspace(0.02 * r_t, 0.9 * r_t, 25)
    xi = r / prof.r_c
    psi = jnp.interp(xi, prof.xi_grid, prof.psi_grid, left=prof.W0, right=0.0)

    rho = np.asarray(prof.density(r))
    rho_oracle = np.asarray([_direct_velocity_integral(float(p)) for p in psi])

    rho_n = rho / rho[0]
    oracle_n = rho_oracle / rho_oracle[0]
    rel = np.abs(rho_n - oracle_n) / (np.abs(oracle_n) + 1e-12)
    max_rel = float(np.max(rel))
    tol = 5e-3
    passed = max_rel < tol

    print(f"  max relative deviation (method vs oracle): {max_rel:.2e}  "
          f"(tol < {tol:.0e})  -> {'PASS' if passed else 'FAIL'}")

    # sampled histogram
    key = jax.random.PRNGKey(SEED)
    masses = jnp.ones(N_SAMPLES)
    radii = jnp.linalg.norm(prof.sample_positions(masses, key), axis=1)
    bins = np.linspace(0.0, r_t, 40)
    centers = 0.5 * (bins[1:] + bins[:-1])
    hist, _ = np.histogram(np.asarray(radii), bins=bins)
    shell_v = 4.0 / 3.0 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    rho_hist = hist / (shell_v + 1e-30)
    valid = rho_hist > 0
    # scale histogram onto the method curve (median ratio at overlapping radii)
    method_at_centers = np.interp(centers, np.asarray(r), rho_n)
    scale = np.median(method_at_centers[valid] / rho_hist[valid])
    rho_hist = rho_hist * scale

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))

    axA.semilogy(np.asarray(r) / prof.r_c, oracle_n, "-", color=OI["black"],
                 lw=2.5, label="direct $\\int v^2(e^{W-v^2/2}-1)\\,dv$ (oracle)")
    axA.semilogy(np.asarray(r) / prof.r_c, rho_n, "--", color=OI["blue"], lw=2,
                 label="progenax  $\\rho(r)$ (lowered-Maxwellian)")
    axA.semilogy(centers[valid] / float(prof.r_c), rho_hist[valid], "o",
                 color=OI["green"], ms=4, alpha=0.7,
                 label=f"sampled (N={N_SAMPLES:,})")
    axA.set_xlabel("$r / r_c$")
    axA.set_ylabel("$\\rho(r) / \\rho_0$")
    axA.set_title("King volume density vs independent oracle ($W_0=7$)")
    axA.set_ylim(1e-4, 2)
    axA.legend(loc="lower left")
    _panel_label(axA, "a")

    axB.axhline(tol, color=OI["green"], ls="--", lw=1.5,
                label=f"tolerance {tol:.0e}")
    axB.plot(np.asarray(r) / prof.r_c, rel, "o-", color=OI["blue"], ms=4)
    axB.set_yscale("log")
    axB.set_xlabel("$r / r_c$")
    axB.set_ylabel("relative deviation |method - oracle| / oracle")
    axB.set_title(f"Agreement with oracle (max = {max_rel:.1e})")
    axB.legend(loc="upper left")
    _panel_label(axB, "b")

    fig.tight_layout()
    path = f"{output_dir}/king_density_oracle.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")
    return passed


# ============================================================================
# Figure 3 -- velocity-space equilibrium triptych
# ============================================================================
def _u2_mean(W, n_u=4000):
    """Analytic lowered-Maxwellian normalized 2nd moment <u^2> (matches test)."""
    u = jnp.linspace(0.0, jnp.sqrt(2.0 * W), n_u)
    g = u**2 * (jnp.exp(W - u**2 / 2.0) - 1.0)
    return float(jnp.trapezoid(u**2 * g, u) / jnp.trapezoid(g, u))


def fig_velocity_equilibrium(output_dir):
    print("\n" + "=" * 60)
    print("FIG 3: velocity-space equilibrium (W0=7)")
    print("=" * 60)

    W0, r_c = 7.0, 1.0
    G = STELLAR.G
    prof = KingProfile.from_W0_rc(W0, r_c)
    df = KingVelocityDF(W0=W0, r_c=r_c, r_t=float(prof.r_t))

    # --- dispersion profile (large N) ---
    m_d = jnp.ones(N_DISPERSION)
    kp, kv = jax.random.split(jax.random.PRNGKey(SEED))
    pos_d = prof.sample_positions(m_d, kp)
    vel_d = df.sample_velocities(pos_d, m_d, kv, G=G)
    sigma = float(df._sigma(jnp.sum(m_d), G))
    r_d = jnp.linalg.norm(pos_d, axis=1)
    v2_d = jnp.sum(vel_d**2, axis=1)

    bins_phys = [(0.5, 1.5), (2.0, 4.0), (5.0, 9.0)]
    bin_mid, sig_samp, sig_ana, sig_rel = [], [], [], []
    for lo, hi in bins_phys:
        msk = (r_d >= lo) & (r_d < hi)
        W_bin = float(jnp.mean(jnp.interp(
            r_d[msk] / prof.r_c, df.xi_grid, df.psi_grid, left=df.W0, right=0.0)))
        s_s = float(jnp.sqrt(jnp.mean(v2_d[msk]) / 3.0))
        s_a = sigma * jnp.sqrt(_u2_mean(W_bin) / 3.0)
        bin_mid.append(0.5 * (lo + hi))
        sig_samp.append(s_s)
        sig_ana.append(float(s_a))
        sig_rel.append(abs(s_s - s_a) / s_a)
    disp_pass = all(x < 0.12 for x in sig_rel)

    # --- boundedness + virial (separate seeds, matching the tests) ---
    m_v = jnp.ones(N_VIRIAL)
    kp2, kv2 = jax.random.split(jax.random.PRNGKey(0))
    pos_v = prof.sample_positions(m_v, kp2)
    vel_v = df.sample_velocities(pos_v, m_v, kv2, G=G)
    r_v = jnp.linalg.norm(pos_v, axis=1)
    v_v = jnp.linalg.norm(vel_v, axis=1)
    W_v = jnp.interp(r_v / prof.r_c, df.xi_grid, df.psi_grid, left=df.W0, right=0.0)
    v_esc = df._sigma(jnp.sum(m_v), G) * jnp.sqrt(2.0 * jnp.maximum(W_v, 0.0))
    ratio = np.asarray(v_v / v_esc)
    bound_frac = float(jnp.mean(v_v <= v_esc + 1e-9))
    T = compute_kinetic_energy(vel_v, m_v)
    V = compute_potential_energy(pos_v, m_v, G=G)
    Q = float(T / jnp.abs(V))
    bound_pass = bound_frac == 1.0
    q_pass = abs(Q - 0.5) < 0.05
    passed = disp_pass and bound_pass and q_pass

    print(f"  sigma_1d(r) vs King moment:")
    for (lo, hi), ss, sa, rl in zip(bins_phys, sig_samp, sig_ana, sig_rel):
        print(f"    r in [{lo},{hi}): sampled={ss:.3f} analytic={sa:.3f} "
              f"rel={rl:.2%} {'PASS' if rl < 0.12 else 'FAIL'}")
    print(f"  bound fraction (v<=v_esc): {bound_frac*100:.2f}%  "
          f"-> {'PASS' if bound_pass else 'FAIL'}")
    print(f"  unscaled virial Q=T/|V|: {Q:.3f} (expect 0.5+-0.05)  "
          f"-> {'PASS' if q_pass else 'FAIL'}")

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(14.5, 4.4))

    axA.plot(bin_mid, sig_ana, "-o", color=OI["black"], ms=7,
             label="analytic King 2nd moment")
    axA.plot(bin_mid, sig_samp, "s", color=OI["blue"], ms=8,
             label=f"sampled (N={N_DISPERSION:,})")
    for x, ss, rl in zip(bin_mid, sig_samp, sig_rel):
        axA.annotate(f"{rl:.1%}", (x, ss), textcoords="offset points",
                     xytext=(6, 6), fontsize=8, color=OI["vermilion"])
    axA.set_xlabel("$r$ [pc]")
    axA.set_ylabel("$\\sigma_{1d}(r)$ [km/s, code units]")
    axA.set_title("Dispersion vs King moment (rel < 12%)")
    axA.legend(loc="upper right")
    _panel_label(axA, "a")

    axB.hist(ratio, bins=50, color=OI["sky"], edgecolor="k", linewidth=0.3)
    axB.axvline(1.0, color=OI["vermilion"], ls="--", lw=2,
                label="$v = v_{esc}$")
    axB.set_xlabel("$v / v_{esc}(r)$")
    axB.set_ylabel("count")
    axB.set_title(f"Boundedness: {bound_frac*100:.1f}% have $v \\leq v_{{esc}}$")
    axB.legend(loc="upper right")
    _panel_label(axB, "b")

    axC.bar([0], [Q], width=0.5, color=OI["green"], edgecolor="k")
    axC.axhline(0.5, color=OI["black"], ls="--", lw=2, label="equilibrium $Q=0.5$")
    axC.axhspan(0.45, 0.55, color=OI["green"], alpha=0.15, label="$\\pm0.05$ tol")
    axC.set_xticks([0])
    axC.set_xticklabels(["unscaled IC"])
    axC.set_ylim(0, 0.7)
    axC.set_ylabel("virial ratio  $Q = T/|V|$")
    axC.set_title(f"Unscaled virial $Q = {Q:.3f}$")
    axC.annotate(f"{Q:.3f}", (0, Q), textcoords="offset points",
                 xytext=(0, 6), ha="center", fontsize=11, fontweight="bold")
    axC.legend(loc="upper right")
    _panel_label(axC, "c")

    fig.tight_layout()
    path = f"{output_dir}/king_velocity_equilibrium.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")
    return passed


# ============================================================================
# Figure 4 -- anchored W0 concentration sweep
# ============================================================================
def fig_w0_sweep(output_dir):
    print("\n" + "=" * 60)
    print("FIG 4: anchored W0 concentration sweep")
    print("=" * 60)

    W0s = [3.0, 5.0, 7.0, 9.0]
    colors = [OI["sky"], OI["green"], OI["orange"], OI["vermilion"]]
    tol = 0.03
    all_pass = True

    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    print(f"  {'W0':>5} {'xi_t_prog':>10} {'c_prog':>8} {'c_TableII':>10} "
          f"{'delta':>8} {'pass':>6}")
    for W0, col in zip(W0s, colors):
        prof = KingProfile.from_W0_rc(W0, 1.0, xi_max=400.0, n_ode_points=8000)
        xi_t = float(prof.r_t / prof.r_c)
        c_prog = np.log10(xi_t)
        c_ref = np.log10(TABLE_II[W0])
        d = c_prog - c_ref
        ok = abs(d) <= tol
        all_pass = all_pass and ok
        print(f"  {W0:>5.1f} {xi_t:>10.2f} {c_prog:>8.3f} {c_ref:>10.3f} "
              f"{d:>+8.3f} {'PASS' if ok else 'FAIL':>6}")

        r = jnp.linspace(0.01, xi_t * 0.999, 400) * prof.r_c
        rho = np.asarray(prof.density(r))
        rho = rho / rho[0]
        xi = np.asarray(r / prof.r_c)
        ax.semilogy(xi, rho, "-", color=col, lw=2,
                    label=f"$W_0={W0:.0f}$  ($\\xi_t={xi_t:.1f}$, $c={c_prog:.2f}$)")
        ax.axvline(xi_t, color=col, ls=":", lw=1.2, alpha=0.8)

    ax.set_xlabel("$r / r_c$")
    ax.set_ylabel("$\\rho(r) / \\rho_0$")
    ax.set_title("King profile concentration sweep "
                 "(dotted = natural $\\xi_t = r_t/r_c$ vs Table II)")
    ax.set_xlim(0, 140)
    ax.set_ylim(1e-7, 2)
    ax.legend(loc="upper right")

    fig.tight_layout()
    path = f"{output_dir}/king_w0_sweep.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  xi_t matches Table II for all W0  -> "
          f"{'PASS' if all_pass else 'FAIL'}")
    print(f"  saved {path}")
    return all_pass


def main():
    print("\n" + "=" * 70)
    print("PROGENAX KING (1966) PROFILE + VELOCITY-DF VALIDATION FIGURES")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {
        "Fig 1  concentration c(W0) vs Table II": fig_concentration(OUTPUT_DIR),
        "Fig 2  density vs direct-integral oracle": fig_density_oracle(OUTPUT_DIR),
        "Fig 3  velocity-space equilibrium": fig_velocity_equilibrium(OUTPUT_DIR),
        "Fig 4  anchored W0 sweep": fig_w0_sweep(OUTPUT_DIR),
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    all_ok = all(results.values())
    print("=" * 70)
    print("  ALL KING VALIDATION FIGURES PASS" if all_ok
          else "  SOME KING VALIDATION FIGURES FAILED")
    print("=" * 70)
    print(f"\nFigures written to {OUTPUT_DIR}/king_*.png")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

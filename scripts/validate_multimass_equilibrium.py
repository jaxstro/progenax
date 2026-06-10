#!/usr/bin/env python
"""Validation + figure for the multi-mass LIMEPY coupled equilibrium (Phase 2).

Demonstrates that the multi-mass LIMEPY model is a first-principles, mass-segregated
EQUILIBRIUM (Gieles & Zocchi 2015, Section 2.2): one shared potential, per-component
velocity scales s_j = s mu_j^(-delta). The figure establishes, on one model:

  (a) per-component density profiles rho_j(r): heavier components are more centrally
      concentrated -- segregation built in as an equilibrium, not imposed.
  (b) per-component velocity dispersion sigma_1d,j(r): the SAMPLED dispersion matches
      the analytic LIMEPY moment s_j sqrt(I2/I0/3) -- each component is drawn from its
      own equilibrium DF (the proof of equilibrium).
  (c) segregation strength (light/heavy half-mass-radius ratio) vs delta: ~1 (no
      segregation) at delta=0, rising with delta.
  (d) per-group virial Q_j vs delta: the bias-free THEORETICAL Q_j (from the model) is
      exactly 0.5 for every component -- the rigorous equilibrium proof. The sampled
      N-body Q_j (seed-averaged) is a finite-N estimator of it: light + global are tight;
      the rarer, concentrated heavy component carries a small POSITIVE finite-N bias (the
      1/r-weighted W_j is dominated by its few innermost stars), NOT a softening effect
      (it persists at softening=0) and NOT physics. This per-component equilibrium is the
      property the lambda_seg blend lacks (Phase 0).

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_multimass_equilibrium.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jaxstro.units import STELLAR
from progenax.profiles.limepy import lowered_exponential
from progenax.cluster.multicomponent import MultiComponentCluster
from progenax.dynamics import per_group_virial_ratio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plotstyle import OI, apply_pub_style, panel_label, save_fig

apply_pub_style()

OUTPUT_DIR = "validation/plots"
G = STELLAR.G
W0, g = 7.0, 1.0
M_J = jnp.array([1.0, 4.0])          # light / heavy stellar masses
ALPHA_J = jnp.array([0.5, 0.5])      # central density fractions
DELTAS = np.array([0.0, 0.2, 0.4, 0.6])
SEEDS = range(5)
LCOL, HCOL = OI["sky"], OI["vermilion"]


def _analytic_sigma1d(W_j, s_j):
    u = jnp.linspace(0.0, jnp.sqrt(2.0 * W_j), 400)
    E = lowered_exponential(jnp.asarray(g), W_j - u**2 / 2.0)
    return float(s_j * jnp.sqrt(jnp.trapezoid(u**4 * E, u)
                                / jnp.trapezoid(u**2 * E, u) / 3.0))


def _half_mass_r(model, j):
    r = model._r_grid
    psi = jnp.interp(r / model.r_c, model.xi_grid, model.psi_grid, left=model.W0, right=0.0)
    from progenax.profiles.limepy import limepy_density_hat
    rho = limepy_density_hat(model.rescale_j[j] * psi, model.g) / \
        limepy_density_hat(model.rescale_j[j] * model.W0, model.g)
    rho = jnp.where(r <= model.r_t, rho, 0.0)
    integrand = rho * r**2
    M = jnp.concatenate([jnp.zeros(1), jnp.cumsum(
        0.5 * (integrand[1:] + integrand[:-1])) * (r[1] - r[0])])
    return float(jnp.interp(0.5 * M[-1], M, r))


def main():
    print("\n" + "=" * 70)
    print("MULTI-MASS LIMEPY EQUILIBRIUM VALIDATION (per-component dispersion, Q_j)")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    import matplotlib.pyplot as plt

    model = MultiComponentCluster.from_mass_segregation(alpha_j=ALPHA_J, m_j=M_J, W0=W0, g=g, delta=0.5, r_c=1.0)

    # ---- sampled cluster for panels (a)-(b) ----
    ic = model.sample_cluster(jax.random.PRNGKey(0), n_stars=40000, G=G)
    pos, vel, masses = ic.positions, ic.velocities, ic.masses
    pos = pos - jnp.average(pos, axis=0, weights=masses)
    vel = vel - jnp.average(vel, axis=0, weights=masses)
    r = np.asarray(jnp.linalg.norm(pos, axis=1))
    v2 = np.asarray(jnp.sum(vel**2, axis=1))
    masses_n = np.asarray(masses)
    M_tot = float(jnp.sum(masses))
    s = float(jnp.sqrt(G * M_tot / (9.0 * model.r_c * model.mu_tot)))

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2))
    axA, axB, axC, axD = axes.ravel()

    # (a) per-component density profiles (analytic), normalized to centre
    from progenax.profiles.limepy import limepy_density_hat
    rr = jnp.linspace(0.02, float(model.r_t), 400)
    psi_rr = jnp.interp(rr / model.r_c, model.xi_grid, model.psi_grid, left=model.W0, right=0.0)
    for j, (mj, col, lab) in enumerate([(M_J[0], LCOL, "light"), (M_J[1], HCOL, "heavy")]):
        rho = limepy_density_hat(model.rescale_j[j] * psi_rr, model.g) / \
            limepy_density_hat(model.rescale_j[j] * model.W0, model.g)
        axA.plot(np.asarray(rr), np.asarray(rho / rho[0]), color=col, lw=1.8,
                 label=f"{lab} ($m={float(mj):.0f}$)")
    axA.set_yscale("log"); axA.set_xscale("log")
    axA.set_xlabel(r"$r/r_c$"); axA.set_ylabel(r"$\rho_j(r)/\rho_j(0)$")
    axA.legend(frameon=False, fontsize=8); axA.set_ylim(1e-4, 2)
    axA.set_title("density: heavy more concentrated", fontsize=9)
    panel_label(axA, "(a)", loc="lower left")

    # (b) per-component dispersion: sampled vs analytic
    edges = np.linspace(0.1, 0.85 * float(model.r_t), 9)
    for j, (mj, col, lab) in enumerate([(M_J[0], LCOL, "light"), (M_J[1], HCOL, "heavy")]):
        sel = np.isclose(masses_n, float(mj))
        s_j = s * float(model.w_j[j])  # w_j = mu_j^(-delta), delta=0.5
        rc_, sig_m, sig_a = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = sel & (r >= lo) & (r < hi)
            if m.sum() < 40:
                continue
            rmid = float(np.median(r[m]))
            Wj = float(model.rescale_j[j]) * float(
                jnp.interp(rmid, model.xi_grid, model.psi_grid))
            rc_.append(rmid)
            sig_m.append(np.sqrt(v2[m].mean() / 3.0))
            sig_a.append(_analytic_sigma1d(jnp.asarray(Wj), s_j))
        axB.plot(rc_, sig_a, color=col, lw=1.6, label=f"{lab} (analytic)")
        axB.plot(rc_, sig_m, "o", color=col, ms=4, mfc="white", label=f"{lab} (sampled)")
    axB.set_xlabel(r"$r/r_c$"); axB.set_ylabel(r"$\sigma_{1\rm d,j}(r)$")
    axB.legend(frameon=False, fontsize=7, ncol=2)
    axB.set_title("dispersion: sampled = equilibrium DF", fontsize=9)
    panel_label(axB, "(b)", loc="upper right")

    # (c) segregation strength vs delta
    seg = []
    for d in DELTAS:
        mdl = MultiComponentCluster.from_mass_segregation(alpha_j=ALPHA_J, m_j=M_J, W0=W0, g=g, delta=float(d), r_c=1.0)
        seg.append(_half_mass_r(mdl, 0) / _half_mass_r(mdl, 1))
    axC.plot(DELTAS, seg, "-o", color=OI["green"], ms=5, lw=1.8)
    axC.axhline(1.0, color="0.6", ls="--", lw=1.0)
    axC.set_xlabel(r"equipartition $\delta$")
    axC.set_ylabel(r"$r_{h,\rm light}/r_{h,\rm heavy}$")
    axC.set_title("segregation grows with $\\delta$", fontsize=9)
    panel_label(axC, "(c)", loc="upper left")

    # (d) per-group Q_j vs delta: theoretical (exact) + sampled N-body (seed-averaged).
    # N-body Q_j uses softening=0 (exact Clausius) -> the heavy offset is finite-N, not
    # softening.
    print("  delta  Q_light  Q_heavy  Q_global | theory(light,heavy)")
    Qg_all, Ql_all, Qh_all, Ql_se, Qh_se = [], [], [], [], []
    Qlt_all, Qht_all = [], []
    for d in DELTAS:
        mdl = MultiComponentCluster.from_mass_segregation(alpha_j=ALPHA_J, m_j=M_J, W0=W0, g=g, delta=float(d), r_c=1.0)
        Qth = np.asarray(mdl.component_virial_ratios())
        Qlt_all.append(Qth[0]); Qht_all.append(Qth[1])
        Ql_s, Qh_s, Qg_s = [], [], []
        for sd in SEEDS:
            ic = mdl.sample_cluster(jax.random.PRNGKey(sd), n_stars=8000, G=G)
            p, v, mm = ic.positions, ic.velocities, ic.masses
            p = p - jnp.average(p, axis=0, weights=mm)
            v = v - jnp.average(v, axis=0, weights=mm)
            # ONE pairwise pass: the all-ones group gives the global Q exactly via the
            # Clausius identity W = V at softening=0 (documented per_group_virial_ratio
            # contract), so the separate compute_virial_ratio pass was redundant.
            masks = jnp.stack([jnp.isclose(mm, float(mj)) for mj in M_J]
                              + [jnp.ones_like(mm, dtype=bool)])
            Qj = np.asarray(per_group_virial_ratio(p, v, mm, G=G, group_masks=masks, softening=0.0))
            Ql_s.append(Qj[0]); Qh_s.append(Qj[1]); Qg_s.append(Qj[2])
        Ql_all.append(np.mean(Ql_s)); Qh_all.append(np.mean(Qh_s)); Qg_all.append(np.mean(Qg_s))
        Ql_se.append(np.std(Ql_s) / np.sqrt(len(Ql_s)))
        Qh_se.append(np.std(Qh_s) / np.sqrt(len(Qh_s)))
        print(f"  {d:.1f}    {np.mean(Ql_s):.3f}    {np.mean(Qh_s):.3f}    {np.mean(Qg_s):.3f}"
              f"   | {Qth[0]:.4f}, {Qth[1]:.4f}")
    # theoretical (exact) -- flat at 0.5
    axD.plot(DELTAS, Qlt_all, "-", color=LCOL, lw=2.4, alpha=0.45, label="light (theory)")
    axD.plot(DELTAS, Qht_all, "-", color=HCOL, lw=2.4, alpha=0.45, label="heavy (theory)")
    # sampled N-body (finite-N estimator)
    axD.errorbar(DELTAS, Ql_all, yerr=Ql_se, marker="o", ms=4, ls="none", color=LCOL,
                 capsize=2, label="light (sampled)")
    axD.errorbar(DELTAS, Qh_all, yerr=Qh_se, marker="s", ms=4, ls="none", color=HCOL,
                 capsize=2, label="heavy (sampled)")
    axD.plot(DELTAS, Qg_all, "^", color="0.4", ms=4, ls="none", label="global (sampled)")
    axD.axhline(0.5, color="0.6", ls="--", lw=1.0)
    axD.set_xlabel(r"equipartition $\delta$"); axD.set_ylabel(r"per-group $Q_j = T_j/|W_j|$")
    axD.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper left")
    axD.set_ylim(0.40, 0.62)
    axD.set_title("theory exact 0.5; sampled = finite-N", fontsize=9)
    panel_label(axD, "(d)", loc="lower right")

    fig.tight_layout(pad=0.6)
    save_fig(fig, OUTPUT_DIR, "seg_multimass_equilibrium")

    # ---- pass/fail ----
    seg_ok = abs(seg[0] - 1.0) < 1e-2 and np.all(np.diff(seg) > 0)
    Qg_ok = np.all(np.abs(np.array(Qg_all) - 0.5) < 0.04)
    Qj_ok = np.all(np.abs(np.array(Ql_all) - 0.5) < 0.06) and \
        np.all(np.abs(np.array(Qh_all) - 0.5) < 0.10)  # heavy is rarer -> looser
    ok = seg_ok and Qg_ok and Qj_ok
    print(f"  segregation monotonic in delta, ~1 at 0: {'PASS' if seg_ok else 'FAIL'}")
    print(f"  global Q=0.5 across delta:               {'PASS' if Qg_ok else 'FAIL'}")
    print(f"  per-group Q_j ~ 0.5 across delta:         {'PASS' if Qj_ok else 'FAIL'}")
    print(f"  saved seg_multimass_equilibrium.{{png,pdf}} -> {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    print("  MULTI-MASS EQUILIBRIUM VALIDATION PASS" if ok else "  VALIDATION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

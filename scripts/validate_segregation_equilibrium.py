#!/usr/bin/env python
"""Equilibrium-quality budget of the lambda_seg mass-segregation blend (Phase 0).

Quantifies the page-2 caveat: the lambda_seg blend is a phase-space chord between two
equilibria (lambda in {0,1}), so intermediate states are not exact equilibria. Uses the
per-mass-group virial ratio Q_j = T_j/|W_j| (the diagnostic the global virial rescale
hides) to measure how far each blend drifts from per-group balance.

Anchored to tests/validation/test_segregation_equilibrium_physics.py.

Figures:
  seg_equilibrium_budget.png
    (a) per-group Q_j vs lambda_seg (light->heavy mass groups): endpoints sit on 0.5;
        the blend fans the groups away from equilibrium.
    (b) max_j |Q_j - 0.5| vs lambda_seg — the equilibrium-error budget. The full-
        Baumgardt endpoint (lambda=1) is the cleanest; the blend ~doubles the drift.
        The global |Q-0.5| (rescaled to 0) is shown for contrast.

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_segregation_equilibrium.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jaxstro.units import STELLAR
from progenax.cluster import (
    generate_cluster_ic, SpatialStructureParams, MassSegregationLayer,
)
from progenax.imf import PowerLawIMF
from progenax.dynamics import (
    mass_group_masks, per_group_virial_ratio, compute_virial_ratio,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plotstyle import OI, apply_pub_style, panel_label, save_fig

apply_pub_style()

OUTPUT_DIR = "validation/plots"
G = STELLAR.G
N_STARS = 800
N_GROUPS = 4
SOFT = 0.05
SEEDS = range(10)
LAMBDAS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
GROUP_COLORS = [OI["sky"], OI["blue"], OI["orange"], OI["vermilion"]]  # light -> heavy


def _measure():
    """Return Qj[lambda, group] mean & SE, global Q mean, over seeds."""
    Qj_mean = np.zeros((len(LAMBDAS), N_GROUPS))
    Qj_se = np.zeros((len(LAMBDAS), N_GROUPS))
    Qg_mean = np.zeros(len(LAMBDAS))
    for i, lam in enumerate(LAMBDAS):
        per_seed_Qj, per_seed_Qg = [], []
        for s in SEEDS:
            cl = generate_cluster_ic(
                key=jax.random.PRNGKey(s), N_stars=N_STARS, M_total=float(N_STARS),
                R_half=1.0, imf_params=PowerLawIMF.kroupa(),
                structure_params=SpatialStructureParams(
                    base_profile="plummer",
                    mass_segregation=MassSegregationLayer(lambda_seg=float(lam))), G=G)
            masks = mass_group_masks(cl.masses, n_groups=N_GROUPS)
            per_seed_Qj.append(np.asarray(per_group_virial_ratio(
                cl.positions, cl.velocities, cl.masses, G=G, group_masks=masks, softening=SOFT)))
            per_seed_Qg.append(float(compute_virial_ratio(cl.positions, cl.velocities, cl.masses, G=G)))
        arr = np.array(per_seed_Qj)
        Qj_mean[i] = arr.mean(0)
        Qj_se[i] = arr.std(0) / np.sqrt(len(arr))
        Qg_mean[i] = np.mean(per_seed_Qg)
    return Qj_mean, Qj_se, Qg_mean


def main():
    print("\n" + "=" * 66)
    print("SEGREGATION BLEND EQUILIBRIUM BUDGET (per-group virial Q_j vs lambda_seg)")
    print("=" * 66)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    Qj_mean, Qj_se, Qg_mean = _measure()
    drift = np.max(np.abs(Qj_mean - 0.5), axis=1)  # max over groups, per lambda

    for i, lam in enumerate(LAMBDAS):
        print(f"  lambda={lam:.2f}: Q_j={np.round(Qj_mean[i],3)}  max|Q_j-0.5|={drift[i]:.3f}"
              f"  globalQ={Qg_mean[i]:.3f}")
    drift_seg = drift[LAMBDAS == 1.0][0]
    drift_mid = drift[LAMBDAS == 0.5][0]
    ok_clean = drift_seg < 0.08
    ok_degrade = drift_mid > 1.5 * drift_seg
    ok_global = np.all(np.abs(Qg_mean - 0.5) < 0.02)
    ok = ok_clean and ok_degrade and ok_global
    print(f"  full-Baumgardt clean (drift<0.08): {drift_seg:.3f}  {'PASS' if ok_clean else 'FAIL'}")
    print(f"  blend degrades (mid>1.5x seg): {drift_mid:.3f} vs {drift_seg:.3f}  "
          f"{'PASS' if ok_degrade else 'FAIL'}")
    print(f"  global Q==0.5 all lambda: {'PASS' if ok_global else 'FAIL'}")

    import matplotlib.pyplot as plt
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.4, 4.0))

    # (a) per-group Q_j vs lambda
    for g in range(N_GROUPS):
        axA.errorbar(LAMBDAS, Qj_mean[:, g], yerr=Qj_se[:, g], marker="o", ms=4,
                     lw=1.5, color=GROUP_COLORS[g], capsize=2,
                     label=f"group {g+1}" + (" (lightest)" if g == 0 else
                            " (heaviest)" if g == N_GROUPS-1 else ""))
    axA.axhline(0.5, color="0.4", ls="--", lw=1.0)
    axA.text(0.02, 0.5, "equilibrium", transform=axA.get_yaxis_transform(),
             va="bottom", fontsize=7, color="0.4")
    axA.set_xlabel(r"blend parameter $\lambda_{\rm seg}$")
    axA.set_ylabel(r"per-group virial ratio $Q_j = T_j/|W_j|$")
    axA.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=6.8,
               frameon=False, columnspacing=1.0)
    panel_label(axA, "(a)", loc="lower left")

    # (b) equilibrium-error budget
    axB.plot(LAMBDAS, drift, "-o", color=OI["vermilion"], ms=5, lw=1.8,
             label=r"per-group max$_j|Q_j-0.5|$")
    axB.plot(LAMBDAS, np.abs(Qg_mean - 0.5), "-s", color="0.5", ms=4, lw=1.2,
             label=r"global $|Q-0.5|$ (rescaled)")
    axB.axvline(1.0, color=OI["green"], ls=":", lw=1.0)
    axB.text(0.99, 0.92, "full Baumgardt\n(cleanest)", transform=axB.get_xaxis_transform(),
             ha="right", va="top", fontsize=7, color=OI["green"])
    axB.set_xlabel(r"blend parameter $\lambda_{\rm seg}$")
    axB.set_ylabel(r"equilibrium error")
    axB.set_ylim(0, max(drift) * 1.35)
    axB.legend(loc="upper left", fontsize=7.2)
    panel_label(axB, "(b)", loc="upper right")

    fig.tight_layout(pad=0.5, w_pad=1.5)
    save_fig(fig, OUTPUT_DIR, "seg_equilibrium_budget")
    print(f"  saved seg_equilibrium_budget.{{png,pdf}}  ->  {'PASS' if ok else 'FAIL'}")
    print("=" * 66)
    print("  EQUILIBRIUM BUDGET FIGURE PASS" if ok else "  EQUILIBRIUM BUDGET FIGURE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

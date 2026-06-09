#!/usr/bin/env python
"""
Azimuthal-density-variation substructure diagnostic (Kupper et al. 2011) figures.

sigma_Sigma/<Sigma> = relative scatter of star counts in azimuthal sectors -- a cheap
O(N) substructure proxy. Three figures anchored to passing tests in
``tests/validation/test_azimuthal_variation_physics.py``.

The fractal generator was removed (2026-06), so the Kupper D-slope is shown only as a
REFERENCE line; we validate the Poisson floor, monotonic rise with clumpiness, the
span over the Kupper range, and the anti-correlation with the CW04 Q diagnostic.

Figures:
  1. azimuthal_vs_clumpiness.png  sigma_Sigma/<Sigma> vs clump fraction; Poisson floor + Kupper range
  2. azimuthal_vs_cw04q.png       sigma_Sigma/<Sigma> vs CW04 Q (anti-correlation)
  3. azimuthal_histograms.png     sector-count histograms: smooth (flat) vs clumpy (spiky)

Reference: Kupper et al. (2011), MNRAS 417, 2300.

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_azimuthal_variation.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from progenax.diagnostics.substructure import (
    compute_azimuthal_variation,
    compute_q_parameter,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plotstyle import OI, apply_pub_style, panel_label, save_fig

apply_pub_style()

OUTPUT_DIR = "validation/plots"
N_BINS = 12


def _smooth(n, seed):
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(0, 1, n)); phi = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(-1, 1, n)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _azimuthal_clumps(n, seed, n_clumps=4, frac=0.7, width=0.15):
    rng = np.random.default_rng(seed)
    n_cl = int(frac * n)
    centers = rng.uniform(0, 2 * np.pi, n_clumps)
    phi_cl = centers[rng.integers(0, n_clumps, n_cl)] + rng.normal(0, width, n_cl)
    phi = np.concatenate([phi_cl, rng.uniform(0, 2 * np.pi, n - n_cl)])
    r = np.sqrt(rng.uniform(0, 1, n)); z = rng.uniform(-1, 1, n)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def fig_vs_clumpiness(output_dir):
    print("\n" + "=" * 60)
    print("FIG 1: sigma_Sigma/<Sigma> vs clump fraction")
    print("=" * 60)
    N = 2500
    fracs = np.linspace(0.0, 0.9, 10)
    mean, std = [], []
    for f in fracs:
        v = [compute_azimuthal_variation(_azimuthal_clumps(N, s, frac=float(f)), n_bins=N_BINS)
             for s in range(6)]
        mean.append(np.mean(v)); std.append(np.std(v))
    mean, std = np.array(mean), np.array(std)
    floor = np.sqrt(N_BINS / N)
    monotone = bool(np.all(np.diff(mean) > -1e-3))
    print(f"  smooth (frac=0): {mean[0]:.3f} (Poisson floor {floor:.3f})")
    print(f"  clumpy (frac=0.9): {mean[-1]:.3f};  monotonic: {monotone}  "
          f"-> {'PASS' if monotone else 'FAIL'}")

    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    ax.axhspan(0.07, 0.76, color=OI["sky"], alpha=0.12, label="Kupper (2011) range")
    ax.axhline(floor, color="0.5", ls=":", lw=1.1, label=rf"Poisson floor $\sqrt{{n_b/N}}$")
    ax.errorbar(fracs, mean, yerr=std, fmt="o-", color=OI["vermilion"], ms=5, capsize=2,
                lw=1.4, label=r"$\sigma_\Sigma/\langle\Sigma\rangle$")
    ax.set_xlabel("azimuthal clump fraction")
    ax.set_ylabel(r"$\sigma_\Sigma / \langle\Sigma\rangle$")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "azimuthal_vs_clumpiness")
    print("  saved azimuthal_vs_clumpiness.{png,pdf}")
    return monotone


def fig_vs_cw04q(output_dir):
    print("\n" + "=" * 60)
    print("FIG 2: sigma_Sigma/<Sigma> vs CW04 Q (anti-correlation)")
    print("=" * 60)
    sig, Q = [], []
    for f in np.linspace(0.0, 0.85, 7):
        for s in range(4):
            p = _azimuthal_clumps(800, s, frac=float(f))
            sig.append(compute_azimuthal_variation(p)); Q.append(compute_q_parameter(p))
    sig, Q = np.array(sig), np.array(Q)
    corr = float(np.corrcoef(sig, Q)[0, 1])
    passed = corr < -0.5
    print(f"  corr(sigma_Sigma, Q) = {corr:.2f} (expect strongly negative)  "
          f"-> {'PASS' if passed else 'FAIL'}")

    fig, ax = plt.subplots(figsize=(3.9, 3.4))
    sc = ax.scatter(Q, sig, c=sig, cmap="viridis", s=24, edgecolor="white", linewidth=0.3)
    ax.axvline(0.8, color="0.5", ls="--", lw=1.0)
    ax.text(0.805, ax.get_ylim()[1] * 0.9, r"$Q=0.8$", fontsize=8, color="0.4")
    ax.set_xlabel(r"CW04 $Q$")
    ax.set_ylabel(r"$\sigma_\Sigma / \langle\Sigma\rangle$")
    ax.text(0.04, 0.95, rf"$\rho={corr:.2f}$" + "\n(both detect\nsubstructure)",
            transform=ax.transAxes, fontsize=8, va="top")
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "azimuthal_vs_cw04q")
    print("  saved azimuthal_vs_cw04q.{png,pdf}")
    return passed


def fig_histograms(output_dir):
    print("\n" + "=" * 60)
    print("FIG 3: azimuthal sector-count histograms (smooth vs clumpy)")
    print("=" * 60)
    sm = _smooth(2500, 0)
    cl = _azimuthal_clumps(2500, 0, n_clumps=4, frac=0.7)
    edges = np.linspace(-np.pi, np.pi, N_BINS + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    sig_sm = compute_azimuthal_variation(sm); sig_cl = compute_azimuthal_variation(cl)
    print(f"  smooth sigma={sig_sm:.3f}, clumpy sigma={sig_cl:.3f}")

    fig, (axA, axB) = plt.subplots(1, 2, subplot_kw={"projection": "polar"},
                                   figsize=(6.6, 3.3))
    for ax, pos, col, title, sg in [(axA, sm, OI["blue"], "smooth", sig_sm),
                                    (axB, cl, OI["vermilion"], "clumpy", sig_cl)]:
        phi = np.arctan2(pos[:, 1], pos[:, 0])
        counts, _ = np.histogram(phi, bins=edges)
        ax.bar(centers, counts, width=2 * np.pi / N_BINS, color=col, alpha=0.85,
               edgecolor="white", linewidth=0.4)
        ax.axhline(counts.mean(), color="0.4", ls="--", lw=0.9)
        ax.set_title(rf"{title}: $\sigma_\Sigma/\langle\Sigma\rangle={sg:.2f}$", fontsize=9)
        ax.set_yticklabels([])
    fig.tight_layout(pad=0.6)
    save_fig(fig, output_dir, "azimuthal_histograms")
    print("  saved azimuthal_histograms.{png,pdf}")
    return sig_cl > sig_sm


def main():
    print("\n" + "=" * 70)
    print("PROGENAX AZIMUTHAL-VARIATION (KUPPER 2011) DIAGNOSTIC FIGURES")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {
        "Fig 1  sigma vs clumpiness": fig_vs_clumpiness(OUTPUT_DIR),
        "Fig 2  sigma vs CW04 Q": fig_vs_cw04q(OUTPUT_DIR),
        "Fig 3  azimuthal histograms": fig_histograms(OUTPUT_DIR),
    }
    print("\n" + "=" * 70 + "\nSUMMARY\n" + "=" * 70)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    all_ok = all(results.values())
    print("=" * 70)
    print("  ALL AZIMUTHAL-VARIATION FIGURES PASS" if all_ok else "  SOME FIGURES FAILED")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

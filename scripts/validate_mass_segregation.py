#!/usr/bin/env python
"""Validate the Λ_MSR mass-segregation *diagnostic* against analytic ground truth + plots.

Validates ``progenax.diagnostics.compute_lambda_msr`` (Allison et al. 2009) on hand-constructed
configurations with KNOWN answers — no N-body evolution (that is deferred to the gravax session;
see docs/notes/2026-06-08-gravax-segregation-validation-followup.md). This is the human-facing
companion to ``tests/validation/test_mass_segregation_physics.py``.

Λ_MSR = ⟨L_random⟩ / L_massive   (≈1 none, >1 segregated, <1 inverse),  error = σ_random/L_massive.
Definition verified against the held ApJ 700 L99 PDF
(docs/website/99-bibliography/per-paper/allison-2009.md).

History: this script previously imported the removed ``progenax.profiles.mass_segregation`` module
(stale after the diagnostics/cluster refactor) and tested the estimator only *relatively* (it used a
generator to make segregation, then checked the diagnostic saw more — circular). Rewritten 2026-06-08
to validate the diagnostic against *absolute analytic* references. The Baumgardt energy-ordered
*generator* (``progenax.cluster.mass_segregation``) is validated separately in
``tests/unit/cluster/test_mass_segregation.py``.

Usage:
    PYTHONPATH=src python scripts/validate_mass_segregation.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from progenax.diagnostics import compute_lambda_msr  # noqa: E402

PLOT_DIR = Path(__file__).parent.parent / "validation" / "plots"


# ----------------------------------------------------------------- config builders
def _uniform_ball(n, rng):
    p = rng.normal(size=(n, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p * rng.uniform(0, 1, (n, 1)) ** (1 / 3)


def make_config(kind, N=300, N_massive=15, rng=None):
    """Return (positions, masses, massive_idx) for a named regime."""
    rng = rng or np.random.default_rng(0)
    pos = _uniform_ball(N, rng)
    masses = rng.random(N)
    top = np.argsort(-masses)[:N_massive]
    if kind == "unsegregated":
        pass  # massive stars stay at their random positions
    elif kind == "segregated":
        pos[top] = rng.normal(scale=1e-3, size=(N_massive, 3))   # tight central core
    elif kind == "inverse":
        pos = rng.normal(scale=0.05, size=(N, 3))                # everyone tight...
        shell = rng.normal(size=(N_massive, 3))
        pos[top] = shell / np.linalg.norm(shell, axis=1, keepdims=True)  # ...massive on the rim
    return pos, masses, top


# ----------------------------------------------------------------- validations
def validate_regimes():
    """Λ_MSR on the three qualitative regimes + the exact N_massive=2 reference."""
    print("\n=== Λ_MSR diagnostic vs analytic ground truth ===")
    print(f"  {'regime':<16}{'Λ_MSR (meas)':>16}{'expected':>14}{'':>4}verdict")
    rng = np.random.default_rng(0)
    rows, ok_all = [], True
    checks = [("unsegregated", lambda L: 0.85 < L < 1.15, "≈ 1"),
              ("segregated", lambda L: L > 20, "≫ 1"),
              ("inverse", lambda L: L < 0.7, "< 1")]
    for kind, test, exp in checks:
        pos, m, top = make_config(kind, rng=rng)
        # average over mass-shuffles for the unsegregated case (kill single-draw scatter)
        if kind == "unsegregated":
            vals = [compute_lambda_msr(pos, rng.random(len(m)), N_massive=15,
                                       N_random_samples=120, seed=s)[0] for s in range(40)]
            lam, err = float(np.mean(vals)), float(np.std(vals))
        else:
            lam, err = compute_lambda_msr(pos, m, N_massive=15, N_random_samples=300, seed=1)
        ok = test(lam); ok_all &= ok
        rows.append((kind, pos, m, top, lam))
        print(f"  {kind:<16}{lam:>12.3f}±{err:<3.2f}{exp:>14}    {'PASS' if ok else 'FAIL'}")

    # exact N_massive=2 reference (independent: mean of all pair distances / massive-pair distance)
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [3, 0, 0], [0, 0, 4], [2, 2, 1.0]])
    mm = np.array([10, 9, 1, 1, 1, 1.0])
    lam_true = float(np.mean(pdist(pos)))   # d_massive = 1.0
    lam, _ = compute_lambda_msr(pos, mm, N_massive=2, N_random_samples=20000, seed=5)
    ok = abs(lam - lam_true) / lam_true < 0.03; ok_all &= ok
    print(f"  {'exact (N=2)':<16}{lam:>16.4f}{lam_true:>14.4f}    {'PASS' if ok else 'FAIL'}")
    return rows, ok_all


# ----------------------------------------------------------------- plots
def fig_regimes(rows):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (kind, pos, m, top, lam) in zip(axes, rows):
        other = np.setdiff1d(np.arange(len(m)), top)
        ax.scatter(pos[other, 0], pos[other, 1], s=6, c="0.7", label="all stars")
        ax.scatter(pos[top, 0], pos[top, 1], s=45, c="crimson", marker="^",
                   edgecolor="k", label="massive (MST set)")
        ax.set_aspect("equal"); ax.set_xlabel("x (pc)"); ax.set_ylabel("y (pc)")
        ax.set_title(f"{kind}\nΛ_MSR = {lam:.2f}")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Λ_MSR on analytic regimes (▲ = the N_massive used for L_massive)")
    fig.savefig(PLOT_DIR / "lambda_msr_regimes.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_monotonic_and_convergence():
    rng = np.random.default_rng(2)
    pos = _uniform_ball(300, rng); masses = rng.random(300)
    top = np.argsort(-masses)[:15]
    base = pos[top].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) Λ rises monotonically as the massive set is concentrated toward the centre
    degrees = np.linspace(0.0, 0.98, 12)
    lams, errs = [], []
    for d in degrees:
        p = pos.copy(); p[top] = base * (1.0 - d)        # shrink massive set toward centre
        lam, err = compute_lambda_msr(p, masses, N_massive=15, N_random_samples=300, seed=3)
        lams.append(lam); errs.append(err)
    axes[0].errorbar(degrees, lams, yerr=errs, fmt="o-", capsize=3)
    axes[0].axhline(1.0, ls=":", c="gray", label="Λ=1 (no segregation)")
    axes[0].set_xlabel("segregation degree (massive-set contraction)")
    axes[0].set_ylabel("Λ_MSR"); axes[0].set_title("Λ_MSR increases monotonically with segregation")
    axes[0].legend(fontsize=9)

    # (b) estimator converges to the exact value as N_random_samples grows (N=2 config)
    epos = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [3, 0, 0], [0, 0, 4], [2, 2, 1.0]])
    em = np.array([10, 9, 1, 1, 1, 1.0]); lam_true = float(np.mean(pdist(epos)))
    ns = [10, 30, 100, 300, 1000, 3000]
    spread = [np.std([compute_lambda_msr(epos, em, N_massive=2, N_random_samples=n, seed=s)[0]
                      for s in range(40)]) for n in ns]
    axes[1].loglog(ns, spread, "o-", label="seed-to-seed σ(Λ)")
    axes[1].loglog(ns, spread[0] * (ns[0] / np.array(ns)) ** 0.5, "k--", alpha=0.6,
                   label="∝ 1/√N_samples")
    axes[1].set_xlabel("N_random_samples"); axes[1].set_ylabel("σ(Λ) across seeds")
    axes[1].set_title(f"Estimator converges to exact Λ={lam_true:.3f}")
    axes[1].legend(fontsize=9)
    fig.savefig(PLOT_DIR / "lambda_msr_monotonic_convergence.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_binary_caveat():
    rng = np.random.default_rng(6)
    pos = _uniform_ball(200, rng); masses = rng.random(200)
    top2 = np.argsort(-masses)[:2]
    seps = np.logspace(-4, 0, 16)
    lams = []
    for sep in seps:
        p = pos.copy(); p[top2[1]] = p[top2[0]] + np.array([sep, 0, 0])
        lams.append(compute_lambda_msr(p, masses, N_massive=2, N_random_samples=400, seed=7)[0])
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.loglog(seps, lams, "o-")
    ax.set_xlabel("massive-pair separation (pc)"); ax.set_ylabel("Λ_MSR (N_massive=2)")
    ax.set_title("Binary-contamination caveat: a tight massive pair spuriously inflates Λ\n"
                 "(mitigation: use binary centre-of-mass positions)")
    fig.savefig(PLOT_DIR / "lambda_msr_binary_caveat.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 64)
    print("Λ_MSR diagnostic validation (Allison et al. 2009; analytic ground truth)")
    print("=" * 64)
    rows, ok = validate_regimes()
    print("\n[plots] writing ...")
    fig_regimes(rows)
    fig_monotonic_and_convergence()
    fig_binary_caveat()
    for fn in ["lambda_msr_regimes.png", "lambda_msr_monotonic_convergence.png",
               "lambda_msr_binary_caveat.png"]:
        print("   ", PLOT_DIR / fn)
    print("\nOverall:", "ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

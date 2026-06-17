#!/usr/bin/env python
r"""OED Stage 2 -- the magnitude limit ``m_lim`` as an optimisable design knob for ``M_dyn``.

Stage 1 (``demo_oed.py``) optimised WHERE to put a fixed budget of stars (radius x channel) to
measure the anisotropy radius ``r_a``. Stage 2 promotes the limiting (apparent) magnitude ``m_lim``
from a fixed completeness to a continuous, differentiable DESIGN VARIABLE, optimised JOINTLY with the
radial x channel allocation, and headlines the dynamical mass ``M`` (theta index 1).

The load-bearing physics is a TRADE in depth. Going deeper (fainter ``m_lim``):
  * UNLOCKS supply -- more detectable stars per radial bin (``avail_bins(m_lim)`` rises), and
  * ADDS noise -- the newly-admitted stars are faint and photon-noisier, so the IMF-weighted
    effective per-star error ``eps_eff(m_lim)`` rises.
Rising supply vs rising noise gives ``sigma(M_dyn)/M_dyn`` an INTERIOR minimum in ``m_lim``: a
too-shallow survey is supply-starved (few bright stars, especially in the star-poor outskirts), a
too-deep one is photon-noise-limited. The optimal depth is the survey that best weighs the cluster.

The Stage-1 additive Fisher backbone survives unchanged (single-population, mass-follows-light, so
``sigma_pred`` is ``m_lim``-independent): the per-star Jacobian ``J`` is computed ONCE; ``m_lim`` enters
only through the cheap scalars ``eps_eff`` (per channel) and ``avail_bins`` (per radius), and the
allocation is smoothly capped by availability (``n_eff = avail * tanh(n_design/avail)``).

The budget is ``N_total = 400`` stars -- the SELECTIVELY-BINDING regime: the availability cap binds
at shallow depth but not deep, so depth is a genuine trade (a saturating ``N_total=4000`` would make
every depth availability-limited and the trade degenerates). The headline interior optimum sits near
``m_lim ~ 13.3`` with ``sigma(M)/M ~ 0.10``.

What the demo computes and gates (exit 0 = all pass):
  1. JOINT OPTIMUM: the optimal ``m_lim`` and ``sigma(M)/M = sqrt(criterion)`` there, with the
     realised PM-vs-RV and core-vs-outskirts allocation split.
  2. INTERIOR-OPTIMUM CONTRAST: the joint optimum beats both a too-shallow (``m_lim=10``) and a
     too-deep (``m_lim=16``) fixed depth; a depth sweep confirms the argmin is INTERIOR.
  3. THE DEPTH TRADE DECOMPOSED: total availability and a representative ``eps_eff`` channel vs
     ``m_lim`` -- the rising-supply-vs-rising-noise numbers behind Task 8's figure.
  4. CALIBRATION: realised vs Fisher-predicted ``sigma(M)/M`` on a magnitude-selected mock. The
     depth Fisher is CONSERVATIVE-AND-BOUNDED (realised/predicted variance ratio ~ 0.70, i.e. the
     Fisher is ~19% conservative in sigma) -- the correct, documented result, not a failure: an
     RMS ``eps_eff`` over-states the noise of a bright-star-dominated dispersion estimator, the SAFE
     direction for OED (it never over-promises precision).

Gates:
  * the depth sweep has an INTERIOR argmin (0 < i < len-1);
  * the joint optimum criterion < both fixed-depth (shallow + deep) criteria;
  * the calibration variance ratio realized/predicted in [0.25, 1.15] (conservative-and-bounded).

Figures are Task 8 (next); this CLI computes + gates + writes a JSON run-record only.

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_oed_dynamical_mass.py
    env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_oed_dynamical_mass.py --full   # 64-draw calibration
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _demo_oed as oed  # noqa: E402
import _demo_oed_depth as depth  # noqa: E402

FIGURE_DIR = "docs/website/60-science-demos/figures"
RUN_RECORD = os.path.join(FIGURE_DIR, "demo_oed_dynamical_mass_run_record.json")

# theta = (r_a, M, r_h); Stage-2 target is M (dynamical mass), index 1.
TARGET_M = 1

# Joint multi-start Adam settings (over [z (3K logits), u (1 depth scalar)]).
N_STARTS = 6
N_STEPS = 400

# Fixed reference depths for the interior-optimum contrast (mag): too-shallow vs too-deep.
M_LIM_SHALLOW = 10.0
M_LIM_DEEP = 16.0

# Depth sweep (the trade decomposition + interior-argmin gate). --full -> denser grid.
N_SWEEP_QUICK = 13
N_SWEEP_FULL = 25

# Calibration draws: small by default (each draw MAP-fits a mag-selected mock); --full -> 64.
N_DRAWS_QUICK = 24
N_DRAWS_FULL = 64
# Cheaper sweep optimiser than the headline joint optimiser (sweep evaluates many depths).
SWEEP_STARTS = 3
SWEEP_STEPS = 250

# Calibration band: realized/predicted variance ratio. The depth Fisher is conservative-and-bounded
# (ratio ~ 0.70); the band brackets it with room for MC noise (~18% at 64 draws). The lower bound
# also guards against a regression to the old M-pinned-fit bug (which gave ratio ~ 0).
CAL_RATIO_LO = 0.25
CAL_RATIO_HI = 1.15


def _allocation_split(n_eff):
    """Realised allocation summary from the availability-capped per-cell counts n_eff (3, K).

    Returns (pm_fraction, core_fraction): the proper-motion share (PM_R + PM_T) of the total
    effective budget, and the inner-half ('core') share of the total -- a compact read on WHERE
    and in WHICH channels the optimal design spends its stars.
    """
    total = jnp.sum(n_eff)
    pm_fraction = float(jnp.sum(n_eff[1] + n_eff[2]) / total)
    K = n_eff.shape[1]
    core_fraction = float(jnp.sum(n_eff[:, : K // 2]) / total)
    return pm_fraction, core_fraction


def main(argv=None):
    p = argparse.ArgumentParser(
        description="OED Stage 2: m_lim as an optimisable depth knob for the dynamical mass M_dyn."
    )
    p.add_argument("--full", action="store_true",
                   help="Publication-grade run: 64-draw calibration + denser depth sweep "
                        "(slower) instead of the faster quick defaults.")
    p.add_argument("--seed", type=int, default=0,
                   help="PRNG seed for the joint optimiser + calibration (default 0).")
    p.add_argument("--n-total", type=float, default=400.0,
                   help="Total star budget across (radius x channel). 400 is the selectively-"
                        "binding regime where depth is a genuine trade (default 400).")
    p.add_argument("--out", type=str, default=RUN_RECORD,
                   help=f"Run-record JSON path (default {RUN_RECORD}).")
    args = p.parse_args(argv)

    n_total = float(args.n_total)
    key = jax.random.PRNGKey(args.seed)
    k_opt, k_cal = jax.random.split(key)
    n_draws = N_DRAWS_FULL if args.full else N_DRAWS_QUICK
    n_sweep = N_SWEEP_FULL if args.full else N_SWEEP_QUICK

    print("=" * 78)
    print("OED STAGE 2: m_lim as a depth knob for the dynamical mass M_dyn")
    print("=" * 78)
    print(f"  mock: M={oed.MOCK['M']:.0e} Msun, r_h={oed.MOCK['r_h']} pc, "
          f"r_a={oed.MOCK['r_a']} pc, d={oed.MOCK['d_kpc']} kpc")
    print(f"  depth knob m_lim in [{depth.M_LIM_LO:.0f}, {depth.M_LIM_HI:.0f}] mag  |  "
          f"K={oed.R_BINS.shape[0]} bins  |  N_total={n_total:.0f}  |  "
          f"N_field(intrinsic)={depth.N_FIELD:.0e}")
    print(f"  target = M_dyn (theta index {TARGET_M}); M left FREE in the prior (PRIOR_DIAG_M)")

    # --- (1) JOINT OPTIMUM over [z, m_lim] --------------------------------- #
    print(f"\n  optimising joint [z, m_lim] design "
          f"({N_STARTS} starts x {N_STEPS} steps) ...")
    res = depth.optimize_depth_design(target=TARGET_M, N_total=n_total, key=k_opt,
                                      n_starts=N_STARTS, n_steps=N_STEPS)
    sigM_opt = res.criterion ** 0.5                  # sigma(M)/M at the joint optimum
    pm_frac, core_frac = _allocation_split(res.n_eff)

    # --- (2) INTERIOR-OPTIMUM CONTRAST: optimum vs fixed shallow / deep ---- #
    crit_shallow = depth.crit_at_fixed_depth(m_lim=M_LIM_SHALLOW, target=TARGET_M, N_total=n_total)
    crit_deep = depth.crit_at_fixed_depth(m_lim=M_LIM_DEEP, target=TARGET_M, N_total=n_total)
    sigM_shallow = crit_shallow ** 0.5
    sigM_deep = crit_deep ** 0.5

    # Depth sweep -> confirm the argmin is INTERIOR (not at an endpoint).
    print(f"  sweeping sigma(M)/M over {n_sweep} depths in "
          f"[{depth.M_LIM_LO:.0f}, {depth.M_LIM_HI:.0f}] ...")
    m_grid = jnp.linspace(depth.M_LIM_LO, depth.M_LIM_HI, n_sweep)
    sigM_sweep = depth.sigma_M_vs_depth(m_grid, target=TARGET_M, N_total=n_total,
                                        n_starts=SWEEP_STARTS, n_steps=SWEEP_STEPS)
    i_argmin = int(jnp.argmin(sigM_sweep))
    m_lim_sweep_argmin = float(m_grid[i_argmin])

    # --- (3) THE DEPTH TRADE DECOMPOSED: supply vs noise across the grid --- #
    # Total available stars (summed over bins) rises with depth; a representative eps_eff channel
    # (RV, channel 0) also rises -- the two competing terms behind the interior optimum.
    avail_total = jnp.array([jnp.sum(depth.avail_bins(float(m))) for m in m_grid])
    eps_eff_rv = jnp.array([depth.eps_eff(float(m))[0] for m in m_grid])

    # --- (4) CALIBRATION: realised vs Fisher sigma(M)/M (mag-selected mock) - #
    print(f"\n  calibrating depth Fisher at the optimal design against {n_draws} "
          f"magnitude-selected mock draws{' [--full]' if args.full else ' [quick]'} ...")
    cal = depth.calibrate_depth_fisher(res.z, res.m_lim, n_total, n_draws, k_cal)
    cal_realized = cal.realized ** 0.5               # realised fractional sigma(M)
    cal_predicted = cal.predicted ** 0.5             # Fisher-predicted fractional sigma(M)
    cal_ratio = cal.realized / cal.predicted         # variance ratio; < 1 => Fisher conservative

    # --- quantitative summary --------------------------------------------- #
    print("\n" + "-" * 78)
    print("  (1) JOINT OPTIMUM  [z, m_lim]")
    print(f"      optimal m_lim          = {res.m_lim:.3f} mag")
    print(f"      sigma(M_dyn)/M_dyn     = {sigM_opt:.4f}   (= sqrt(criterion))")
    print(f"      realised allocation    : PM fraction {pm_frac:.3f} "
          f"(RV {1.0 - pm_frac:.3f})  |  core fraction {core_frac:.3f} "
          f"(outskirts {1.0 - core_frac:.3f})")
    print("-" * 78)
    print("  (2) INTERIOR-OPTIMUM CONTRAST  sigma(M_dyn)/M_dyn")
    print(f"      too-shallow m_lim={M_LIM_SHALLOW:<5.1f} -> {sigM_shallow:.4f}")
    print(f"      JOINT OPTIMUM m_lim={res.m_lim:<5.2f}-> {sigM_opt:.4f}   "
          f"(beats shallow x{sigM_shallow / sigM_opt:.2f}, deep x{sigM_deep / sigM_opt:.2f})")
    print(f"      too-deep    m_lim={M_LIM_DEEP:<5.1f} -> {sigM_deep:.4f}")
    print(f"      depth-sweep argmin     : m_lim={m_lim_sweep_argmin:.2f} "
          f"(grid index {i_argmin}/{n_sweep - 1}; INTERIOR if 0 < i < {n_sweep - 1})")
    print("-" * 78)
    print("  (3) THE DEPTH TRADE DECOMPOSED (rising supply vs rising noise)")
    print(f"      {'m_lim':>7s}{'avail_total':>14s}{'eps_eff[RV]':>14s}{'sigma(M)/M':>13s}")
    for j in range(n_sweep):
        print(f"      {float(m_grid[j]):>7.2f}{float(avail_total[j]):>14.1f}"
              f"{float(eps_eff_rv[j]):>14.4f}{float(sigM_sweep[j]):>13.4f}")
    print("-" * 78)
    print(f"  (4) CALIBRATION (mag-selected mock, {n_draws} draws, optimal design)")
    print(f"      realised  sigma(M)/M   = {cal_realized:.4f}")
    print(f"      Fisher    sigma(M)/M   = {cal_predicted:.4f}")
    print(f"      variance ratio realized/predicted = {cal_ratio:.3f} "
          f"(gate [{CAL_RATIO_LO}, {CAL_RATIO_HI}])")
    print("      NOTE: the depth Fisher is CONSERVATIVE-AND-BOUNDED (ratio < 1 => it over-predicts")
    print("      sigma(M)). eps_eff is an RMS over detectable masses, but a dispersion estimator is")
    print("      bright-star (inverse-variance) dominated, so the true per-cell noise sits BELOW the")
    print("      RMS. This is the SAFE direction for OED -- it never over-promises precision -- and is")
    print("      the documented, correct result, NOT a failure.")
    print("-" * 78)

    # --- gates ------------------------------------------------------------- #
    interior_ok = 0 < i_argmin < n_sweep - 1
    beats_fixed_ok = (res.criterion < crit_shallow) and (res.criterion < crit_deep)
    cal_ok = CAL_RATIO_LO < cal_ratio < CAL_RATIO_HI
    rows = [
        ("depth sweep argmin INTERIOR", interior_ok,
         f"i={i_argmin}/{n_sweep - 1}"),
        ("joint optimum < shallow & deep", beats_fixed_ok,
         f"{res.criterion:.3e} < ({crit_shallow:.3e}, {crit_deep:.3e})"),
        (f"calib ratio in [{CAL_RATIO_LO},{CAL_RATIO_HI}]", cal_ok,
         f"{cal_ratio:.3f}"),
    ]
    print(f"  {'CHECK':<34s}{'status':>8s}{'value':>26s}")
    print("-" * 78)
    all_ok = True
    for name, ok, val in rows:
        all_ok &= ok
        print(f"  {name:<34s}{'PASS' if ok else 'FAIL':>8s}{val:>26s}")
    print("-" * 78)

    # --- run-record JSON --------------------------------------------------- #
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    record = {
        "demo": "demo_oed_dynamical_mass (OED Stage 2, M_dyn depth knob)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "params": {
            "seed": args.seed,
            "n_total": n_total,
            "full": bool(args.full),
            "n_draws": n_draws,
            "n_sweep": n_sweep,
            "n_starts": N_STARTS,
            "n_steps": N_STEPS,
            "target": TARGET_M,
            "m_lim_range": [float(depth.M_LIM_LO), float(depth.M_LIM_HI)],
            "m_lim_shallow": M_LIM_SHALLOW,
            "m_lim_deep": M_LIM_DEEP,
            "N_field": float(depth.N_FIELD),
            "K_bins": int(oed.R_BINS.shape[0]),
            "mock": oed.MOCK,
            "prior_diag_M": [float(x) for x in depth.PRIOR_DIAG_M],
        },
        "results": {
            "joint_optimum": {
                "m_lim": float(res.m_lim),
                "criterion": float(res.criterion),
                "frac_sigma_M": float(sigM_opt),
                "pm_fraction": pm_frac,
                "core_fraction": core_frac,
            },
            "contrast": {
                "crit_shallow": float(crit_shallow),
                "crit_deep": float(crit_deep),
                "frac_sigma_M_shallow": float(sigM_shallow),
                "frac_sigma_M_deep": float(sigM_deep),
                "sweep_argmin_index": i_argmin,
                "sweep_argmin_m_lim": m_lim_sweep_argmin,
            },
            "depth_trade": {
                "m_grid": [float(x) for x in m_grid],
                "avail_total": [float(x) for x in avail_total],
                "eps_eff_rv": [float(x) for x in eps_eff_rv],
                "frac_sigma_M_sweep": [float(x) for x in sigM_sweep],
            },
            "calibration": {
                "realized_frac_sigma_M": float(cal_realized),
                "fisher_frac_sigma_M": float(cal_predicted),
                "variance_ratio": float(cal_ratio),
                "gate_lo": CAL_RATIO_LO,
                "gate_hi": CAL_RATIO_HI,
            },
        },
        "design_z": [float(x) for x in res.z],
        "design_u": float(res.u),
        "gates": {name: bool(ok) for name, ok, _ in rows},
        "all_pass": bool(all_ok),
    }
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  run-record -> {out_path}")

    print("=" * 78)
    print("  OED STAGE 2 DEMO: ALL PASS" if all_ok else "  OED STAGE 2 DEMO: FAILED")
    if not all_ok:
        failed = [name for name, ok, _ in rows if not ok]
        print(f"  FAILED gate(s): {', '.join(failed)}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

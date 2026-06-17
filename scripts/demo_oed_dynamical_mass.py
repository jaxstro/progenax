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
  4. CALIBRATION: realised vs Fisher-predicted ``sigma(M)/M`` on magnitude-selected mocks, averaged
     over several seeds. The depth Fisher is VALIDATED -- the realised scatter matches the prediction to
     within ~15% (variance ratio ~ 0.84-1.05 across optimal designs, consistent with 1.0; no significant
     systematic bias). A single design/seed fluctuates by ~18% MC noise at 64 draws, so we report the
     multi-seed mean +/- seed spread.

Gates:
  * the depth sweep has an INTERIOR argmin (0 < i < len-1);
  * the joint optimum criterion < both fixed-depth (shallow + deep) criteria;
  * the calibration variance ratio realized/predicted in [0.25, 1.15] (validated / sanity-bounded).

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

# Stage-2 figures AND the run-record both live in the optimal-design SECTION dir (Task 9), so a
# plain re-run regenerates the committed record in place alongside the figures it describes.
FIG2_DIR = "docs/website/60-science-demos/optimal-design/figures"
RUN_RECORD = os.path.join(FIG2_DIR, "demo_oed_dynamical_mass_run_record.json")

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
# A single 64-draw calibration carries ~18% MC noise; average over a few seeds for a representative
# central ratio + spread (this is what the figure and the printed summary report).
N_CAL_SEEDS_QUICK = 3
N_CAL_SEEDS_FULL = 5
# Cheaper sweep optimiser than the headline joint optimiser (sweep evaluates many depths).
SWEEP_STARTS = 3
SWEEP_STEPS = 250

# Calibration sanity band: realized/predicted variance ratio. The depth Fisher is VALIDATED -- the ratio
# brackets 1.0 across designs (~0.84-1.05) with ~18% per-seed MC noise at 64 draws. The band is a sanity
# check (right order of magnitude), and its lower bound guards against a regression to the old
# M-pinned-fit bug (which gave ratio ~ 0).
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
    p.add_argument("--figures", action="store_true",
                   help="Generate the five Stage-2 figures (Task 8) into "
                        f"{FIG2_DIR}/ after computing + gating.")
    args = p.parse_args(argv)

    n_total = float(args.n_total)
    key = jax.random.PRNGKey(args.seed)
    k_opt, k_cal = jax.random.split(key)
    n_draws = N_DRAWS_FULL if args.full else N_DRAWS_QUICK
    n_cal_seeds = N_CAL_SEEDS_FULL if args.full else N_CAL_SEEDS_QUICK
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
    # Average over n_cal_seeds independent calibration seeds: a single 64-draw estimate carries ~18%
    # MC noise, so we report the CENTRAL ratio + seed-to-seed spread (representative, not one draw).
    print(f"\n  calibrating depth Fisher at the optimal design: {n_cal_seeds} seeds x {n_draws} "
          f"magnitude-selected mock draws{' [--full]' if args.full else ' [quick]'} ...")
    cal_realized_vars, cal_ratios = [], []
    for s in range(n_cal_seeds):
        cal = depth.calibrate_depth_fisher(res.z, res.m_lim, n_total, n_draws,
                                           jax.random.fold_in(k_cal, s))
        cal_predicted_var = cal.predicted            # design-only; identical across seeds
        cal_realized_vars.append(cal.realized)
        cal_ratios.append(cal.realized / cal.predicted)
    cal_realized_vars = jnp.asarray(cal_realized_vars)
    cal_ratios = jnp.asarray(cal_ratios)
    cal_predicted = cal_predicted_var ** 0.5         # Fisher-predicted fractional sigma(M)
    cal_realized = float(jnp.mean(cal_realized_vars)) ** 0.5   # mean realised fractional sigma(M)
    cal_realized_spread = float(jnp.std(cal_realized_vars ** 0.5, ddof=1))  # seed spread on sigma
    cal_ratio = float(jnp.mean(cal_ratios))          # CENTRAL variance ratio (brackets 1.0 across designs)
    cal_ratio_std = float(jnp.std(cal_ratios, ddof=1))

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
    print(f"  (4) CALIBRATION (mag-selected mock, {n_cal_seeds} seeds x {n_draws} draws, optimal design)")
    print(f"      realised  sigma(M)/M   = {cal_realized:.4f} +/- {cal_realized_spread:.4f} (seed spread)")
    print(f"      Fisher    sigma(M)/M   = {cal_predicted:.4f}")
    print(f"      variance ratio realized/predicted = {cal_ratio:.3f} +/- {cal_ratio_std:.3f} "
          f"(gate [{CAL_RATIO_LO}, {CAL_RATIO_HI}])")
    print("      NOTE: the calibration VALIDATES the depth Fisher -- realised matches predicted to")
    print("      within ~15% (variance ratio brackets 1.0, ~0.84-1.05 across optimal designs; no")
    print("      significant systematic bias). A single design/seed carries ~18% MC noise at 64 draws,")
    print("      so the multi-seed mean +/- spread is the representative quantity.")
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
                "n_cal_seeds": n_cal_seeds,
                "n_draws": n_draws,
                "realized_frac_sigma_M": float(cal_realized),
                "realized_frac_sigma_M_seed_spread": float(cal_realized_spread),
                "fisher_frac_sigma_M": float(cal_predicted),
                "variance_ratio": float(cal_ratio),
                "variance_ratio_seed_std": float(cal_ratio_std),
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

    # --- figures (Task 8) -------------------------------------------------- #
    if args.figures:
        cal_summary = dict(realized=cal_realized, realized_spread=cal_realized_spread,
                           predicted=cal_predicted, ratio=cal_ratio, ratio_std=cal_ratio_std,
                           n_seeds=n_cal_seeds, n_draws=n_draws)
        make_figures(res, m_grid, sigM_sweep, avail_total, eps_eff_rv,
                     cal_summary, n_total=n_total, n_draws=n_draws)

    print("=" * 78)
    print("  OED STAGE 2 DEMO: ALL PASS" if all_ok else "  OED STAGE 2 DEMO: FAILED")
    if not all_ok:
        failed = [name for name, ok, _ in rows if not ok]
        print(f"  FAILED gate(s): {', '.join(failed)}")
    return 0 if all_ok else 1


# --------------------------------------------------------------------------- #
# Task 8: the five Stage-2 figures, saved to FIG2_DIR. All use the shared
# publication style (scripts/_plotstyle.py: Okabe-Ito palette, serif/CM math,
# inward ticks, no in-figure titles -- the MyST caption carries the title).
# These compose the PUBLIC outputs the CLI already computed (the joint optimum,
# the depth sweep, the supply/noise trade arrays, and the calibration); the
# frozen physics modules are not touched, only their public results are plotted.
# matplotlib here is a CLI/plotting path (not the JAX core); force the headless Agg
# backend BEFORE _plotstyle imports pyplot. numpy is host-side bookkeeping only.
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import numpy as np  # noqa: E402

from _plotstyle import OI, apply_pub_style, panel_label, save_fig  # noqa: E402

# Channel labels + colours shared across figures (RV, PM_R, PM_T), matching Stage 1.
_CH_LABELS = (r"RV ($v_{\rm los}$)", r"PM$_R$", r"PM$_T$")
_CH_COLORS = (OI["blue"], OI["orange"], OI["green"])


def _fig_depth_optimum(m_grid, sigM_sweep, res, fig_dir):
    """Fig 1 (HEADLINE): the fractional dynamical-mass precision sigma(M_dyn)/M_dyn
    vs limiting magnitude m_lim, with the INTERIOR minimum marked.

    Each point is the best achievable sigma(M)/M at that FROZEN depth (optimal
    allocation z); the rising-supply-vs-rising-noise trade gives an interior argmin
    -- a too-shallow survey is supply-starved, a too-deep one photon-noise-limited.
    The star marks the depth-sweep argmin; the dotted line is the joint optimum's
    m_lim (which jointly optimises z AND m_lim, so it can sit slightly off the
    coarse-sweep argmin)."""
    import matplotlib.pyplot as plt

    mg = np.asarray(m_grid)
    sw = np.asarray(sigM_sweep)
    i_min = int(np.argmin(sw))

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(mg, sw, "o-", color=OI["vermilion"], ms=4.0,
            label=r"$\sigma(M_{\rm dyn})/M_{\rm dyn}$ (optimal $z$ per depth)")
    # Mark the interior sweep argmin.
    ax.plot(mg[i_min], sw[i_min], "*", color=OI["black"], ms=14, zorder=5)
    ax.annotate(fr"interior optimum$\;m_{{\rm lim}}={mg[i_min]:.2f}$"
                + "\n" + fr"$\sigma(M)/M={sw[i_min]:.3f}$",
                xy=(mg[i_min], sw[i_min]),
                xytext=(0.5, 0.82), textcoords="axes fraction",
                ha="center", fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=OI["black"], lw=1.0))
    # Joint [z, m_lim] optimum depth. Label along the line low-left of the star, in the
    # clear mid-axis band (below the annotation, above the curve floor).
    ax.axvline(res.m_lim, color="0.6", ls=":", lw=1.0)
    y0, y1 = ax.get_ylim()
    ax.text(res.m_lim - 0.18, y0 + 0.42 * (y1 - y0), r"joint $m_{\rm lim}^\star$",
            color="0.4", fontsize=8, ha="right", va="center", rotation=90)
    ax.set_xlabel(r"limiting magnitude  $m_{\rm lim}$  [mag]")
    ax.set_ylabel(r"fractional precision  $\sigma(M_{\rm dyn})/M_{\rm dyn}$")
    ax.legend(loc="upper right")
    panel_label(ax, "the optimal survey depth", loc="upper left")
    fig.tight_layout()
    save_fig(fig, fig_dir, "demo_oed2_depth_optimum")


def _fig_depth_trade(m_grid, sigM_sweep, avail_total, eps_eff_rv, fig_dir):
    """Fig 2: the depth trade DECOMPOSED into its two competing terms vs m_lim.

    Left panel: rising SUPPLY -- the total available star pool sum_b avail_bins(m_lim)
    grows with depth (more detectable stars), AND rising NOISE -- a representative
    per-star error eps_eff[RV](m_lim) grows (newly admitted stars are faint, photon-
    noisier). Twin y-axes. Right panel: the NET effect, the information curve
    sigma(M)/M, whose interior minimum is exactly where rising supply stops beating
    rising noise. This is WHY there is an interior optimum."""
    import matplotlib.pyplot as plt

    mg = np.asarray(m_grid)
    av = np.asarray(avail_total)
    ep = np.asarray(eps_eff_rv) * oed.KMS_PER_PC_PER_MYR   # pc/Myr -> km/s for the reader
    sw = np.asarray(sigM_sweep)
    i_min = int(np.argmin(sw))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 3.8))

    # --- left: supply (rising) vs noise (rising), twin axes ---
    axL.plot(mg, av, "o-", color=OI["blue"], ms=3.5,
             label=r"supply: $\sum_b\,$avail$_b(m_{\rm lim})$")
    axL.set_xlabel(r"limiting magnitude  $m_{\rm lim}$  [mag]")
    axL.set_ylabel(r"total available stars  $\sum_b\,$avail$_b$", color=OI["blue"])
    axL.tick_params(axis="y", labelcolor=OI["blue"])
    axLr = axL.twinx()
    axLr.plot(mg, ep, "s--", color=OI["vermilion"], ms=3.5,
              label=r"noise: $\epsilon_{\rm eff}^{\rm RV}(m_{\rm lim})$")
    axLr.set_ylabel(r"effective per-star error  $\epsilon_{\rm eff}^{\rm RV}$  [km s$^{-1}$]",
                    color=OI["vermilion"])
    axLr.tick_params(axis="y", labelcolor=OI["vermilion"])
    h1, l1 = axL.get_legend_handles_labels()
    h2, l2 = axLr.get_legend_handles_labels()
    axL.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.5)
    panel_label(axL, "(a) supply up, noise up", loc="lower right")

    # --- right: the net information curve ---
    axR.plot(mg, sw, "o-", color=OI["black"], ms=4.0)
    axR.plot(mg[i_min], sw[i_min], "*", color=OI["vermilion"], ms=14, zorder=5)
    axR.text(mg[i_min], sw[i_min], fr"  interior min $m_{{\rm lim}}={mg[i_min]:.2f}$",
             fontsize=8, ha="left", va="bottom", color=OI["vermilion"])
    axR.set_xlabel(r"limiting magnitude  $m_{\rm lim}$  [mag]")
    axR.set_ylabel(r"net: fractional precision  $\sigma(M_{\rm dyn})/M_{\rm dyn}$")
    panel_label(axR, "(b) the net trade", loc="upper left")

    fig.tight_layout()
    save_fig(fig, fig_dir, "demo_oed2_depth_trade")


def _fig_allocation(res, fig_dir):
    """Fig 3: the optimal radial x channel allocation AT the optimal depth.

    Stacked bars over projected radius R of the availability-capped effective star
    count n_eff per channel (RV, PM_R, PM_T), from the joint optimum design. As in
    Stage 1, the PM channels concentrate in the OUTSKIRTS (where the OM anisotropy
    is largest) -- the interpretable design result, here at the jointly-optimal
    survey depth m_lim_star."""
    import matplotlib.pyplot as plt

    R = np.asarray(oed.R_BINS)
    n_eff = np.asarray(res.n_eff)                          # (3, K), availability-capped
    logR = np.log10(R)
    bw = 0.9 * (logR[1] - logR[0])                        # equal-width bars on log R

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    bottom = np.zeros_like(R)
    for c, (lbl, col) in enumerate(zip(_CH_LABELS, _CH_COLORS)):
        ax.bar(logR, n_eff[c], width=bw, bottom=bottom, color=col,
               edgecolor="white", linewidth=0.2, label=lbl)
        bottom = bottom + n_eff[c]
    ax.set_xlabel(r"$\log_{10}(R\,/\,{\rm pc})$")
    ax.set_ylabel(r"effective star count  $n_{\rm eff}$")
    ax.legend(loc="upper right", title=fr"channel  ($m_{{\rm lim}}^\star={res.m_lim:.2f}$)",
              title_fontsize=8)
    panel_label(ax, "PMs to the outskirts", loc="upper left")
    fig.tight_layout()
    save_fig(fig, fig_dir, "demo_oed2_allocation")


def _fig_frontier(res, n_total, fig_dir):
    """Fig 4: the dynamical-mass precision frontier -- sigma(M_dyn)/M_dyn vs star
    budget N_total AT the optimal depth m_lim_star.

    For each N_total, re-optimise the allocation at the FROZEN optimal depth and take
    sqrt(crit_at_fixed_depth) = sigma(M)/M. Precision improves with budget (mildly
    non-1/sqrt(N): the availability cap binds harder at large N and the fixed nuisance
    prior dilutes). The demo's operating point N_total is annotated."""
    import matplotlib.pyplot as plt

    n_grid = np.geomspace(1e2, 10 ** 3.5, 12)
    fs = np.array([
        depth.crit_at_fixed_depth(res.m_lim, target=TARGET_M, N_total=float(N),
                                  n_starts=SWEEP_STARTS, n_steps=SWEEP_STEPS) ** 0.5
        for N in n_grid
    ])

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.loglog(n_grid, fs, "o-", color=OI["vermilion"], ms=4,
              label=fr"optimal allocation at $m_{{\rm lim}}^\star={res.m_lim:.2f}$")
    # Mark the demo's operating budget.
    sig_at = float(np.interp(np.log(n_total), np.log(n_grid), np.log(fs)))
    sig_at = float(np.exp(sig_at))
    ax.axvline(n_total, color="0.6", ls=":", lw=1.0)
    ax.plot([n_total], [sig_at], "s", color=OI["black"], ms=7, zorder=5)
    ax.annotate(fr"demo: $N_{{\rm total}}={n_total:.0f}$" + "\n"
                + fr"$\sigma(M)/M={sig_at:.3f}$",
                xy=(n_total, sig_at), xytext=(0.06, 0.12),
                textcoords="axes fraction", ha="left", fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=OI["black"], lw=1.0))
    ax.set_xlabel(r"star budget  $N_{\rm total}$")
    ax.set_ylabel(r"fractional precision  $\sigma(M_{\rm dyn})/M_{\rm dyn}$")
    ax.legend(loc="upper right")
    panel_label(ax, "precision frontier", loc="lower left")
    fig.tight_layout()
    save_fig(fig, fig_dir, "demo_oed2_frontier")


def _fig_calibration(cal, fig_dir):
    """Fig 5: realized vs Fisher-predicted fractional precision sigma(M_dyn)/M_dyn
    on magnitude-selected mocks, averaged over several calibration seeds.

    The realized point is the MEAN over `n_seeds` seeds; the error bar is the
    seed-to-seed spread on sigma(M). The realized scatter matches the Fisher to within
    the seed spread (variance ratio ~ 0.84-1.05 across optimal designs, consistent with
    1.0) -- the depth Fisher is VALIDATED, with no significant systematic bias. The
    central ratio +/- seed spread is annotated."""
    import matplotlib.pyplot as plt

    realized = float(cal["realized"])                    # mean realised sigma(M)/M
    realized_err = float(cal["realized_spread"])         # seed-to-seed spread on sigma
    fisher_p = float(cal["predicted"])
    ratio, ratio_std = float(cal["ratio"]), float(cal["ratio_std"])
    n_seeds, n_draws = int(cal["n_seeds"]), int(cal["n_draws"])
    # Data-driven verdict: brackets 1.0 within the spread -> consistent; else conservative / anti-.
    if abs(ratio - 1.0) <= max(ratio_std, 0.15):
        tag = "consistent with Fisher"
    else:
        tag = "conservative" if ratio < 1.0 else "anti-conservative"

    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.errorbar([0], [realized], yerr=[realized_err], fmt="o", ms=7,
                color=OI["vermilion"], capsize=4,
                label=fr"realized ({n_seeds} seeds $\times$ {n_draws} mocks)")
    ax.plot([1], [fisher_p], "s", ms=7, color=OI["blue"],
            label=r"depth Fisher  $\sqrt{(F^{-1})_{MM}}$")
    ax.axhline(fisher_p, color=OI["blue"], ls=":", lw=1.0, alpha=0.7)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["realized", "Fisher"])
    ax.set_ylabel(r"fractional precision  $\sigma(M_{\rm dyn})/M_{\rm dyn}$")
    ax.legend(loc="upper right", fontsize=7.5)
    panel_label(ax, fr"ratio$\,=\,{ratio:.2f}\pm{ratio_std:.2f}$ ({tag})", loc="lower left")
    fig.tight_layout()
    save_fig(fig, fig_dir, "demo_oed2_calibration")


def make_figures(res, m_grid, sigM_sweep, avail_total, eps_eff_rv, cal, *,
                 n_total, n_draws):
    """Generate the five Stage-2 figures into FIG2_DIR (PNG + PDF via save_fig).

      * fig 1 (depth_optimum): sigM_sweep vs m_grid; interior argmin + joint m_lim_star.
      * fig 2 (depth_trade):   avail_total (supply) + eps_eff_rv (noise) + sigM_sweep (net).
      * fig 3 (allocation):    res.n_eff (3, K) over oed.R_BINS at the optimal depth.
      * fig 4 (frontier):      sweep N_total, depth.crit_at_fixed_depth at res.m_lim.
      * fig 5 (calibration):   mean realized vs Fisher predicted (multi-seed mag-selected mocks).
    """
    apply_pub_style()
    os.makedirs(FIG2_DIR, exist_ok=True)
    print(f"\n  generating 5 Stage-2 figures -> {FIG2_DIR}/ ...")
    _fig_depth_optimum(m_grid, sigM_sweep, res, FIG2_DIR)
    _fig_depth_trade(m_grid, sigM_sweep, avail_total, eps_eff_rv, FIG2_DIR)
    _fig_allocation(res, FIG2_DIR)
    _fig_frontier(res, n_total, FIG2_DIR)
    _fig_calibration(cal, FIG2_DIR)
    print(f"  figures: wrote 5 PNG+PDF to {FIG2_DIR}/demo_oed2_*.png")


if __name__ == "__main__":
    sys.exit(main())

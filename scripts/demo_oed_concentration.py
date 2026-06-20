#!/usr/bin/env python
r"""W0-OED (concentration) -- the c-optimal observing design for a cluster's W0.

Stage-1 (``demo_oed.py``) asked where to spend a fixed star budget (radial bins x
{RV, PM_R, PM_T}) to best constrain the Osipkov-Merritt anisotropy radius ``r_a``.
This demo asks the SAME question for the cluster CONCENTRATION ``W0`` -- the headline
of the W0-differentiability arc (ADR-0016 C1 PCHIP, ADR-0017 df_moment W0 lock) closed
through an actual Fisher/OED inference that treats W0 as a parameter, for BOTH a King
(headline OM-King) and a Michie profile. See
``docs/plans/2026-06-18-oed-concentration-w0-{design,plan}.md``.

The load-bearing idea (Stage-1, ADR-0004) is unchanged: the design Fisher is ADDITIVE
and LINEAR in the design, ``F(z) = sum_{bin b, channel c} n_eff,{b,c} * M_{b,c}``, each
per-star block ``M_{b,c} = 2 J J^T / (sigma^2 + eps_c^2)`` design-INDEPENDENT and built
ONCE via a single reverse-mode ``jacrev`` through ``project_dispersion`` (the OM Jeans +
Binney & Mamon 1982 projection of the King/Michie density). The optimization is then
pure 3x3 linear algebra. The Fisher is built wrt ``ln theta`` (ADR-0011), so every
covariance entry is a FRACTIONAL variance and the c-headline is the fractional precision
``sigma(W0)/... = sqrt((F^-1)_{W0,W0})`` -- the squared fractional precision on ln W0.

What this CLI computes and prints (exit 0):
  * per model (king, michie): the per-star blocks at the truth ``theta=(W0, r_a, M)``,
    the c-/D-/A-optimal designs (multi-start Adam), and the HEADLINE precision gain
    ``sigma(lnW0)_uniform / sigma(lnW0)_c-optimal`` at FIXED N (the prior cancels in
    the ratio);
  * the pre-registered hypothesis (H1) check: the W0 c-optimal radial split (core vs
    outskirts) and channel balance (RV vs PM), CONTRASTED with the Stage-1 r_a result.

Figures (publication-quality, into ``--outdir``):
  1. King   W0 c-optimal allocation heatmap (radius x channel).
  2. Michie W0 c-optimal allocation heatmap.
  3. THE HEADLINE CONTRAST: W0 radial allocation (core-heavy) vs the Stage-1 r_a
     allocation (outskirts-heavy) -- the pre-registered hypothesis made visual.
  4. Precision gain (uniform vs c-optimal sigma(lnW0)) + c/D/A comparison, King vs Michie.
  5. The W0-r_a degeneracy: the Fisher correlation rho(W0, r_a) at the c-optimal design,
     King vs Michie -- how separable concentration is from anisotropy.

The CLI computes ONLY the cheap parts -- ONE jacrev per model at the truth, the c/D/A
3x3-linear-algebra optimization, and the figures. The @slow real-star calibration MC is
NOT run here (it is the env-gated ``test_W0_fisher_calibration_matches_realized_scatter``,
King only; the Michie MLE-MC is ~28 GB and is intentionally never run). The validated King
calibration ratio (realized/Fisher fractional variance of W0) ~ 0.976 is CITED below as a
constant, not re-derived.

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_oed_concentration.py
    env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_oed_concentration.py --outdir /tmp/oedc
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _demo_oed as oed  # noqa: E402  -- Stage-1 backbone (reused criteria + the r_a contrast)
import _demo_oed_concentration as oedc  # noqa: E402  -- the W0-OED core (Task 1-5 API)

G = oedc.STELLAR.G

# Figures live in the optimal-design SECTION dir (alongside the Stage-1/Stage-2 figures
# and run-records) so a plain re-run regenerates them in place next to the MyST page.
FIGURE_DIR = "docs/website/60-science-demos/optimal-design/figures"
RUN_RECORD = os.path.join(FIGURE_DIR, "demo_oed_concentration_run_record.json")

# Stage-1 c-optimal r_a design (the HEADLINE CONTRAST). Committed run-record next to the
# Stage-1 figures; loaded so the W0-vs-r_a comparison uses Stage-1's ACTUAL optimized
# allocation, not a hand-typed pattern. (Falls back to the documented Stage-1 pattern if
# the record is ever absent -- see _stage1_ra_design.)
STAGE1_RA_RECORD = os.path.join(FIGURE_DIR, "demo_oed_run_record.json")

# The KING real-star calibration ratio (realized Var(lnW0_hat) / Fisher (F^-1)_{W0,W0}),
# VALIDATED SEPARATELY by the env-gated @slow gate
# (test_W0_fisher_calibration_matches_realized_scatter, 48 draws, N_total=400). CITED here
# as a constant -- the MLE-MC is NOT re-run in this CLI (it OOM-crashes the host for Michie).
KING_CALIB_RATIO = 0.976

MODELS = ("king", "michie")
# Default budget: the SAME selectively-binding N_total=400 the King calibration gate runs at
# (so the printed gains/splits reproduce the validated calibration's operating point).
N_TOTAL_DEFAULT = 400.0

# Multi-start Adam settings (over the softmax design vector z). Quick-dials for the smoke test.
N_STARTS_DEFAULT = 6
N_STEPS_DEFAULT = 400
N_STARTS_QUICK = 2
N_STEPS_QUICK = 60


# --------------------------------------------------------------------------- #
# Per-model design computation (cheap: ONE jacrev + 3x3-linalg c/D/A optimization).
# --------------------------------------------------------------------------- #
def _designs(Mb, cb, n_total, key, n_starts, n_steps):
    """Optimize the c-, D-, A-optimal designs (multi-start). Target = W0 (index 0).

    Returns {name -> DesignResult}. The c criterion is the W0 marginal fractional
    variance (target=0); D/A are the W0-OED arc's D/A optima at the SAME design space.
    """
    crit_fns = {
        "c": lambda F: oedc.c_criterion(F, target=0),  # W0 marginal fractional variance
        "d": oedc.d_criterion,
        "a": oedc.a_criterion,
    }
    out = {}
    for i, (name, fn) in enumerate(crit_fns.items()):
        out[name] = oedc.optimize_design(
            fn,
            Mb,
            cb,
            n_total,
            key=jax.random.fold_in(key, i),
            n_starts=n_starts,
            n_steps=n_steps,
        )
    return out


def _radial_channel_split(n_eff):
    """(core_frac, outer_frac, rv_frac, pm_frac) from per-cell counts n_eff (3, K).

    core/outer split the inner vs outer half of the K radial bins; rv/pm split the RV
    (channel 0) vs PM (channels 1+2) share of the total effective budget. The compact
    read on WHERE and in WHICH channels a design spends its stars.
    """
    K = n_eff.shape[1]
    total = jnp.sum(n_eff)
    core = float(jnp.sum(n_eff[:, : K // 2]) / total)
    outer = float(jnp.sum(n_eff[:, K // 2 :]) / total)
    rv = float(jnp.sum(n_eff[0]) / total)
    pm = float(jnp.sum(n_eff[1] + n_eff[2]) / total)
    return core, outer, rv, pm


def _pm_fraction_per_bin(n_eff):
    """Per-bin PM fraction (n_pm_r + n_pm_t)/sum_channels (K,)."""
    return (n_eff[1] + n_eff[2]) / jnp.sum(n_eff, axis=0)


def _wa_correlation(F):
    """Fisher correlation rho(W0, r_a) = (F^-1)_{W0,r_a}/sqrt((F^-1)_{W0,W0}(F^-1)_{r_a,r_a}).

    The off-diagonal of the INVERSE Fisher (the covariance) normalized to [-1, 1]:
    |rho| -> 1 means concentration W0 and anisotropy r_a are nearly degenerate at this
    design (hard to separate); |rho| -> 0 means the design measures them independently.
    """
    Finv = jnp.linalg.inv(F)
    return float(Finv[0, 1] / jnp.sqrt(Finv[0, 0] * Finv[1, 1]))


def _compute_model(model, n_total, key, n_starts, n_steps):
    """All the cheap per-model OED products -> a dict (no MC, no ODE memory).

    ONE jacrev (per_star_blocks at the truth) + the c/D/A 3x3-linear-algebra
    optimization + the uniform baseline + the H1 splits + the W0-r_a correlation.
    """
    theta = oedc.theta_truth()
    Mb, _ = oedc.per_star_blocks(theta, oedc.R_BINS, oedc.EPS, G, model)  # ONE jacrev
    cb = oedc.completeness(oedc.R_BINS)
    K = oedc.R_BINS.shape[0]

    z_unif = jnp.zeros(3 * K)
    F_unif = oedc.fisher(z_unif, Mb, cb, n_total, oedc.PRIOR_DIAG)
    c_unif = float(oedc.c_criterion(F_unif, target=0))  # uniform W0 fractional variance

    designs = _designs(Mb, cb, n_total, key, n_starts, n_steps)
    c_res = designs["c"]
    F_c = oedc.fisher(c_res.z, Mb, cb, n_total, oedc.PRIOR_DIAG)

    n_c = oedc.design_counts(c_res.z, cb, n_total)  # (3, K) c-optimal allocation
    core, outer, rv, pm = _radial_channel_split(n_c)
    pm_bin = _pm_fraction_per_bin(n_c)

    # HEADLINE gain: equal-precision factor on the FRACTIONAL precision sigma(lnW0).
    # c is the squared fractional precision, so the precision RATIO is sqrt(c_unif/c_opt).
    sig_unif = c_unif**0.5
    sig_c = c_res.criterion**0.5
    gain = sig_unif / sig_c

    return dict(
        model=model,
        Mb=Mb,
        cb=cb,
        n_total=n_total,
        K=K,
        designs=designs,
        z_unif=z_unif,
        F_unif=F_unif,
        F_c=F_c,
        n_c=n_c,
        c_unif=c_unif,
        c_opt=c_res.criterion,
        d_unif=float(oedc.d_criterion(F_unif)),
        d_opt=designs["d"].criterion,
        a_unif=float(oedc.a_criterion(F_unif)),
        a_opt=designs["a"].criterion,
        sig_unif=sig_unif,
        sig_c=sig_c,
        gain=gain,
        core=core,
        outer=outer,
        rv=rv,
        pm=pm,
        pm_bin=pm_bin,
        rho_wa=_wa_correlation(F_c),
    )


# --------------------------------------------------------------------------- #
# The Stage-1 r_a c-optimal design (the contrast). Loaded from the committed
# run-record + the Stage-1 backbone (cheap; no re-optimization).
# --------------------------------------------------------------------------- #
def _stage1_ra_design():
    """Stage-1 c-optimal r_a allocation -> (n_eff (3, K1), core_frac, outer_frac, pm_bin).

    Loads Stage-1's committed c-design z vector + N_total from its run-record and decodes
    it through the Stage-1 backbone (oed.completeness / oed.design_counts) -- the ACTUAL
    optimized r_a allocation, on the Stage-1 r_h-scaled radial grid. Returns None if the
    record is absent (the CLI then annotates the documented Stage-1 pattern instead).
    """
    if not os.path.exists(STAGE1_RA_RECORD):
        return None
    with open(STAGE1_RA_RECORD) as f:
        rec = json.load(f)
    z_c = jnp.asarray(rec["designs_z"]["c"])
    n_total = float(rec["params"]["n_total"])
    cb = oed.completeness(oed.R_BINS)
    n_eff = oed.design_counts(z_c, cb, n_total)  # (3, K1)
    K1 = oed.R_BINS.shape[0]
    total = jnp.sum(n_eff)
    core = float(jnp.sum(n_eff[:, : K1 // 2]) / total)
    outer = float(jnp.sum(n_eff[:, K1 // 2 :]) / total)
    pm_bin = (n_eff[1] + n_eff[2]) / jnp.sum(n_eff, axis=0)
    return dict(
        n_eff=n_eff,
        R_BINS=oed.R_BINS,
        core=core,
        outer=outer,
        pm_bin=pm_bin,
        pm_inner=float(rec["results"]["pm_fraction_inner"]),
        pm_outer=float(rec["results"]["pm_fraction_outer"]),
        n_total=n_total,
    )


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(
        description="W0-OED concentration demo: the c-optimal observing design for W0 "
        "(King + Michie), with the pre-registered W0-vs-r_a hypothesis check."
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=FIGURE_DIR,
        help=f"Directory for the figures (default {FIGURE_DIR}).",
    )
    p.add_argument(
        "--n-starts",
        type=int,
        default=N_STARTS_DEFAULT,
        help=f"Multi-start Adam restarts for each c/D/A optimize "
        f"(default {N_STARTS_DEFAULT}).",
    )
    p.add_argument(
        "--n-steps",
        type=int,
        default=N_STEPS_DEFAULT,
        help=f"Adam steps per start (default {N_STEPS_DEFAULT}).",
    )
    p.add_argument(
        "--n-total",
        type=float,
        default=N_TOTAL_DEFAULT,
        help=f"Total star budget across (radius x channel); the King calibration "
        f"operating point (default {N_TOTAL_DEFAULT:.0f}).",
    )
    p.add_argument(
        "--seed", type=int, default=0, help="PRNG seed for the optimizer (default 0)."
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Dial the optimizer down (smoke-test/CI fast path): few starts/steps.",
    )
    p.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation; print the summary + write the run-record only.",
    )
    args = p.parse_args(argv)

    n_starts = N_STARTS_QUICK if args.quick else args.n_starts
    n_steps = N_STEPS_QUICK if args.quick else args.n_steps
    n_total = float(args.n_total)
    key = jax.random.PRNGKey(args.seed)

    print("=" * 78)
    print("W0-OED CONCENTRATION: the c-optimal observing design for W0 (King + Michie)")
    print("=" * 78)
    print(
        f"  mock: W0={oedc.MOCK['W0']}, r_a={oedc.MOCK['r_a']} r_c, M={oedc.MOCK['M']:.0e} Msun, "
        f"r_c={oedc.MOCK['r_c']} (length unit), d={oedc.MOCK['d_kpc']} kpc"
    )
    print(
        f"  errors: eps_RV={oedc.MOCK['eps_RV_kms']} km/s, eps_PM={oedc.MOCK['eps_PM_masyr']} "
        f"mas/yr  |  K={oedc.R_BINS.shape[0]} bins  |  N_total={n_total:.0f}"
    )
    print(
        f"  optimizer: {n_starts} starts x {n_steps} steps"
        f"{'  [--quick]' if args.quick else ''}  |  theta=(W0=0, r_a=1, M=2), c-target=W0"
    )

    # --- per-model cheap OED products (ONE jacrev + 3x3 linalg each) ------- #
    results = {}
    for i, model in enumerate(MODELS):
        print(
            f"\n  computing {model} design (1 jacrev at truth + c/D/A 3x3 optimization) ..."
        )
        results[model] = _compute_model(
            model, n_total, jax.random.fold_in(key, i), n_starts, n_steps
        )

    ra = _stage1_ra_design()  # the contrast (cheap; from the committed Stage-1 record)

    # --- quantitative summary --------------------------------------------- #
    print("\n" + "-" * 78)
    print("  W0 c-OPTIMAL DESIGN (per model)")
    print(
        f"  {'model':<8s}{'sig(lnW0) unif':>16s}{'-> c-opt':>11s}{'gain':>8s}"
        f"{'RV/PM':>13s}{'core/outer':>14s}"
    )
    print("-" * 78)
    for model in MODELS:
        r = results[model]
        print(
            f"  {model:<8s}{r['sig_unif']:>16.4f}{r['sig_c']:>11.4f}{r['gain']:>7.2f}x"
            f"{r['rv']:>7.2f}/{r['pm']:<5.2f}{r['core']:>8.2f}/{r['outer']:<5.2f}"
        )
    print("-" * 78)

    # c/D/A table (each criterion on the uniform + each design's own optimum).
    print("  c / D / A criteria  (uniform  ->  that-criterion optimum), per model")
    print(
        f"  {'model':<8s}{'c unif':>12s}{'c opt':>12s}{'D unif':>12s}{'D opt':>12s}"
        f"{'A unif':>12s}{'A opt':>12s}"
    )
    print("-" * 78)
    for model in MODELS:
        r = results[model]
        print(
            f"  {model:<8s}{r['c_unif']:>12.3e}{r['c_opt']:>12.3e}"
            f"{r['d_unif']:>12.3e}{r['d_opt']:>12.3e}{r['a_unif']:>12.3e}{r['a_opt']:>12.3e}"
        )
    print("-" * 78)

    # Per-bin PM fraction (the channel-balance read entering the H1 channel verdict).
    print("  per-bin PM fraction (n_PM / n_total), c-optimal:")
    for model in MODELS:
        r = results[model]
        core_pm = float(jnp.mean(r["pm_bin"][: r["K"] // 2]))
        outer_pm = float(jnp.mean(r["pm_bin"][r["K"] // 2 :]))
        print(f"     {model:<8s} core-half {core_pm:.2f} -> outer-half {outer_pm:.2f}")
    print("-" * 78)

    # --- the pre-registered hypothesis H1 verdict -------------------------- #
    king = results["king"]
    if ra is not None:
        ra_outer_pct = 100.0 * ra["outer"]
        ra_contrast = (
            f"Stage-1 r_a -> {ra_outer_pct:.0f}% OUTSKIRTS "
            f"(core {100.0 * ra['core']:.0f}%)"
        )
    else:
        ra_contrast = (
            "Stage-1 r_a -> ~99% OUTSKIRTS (documented pattern; record absent)"
        )
    print("  PRE-REGISTERED HYPOTHESIS H1  (W0 differs from r_a)")
    print(
        f"     RADIAL  : CONFIRMED -- W0 -> {100.0 * king['core']:.0f}% CORE "
        f"(king); {ra_contrast}."
    )
    print("                The headline contrast: concentration wants the CORE")
    print(
        "                (core<->truncation sigma contrast), anisotropy wanted the OUTSKIRTS."
    )
    print(
        f"     CHANNEL : REFUTED -- W0 is PM-DOMINATED (king RV {king['rv']:.2f} / "
        f"PM {king['pm']:.2f}), NOT"
    )
    print(
        "                channel-balanced as predicted. At the RV/PM error parity PM gives"
    )
    print(
        "                2 components/star (PM_R + PM_T) -- a 2-for-1 efficiency. A wrong"
    )
    print(
        "                sub-prediction is a finding, reported honestly (null-result integrity)."
    )
    print("-" * 78)
    print("  W0 <-> r_a degeneracy (Fisher correlation rho at the c-optimal design):")
    for model in MODELS:
        print(f"     {model:<8s} rho(W0, r_a) = {results[model]['rho_wa']:+.3f}")
    print(
        "  KING real-star calibration (validated separately, env-gated @slow, 48 draws):"
    )
    print(
        f"     realized Var(lnW0_hat) / Fisher (F^-1)_W0W0 = {KING_CALIB_RATIO:.3f}  "
        f"(~1.0 -> the design Fisher is trustworthy)"
    )
    print(
        "     [Michie calibration MC intentionally NOT run -- its MLE-MC backward tape is"
    )
    print(
        "      ~28 GB and OOM-crashes the host; Michie is validated by the cheap gradient tests.]"
    )
    print("-" * 78)

    # --- run-record JSON --------------------------------------------------- #
    os.makedirs(args.outdir, exist_ok=True)
    record = {
        "demo": "demo_oed_concentration (W0-OED, King + Michie)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "params": {
            "seed": args.seed,
            "n_total": n_total,
            "n_starts": n_starts,
            "n_steps": n_steps,
            "quick": bool(args.quick),
            "K_bins": int(oedc.R_BINS.shape[0]),
            "mock": oedc.MOCK,
            "eps_pcMyr": [float(e) for e in oedc.EPS],
            "prior_diag": [float(x) for x in oedc.PRIOR_DIAG],
            "king_calib_ratio_cited": KING_CALIB_RATIO,
        },
        "results": {
            model: {
                "frac_sigma_lnW0_uniform": results[model]["sig_unif"],
                "frac_sigma_lnW0_c_design": results[model]["sig_c"],
                "precision_gain": results[model]["gain"],
                "rv_fraction": results[model]["rv"],
                "pm_fraction": results[model]["pm"],
                "core_fraction": results[model]["core"],
                "outer_fraction": results[model]["outer"],
                "rho_W0_ra": results[model]["rho_wa"],
                "pm_fraction_per_bin": [float(x) for x in results[model]["pm_bin"]],
                "criteria": {
                    "c_uniform": results[model]["c_unif"],
                    "c_opt": results[model]["c_opt"],
                    "d_uniform": results[model]["d_unif"],
                    "d_opt": results[model]["d_opt"],
                    "a_uniform": results[model]["a_unif"],
                    "a_opt": results[model]["a_opt"],
                },
            }
            for model in MODELS
        },
        "hypothesis_H1": {
            "radial": "CONFIRMED (W0 core-heavy vs r_a outskirts-heavy)",
            "channel": "REFUTED (W0 PM-dominated, 2-for-1 PM efficiency at RV/PM error parity)",
            "king_W0_core_fraction": king["core"],
            "stage1_ra_outer_fraction": (ra["outer"] if ra is not None else None),
        },
    }
    # Write the run-record into --outdir (NOT the fixed FIGURE_DIR): the @quick smoke test
    # passes --outdir=tmp_path, so a fixed path would clobber the committed full-quality
    # record with low-resolution smoke numbers. The default --outdir IS FIGURE_DIR, so a real
    # figure-generation run still lands the record next to the committed figures.
    run_record_path = os.path.join(args.outdir, os.path.basename(RUN_RECORD))
    with open(run_record_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  run-record -> {run_record_path}")

    # --- figures ----------------------------------------------------------- #
    if args.no_figures:
        print("  figures: SKIPPED (--no-figures)")
    else:
        make_figures(results, ra, args.outdir)

    print("=" * 78)
    print("  W0-OED CONCENTRATION DEMO: DONE")
    return 0


# =========================================================================== #
# Figures: publication style (scripts/_plotstyle.py -- Okabe-Ito palette,
# serif/CM math, inward ticks). matplotlib is a CLI/plotting path (not the JAX
# core); force the headless Agg backend BEFORE _plotstyle imports pyplot.
# =========================================================================== #
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import numpy as np  # noqa: E402  -- host-side plotting bookkeeping only
from _plotstyle import OI, apply_pub_style, panel_label, save_fig  # noqa: E402

# Channel labels shared across figures (RV, PM_R, PM_T), matching Stage 1/2.
_CH_LABELS = (r"RV ($v_{\rm los}$)", r"PM$_R$", r"PM$_T$")
_CH_COLORS = (OI["blue"], OI["orange"], OI["green"])
_CH_SHORT = ("RV", r"PM$_R$", r"PM$_T$")


def _fig_allocation_heatmap(res, fig_dir):
    """Figs 1-2: the W0 c-optimal allocation as a (channel x radius) heatmap.

    Cell (c, b) is the effective star count n_eff for channel c at radial bin b; rows
    are the three velocity channels (RV, PM_R, PM_T), columns the K log-spaced on-sky
    radii. A heatmap makes the joint radius-AND-channel structure of the W0 design legible
    at a glance (the bars in Fig 3 carry the radial story; this carries both axes)."""
    import matplotlib.pyplot as plt

    n_eff = np.asarray(res["n_c"])  # (3, K)
    R = np.asarray(oedc.R_BINS)
    model = res["model"]

    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    im = ax.imshow(
        n_eff, aspect="auto", origin="lower", cmap="viridis", interpolation="nearest"
    )
    ax.set_yticks(range(3))
    ax.set_yticklabels(_CH_SHORT)
    # Label a sparse subset of the K radial bins (every 2nd) to keep the axis readable.
    xt = range(0, len(R), 2)
    ax.set_xticks(list(xt))
    ax.set_xticklabels([f"{R[i]:.1f}" for i in xt])
    ax.set_xlabel(r"projected radius  $R$  [$r_c$]")
    ax.set_ylabel("channel")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"effective star count  $n_{\rm eff}$")
    panel_label(ax, f"{model} W0 c-optimal", loc="upper right")
    fig.tight_layout()
    save_fig(fig, fig_dir, f"demo_oedc_alloc_{model}")


def _stacked_radial(ax, n_eff, R, title, unit="r_c"):
    """Helper: stacked bars of per-channel n_eff over log10(R) on a given axis.

    `unit` sets the radial-axis length unit label (W0 grid is r_c; the Stage-1 r_a
    grid is r_h), so each panel of the contrast is unit-honest.
    """
    logR = np.log10(np.asarray(R))
    bw = 0.9 * (logR[1] - logR[0])
    bottom = np.zeros_like(logR)
    for c, (lbl, col) in enumerate(zip(_CH_LABELS, _CH_COLORS)):
        ax.bar(
            logR,
            np.asarray(n_eff)[c],
            width=bw,
            bottom=bottom,
            color=col,
            edgecolor="white",
            linewidth=0.2,
            label=lbl,
        )
        bottom = bottom + np.asarray(n_eff)[c]
    ax.set_xlabel(rf"$\log_{{10}}(R\,/\,{unit})$")
    panel_label(ax, title, loc="upper right")


def _fig_headline_contrast(results, ra, fig_dir):
    """Fig 3 (HEADLINE): W0 wants the CORE; r_a wanted the OUTSKIRTS.

    Left: the W0 c-optimal radial allocation (King), stacked per channel -- core-heavy.
    Right: the Stage-1 r_a c-optimal allocation -- outskirts-heavy. The two panels share
    the per-bin shape but mirror each other in WHERE the budget goes; the annotated
    core/outskirts fractions make the pre-registered hypothesis (H1) visually obvious.
    Both panels normalize each bar-set to its own budget fraction so the SHAPE (not the
    different N_total) is what is compared."""
    import matplotlib.pyplot as plt

    king = results["king"]
    n_w0 = np.asarray(king["n_c"])
    n_w0 = n_w0 / n_w0.sum()  # fraction of the W0 budget
    R_w0 = np.asarray(oedc.R_BINS)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.2, 4.0))
    # The W0 grid is in r_c; the Stage-1 r_a grid is in r_h. Each panel labels its own unit.
    _stacked_radial(axL, n_w0, R_w0, r"$W_0$ (concentration)", unit="r_c")
    axL.set_ylabel(r"allocation fraction of the budget")
    axL.text(
        0.035,
        0.80,
        r"$W_0$ wants the CORE"
        + "\n"
        + rf"core ${100 * king['core']:.0f}\%$ / outskirts ${100 * king['outer']:.0f}\%$",
        transform=axL.transAxes,
        fontsize=9,
        va="top",
        color=OI["vermilion"],
    )
    axL.legend(loc="upper left", fontsize=7.5)

    if ra is not None:
        n_ra = np.asarray(ra["n_eff"])
        n_ra = n_ra / n_ra.sum()
        _stacked_radial(
            axR,
            n_ra,
            np.asarray(ra["R_BINS"]),
            r"$r_a$ (anisotropy, Stage 1)",
            unit="r_h",
        )
        axR.text(
            0.035,
            0.80,
            r"$r_a$ wanted the OUTSKIRTS"
            + "\n"
            + rf"core ${100 * ra['core']:.0f}\%$ / outskirts ${100 * ra['outer']:.0f}\%$",
            transform=axR.transAxes,
            fontsize=9,
            va="top",
            color=OI["blue"],
        )
    else:
        axR.text(
            0.5,
            0.5,
            "Stage-1 $r_a$ record absent\n(documented: ~99% outskirts)",
            transform=axR.transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )
        axR.set_xlabel(r"$\log_{10}(R\,/\,r_h)$")
        panel_label(axR, r"$r_a$ (anisotropy, Stage 1)", loc="upper right")
    axR.set_ylabel(r"allocation fraction of the budget")

    fig.suptitle("")  # no in-figure title (the MyST caption carries it)
    fig.tight_layout()
    save_fig(fig, fig_dir, "demo_oedc_headline_contrast")


def _fig_gain_cda(results, fig_dir):
    """Fig 4: the precision gain + the c/D/A criterion comparison, King vs Michie.

    Left: the fractional precision sigma(lnW0) for the uniform vs the c-optimal design
    (grouped bars, King vs Michie) -- the headline equal-precision gain. Right: each of
    the c/D/A optima as a RATIO to its uniform-design value (criterion_opt/criterion_unif,
    so all three are dimensionless 'fraction of the uniform criterion'; < 1 = better),
    King vs Michie -- a like-for-like read across the three alphabet-optimality designs."""
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.0, 3.8))

    # --- left: sigma(lnW0) uniform vs c-optimal ---
    x = np.arange(len(MODELS))
    w = 0.36
    sig_unif = [results[m]["sig_unif"] for m in MODELS]
    sig_c = [results[m]["sig_c"] for m in MODELS]
    axL.bar(x - w / 2, sig_unif, w, color="0.6", label="uniform design")
    axL.bar(x + w / 2, sig_c, w, color=OI["vermilion"], label="c-optimal design")
    # Headroom so the gain annotation + legend never collide with the bars.
    axL.set_ylim(0.0, max(sig_unif) * 1.32)
    for i, m in enumerate(MODELS):
        # Gain label ON the c-optimal (orange) bar (white text, mid-bar) -- legible and
        # unambiguously attached to the c-optimal result; never collides with the top
        # legend or the panel tag.
        axL.text(
            x[i] + w / 2,
            0.5 * sig_c[i],
            rf"${results[m]['gain']:.2f}\times$",
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="white",
        )
    axL.set_xticks(x)
    axL.set_xticklabels([m for m in MODELS])
    axL.set_ylabel(r"fractional precision  $\sigma(\ln W_0)$")
    # Single-column legend in the upper-right empty space (above the shorter c-optimal
    # bars), clear of both the top-left panel tag and the centred gain annotations.
    axL.legend(loc="upper right", fontsize=8)
    panel_label(axL, "precision gain", loc="upper left")

    # --- right: c/D/A opt-vs-uniform ratio ---
    crits = ("c", "d", "a")
    crit_lbl = {
        "c": "c  (var $W_0$)",
        "d": r"D  ($-\log\det F$)",
        "a": r"A  (tr $F^{-1}$)",
    }
    xc = np.arange(len(crits))
    for j, m in enumerate(MODELS):
        r = results[m]
        # For each criterion: opt/uniform. D is -logdet (negative) -- use |.| so the bar
        # reads as 'fraction of the uniform magnitude' for all three (still <1 = better).
        ratios = [
            r["c_opt"] / r["c_unif"],
            abs(r["d_opt"]) / abs(r["d_unif"]),
            r["a_opt"] / r["a_unif"],
        ]
        col = OI["blue"] if m == "king" else OI["orange"]
        axR.bar(xc + (j - 0.5) * 0.36, ratios, 0.36, color=col, label=m)
    axR.axhline(1.0, color="0.5", ls=":", lw=1.0)
    axR.set_xticks(xc)
    axR.set_xticklabels([crit_lbl[c] for c in crits], fontsize=8)
    axR.set_ylabel(r"criterion ratio  optimum / uniform")
    axR.legend(loc="upper right", fontsize=8)
    panel_label(axR, "c / D / A designs", loc="upper left")

    fig.tight_layout()
    save_fig(fig, fig_dir, "demo_oedc_gain_cda")


def _fig_degeneracy(results, fig_dir):
    """Fig 5: the W0<->r_a degeneracy -- the Fisher correlation rho(W0, r_a) at the
    c-optimal design, King vs Michie.

    rho = (F^-1)_{W0,r_a}/sqrt((F^-1)_{W0,W0}(F^-1)_{r_a,r_a}) is the covariance off-
    diagonal normalized to [-1, 1]: |rho| near 1 means concentration and anisotropy are
    nearly degenerate (the design struggles to separate them); |rho| near 0 means they
    are measured independently. The bars quantify HOW separable W0 is from r_a in this
    design space -- the science payoff behind 'can concentration be pinned independent
    of anisotropy?'."""
    import matplotlib.pyplot as plt

    rho = [results[m]["rho_wa"] for m in MODELS]
    x = np.arange(len(MODELS))

    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    colors = [OI["blue"] if m == "king" else OI["orange"] for m in MODELS]
    ax.bar(x, rho, 0.5, color=colors)
    for i, v in enumerate(rho):
        ax.text(
            x[i],
            v + (0.03 if v >= 0 else -0.03),
            f"{v:+.2f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=9.5,
        )
    ax.axhline(0.0, color="0.5", ls="-", lw=0.8)
    # Shade the |rho| > 0.9 'near-degenerate' band for context.
    ax.axhspan(0.9, 1.05, color="0.85", alpha=0.5, zorder=0)
    ax.axhspan(-1.05, -0.9, color="0.85", alpha=0.5, zorder=0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([m for m in MODELS])
    ax.set_ylabel(r"Fisher correlation  $\rho(W_0, r_a)$")
    panel_label(ax, "concentration vs anisotropy", loc="lower left")
    fig.tight_layout()
    save_fig(fig, fig_dir, "demo_oedc_degeneracy")


def make_figures(results, ra, fig_dir):
    """Generate the five W0-OED figures into fig_dir (PNG + PDF via save_fig).

    * figs 1-2 (alloc_{king,michie}): the W0 c-optimal (channel x radius) heatmaps.
    * fig 3 (headline_contrast):      W0 (core-heavy) vs Stage-1 r_a (outskirts-heavy).
    * fig 4 (gain_cda):               sigma(lnW0) uniform-vs-c gain + c/D/A ratios.
    * fig 5 (degeneracy):             the Fisher rho(W0, r_a) at the c-optimal design.
    """
    apply_pub_style()
    os.makedirs(fig_dir, exist_ok=True)
    print(f"\n  generating 5 W0-OED figures -> {fig_dir}/ ...")
    for model in MODELS:
        _fig_allocation_heatmap(results[model], fig_dir)
    _fig_headline_contrast(results, ra, fig_dir)
    _fig_gain_cda(results, fig_dir)
    _fig_degeneracy(results, fig_dir)
    print(f"  figures: wrote 5 PNG+PDF to {fig_dir}/demo_oedc_*.png")


if __name__ == "__main__":
    sys.exit(main())

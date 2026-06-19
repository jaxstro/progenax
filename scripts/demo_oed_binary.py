#!/usr/bin/env python
r"""Binary-misspecification OED (H1) -- the false-confidence disaster, made visual.

A binary-UNAWARE analyst designs a young-massive-cluster (YMC) RV survey to best
constrain the dynamical mass ``M``: a c-optimal-for-M radial allocation under the
BINARY-FREE EFF-OM forward model (``sigma_obs = cluster_sigma_los``, the
Binney & Mamon 1982 projection of the Elson-Fall-Freeman density + Osipkov-Merritt
anisotropy; ``progenax.project_dispersion``, RV channel only). The cluster ACTUALLY
has unresolved binaries, so the observed second moment carries a flat pedestal
``sigma_obs^2 = sigma_cluster^2(M, r_a, gamma, a) + f_bin * V_bin + eps_RV^2``. Fitting
the binary-free model to that contaminated data biases ``M_hat`` HIGH -- and, because
the design concentrated the budget in the cold contaminated OUTSKIRTS, the forecast
error bar is tiny: a false-confidence disaster. See
``docs/plans/2026-06-19-oed-binary-misspecification-{design,plan}.md`` (Task 1.6).

The headline (H1, pre-registration LOCKED 2026-06-19): the naive binary-free
c-optimal-for-M design, fit binary-free on binary-contaminated mocks, biases ``M_hat``
by MORE than 2x its OWN forecast ``sigma(M)/M`` (cross-model MC, env-gated @slow). The
``f_bin = 0`` baseline -- the SAME forward model the fit uses -- recovers ``M`` unbiased
(to ~0.5%, a small-sample realized-scatter-weighted-fit effect, far below the forecast).

What this CLI computes and prints (exit 0):
  * the NAIVE binary-free c-optimal-for-M radial design (multi-start Adam over the
    per-bin RV allocation) and its OWN binary-free Fisher forecast ``sigma(M)/M``;
  * the per-bin cluster ``sigma_los`` (truth) and the binary-inflated observable
    ``sqrt(sigma_cluster^2 + f_bin*V_bin)`` -- the mechanism;
  * in FULL mode (with the cross-model MC enabled): the realized cross-model bias
    ``M_hat/M`` of the naive design WITH binaries and the ``f_bin=0`` baseline, the
    headline ``ratio = bias / forecast``, and the pre-registered H1 accept verdict.

Figures (publication-quality, into ``--outdir``):
  1. demo_oedb_false_confidence : THE HEADLINE -- ``M_hat/M`` for the naive design WITH
     binaries (~2.85, with its TINY +/-forecast-sigma error bar), the unbiased truth
     line (1.0), and the ``f_bin=0`` baseline (~1.0). The claimed error bar is dwarfed
     by the bias. (Needs the MC; produced only in full MC mode.)
  2. demo_oedb_mechanism : WHY -- the truth cluster ``sigma_los(R)`` (falls steeply
     core->outskirts), the binary-inflated observable (a ~flat pedestal lifting the cold
     outskirts enormously), and the naive design's per-bin allocation (bars, twin axis)
     showing it concentrates the budget EXACTLY in the contaminated outer bins. (No MC.)

The cross-model calibration MC is the env-gated @slow gate (``test_H1_*`` in
``tests/unit/test_demo_oed_binary.py``, ``PROGENAX_RUN_OED_BINARY=1``) -- OUT of CI. This
CLI runs it ONLY when ``--run-mc`` is passed or ``PROGENAX_RUN_OED_BINARY`` is set; the
default (and ``--quick``) path is the cheap design+forecast+mechanism-figure, exit 0.

Usage::

    # Cheap (CI-safe): design + forecast + the no-MC mechanism figure.
    env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_oed_binary.py
    # Full (env-gated heavy MC): + the cross-model bias + the headline figure.
    PROGENAX_RUN_OED_BINARY=1 env -u VIRTUAL_ENV uv run --no-sync \
        python scripts/demo_oed_binary.py --run-mc --outdir /tmp/oedb
    # Fast smoke (no MC, dialed-down optimizer, exit 0): the mechanism figure only.
    env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_oed_binary.py --quick
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _demo_oed_binary as oedb  # noqa: E402  -- the binary-misspec OED core (Task 0-1.5 API)

G = oedb.STELLAR.G

# Figures live in the optimal-design SECTION dir (alongside the Stage-1/Stage-2 + W0-OED
# figures and run-records) so a plain re-run regenerates them in place next to the MyST page.
FIGURE_DIR = "docs/website/60-science-demos/optimal-design/figures"
RUN_RECORD = os.path.join(FIGURE_DIR, "demo_oed_binary_run_record.json")

# Multi-start Adam settings (over the K-bin softmax design vector z). Quick-dials for smoke.
N_STARTS_DEFAULT = 8
N_STEPS_DEFAULT = 500
N_STARTS_QUICK = 2
N_STEPS_QUICK = 60


def _mc_enabled(args):
    """Whether the env-gated cross-model calibration MC runs in this invocation.

    The MC is @slow/OOM-bounded and OUT of CI: it runs ONLY when ``--run-mc`` is passed
    or ``PROGENAX_RUN_OED_BINARY`` is set in the environment (the SAME gate the @slow
    test uses), and NEVER under ``--quick`` (the fast smoke path must not need the MC).
    """
    if args.quick:
        return False
    return bool(args.run_mc or os.environ.get("PROGENAX_RUN_OED_BINARY"))


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Binary-misspecification OED (H1): the naive binary-free "
        "c-optimal-for-M design biases M_hat HIGH with a tiny forecast error bar "
        "(false confidence) when the cluster has binaries."
    )
    p.add_argument("--outdir", type=str, default=FIGURE_DIR,
                   help=f"Directory for the figures + run-record (default {FIGURE_DIR}).")
    p.add_argument("--n-starts", type=int, default=N_STARTS_DEFAULT,
                   help=f"Multi-start Adam restarts for the c-optimal-for-M design "
                        f"(default {N_STARTS_DEFAULT}).")
    p.add_argument("--n-steps", type=int, default=N_STEPS_DEFAULT,
                   help=f"Adam steps per start (default {N_STEPS_DEFAULT}).")
    p.add_argument("--n-total", type=float, default=oedb.N_TOTAL,
                   help=f"Total RV-measurement budget across the K radial bins "
                        f"(default {oedb.N_TOTAL:.0f}).")
    p.add_argument("--n-draws", type=int, default=oedb.N_DRAWS_H1,
                   help=f"Cross-model MC draws (full mode only; default {oedb.N_DRAWS_H1}).")
    p.add_argument("--seed", type=int, default=0, help="PRNG seed (default 0).")
    p.add_argument("--run-mc", action="store_true",
                   help="Run the env-gated cross-model calibration MC (the H1 bias + the "
                        "headline figure). Equivalent to setting PROGENAX_RUN_OED_BINARY. "
                        "Slow (~80 s, ~2.3 GB peak); OUT of CI.")
    p.add_argument("--quick", action="store_true",
                   help="Smoke/CI fast path: dial the optimizer down, NO MC, mechanism "
                        "figure only. Exits 0 quickly without the MC env var.")
    p.add_argument("--no-figures", action="store_true",
                   help="Skip figure generation; print the summary + write the run-record only.")
    args = p.parse_args(argv)

    n_starts = N_STARTS_QUICK if args.quick else args.n_starts
    n_steps = N_STEPS_QUICK if args.quick else args.n_steps
    n_total = float(args.n_total)
    run_mc = _mc_enabled(args)
    key = jax.random.PRNGKey(args.seed)
    k_design, k_mc = jax.random.split(key)

    print("=" * 80)
    print("BINARY-MISSPECIFICATION OED (H1): the false-confidence disaster")
    print("=" * 80)
    print(f"  YMC truth: M={oedb.M_FID:.1e} Msun, gamma={oedb.GAMMA_FID}, a={oedb.A_FID} pc, "
          f"r_t={oedb.R_T_FID} pc, r_a={oedb.R_A_FID} pc, f_bin={oedb.F_BIN_TRUTH}")
    sig_bin = float(jnp.sqrt(oedb.V_BIN))
    sig_clu_central = float(oedb.sigma_cluster_ref())
    print(f"  scales: sigma_bin={sig_bin:.2f} km/s, sigma_cluster,central={sig_clu_central:.2f} "
          f"km/s (ratio {sig_bin / sig_clu_central:.2f}), eps_RV={oedb.EPS_RV_KMS} km/s")
    print(f"  budget: K={oedb.R_BINS.shape[0]} bins, N_total={n_total:.0f}  |  "
          f"optimizer: {n_starts} starts x {n_steps} steps{'  [--quick]' if args.quick else ''}")

    # --- the NAIVE binary-free c-optimal-for-M design + its own forecast sigma(M)/M --- #
    print("\n  computing the naive binary-free c-optimal-for-M design "
          "(multi-start Adam, cached jacrev) ...")
    design = oedb.optimize_design_M(n_total, key=k_design, n_starts=n_starts, n_steps=n_steps)
    forecast_sigma_M_frac = float(design.sigma_M_over_M)
    n_eff = design.n_eff

    # --- the mechanism: per-bin cluster sigma_los (truth) + the binary-inflated observable --- #
    sig_cluster = oedb.cluster_sigma_los(oedb.theta_truth_clusteronly(), oedb.R_BINS, G)  # (K,)
    sig_obs = oedb.predict_sigma_obs(oedb.theta_truth(), oedb.R_BINS, G)                   # (K,)

    print("\n" + "-" * 80)
    print("  NAIVE c-OPTIMAL-FOR-M DESIGN (per radial bin)")
    print(f"  {'R [pc]':>9s}{'sigma_clu':>11s}{'sigma_obs':>11s}{'n_eff':>10s}{'frac':>9s}")
    print("-" * 80)
    total = float(jnp.sum(n_eff))
    for b in range(oedb.R_BINS.shape[0]):
        ne = float(n_eff[b])
        marker = "  <--" if ne >= oedb.N_MIN_FIT else ""
        print(f"  {float(oedb.R_BINS[b]):>9.2f}{float(sig_cluster[b]):>11.3f}"
              f"{float(sig_obs[b]):>11.3f}{ne:>10.1f}{ne / total:>9.3f}{marker}")
    print("-" * 80)
    kept = [b for b in range(oedb.R_BINS.shape[0]) if float(n_eff[b]) >= oedb.N_MIN_FIT]
    kept_R = [round(float(oedb.R_BINS[b]), 1) for b in kept]
    print(f"  forecast sigma(M)/M (binary-free Fisher, c-optimal) = {forecast_sigma_M_frac:.4f}  "
          f"({100 * forecast_sigma_M_frac:.1f}%)")
    print(f"  the design POPULATES {len(kept)} bins (n_eff >= {oedb.N_MIN_FIT}): {kept_R} pc "
          f"-- it concentrates the budget in the COLD outskirts where binaries dominate.")
    print("-" * 80)

    # --- the cross-model bias (full mode only -- the env-gated @slow MC) --------------- #
    h1 = None
    base = None
    if run_mc:
        print(f"\n  running the cross-model calibration MC (n_draws={args.n_draws}) ... "
              f"[slow; ~80 s, ~2.3 GB peak]")
        h1 = oedb.run_H1(n_draws=args.n_draws, key=key, N_total=n_total)
        base = oedb.cross_model_bias(n_eff, n_draws=args.n_draws, key=k_mc, f_bin_truth=0.0)
        print("\n" + "-" * 80)
        print("  CROSS-MODEL BIAS (generate WITH Moe binaries, fit the BINARY-FREE model)")
        print("-" * 80)
        print(f"  naive design + binaries (f_bin={oedb.F_BIN_TRUTH}):")
        print(f"     M_hat/M_true          = {1.0 + h1.bias_M_frac:.3f}  "
              f"(bias {h1.bias_M_frac:+.3f} = {100 * h1.bias_M_frac:+.0f}%)")
        print(f"     forecast sigma(M)/M   = {h1.forecast_sigma_M_frac:.4f}  (the claimed error bar)")
        print(f"     bias / forecast RATIO = {h1.ratio:.1f}x   <-- the false-confidence headline")
        print(f"     SEM(bias)             = {h1.sem:.4f}  (2*SEM = {2 * h1.sem:.4f} << bias)")
        print(f"     M-step unconverged    = {h1.n_unconverged}/{args.n_draws} draws")
        print(f"  f_bin=0 baseline (mock == fit model):")
        print(f"     M_hat/M_true          = {1.0 + base.bias_M_frac:.3f}  "
              f"(bias {base.bias_M_frac:+.4f}; unbiased to ~0.5%, << forecast)")
        print(f"  PRE-REGISTERED H1: ACCEPT iff bias > 2*forecast AND bias > 0  ->  "
              f"{'ACCEPT' if h1.accept else 'REJECT'}")
        print("-" * 80)
    else:
        print("\n  [cross-model MC NOT run: pass --run-mc or set PROGENAX_RUN_OED_BINARY for "
              "the\n   H1 bias + the headline false-confidence figure. The mechanism figure "
              "(no MC)\n   and the design/forecast above are always produced.]")

    # --- run-record JSON (into --outdir, NOT the fixed FIGURE_DIR; Stage-3 CLI lesson) -- #
    # The smoke test passes --outdir=tmp_path; a FIXED path would clobber the committed
    # full-quality record with low-res smoke numbers. The default --outdir IS FIGURE_DIR, so
    # a real run still lands the record next to the committed figures.
    os.makedirs(args.outdir, exist_ok=True)
    record = {
        "demo": "demo_oed_binary (binary-misspecification OED, H1)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "params": {
            "seed": args.seed, "n_total": n_total, "n_starts": n_starts, "n_steps": n_steps,
            "quick": bool(args.quick), "mc_run": bool(run_mc), "n_draws": int(args.n_draws),
            "K_bins": int(oedb.R_BINS.shape[0]),
            "M_fid": oedb.M_FID, "gamma_fid": oedb.GAMMA_FID, "a_fid": oedb.A_FID,
            "r_t_fid": oedb.R_T_FID, "r_a_fid": oedb.R_A_FID, "f_bin_truth": oedb.F_BIN_TRUTH,
            "eps_RV_kms": oedb.EPS_RV_KMS, "V_bin_kms2": float(oedb.V_BIN),
            "sigma_bin_kms": sig_bin, "sigma_cluster_central_kms": sig_clu_central,
        },
        "design": {
            "R_bins_pc": [float(x) for x in oedb.R_BINS],
            "n_eff": [float(x) for x in n_eff],
            "kept_bins_pc": kept_R,
            "forecast_sigma_M_over_M": forecast_sigma_M_frac,
        },
        "mechanism": {
            "sigma_cluster_los_kms": [float(x) for x in sig_cluster],
            "sigma_obs_kms": [float(x) for x in sig_obs],
        },
    }
    if h1 is not None:
        record["H1"] = {
            "bias_M_frac": h1.bias_M_frac, "mhat_over_mtrue": 1.0 + h1.bias_M_frac,
            "forecast_sigma_M_frac": h1.forecast_sigma_M_frac, "ratio": h1.ratio,
            "sem": h1.sem, "std_M_frac": h1.std_M_frac, "accept": bool(h1.accept),
            "n_unconverged": int(h1.n_unconverged),
            "bias_other_ra_gamma_a": [float(x) for x in h1.bias_other],
            "baseline_fbin0_bias_M_frac": base.bias_M_frac,
        }
    run_record_path = os.path.join(args.outdir, os.path.basename(RUN_RECORD))
    with open(run_record_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  run-record -> {run_record_path}")

    # --- figures ----------------------------------------------------------------------- #
    if args.no_figures:
        print("  figures: SKIPPED (--no-figures)")
    else:
        make_figures(record, h1, args.outdir)

    print("=" * 80)
    print("  BINARY-MISSPECIFICATION OED DEMO: DONE")
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


def _fig_false_confidence(record, h1, fig_dir):
    """Fig 1 (HEADLINE): M_hat/M_true for the naive design -- the false-confidence disaster.

    Three points on a single M_hat/M axis: (i) the naive binary-free design fit on
    binary-CONTAMINATED data (~2.85x truth), drawn WITH its OWN forecast-sigma error bar
    (+/- forecast_sigma_M_frac -- visually TINY); (ii) the unbiased truth line at 1.0;
    (iii) the f_bin=0 baseline (mock == fit model -> ~1.0). The whole point is the visual
    contrast: the claimed (forecast) error bar is DWARFED by the realized bias. The bias
    (+185%) and the bias/forecast ratio (41x) are annotated. Needs the MC (h1 not None).
    """
    import matplotlib.pyplot as plt

    H = record["H1"]
    mhat = H["mhat_over_mtrue"]                       # ~2.85
    forecast = H["forecast_sigma_M_frac"]            # ~0.045 (FRACTIONAL -> same unit as M_hat/M)
    base = 1.0 + H["baseline_fbin0_bias_M_frac"]     # ~1.0 (f_bin=0 baseline)
    ratio = H["ratio"]
    bias_pct = 100.0 * H["bias_M_frac"]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))

    # (ii) the unbiased truth.
    ax.axhline(1.0, color="0.35", ls="--", lw=1.2, zorder=1)
    ax.text(0.5, 1.0, "  unbiased truth $\\hat M / M = 1$", color="0.35", fontsize=8.5,
            va="bottom", ha="left", transform=ax.get_yaxis_transform())

    # (i) the naive design WITH binaries: the +/- forecast-sigma error bar is the claimed
    #     precision (forecast is fractional sigma(M)/M, so the absolute bar on M_hat/M is
    #     mhat*forecast). It is TINY next to the bias -- that is the disaster.
    err = mhat * forecast
    ax.errorbar([0], [mhat], yerr=[err], fmt="o", ms=11, color=OI["vermilion"],
                ecolor=OI["vermilion"], elinewidth=2.4, capsize=7, capthick=2.4,
                zorder=4, label="naive design + binaries")

    # (iii) the f_bin=0 baseline (mock == fit model -> unbiased).
    ax.errorbar([1], [base], yerr=[base * forecast], fmt="s", ms=9, color=OI["blue"],
                ecolor=OI["blue"], elinewidth=2.0, capsize=6, capthick=2.0,
                zorder=4, label="$f_{\\rm bin}=0$ baseline")

    # The bias annotation: a vertical span from truth to M_hat, labelled.
    ax.annotate("", xy=(0.22, mhat), xytext=(0.22, 1.0),
                arrowprops=dict(arrowstyle="<->", color="0.25", lw=1.4))
    ax.text(0.27, 0.5 * (1.0 + mhat),
            f"bias $= {bias_pct:+.0f}\\%$\n$\\,$($\\hat M/M = {mhat:.2f}$)",
            color="0.15", fontsize=10, va="center", ha="left")

    # The headline ratio, in a callout box.
    ax.text(0.5, 0.93,
            f"forecast error bar $\\pm{100 * forecast:.1f}\\%$\n"
            f"is dwarfed: bias is $\\mathbf{{{ratio:.0f}\\times}}$ the forecast $\\sigma(M)$",
            transform=ax.transAxes, fontsize=10.5, va="top", ha="center",
            bbox=dict(boxstyle="round,pad=0.4", fc=OI["yellow"], ec="0.4", alpha=0.85))

    ax.set_xlim(-0.5, 1.6)
    ax.set_ylim(0.0, mhat * 1.22)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["naive design\n+ binaries", "$f_{\\rm bin}=0$\nbaseline"])
    ax.set_ylabel(r"recovered dynamical mass  $\hat M / M_{\rm true}$")
    ax.legend(loc="center right", fontsize=8.5)
    panel_label(ax, "false confidence", loc="upper left")
    fig.tight_layout()
    save_fig(fig, fig_dir, "demo_oedb_false_confidence")


def _fig_mechanism(record, fig_dir):
    """Fig 2 (MECHANISM): WHY the bias happens -- sigma_los(R) + the binary pedestal + the
    design's allocation.

    Left axis: the truth cluster sigma_los(R) (steeply DECLINING core->outskirts, where the
    M information lives) and the binary-inflated observable sqrt(sigma_cluster^2 +
    f_bin*V_bin) (a ~FLAT pedestal that lifts the cold outskirts enormously). Right (twin)
    axis: the naive c-optimal-for-M design's per-bin allocation (bars) -- it concentrates the
    budget in the OUTER bins, EXACTLY where the binary pedestal dominates the signal. That is
    the mechanism: the design spends its stars where binaries, not the cluster, set the
    second moment, so the binary-free fit reads the pedestal as extra mass. No MC needed.
    """
    import matplotlib.pyplot as plt

    R = np.asarray(record["design"]["R_bins_pc"])
    sig_clu = np.asarray(record["mechanism"]["sigma_cluster_los_kms"])
    sig_obs = np.asarray(record["mechanism"]["sigma_obs_kms"])
    n_eff = np.asarray(record["design"]["n_eff"])
    sig_bin = record["params"]["sigma_bin_kms"]

    fig, axL = plt.subplots(figsize=(7.2, 4.6))

    # --- right (twin) axis FIRST so the bars sit BEHIND the sigma curves --- #
    axR = axL.twinx()
    logR = np.log10(R)
    bw = 0.9 * (logR[1] - logR[0])
    axR.bar(logR, n_eff, width=bw, color=OI["sky"], alpha=0.55, edgecolor="white",
            linewidth=0.3, zorder=1, label="design allocation")
    axR.set_ylabel(r"design allocation  $n_{\rm eff}$  per bin", color=OI["blue"])
    axR.tick_params(axis="y", labelcolor=OI["blue"])
    axR.set_ylim(0, n_eff.max() * 1.25)

    # --- left axis: the two sigma profiles (the mechanism) --- #
    axL.set_zorder(axR.get_zorder() + 1)   # sigma curves drawn ON TOP of the bars
    axL.patch.set_visible(False)           # but let the bars show through
    axL.plot(logR, sig_clu, "-o", ms=5, color=OI["vermilion"], zorder=3,
             label=r"truth cluster $\sigma_{\rm los}$")
    axL.plot(logR, sig_obs, "-s", ms=5, color=OI["green"], zorder=3,
             label=r"binary-inflated $\sigma_{\rm obs}=\sqrt{\sigma_{\rm clu}^2+f_{\rm bin}V_{\rm bin}}$")
    # The binary pedestal floor sqrt(f_bin*V_bin) as a horizontal reference.
    pedestal = float(np.sqrt(record["params"]["f_bin_truth"] * record["params"]["V_bin_kms2"]))
    axL.axhline(pedestal, color=OI["green"], ls=":", lw=1.1, zorder=2)
    axL.text(logR[0], pedestal, f"  binary pedestal $\\sqrt{{f_{{\\rm bin}}V_{{\\rm bin}}}}$ "
             f"$= {pedestal:.1f}$ km/s", color=OI["green"], fontsize=8, va="bottom", ha="left")

    axL.set_xlabel(r"$\log_{10}(R\,/\,{\rm pc})$")
    axL.set_ylabel(r"line-of-sight dispersion  $\sigma$  [km/s]")
    axL.set_ylim(0, max(sig_obs.max(), sig_clu.max()) * 1.18)

    # Annotate the core-vs-outskirt contrast that the design IGNORES.
    axL.annotate(f"cluster $\\sigma_{{\\rm los}}$ falls\n"
                 f"{sig_clu.max():.1f} $\\to$ {sig_clu.min():.2f} km/s",
                 xy=(logR[-1], sig_clu[-1]), xytext=(logR[len(R) // 2], 0.55 * sig_obs.max()),
                 fontsize=8.5, color=OI["vermilion"], ha="center",
                 arrowprops=dict(arrowstyle="->", color=OI["vermilion"], lw=1.0))

    # Merge the two axes' legends into one.
    hL, lL = axL.get_legend_handles_labels()
    hR, lR = axR.get_legend_handles_labels()
    axL.legend(hL + hR, lL + lR, loc="upper right", fontsize=8)
    panel_label(axL, "mechanism", loc="upper left")
    fig.tight_layout()
    save_fig(fig, fig_dir, "demo_oedb_mechanism")


def make_figures(record, h1, fig_dir):
    """Generate the binary-misspecification figures into fig_dir (PNG + PDF via save_fig).

      * fig 1 (false_confidence): M_hat/M for the naive design + binaries (~2.85, tiny
        forecast error bar) vs truth vs the f_bin=0 baseline. ONLY in full MC mode (h1 set).
      * fig 2 (mechanism): sigma_los(R) + the binary pedestal + the design's allocation --
        WHY the bias happens. Always produced (no MC).
    """
    apply_pub_style()
    os.makedirs(fig_dir, exist_ok=True)
    print(f"\n  generating figures -> {fig_dir}/ ...")
    _fig_mechanism(record, fig_dir)
    if h1 is not None:
        _fig_false_confidence(record, h1, fig_dir)
        print(f"  figures: wrote 2 PNG+PDF (mechanism + false_confidence) to "
              f"{fig_dir}/demo_oedb_*.png")
    else:
        print(f"  figures: wrote 1 PNG+PDF (mechanism; no MC -> no false-confidence figure) "
              f"to {fig_dir}/demo_oedb_mechanism.png")


if __name__ == "__main__":
    sys.exit(main())

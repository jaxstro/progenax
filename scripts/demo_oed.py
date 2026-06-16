#!/usr/bin/env python
r"""B14 -- Pre-data c-optimal Bayesian experimental design for the anisotropy radius.

A pure pre-DATA optimal-experimental-design (OED) demo. We have a fixed budget of
``N_total`` stars to allocate across (projected radius x {RV, PM_R, PM_T}) and we
want to MINIMIZE the marginal (fractional) variance of the Osipkov-Merritt
anisotropy radius ``r_a`` of a mock GC-scale cluster -- a c-optimal design.

The load-bearing idea (ADR 0004): the design Fisher is ADDITIVE and LINEAR in the
design, ``F(design) = sum_{bin b, channel c} n_eff,{b,c} * M_{b,c}``, where each
per-star block ``M_{b,c} = 2 J J^T / (sigma^2 + eps_c^2)`` is design-INDEPENDENT
and computed ONCE via a single reverse-mode ``jacrev`` through the packaged
``project_dispersion`` (Binney & Mamon 1982 projection of the OM-Plummer Jeans
model). The optimization is then pure 3x3 linear algebra. The Fisher is built wrt
``ln theta`` (ADR 0011), so it is dimensionless and every covariance entry is a
FRACTIONAL variance; the c-headline is the fractional precision ``sigma(r_a)/r_a``.

What the demo computes and gates:
  * the per-star blocks once at the truth ``theta = (r_a, M, r_h)``;
  * the c-, D-, and A-optimal designs (multi-start Adam);
  * the HEADLINE equal-precision factor ``c_uniform / c_designed`` at FIXED N
    (the c-design's fractional variance on r_a is this factor smaller than the
    uniform design's, AT THE SAME N -- the prior cancels exactly in the ratio);
  * the INTERPRETABILITY payoff: the per-bin proper-motion fraction
    ``(n_pm_r + n_pm_t) / sum_channels`` rises from the core to the outskirts --
    the design DISCOVERS that PM stars belong in the outskirts (where the OM
    anisotropy beta(r) = r^2/(r^2+r_a^2) is largest);
  * a small calibration ensemble confirming the design Fisher predicts the
    realized scatter of r_a_hat (use ``--full`` for the publication-grade run).

Gates (exit 0 = all pass):
  * headline factor c_uniform/c_designed > 1.3  (actual ~3.65);
  * PM-fraction outer-half mean > inner-half mean (PMs favored outward);
  * calibration realized-vs-Fisher fractional sigma(r_a) agrees within tolerance.

Caveats printed below: the factor is exact only AT FIXED N (the fixed nuisance
prior dilutes as N grows, so an "equivalent uniform star count" gloss is only
approximate); the calibration Fisher is mildly conservative (realized scatter
sits slightly below the Fisher prediction because the binned-dispersion estimator
loses a little information vs the idealized per-star Fisher).

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_oed.py
    env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_oed.py --full   # 64-draw calibration
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import jax
import jax.numpy as jnp

from jaxstro.units import STELLAR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _demo_oed as oed  # noqa: E402

FIGURE_DIR = "docs/website/60-science-demos/figures"
RUN_RECORD = os.path.join(FIGURE_DIR, "demo_oed_run_record.json")
G = STELLAR.G

# Optimizer settings (multi-start Adam over the design simplex).
N_STARTS = 6
N_STEPS = 400
# Calibration draws: small by default (calibration is ~3.75 s/draw); --full -> 64.
N_DRAWS_QUICK = 12
N_DRAWS_FULL = 64


def _designs(Mb, cb, n_total, key):
    """Optimize the c-, D-, and A-optimal designs (multi-start). Returns a dict
    keyed by criterion name -> DesignResult."""
    crit_fns = {"c": oed.c_criterion, "d": oed.d_criterion, "a": oed.a_criterion}
    out = {}
    for i, (name, fn) in enumerate(crit_fns.items()):
        out[name] = oed.optimize_design(
            fn, Mb, cb, n_total, key=jax.random.fold_in(key, i),
            n_starts=N_STARTS, n_steps=N_STEPS,
        )
    return out


def _pm_fraction(z, cb, n_total):
    """Per-bin proper-motion fraction (n_pm_r + n_pm_t)/sum_channels at design z."""
    n = oed.design_counts(z, cb, n_total)            # (3, K)
    return (n[1] + n[2]) / jnp.sum(n, axis=0)        # (K,)


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description="B14 c-optimal experimental-design demo.")
    p.add_argument("--full", action="store_true",
                   help="64-draw calibration ensemble (publication grade; ~4 min) "
                        "instead of the 12-draw quick default.")
    p.add_argument("--seed", type=int, default=0,
                   help="PRNG seed for the optimizer + calibration (default 0).")
    p.add_argument("--n-total", type=float, default=4000.0,
                   help="Total star budget allocated across (radius x channel) "
                        "(default 4000).")
    p.add_argument("--no-figures", action="store_true",
                   help="Skip figure generation (Task 7); write the run-record only.")
    args = p.parse_args(argv)

    n_total = float(args.n_total)
    key = jax.random.PRNGKey(args.seed)
    k_opt, k_cal = jax.random.split(key)

    print("=" * 78)
    print("OPTIMAL EXPERIMENTAL DESIGN: c-optimal r_a, PMs to the outskirts (B14)")
    print("=" * 78)
    print(f"  mock: M={oed.MOCK['M']:.0e} Msun, r_h={oed.MOCK['r_h']} pc, "
          f"r_a={oed.MOCK['r_a']} pc, d={oed.MOCK['d_kpc']} kpc")
    print(f"  errors: eps_RV={oed.MOCK['eps_RV_kms']} km/s, "
          f"eps_PM={oed.MOCK['eps_PM_masyr']} mas/yr  |  K={oed.R_BINS.shape[0]} bins  |  "
          f"N_total={n_total:.0f}")

    # --- per-star blocks (ONE jacrev) + uniform baseline ------------------- #
    theta = oed.theta_truth()
    Mb, _ = oed.per_star_blocks(theta, oed.R_BINS, oed.EPS, G)
    cb = oed.completeness(oed.R_BINS)
    K = oed.R_BINS.shape[0]
    z_unif = jnp.zeros(3 * K)
    F_unif = oed.fisher(z_unif, Mb, cb, n_total, oed.PRIOR_DIAG)
    c_unif = float(oed.c_criterion(F_unif))
    a_unif = float(oed.a_criterion(F_unif))
    d_unif = float(oed.d_criterion(F_unif))

    # --- optimize c / D / A designs ---------------------------------------- #
    designs = _designs(Mb, cb, n_total, k_opt)
    c_res, d_res, a_res = designs["c"], designs["d"], designs["a"]

    # --- headline equal-precision factor (c-design vs uniform, FIXED N) ---- #
    factor = c_unif / c_res.criterion
    frac_sig_unif = c_unif ** 0.5            # sigma(r_a)/r_a, uniform design
    frac_sig_c = c_res.criterion ** 0.5      # sigma(r_a)/r_a, c-optimal design

    # --- interpretability: PM fraction inner vs outer (c-design) ----------- #
    pm_frac = _pm_fraction(c_res.z, cb, n_total)
    pm_inner = float(jnp.mean(pm_frac[: K // 2]))
    pm_outer = float(jnp.mean(pm_frac[K // 2:]))

    # --- calibration ensemble (small unless --full) ------------------------ #
    n_draws = N_DRAWS_FULL if args.full else N_DRAWS_QUICK
    print(f"\n  calibrating design Fisher against {n_draws} mock draws "
          f"(~3.75 s/draw){' [--full]' if args.full else ' [quick; use --full for 64]'} ...")
    cal = oed.calibrate_fisher(z=z_unif, N_total=n_total, n_draws=n_draws, key=k_cal)
    cal_realized = cal.realized_var_ra ** 0.5    # realized fractional sigma(r_a)
    cal_fisher = cal.fisher_var_ra ** 0.5        # Fisher-predicted fractional sigma(r_a)
    cal_dev = abs(cal.realized_var_ra - cal.fisher_var_ra) / cal.fisher_var_ra
    # The realized quantity is a sample VARIANCE from n_draws draws, so its own MC
    # error is ~sqrt(2/n_draws) (~18% at 64 draws, ~41% at the 12-draw quick
    # default). Gate the fractional deviation at 2x that MC error so the quick run
    # is not failed by sampling noise; the @slow unit test uses the tighter 35%.
    cal_tol = 2.0 * (2.0 / n_draws) ** 0.5

    # --- summary table: each criterion evaluated on each design ------------ #
    # Columns are the four designs (uniform + the three optimized z); rows are the
    # three criteria. The diagonal of the optimized block is each design's own
    # objective (lowest in its column, by construction of the optimization).
    design_z = {"uniform": z_unif, "c-opt": c_res.z, "d-opt": d_res.z, "a-opt": a_res.z}
    crit_of = {"c  (var r_a)": oed.c_criterion,
               "D  (-logdet)": oed.d_criterion,
               "A  (tr Finv)": oed.a_criterion}
    print("\n" + "-" * 78)
    print(f"  {'criterion':<14s}{'uniform':>14s}{'c-opt':>14s}{'D-opt':>14s}{'A-opt':>14s}")
    print("-" * 78)
    for cname, cfn in crit_of.items():
        vals = [float(cfn(oed.fisher(z, Mb, cb, n_total, oed.PRIOR_DIAG)))
                for z in design_z.values()]
        print(f"  {cname:<14s}" + "".join(f"{v:>14.4e}" for v in vals))
    print("-" * 78)
    print(f"  HEADLINE  sigma(r_a)/r_a : uniform {frac_sig_unif:.4f}  ->  "
          f"c-optimal {frac_sig_c:.4f}")
    print(f"  equal-precision factor   : c_uniform / c_designed = {factor:.3f}x  "
          f"(FIXED N -- prior cancels)")
    print("     NOTE: this factor is EXACT at fixed N. An 'equivalent uniform star count'")
    print("     gloss is only APPROXIMATE: the fixed nuisance prior dilutes as N grows")
    print("     (c*N drifts ~18% over N=1e3..8e3), so the frontier is mildly non-1/N.")
    print(f"  PM fraction (c-design)   : inner-half {pm_inner:.3f}  ->  "
          f"outer-half {pm_outer:.3f}  (PMs to the outskirts)")
    print("-" * 78)
    print(f"  calibration (uniform, {n_draws} draws):")
    print(f"     realized  sigma(r_a)/r_a = {cal_realized:.4f}")
    print(f"     Fisher    sigma(r_a)/r_a = {cal_fisher:.4f}   "
          f"(fractional dev {cal_dev*100:.1f}%, gate {cal_tol*100:.0f}% = 2 sqrt(2/{n_draws}))")
    print("     NOTE: the Fisher is mildly CONSERVATIVE -- with the binned-dispersion")
    print("     estimator the realized scatter tracks the per-star Fisher prediction to")
    print("     within its MC error; at the @slow 64-draw setting it sits slightly below it.")
    print("-" * 78)

    # --- gates ------------------------------------------------------------- #
    headline_ok = factor > 1.3
    pm_ok = pm_outer > pm_inner
    cal_ok = cal_dev < cal_tol
    rows = [
        ("headline factor > 1.3", "PASS" if headline_ok else "FAIL",
         f"{factor:.2f}x", headline_ok),
        ("PM fraction outer > inner", "PASS" if pm_ok else "FAIL",
         f"{pm_outer:.2f}>{pm_inner:.2f}", pm_ok),
        (f"calibration dev < {cal_tol*100:.0f}%", "PASS" if cal_ok else "FAIL",
         f"{cal_dev*100:.0f}%", cal_ok),
    ]
    print(f"  {'CHECK':<30s}{'status':>8s}{'value':>14s}")
    print("-" * 78)
    all_ok = True
    for name, status, val, ok in rows:
        all_ok &= ok
        print(f"  {name:<30s}{status:>8s}{val:>14s}")
    print("-" * 78)

    # --- run-record JSON --------------------------------------------------- #
    os.makedirs(FIGURE_DIR, exist_ok=True)
    record = {
        "demo": "demo_oed (B14, Stage 1)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "params": {
            "seed": args.seed,
            "n_total": n_total,
            "full_calibration": bool(args.full),
            "n_draws": n_draws,
            "n_starts": N_STARTS,
            "n_steps": N_STEPS,
            "K_bins": K,
            "mock": oed.MOCK,
            "eps_pcMyr": [float(e) for e in oed.EPS],
            "prior_diag": [float(x) for x in oed.PRIOR_DIAG],
        },
        "results": {
            "criteria": {
                "uniform": {"c": c_unif, "d": d_unif, "a": a_unif},
                "c_opt": c_res.criterion,
                "d_opt": d_res.criterion,
                "a_opt": a_res.criterion,
            },
            "headline_factor": factor,
            "frac_sigma_ra_uniform": frac_sig_unif,
            "frac_sigma_ra_c_design": frac_sig_c,
            "pm_fraction_inner": pm_inner,
            "pm_fraction_outer": pm_outer,
            "pm_fraction_per_bin": [float(x) for x in pm_frac],
            "calibration": {
                "realized_frac_sigma_ra": cal_realized,
                "fisher_frac_sigma_ra": cal_fisher,
                "fractional_dev": cal_dev,
                "gate_tol": cal_tol,
            },
        },
        "designs_z": {
            "c": [float(x) for x in c_res.z],
            "d": [float(x) for x in d_res.z],
            "a": [float(x) for x in a_res.z],
        },
        "gates": {name: ok for name, _, _, ok in rows},
        "all_pass": bool(all_ok),
    }
    with open(RUN_RECORD, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  run-record -> {RUN_RECORD}")

    # --- figures (Task 7) -------------------------------------------------- #
    if args.no_figures:
        print("  figures: SKIPPED (--no-figures)")
    else:
        make_figures(
            Mb, cb, n_total, designs, z_unif, pm_frac, cal,
            factor=factor, frac_sig_unif=frac_sig_unif, frac_sig_c=frac_sig_c,
        )

    print("=" * 78)
    print("  OED DEMO: ALL PASS" if all_ok else "  OED DEMO: FAILED")
    return 0 if all_ok else 1


# --------------------------------------------------------------------------- #
# Task 7 will implement the five figures, saved to FIGURE_DIR:
#   1. demo_oed_optpath.png     -- c-criterion vs iteration, multi-start traces
#   2. demo_oed_headline.png    -- optimal radial weighting + RV/PM split over sigma(r)/beta(r)
#   3. demo_oed_cda.png         -- c-vs-D-vs-A allocations side-by-side
#   4. demo_oed_frontier.png    -- measured sigma(r_a)/r_a vs N, designed vs uniform
#   5. demo_oed_calibration.png -- realized MLE Cov vs F^-1
# All the data handles each figure needs are passed into make_figures() below
# (DesignResult.trace for fig 1; Mb/cb/designs/z_unif for figs 2-3; the c-design
# + uniform criteria as a function of N for fig 4 via oed.fisher; the CalibResult
# for fig 5). For now this is a clearly-marked stub so the CLI runs end-to-end and
# exits 0 WITHOUT the figures; Task 7 fills it in using scripts/_plotstyle.py.
def make_figures(Mb, cb, n_total, designs, z_unif, pm_frac, cal, *,
                 factor, frac_sig_unif, frac_sig_c):
    """STUB (Task 7): generate the five Stage-1 figures into FIGURE_DIR.

    Handles available for each figure:
      * fig 1 (optpath):     designs['c'/'d'/'a'].trace (per-step criterion).
      * fig 2 (headline):    designs['c'].z + cb + n_total -> oed.design_counts;
                             oed.predict_sigma(theta_truth, R_BINS, G) for sigma(r);
                             beta_OM(r) = R_BINS^2/(R_BINS^2 + r_a^2).
      * fig 3 (cda):         oed.design_counts for each of designs['c'/'d'/'a'].z.
      * fig 4 (frontier):    sweep N, recompute oed.fisher/c_criterion for the
                             c-design fractions and the uniform design.
      * fig 5 (calibration): cal.realized_var_ra vs cal.fisher_var_ra.
    """
    print("  figures: TODO (Task 7) -- handles wired in make_figures(); skipping for now")


if __name__ == "__main__":
    sys.exit(main())

"""M1 shot-noise bake-off: which shot model recovers turbulence slope beta UNBIASED?

SCRATCH validation script (NOT production, NOT committed). Decides whether the
forward model for inferring beta (P(k) propto k^-beta of the natal turbulent
density field) from a 2D projected star-count map should subtract shot noise in

  Option A -- raw-count (delta) space:  P_meas(k) = P_signal(k) + 1/nbar  (flat white plateau)
  Option B -- rank-Gaussianized space (Neyrinck+2011 Eq.1), shot via Eq.3.

Identical recovery estimator for both (log-log slope LSQ over signal-dominated
bins after shot subtraction), so the ONLY difference is the shot model + field space.

numpy/scipy are allowed on the measurement side (validation, non-differentiable).

Run:
  PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync \
      python src/experimental/gravoturb_fdf/validation/_m1_shot_bakeoff.py
"""

import os

import jax
import numpy as np
from scipy.special import erfinv

from gravoturb_fdf.field.field import gaussian_random_field, rank_copula_field
from gravoturb_fdf.field.sampling import sample_cic_counts
from gravoturb_fdf.validation.measure import (
    measure_angular_bandpowers_2d,
    project_counts_los,
)

# ----------------------------------------------------------------------------- #
# Experimental design
# ----------------------------------------------------------------------------- #
SHAPE = (96, 96, 96)        # 3D field grid
CELL_SIZE = 4               # -> M = 96//4 = 24 count cells per side (sky grid 24x24)
M_SKY = SHAPE[0] // CELL_SIZE   # 24
BOX = 1.0                   # unit box; sky-cell area = (BOX / M_SKY)^2

MACH, B_TURB, ALPHA = 8.0, 0.4, 2.5
BETAS = (2.5, 3.0, 3.5)
N_STARS_LIST = (1000, 3000, 10000, 100000)
N_REAL = 30

N_SKY_CELLS = M_SKY * M_SKY  # 576
N_3D_CELLS = M_SKY ** 3      # 13824
V_CELL = (BOX / M_SKY) ** 2  # 2D sky-cell area (consistent with box-normalized P(k))

# k_edges on the 24x24 sky grid: exclude DC, 8 log-spaced bins up to ~M/2 = 12.
N_KBINS = 8
K_MIN, K_MAX = 1.0, M_SKY / 2.0   # 1 .. 12
K_EDGES = np.geomspace(K_MIN, K_MAX, N_KBINS + 1)
K_CENT = np.sqrt(K_EDGES[:-1] * K_EDGES[1:])  # geometric bin centers

PLOTDIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTDIR, exist_ok=True)


# ----------------------------------------------------------------------------- #
# Neyrinck+2011 Gaussianization (Eq.1) and empirical shot estimator (Eq.3)
# ----------------------------------------------------------------------------- #
def gaussianize_neyrinck(field2d):
    """Eq.(1): G(delta) = sqrt(2)*sigma*erfinv(2 f_<delta - 1 + 1/N), sigma=1.

    f_<delta estimated by the plotting position (rank+0.5)/N (fraction of cells
    LESS dense). The +1/N offset in the argument is supplied as stated; with the
    (rank+0.5)/N estimator this keeps the erfinv argument strictly inside (-1, 1).
    """
    flat = np.asarray(field2d, dtype=float).ravel()
    n = flat.size
    ranks = np.argsort(np.argsort(flat))           # 0..n-1
    f_less = (ranks + 0.5) / n                      # plotting-position estimate of f_<delta
    arg = 2.0 * f_less - 1.0 + 1.0 / n
    arg = np.clip(arg, -1.0 + 1e-12, 1.0 - 1e-12)   # guard erfinv domain
    g = np.sqrt(2.0) * 1.0 * erfinv(arg)
    return g.reshape(np.asarray(field2d).shape)


def shot_eq3(field2d):
    """Neyrinck+2011 Eq.(3) shot of the Gaussianized field, as a periodogram plateau.

    Eq.(3):  1/n_eff = V_cell * sum_i f(delta_i) * (G(delta_{i+1}) - G(delta_i)),
    f(delta_i)=1/N per distinct cell, (G_{i+1}-G_i)=spacing of consecutive sorted
    Gaussianized values. Eq.(3) is Neyrinck's box-normalized shot POWER
    (P = V_cell * per-cell-variance). measure_angular_bandpowers_2d returns a per-cell
    variance (<|fft2|^2/size>, NO V_cell factor), so the matching flat plateau is
    1/n_eff DIVIDED by V_cell, i.e. plateau_B = sum_i f_i * (G_{i+1}-G_i) -- the
    per-cell shot VARIANCE of the Gaussianized field. (Computing 1/n_eff with V_cell
    and then dividing it back out is algebraically the same; we report the plateau
    directly to keep A and B in the identical per-cell-variance convention.)

    KEY Neyrinck physics this captures: Gaussianization INCREASES the shot relative to
    raw-delta space (the sum of Gaussian-gap spacings is set by the fat tail), and it is
    mildly scale-dependent -- but we model it as the best flat plateau, as Eq.(3) gives.
    """
    flat = np.asarray(field2d, dtype=float).ravel()
    n = flat.size
    order = np.argsort(flat)
    g_sorted = gaussianize_neyrinck(flat)[order]    # G at sorted-by-density cells
    dG = np.diff(g_sorted)                           # G_{i+1} - G_i, length n-1
    f_i = 1.0 / n                                    # fraction at each distinct value
    plateau = f_i * np.sum(dG)                       # per-cell shot variance (= 1/n_eff / V_cell)
    return float(plateau)


# ----------------------------------------------------------------------------- #
# Recovery estimator (IDENTICAL for both options)
# ----------------------------------------------------------------------------- #
def recover_beta(bandpowers, shot_level, k_cent, snr_min=1.0):
    """Shot-subtract then log-log slope LSQ over signal-dominated bins.

    Subtract the (flat) shot level, keep bins with (P_meas - shot)/shot > snr_min
    AND positive residual, fit log P_sig = log A - beta log k. Returns
    (beta_rec, used_mask). Same estimator for A and B; only `shot_level` and the
    `bandpowers` (which space) differ.
    """
    bp = np.asarray(bandpowers, dtype=float)
    sig = bp - shot_level
    # signal/shot ratio test (shot>0 always here)
    snr = sig / max(shot_level, 1e-300)
    used = (sig > 0) & (snr > snr_min)
    if used.sum() < 2:
        # too few signal-dominated bins -> fall back to positive-residual bins
        used = sig > 0
        if used.sum() < 2:
            return np.nan, used
    x = np.log(k_cent[used])
    y = np.log(sig[used])
    slope, _ = np.polyfit(x, y, 1)
    return -slope, used


# ----------------------------------------------------------------------------- #
# One realization -> (bandpowers_A, nbar_2d, bandpowers_B, shot_B)
# ----------------------------------------------------------------------------- #
def one_realization(beta, n_bar_3d, key):
    """Build field once, Poisson-sample CIC counts, project full-depth, measure both options."""
    s = rank_copula_field(gaussian_random_field(SHAPE, beta, key), MACH, B_TURB, ALPHA)
    counts3d = np.asarray(sample_cic_counts(s, n_bar_3d, CELL_SIZE, jax.random.fold_in(key, 1)))
    # full-depth LOS projection over all M_SKY slices -> (M_SKY, M_SKY) count map
    cmap = project_counts_los(counts3d, depth=M_SKY, los_axis=2).astype(float)

    nbar_2d = cmap.mean()  # mean stars per SKY cell (known from data)

    # --- Option A: raw-count space, flat 1/nbar plateau ---
    # measure_angular_bandpowers_2d returns <|fft2(f-<f>)|^2/size>, which for white noise
    # equals the per-cell variance. A Poisson count map has per-cell shot variance =
    # nbar_2d, so the flat white plateau in THIS periodogram convention is exactly nbar_2d
    # (this IS the "1/nbar" plateau, expressed as a count-space power). Exact in raw space.
    bp_A = measure_angular_bandpowers_2d(cmap, K_EDGES)
    shot_A = float(nbar_2d)

    # --- Option B: rank-Gaussianized space, Eq.3 shot (per-cell variance plateau) ---
    g_map = gaussianize_neyrinck(cmap)
    bp_B = measure_angular_bandpowers_2d(g_map, K_EDGES)
    shot_B = shot_eq3(cmap)

    return bp_A, shot_A, bp_B, shot_B


# ----------------------------------------------------------------------------- #
# Driver
# ----------------------------------------------------------------------------- #
def run():
    base = jax.random.PRNGKey(20260607)
    # results[(beta, N)] = dict(A=[...], B=[...], bp_A=mean, bp_B=mean, shotA, shotB, used_A, used_B)
    results = {}
    # also store per-(beta,N) representative band-powers for the diagnostic plot
    store = {}

    for beta in BETAS:
        for N in N_STARS_LIST:
            n_bar_3d = N / N_3D_CELLS  # mean count per 3D cell so projected total ~ N
            kb = jax.random.fold_in(jax.random.fold_in(base, int(beta * 10)), N)
            betaA, betaB = [], []
            bpA_acc, bpB_acc = [], []
            shotA_acc, shotB_acc = [], []
            usedA_acc, usedB_acc = [], []
            for r in range(N_REAL):
                k = jax.random.fold_in(kb, r)
                bp_A, shot_A, bp_B, shot_B = one_realization(beta, n_bar_3d, k)
                bA, uA = recover_beta(bp_A, shot_A, K_CENT)
                bB, uB = recover_beta(bp_B, shot_B, K_CENT)
                betaA.append(bA); betaB.append(bB)
                bpA_acc.append(bp_A); bpB_acc.append(bp_B)
                shotA_acc.append(shot_A); shotB_acc.append(shot_B)
                usedA_acc.append(uA); usedB_acc.append(uB)
            betaA = np.array(betaA); betaB = np.array(betaB)
            results[(beta, N)] = dict(
                A_mean=np.nanmean(betaA), A_std=np.nanstd(betaA, ddof=1),
                B_mean=np.nanmean(betaB), B_std=np.nanstd(betaB, ddof=1),
            )
            store[(beta, N)] = dict(
                bp_A=np.mean(bpA_acc, axis=0), bp_B=np.mean(bpB_acc, axis=0),
                shot_A=np.mean(shotA_acc), shot_B=np.mean(shotB_acc),
                used_A=np.array(usedA_acc).mean(axis=0), used_B=np.array(usedB_acc).mean(axis=0),
            )
    return results, store


# ----------------------------------------------------------------------------- #
# Reporting + PASS/FAIL
# ----------------------------------------------------------------------------- #
def report(results):
    print("\n" + "=" * 96)
    print("M1 SHOT-NOISE BAKE-OFF  (Mach=8.0, b=0.4, alpha=2.5; shape=96^3; sky 24x24; "
          f"n_real={N_REAL})")
    print(f"k-bins: {N_KBINS} log-spaced in [{K_MIN:.1f},{K_MAX:.1f}] (DC excluded); "
          "signal/shot>1 bins used")
    print("=" * 96)
    header = (f"{'beta_t':>6} {'N':>7} | {'opt':>3} {'mean(b_rec)':>11} {'sigma':>7} "
              f"{'bias':>8} {'z=bias/(s/sqrtn)':>17}")
    print(header)
    print("-" * 96)
    sqrtn = np.sqrt(N_REAL)
    pass_A, pass_B = True, True
    for beta in BETAS:
        for N in N_STARS_LIST:
            r = results[(beta, N)]
            for opt, mean, std in (("A", r["A_mean"], r["A_std"]),
                                   ("B", r["B_mean"], r["B_std"])):
                bias = mean - beta
                z = bias / (std / sqrtn) if std > 0 else np.nan
                print(f"{beta:>6.1f} {N:>7d} | {opt:>3} {mean:>11.3f} {std:>7.3f} "
                      f"{bias:>8.3f} {z:>17.2f}")
                # PASS if |bias| within ~2 sigma/sqrt(n) of zero, i.e. |z| <= 2
                if opt == "A" and abs(z) > 2.0:
                    pass_A = False
                if opt == "B" and abs(z) > 2.0:
                    pass_B = False
            print("-" * 96)
    print(f"\nPASS/FAIL (|z| = |bias|/(sigma/sqrt(n_real)) <= 2 across ALL N and beta_true):")
    print(f"  Option A (raw-count, flat 1/nbar)      : {'PASS' if pass_A else 'FAIL'}")
    print(f"  Option B (rank-Gaussianized, Eq.3 shot): {'PASS' if pass_B else 'FAIL'}")
    return pass_A, pass_B


# ----------------------------------------------------------------------------- #
# Plots
# ----------------------------------------------------------------------------- #
def make_plots(results, store):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Plot 1: beta_rec vs N, one panel per beta_true ---
    fig, axes = plt.subplots(1, len(BETAS), figsize=(5 * len(BETAS), 4.2), sharey=False)
    for ax, beta in zip(axes, BETAS):
        Ns = np.array(N_STARS_LIST, dtype=float)
        Am = [results[(beta, N)]["A_mean"] for N in N_STARS_LIST]
        As = [results[(beta, N)]["A_std"] for N in N_STARS_LIST]
        Bm = [results[(beta, N)]["B_mean"] for N in N_STARS_LIST]
        Bs = [results[(beta, N)]["B_std"] for N in N_STARS_LIST]
        ax.errorbar(Ns * 0.95, Am, yerr=As, fmt="o-", capsize=3, label="A: raw-count, 1/nbar")
        ax.errorbar(Ns * 1.05, Bm, yerr=Bs, fmt="s-", capsize=3, label="B: Gaussianized, Eq.3")
        ax.axhline(beta, color="k", ls="--", lw=1, label=f"truth = {beta}")
        ax.set_xscale("log")
        ax.set_xlabel("N_stars")
        ax.set_ylabel(r"recovered $\beta$")
        ax.set_title(rf"$\beta_{{\rm true}} = {beta}$")
        ax.legend(fontsize=8)
    fig.suptitle("M1: turbulence-slope recovery vs star count (shot-model bake-off)")
    fig.tight_layout()
    p1 = os.path.join(PLOTDIR, "m1_beta_recovery_vs_N.png")
    fig.savefig(p1, dpi=130); plt.close(fig)

    # --- Plot 2: band-powers + shot + fit, beta_true=3.0, a few N ---
    beta0 = 3.0
    Ns_show = N_STARS_LIST
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(Ns_show)))
    for opt, ax, key_bp, key_shot, key_used in (
        ("A (raw-count)", axes[0], "bp_A", "shot_A", "used_A"),
        ("B (Gaussianized)", axes[1], "bp_B", "shot_B", "used_B"),
    ):
        for c, N in zip(cmap, Ns_show):
            st = store[(beta0, N)]
            bp = st[key_bp]; shot = st[key_shot]
            ax.loglog(K_CENT, bp, "o-", color=c, alpha=0.8, label=f"N={N} meas")
            ax.axhline(shot, color=c, ls=":", lw=1)
            sig = bp - shot
            m = sig > 0
            if m.sum() >= 2:
                ax.loglog(K_CENT[m], sig[m], "x--", color=c, alpha=0.6)
        # reference k^-beta line
        kref = K_CENT
        ref = (kref / kref[0]) ** (-beta0)
        ax.loglog(kref, ref * store[(beta0, Ns_show[-1])]["bp_A"][0], "k-", lw=1.5,
                  label=rf"$k^{{-{beta0}}}$ ref")
        ax.set_xlabel("k (sky-grid)")
        ax.set_ylabel("band-power")
        ax.set_title(rf"Option {opt}, $\beta_{{\rm true}}={beta0}$")
        ax.legend(fontsize=7)
    fig.suptitle("M1: measured band-powers (o), shot level (:), shot-subtracted signal (x)")
    fig.tight_layout()
    p2 = os.path.join(PLOTDIR, "m1_bandpowers_shot.png")
    fig.savefig(p2, dpi=130); plt.close(fig)

    # --- Plot 3: bias vs N, both options, all beta ---
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sqrtn = np.sqrt(N_REAL)
    markers = {2.5: "o", 3.0: "s", 3.5: "^"}
    for beta in BETAS:
        Ns = np.array(N_STARS_LIST, dtype=float)
        biasA = [results[(beta, N)]["A_mean"] - beta for N in N_STARS_LIST]
        errA = [results[(beta, N)]["A_std"] / sqrtn for N in N_STARS_LIST]
        biasB = [results[(beta, N)]["B_mean"] - beta for N in N_STARS_LIST]
        errB = [results[(beta, N)]["B_std"] / sqrtn for N in N_STARS_LIST]
        ax.errorbar(Ns * 0.95, biasA, yerr=errA, fmt=markers[beta] + "-", color="C0",
                    capsize=3, label=f"A b={beta}")
        ax.errorbar(Ns * 1.05, biasB, yerr=errB, fmt=markers[beta] + "--", color="C1",
                    capsize=3, label=f"B b={beta}")
    ax.axhline(0, color="k", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("N_stars")
    ax.set_ylabel(r"bias = mean($\beta_{\rm rec}$) - $\beta_{\rm true}$")
    ax.set_title("M1: recovery bias vs N (error bars = sigma/sqrt(n_real))")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    p3 = os.path.join(PLOTDIR, "m1_bias_vs_N.png")
    fig.savefig(p3, dpi=130); plt.close(fig)

    return p1, p2, p3


if __name__ == "__main__":
    results, store = run()
    pass_A, pass_B = report(results)
    p1, p2, p3 = make_plots(results, store)
    print("\nPNGs written:")
    for p in (p1, p2, p3):
        print(f"  {p}")

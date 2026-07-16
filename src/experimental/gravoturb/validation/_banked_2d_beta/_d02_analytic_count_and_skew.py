r"""D02 — sharpen the gate: (1) HIGH-N per-bin band-power skew (properly powered, unlike D01's
N=24 where sample skew is capped at ~sqrt(N)); (2) does the ANALYTIC count predictor
``angular_bandpowers_2d_limber + add_poisson_shot`` reproduce the simulator's count band-power MEAN
bin-by-bin across beta?  If YES, the analytic forward MEAN is valid for the observable and only the
likelihood SHAPE is broken (=> Path A viable with a correct non-Gaussian likelihood).

EXPERIMENTAL scratch; no production edits, no commits.
Run: PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync python -u \
     src/experimental/gravoturb_fdf/validation/_d02_analytic_count_and_skew.py
"""
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy import special, stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gravoturb_fdf.field.field import gaussian_random_field
from gravoturb_fdf.field.sampling import sample_cic_counts
from gravoturb_fdf.inference.covariance import add_poisson_shot, angular_bandpowers_2d_limber
from gravoturb_fdf.validation.measure import (
    measure_angular_bandpowers_2d,
    project_counts_los,
    smooth_copula_field,
)

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
SHAPE, DEPTH = (64, 64, 64), 64
B_FIXED, ALPHA, M_FID = 0.4, 2.5, 8.0
K_EDGES = np.linspace(1.0, 28.0, 11)
N_STARS = 10**5
N_BAR_3D = N_STARS / (SHAPE[0] ** 3)
N_BAR_SKY = N_STARS / (SHAPE[0] ** 2)            # = N_BAR_3D * DEPTH
BETAS = np.array([2.0, 2.5, 3.0, 3.5, 11.0 / 3.0])
N_MAX = 14


def rank_gaussianize_2d(map2d):
    f = np.asarray(map2d, float).ravel()
    ranks = np.argsort(np.argsort(f))
    u = (ranks + 0.5) / f.size
    return (np.sqrt(2.0) * special.erfinv(2.0 * u - 1.0)).reshape(map2d.shape)


def log_plus(N, n_bar):
    d = np.asarray(N, float) / n_bar - 1.0
    return np.where(d > 0.0, np.log1p(np.where(d > 0.0, d, 0.0)), d)


def hamimeche_lewis(x):
    x = np.asarray(x, float)
    xs = np.where(x > 1e-12, x, 1e-12)
    return np.sign(xs - 1.0) * np.sqrt(2.0 * (xs - np.log(xs) - 1.0))


def count_map(beta, M, key):
    s = smooth_copula_field(gaussian_random_field(SHAPE, beta, key), M, B_FIXED, ALPHA)
    cnt = np.asarray(sample_cic_counts(jnp.asarray(s), N_BAR_3D, 1, jax.random.fold_in(key, 1)))
    return project_counts_los(cnt, DEPTH, los_axis=2).astype(float)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    k_cent = 0.5 * (K_EDGES[:-1] + K_EDGES[1:])
    t0 = time.time()

    # ---------------- (1) HIGH-N per-bin skew at beta=3 ----------------
    N_SKEW = 400
    print(f"D02 (1) per-bin skew at beta=3, N_real={N_SKEW} (sample-skew cap ~sqrt(N)={np.sqrt(N_SKEW):.0f})")
    raw = np.zeros((N_SKEW, len(k_cent)))
    rg = np.zeros((N_SKEW, len(k_cent)))
    lp = np.zeros((N_SKEW, len(k_cent)))
    for r in range(N_SKEW):
        key = jax.random.fold_in(jax.random.PRNGKey(31415), r)
        cm = count_map(3.0, M_FID, key)
        nb = cm.mean()
        raw[r] = measure_angular_bandpowers_2d(cm, K_EDGES)
        rg[r] = measure_angular_bandpowers_2d(rank_gaussianize_2d(cm), K_EDGES)
        lp[r] = measure_angular_bandpowers_2d(log_plus(cm, nb), K_EDGES)
        if (r + 1) % 100 == 0:
            print(f"    {r+1}/{N_SKEW} ({time.time()-t0:.0f}s)")
    mu_raw = raw.mean(axis=0)
    print(f"  {'k':>6} {'raw_sk':>7} {'raw_exk':>8} {'log_sk':>7} {'HL_sk':>7} {'logp_sk':>8} {'rankG_sk':>9}")
    for j in range(len(k_cent)):
        logc = np.log(np.where(raw[:, j] > 0, raw[:, j], np.nan))
        hlc = hamimeche_lewis(raw[:, j] / mu_raw[j])
        print(f"  {k_cent[j]:6.1f} {stats.skew(raw[:,j]):7.2f} {stats.kurtosis(raw[:,j]):8.1f} "
              f"{stats.skew(logc, nan_policy='omit'):7.2f} {stats.skew(hlc):7.2f} "
              f"{stats.skew(lp[:,j]):8.2f} {stats.skew(rg[:,j]):9.2f}")

    # ---------------- (2) analytic count predictor vs sim count mean, bin-by-bin ----------------
    print(f"\nD02 (2) analytic count mean [A_rho + add_poisson_shot] vs sim E[O_raw], "
          f"N_real=48/beta  (n_bar_sky={N_BAR_SKY:.2f})")
    N_MEAN = 48
    print(f"  {'beta':>5}  {'max|%err| bin-by-bin':>20}  {'analytic slope':>14}  {'sim slope':>10}")
    pcts = []
    for beta in BETAS:
        _kc, Pc, _nm = angular_bandpowers_2d_limber(SHAPE, float(beta), M_FID, B_FIXED, ALPHA,
                                                    DEPTH, K_EDGES, n_max=N_MAX)
        A_count = np.asarray(add_poisson_shot(Pc, N_BAR_SKY, DEPTH))
        rows = np.array([measure_angular_bandpowers_2d(count_map(float(beta), M_FID,
                        jax.random.fold_in(jax.random.PRNGKey(271828), int(beta * 1000) + r)), K_EDGES)
                        for r in range(N_MEAN)])
        E_raw = rows.mean(axis=0)
        pct = 100.0 * np.abs(A_count - E_raw) / E_raw
        pcts.append(pct)
        sa = np.polyfit(np.log(k_cent), np.log(A_count), 1)[0]
        ss = np.polyfit(np.log(k_cent), np.log(E_raw), 1)[0]
        print(f"  {beta:5.2f}  {pct.max():19.1f}%  {sa:14.3f}  {ss:10.3f}")
    pcts = np.array(pcts)

    # ---------------- figure ----------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for label, arr in [("raw", raw), ("rank-G", rg), ("log+", lp)]:
        ax[0].plot(k_cent, [stats.skew(arr[:, j]) for j in range(len(k_cent))], marker="s", label=label)
    ax[0].plot(k_cent, [stats.skew(hamimeche_lewis(raw[:, j] / mu_raw[j])) for j in range(len(k_cent))],
               marker="^", label="H&L(raw)")
    ax[0].plot(k_cent, [stats.skew(np.log(np.where(raw[:, j] > 0, raw[:, j], np.nan)), nan_policy='omit')
               for j in range(len(k_cent))], marker="v", label="log(raw)")
    ax[0].axhline(0, color="k", lw=0.5); ax[0].set_xlabel("|k|"); ax[0].set_ylabel("band-power skew")
    ax[0].set_title(f"D02(1): per-bin skew vs |k| (beta=3, N={N_SKEW})"); ax[0].legend(fontsize=8)
    for bi, beta in enumerate(BETAS):
        ax[1].plot(k_cent, pcts[bi], marker="o", label=f"beta={beta:.2f}")
    ax[1].set_xlabel("|k|"); ax[1].set_ylabel("|analytic - sim|/sim  [%]")
    ax[1].set_title("D02(2): analytic count mean vs sim E[O_raw]"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "d02_analytic_count_and_skew.png")
    fig.savefig(path, dpi=140); plt.close(fig)
    print(f"\nfigure: {path}\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

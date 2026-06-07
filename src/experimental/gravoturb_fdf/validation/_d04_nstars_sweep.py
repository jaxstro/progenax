r"""D04 — Anna's question: is the beta fit shot-noise-limited (richer clusters help) or
cosmic-variance-limited (only stacking helps)?  Sweep N_stars, generating each cosmic field ONCE and
re-sampling Poisson counts at every N_stars (isolates shot from cosmic variance). Verify BOTH log_plus
and rank-G observables (Anna: keep/verify both).

For each observable report vs N_stars:
  * gain = d(slope)/d(beta)         -> does the shot-suppressed beta-response recover toward the
                                       no-shot field value (S_inf = log/rankG of projected DENSITY)?
  * sigma(beta)_per_cluster = std_fields(slope at beta=3) / |gain|
                                    -> FLAT in N_stars => cosmic-variance-limited (stack to improve);
                                       DROPS with N_stars => shot-limited (richer clusters improve).

EXPERIMENTAL scratch; no production edits, no commits.
Run: PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync python -u \
     src/experimental/gravoturb_fdf/validation/_d04_nstars_sweep.py
"""
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy import special

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gravoturb_fdf.field.field import gaussian_random_field
from gravoturb_fdf.field.sampling import sample_cic_counts
from gravoturb_fdf.validation.measure import (
    measure_angular_bandpowers_2d,
    project_counts_los,
    smooth_copula_field,
)

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
SHAPE, DEPTH = (64, 64, 64), 64
B_FIXED, ALPHA, M_FID = 0.4, 2.5, 8.0
K_EDGES = np.linspace(1.0, 28.0, 11)
BETAS = np.array([2.5, 3.0, 3.5])
NSTARS = [10**4, 10**5, 10**6, 10**7]
N_REAL = 24


def rank_g(map2d):
    f = np.asarray(map2d, float).ravel()
    ranks = np.argsort(np.argsort(f))
    u = (ranks + 0.5) / f.size
    return (np.sqrt(2.0) * special.erfinv(2.0 * u - 1.0)).reshape(map2d.shape)


def log_plus(N, n_bar):
    d = np.asarray(N, float) / n_bar - 1.0
    return np.where(d > 0.0, np.log1p(np.where(d > 0.0, d, 0.0)), d)


def slope(k, P):
    P = np.asarray(P); g = (P > 0) & np.isfinite(P)
    return np.polyfit(np.log(np.asarray(k)[g]), np.log(P[g]), 1)[0] if g.sum() > 1 else np.nan


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    k = 0.5 * (K_EDGES[:-1] + K_EDGES[1:])
    t0 = time.time()
    cols = NSTARS + ["inf"]
    # slopes[obs][col] = (nbeta, nreal)
    slopes = {obs: {c: np.zeros((len(BETAS), N_REAL)) for c in cols} for obs in ["logp", "rankg"]}

    for bi, beta in enumerate(BETAS):
        for r in range(N_REAL):
            key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(123), bi), r)
            s = smooth_copula_field(gaussian_random_field(SHAPE, float(beta), key), M_FID, B_FIXED, ALPHA)
            projSig = np.exp(s).sum(axis=2)                         # no-shot projected density
            slopes["logp"]["inf"][bi, r] = slope(k, measure_angular_bandpowers_2d(np.log(projSig), K_EDGES))
            slopes["rankg"]["inf"][bi, r] = slope(k, measure_angular_bandpowers_2d(rank_g(projSig), K_EDGES))
            for N in NSTARS:
                nb3 = N / SHAPE[0] ** 3
                cnt = np.asarray(sample_cic_counts(jnp.asarray(s), nb3, 1, jax.random.fold_in(key, N % 97 + 1)))
                pc = project_counts_los(cnt, DEPTH, los_axis=2).astype(float)
                slopes["logp"][N][bi, r] = slope(k, measure_angular_bandpowers_2d(log_plus(pc, pc.mean()), K_EDGES))
                slopes["rankg"][N][bi, r] = slope(k, measure_angular_bandpowers_2d(rank_g(pc), K_EDGES))
        print(f"  beta={beta:.2f} done ({time.time()-t0:.0f}s)")

    bi3 = int(np.argmin(np.abs(BETAS - 3.0)))
    print(f"\n  {'observable':<8} {'N_stars':>10} {'n_bar_sky':>9} {'gain':>8} "
          f"{'mean_slope(b=3)':>15} {'sig_slope':>10} {'sigma(beta)/cluster':>19}")
    summary = {obs: {"N": [], "gain": [], "sigb": []} for obs in slopes}
    for obs in ["logp", "rankg"]:
        for c in cols:
            mean_sl = np.array([slopes[obs][c][bi].mean() for bi in range(len(BETAS))])
            gain = np.polyfit(BETAS, mean_sl, 1)[0]
            sig_sl = slopes[obs][c][bi3].std(ddof=1)
            sigb = sig_sl / abs(gain) if gain != 0 else np.nan
            label = f"{c}" if c == "inf" else f"{c:.0e}"
            nbsky = (float(c) / SHAPE[0] ** 2) if c != "inf" else np.inf
            print(f"  {obs:<8} {label:>10} {nbsky:9.1f} {gain:8.3f} {mean_sl[bi3]:15.3f} "
                  f"{sig_sl:10.3f} {sigb:19.3f}")
            summary[obs]["N"].append(float(c) if c != "inf" else NSTARS[-1] * 30)
            summary[obs]["gain"].append(gain)
            summary[obs]["sigb"].append(sigb)
        print()

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for obs, mk in [("logp", "o"), ("rankg", "s")]:
        ax[0].semilogx(summary[obs]["N"][:-1], np.abs(summary[obs]["gain"][:-1]), mk + "-", label=f"{obs}")
        ax[0].axhline(abs(summary[obs]["gain"][-1]), ls=":", color="gray")
        ax[1].semilogx(summary[obs]["N"][:-1], summary[obs]["sigb"][:-1], mk + "-", label=f"{obs}")
        ax[1].axhline(summary[obs]["sigb"][-1], ls=":", color="gray")
    ax[0].set_xlabel("N_stars"); ax[0].set_ylabel("|gain| = |d slope / d beta|")
    ax[0].set_title("D04: beta-response vs N_stars (dotted = no-shot inf)"); ax[0].legend()
    ax[1].set_xlabel("N_stars"); ax[1].set_ylabel("implied sigma(beta) per cluster")
    ax[1].set_title("D04: sigma(beta) vs N_stars (flat=>cosmic-var-limited)"); ax[1].legend()
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "d04_nstars_sweep.png")
    fig.savefig(path, dpi=140); plt.close(fig)
    print(f"figure: {path}\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

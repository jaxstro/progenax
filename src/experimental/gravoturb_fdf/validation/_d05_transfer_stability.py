r"""D05 — the make-or-break design check for the analytic path: is the per-bin amplitude offset
between an ANALYTIC predictor and the SIMULATOR observable beta-INDEPENDENT?

If T(k,beta) = E[observable(k,beta)] / analytic(k,beta) is ~constant in beta, then
    mu(beta) = analytic_shape(beta) * T_fixed(k)
is an SBC-grade forward model whose beta-RESPONSE stays purely analytic (no emulated slope -> avoids
v2h's fatal flaw). T_fixed(k) is a physical, calibrate-ONCE projection/marginal transfer, NOT a fudge.
We measure the per-bin coefficient-of-variation of T across beta (CV = std_beta(T)/mean_beta(T)); a
small CV (<~few %) for some (analytic, observable) pair => that pair gives a clean analytic forward
model. Candidates: analytic {A_logSig, A_s, A_rho} x observables {S_logSig (no shot), O_logp@1e6,
O_rg@1e6}. (Anna: keep/verify both log+ and rank-G.)

EXPERIMENTAL scratch; no production edits, no commits.
Run: PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync python -u \
     src/experimental/gravoturb_fdf/validation/_d05_transfer_stability.py
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
from gravoturb_fdf.inference.covariance import (
    _angular_bandpowers_from_xi_rho_2d,
    _xi_rho_grid,
    angular_bandpowers_2d_limber,
)
from gravoturb_fdf.theory.gaussianization import bm19_hermite_coefficients, gaussianized_xi
from gravoturb_fdf.theory.projection import gaussian_correlation_grid, limber_project_slab
from gravoturb_fdf.validation.measure import (
    measure_angular_bandpowers_2d,
    project_counts_los,
    smooth_copula_field,
)

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
SHAPE, DEPTH = (64, 64, 64), 64
B_FIXED, ALPHA, M_FID = 0.4, 2.5, 8.0
K_EDGES = np.linspace(1.0, 28.0, 11)
BETAS = np.array([2.0, 2.5, 3.0, 3.5, 11.0 / 3.0])
N_REAL = 32
N_STARS_HI = 10**6
N_MAX = 14


def rank_g(m):
    f = np.asarray(m, float).ravel(); r = np.argsort(np.argsort(f)); u = (r + 0.5) / f.size
    return (np.sqrt(2.0) * special.erfinv(2.0 * u - 1.0)).reshape(m.shape)


def log_plus(N, nb):
    d = np.asarray(N, float) / nb - 1.0
    return np.where(d > 0.0, np.log1p(np.where(d > 0.0, d, 0.0)), d)


def A_logSig(beta):
    xr = _xi_rho_grid(SHAPE, float(beta), M_FID, B_FIXED, ALPHA, N_MAX, 256)
    xs = limber_project_slab(xr, DEPTH, los_axis=2)
    xl = jnp.log1p(xs / float(DEPTH) ** 2)
    return np.asarray(_angular_bandpowers_from_xi_rho_2d(xl, K_EDGES)[1])


def A_s(beta):
    rg = gaussian_correlation_grid(SHAPE, float(beta))
    c = bm19_hermite_coefficients(M_FID, B_FIXED, ALPHA, N_MAX)
    xs = limber_project_slab(gaussianized_xi(rg, c), DEPTH, los_axis=2)
    return np.asarray(_angular_bandpowers_from_xi_rho_2d(xs, K_EDGES)[1])


def A_rho(beta):
    return np.asarray(angular_bandpowers_2d_limber(SHAPE, float(beta), M_FID, B_FIXED, ALPHA,
                                                   DEPTH, K_EDGES, n_max=N_MAX)[1])


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    k = 0.5 * (K_EDGES[:-1] + K_EDGES[1:])
    nb = len(k); t0 = time.time()
    nb3 = N_STARS_HI / SHAPE[0] ** 3

    obs = {c: np.zeros((len(BETAS), nb)) for c in ["S_logSig", "O_logp", "O_rg"]}
    for bi, beta in enumerate(BETAS):
        acc = {c: np.zeros(nb) for c in obs}
        for r in range(N_REAL):
            key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(5), bi), r)
            s = smooth_copula_field(gaussian_random_field(SHAPE, float(beta), key), M_FID, B_FIXED, ALPHA)
            acc["S_logSig"] += measure_angular_bandpowers_2d(np.log(np.exp(s).sum(axis=2)), K_EDGES)
            cnt = np.asarray(sample_cic_counts(jnp.asarray(s), nb3, 1, jax.random.fold_in(key, 1)))
            pc = project_counts_los(cnt, DEPTH, los_axis=2).astype(float)
            acc["O_logp"] += measure_angular_bandpowers_2d(log_plus(pc, pc.mean()), K_EDGES)
            acc["O_rg"] += measure_angular_bandpowers_2d(rank_g(pc), K_EDGES)
        for c in obs:
            obs[c][bi] = acc[c] / N_REAL
        print(f"  beta={beta:.3f} ({time.time()-t0:.0f}s)")

    A = {"A_logSig": np.array([A_logSig(b) for b in BETAS]),
         "A_s": np.array([A_s(b) for b in BETAS]),
         "A_rho": np.array([A_rho(b) for b in BETAS])}

    print(f"\nTransfer-stability: T(k,beta)=E[obs]/analytic; report per-bin CV across beta = "
          f"std_beta(T)/mean_beta(T)  (small => beta-independent => clean analytic shape x fixed T)")
    print(f"  N_stars_hi={N_STARS_HI:.0e}, N_real={N_REAL}, betas={BETAS.round(2)}")
    print(f"\n  {'analytic':>9} x {'observable':<9}  {'median CV':>10}  {'max CV':>8}  {'per-bin CV (%)':>14}")
    best = None
    for an in A:
        for ob in obs:
            T = obs[ob] / A[an]                       # (nbeta, nbins)
            cv = T.std(axis=0, ddof=1) / np.abs(T.mean(axis=0))   # per-bin CV across beta
            medcv, maxcv = float(np.median(cv)) * 100, float(cv.max()) * 100
            tag = f"{an:>9} x {ob:<9}"
            print(f"  {tag}  {medcv:9.1f}%  {maxcv:7.1f}%  " + " ".join(f"{c*100:4.1f}" for c in cv))
            if best is None or medcv < best[0]:
                best = (medcv, an, ob, cv, T)
    print(f"\n  BEST (most beta-stable transfer): {best[1]} x {best[2]}  median CV={best[0]:.1f}%")
    print("  => if median CV <~ few %, mu(beta)=analytic_shape(beta) * T_fixed(k) is SBC-grade "
          "with an analytic beta-response.")

    # figure: best transfer T(k) for each beta (overlap => beta-independent)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    _, an, ob, cv, T = best
    for bi, beta in enumerate(BETAS):
        ax[0].plot(k, T[bi], marker="o", label=f"beta={beta:.2f}")
    ax[0].set_xlabel("|k|"); ax[0].set_ylabel(f"T(k)=E[{ob}]/{an}")
    ax[0].set_title(f"D05 best transfer {an} x {ob} (overlap=>beta-indep)"); ax[0].legend(fontsize=8)
    for an2 in A:
        cvs = [(obs[ob2] / A[an2]).std(axis=0, ddof=1) / np.abs((obs[ob2] / A[an2]).mean(axis=0))
               for ob2 in obs]
        ax[1].plot(k, np.min(cvs, axis=0) * 100, marker="s", label=f"{an2} (best obs)")
    ax[1].set_xlabel("|k|"); ax[1].set_ylabel("min per-bin CV across beta [%]")
    ax[1].set_title("D05: transfer beta-stability by analytic base"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "d05_transfer_stability.png")
    fig.savefig(path, dpi=140); plt.close(fig)
    print(f"\nfigure: {path}\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

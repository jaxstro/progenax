r"""D03 — the decisive gate test: does the (Gaussianizing, differentiable) log observable have an
ANALYTIC 2-pt predictor?  log_plus Gaussianizes the band-powers (D02) and is deterministic, so if its
2-pt is analytic we get analytic + Gaussian-likelihood + differentiable (the best outcome, Path A
revived via log_+ instead of rank-G).

Construct an ANALYTIC lognormal-copula predictor for the LOG of the projected density:
    xi_rho_3d   = sum_{n>=1} d_n^2/n! rho_g(r)^n           (exact BM19 density 2-pt, n_max=14)
    xi_Sigma(r) = limber_project_slab(xi_rho, depth)        (real-space projected-density 2-pt)
    <Sigma>     = depth         (rho is mean-1 -> projected mean = depth)
    xi_logSigma(r) = ln(1 + xi_Sigma(r)/<Sigma>^2)          (Coles&Jones 1991: lognormal log-2pt)
    A_logSig    = bin( fft2( xi_logSigma ) )
Compare its log-log beta-slope to:
    S_logSig : sim band-powers of log(exp(s).sum(LOS))      (log of projected density, NO shot)
    A_s      : analytic projected LOG-density (xi_s Limber)  (the field-level upper bound)
    O_logp   : sim band-powers of log_plus(project counts)   (the real observable, WITH shot)
If A_logSig tracks S_logSig across beta -> the lognormal-copula analytic predictor for the log
observable WORKS (shot still to be added). If not, the log-vs-projection non-commute is not analytic.

EXPERIMENTAL scratch; no production edits, no commits.
Run: PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync python -u \
     src/experimental/gravoturb_fdf/validation/_d03_logobs_analytic.py
"""
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

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
N_STARS = 10**5
N_BAR_3D = N_STARS / (SHAPE[0] ** 3)
BETAS = np.array([2.0, 2.5, 3.0, 3.5, 11.0 / 3.0])
N_REAL = 24
N_MAX = 14


def log_plus(N, n_bar):
    d = np.asarray(N, float) / n_bar - 1.0
    return np.where(d > 0.0, np.log1p(np.where(d > 0.0, d, 0.0)), d)


def analytic_logsigma_bp(beta, M):
    """A_logSig: analytic lognormal-copula 2-pt of the LOG of the projected density."""
    xi_rho = _xi_rho_grid(SHAPE, float(beta), M, B_FIXED, ALPHA, N_MAX, 256)
    xi_Sigma = limber_project_slab(xi_rho, DEPTH, los_axis=2)        # real-space projected-density 2-pt
    mean_Sigma = float(DEPTH)                                        # rho mean-1 -> projected mean = depth
    xi_logSigma = jnp.log1p(xi_Sigma / mean_Sigma**2)               # Coles&Jones lognormal log-2pt
    _kc, P, _nm = _angular_bandpowers_from_xi_rho_2d(xi_logSigma, K_EDGES)
    return np.asarray(P)


def analytic_logdensity_bp(beta, M):
    """A_s: analytic projected LOG-density band-powers (Limber of xi_s)."""
    rho_g = gaussian_correlation_grid(SHAPE, float(beta))
    c = bm19_hermite_coefficients(M, B_FIXED, ALPHA, N_MAX)
    xi_s = gaussianized_xi(rho_g, c)
    xi_Sigma = limber_project_slab(xi_s, DEPTH, los_axis=2)
    _kc, P, _nm = _angular_bandpowers_from_xi_rho_2d(xi_Sigma, K_EDGES)
    return np.asarray(P)


def sim_channels(beta, M, key):
    s = smooth_copula_field(gaussian_random_field(SHAPE, float(beta), key), M, B_FIXED, ALPHA)
    rho = np.exp(s)
    projSig = rho.sum(axis=2)
    cnt = np.asarray(sample_cic_counts(jnp.asarray(s), N_BAR_3D, 1, jax.random.fold_in(key, 1)))
    pc = project_counts_los(cnt, DEPTH, los_axis=2)
    return {
        "S_logSig": measure_angular_bandpowers_2d(np.log(projSig), K_EDGES),
        "O_logp": measure_angular_bandpowers_2d(log_plus(pc.astype(float), pc.mean()), K_EDGES),
    }


def slope(k, P):
    P = np.asarray(P); g = (P > 0) & np.isfinite(P)
    return np.polyfit(np.log(np.asarray(k)[g]), np.log(P[g]), 1)[0] if g.sum() > 1 else np.nan


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    k = 0.5 * (K_EDGES[:-1] + K_EDGES[1:])
    t0 = time.time()
    print(f"D03 log-observable analytic gate  shape={SHAPE} depth={DEPTH} M={M_FID} N_real={N_REAL}")

    A_logSig = {b: analytic_logsigma_bp(b, M_FID) for b in BETAS}
    A_s = {b: analytic_logdensity_bp(b, M_FID) for b in BETAS}
    A_rho = {b: np.asarray(angular_bandpowers_2d_limber(SHAPE, float(b), M_FID, B_FIXED, ALPHA,
             DEPTH, K_EDGES, n_max=N_MAX)[1]) for b in BETAS}

    sim = {c: np.zeros((len(BETAS), N_REAL, len(k))) for c in ["S_logSig", "O_logp"]}
    for bi, b in enumerate(BETAS):
        for r in range(N_REAL):
            d = sim_channels(b, M_FID, jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(99), bi), r))
            for c in sim:
                sim[c][bi, r] = d[c]
        print(f"  beta={b:.3f} ({time.time()-t0:.0f}s)")

    def gain(sl):
        return np.polyfit(BETAS, np.array(sl), 1)[0]

    sl_AlogSig = [slope(k, A_logSig[b]) for b in BETAS]
    sl_As = [slope(k, A_s[b]) for b in BETAS]
    sl_Arho = [slope(k, A_rho[b]) for b in BETAS]
    sl_SlogSig = [slope(k, sim["S_logSig"][bi].mean(axis=0)) for bi in range(len(BETAS))]
    sl_Ologp = [slope(k, sim["O_logp"][bi].mean(axis=0)) for bi in range(len(BETAS))]

    print(f"\n  {'channel':<26}" + " ".join(f"b={b:5.2f}" for b in BETAS) + "   gain")
    for name, sl in [("A_logSig (analytic logSig)", sl_AlogSig), ("S_logSig (sim log proj-dens)", sl_SlogSig),
                     ("A_s (analytic proj-logdens)", sl_As), ("A_rho(analytic proj-dens)", sl_Arho),
                     ("O_logp (sim log+ counts)", sl_Ologp)]:
        print(f"  {name:<26}" + " ".join(f"{x:7.3f}" for x in sl) + f"   {gain(sl):6.3f}")

    print("\n  bin-by-bin: A_logSig vs S_logSig (analytic lognormal-copula log predictor vs sim):")
    print(f"  {'beta':>5}  {'max|%err|':>10}  {'gain ratio A_logSig/S_logSig':>28}")
    for bi, b in enumerate(BETAS):
        E = sim["S_logSig"][bi].mean(axis=0)
        pct = 100.0 * np.abs(A_logSig[b] - E) / np.abs(E)
        print(f"  {b:5.2f}  {pct.max():9.1f}%")
    print(f"\n  transfer gains: A_logSig {gain(sl_AlogSig):.3f}  S_logSig {gain(sl_SlogSig):.3f}  "
          f"O_logp {gain(sl_Ologp):.3f}  A_s {gain(sl_As):.3f}")
    print(f"  => A_logSig predicts S_logSig gain to {gain(sl_AlogSig)/gain(sl_SlogSig):.3f} "
          f"(==1 ideal); O_logp/S_logSig = {gain(sl_Ologp)/gain(sl_SlogSig):.3f} (shot gap)")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for name, sl, st in [("A_logSig", sl_AlogSig, "--"), ("S_logSig", sl_SlogSig, "-"),
                         ("A_s", sl_As, "--"), ("O_logp", sl_Ologp, "-"), ("A_rho", sl_Arho, ":")]:
        ax[0].plot(BETAS, sl, st, marker="o", label=name)
    ax[0].set_xlabel("beta"); ax[0].set_ylabel("log-log slope"); ax[0].legend(fontsize=8)
    ax[0].set_title("D03: slope vs beta (analytic dashed)")
    bi3 = int(np.argmin(np.abs(BETAS - 3.0)))
    ax[1].loglog(k, A_logSig[BETAS[bi3]], "--o", label="A_logSig (analytic)")
    ax[1].loglog(k, sim["S_logSig"][bi3].mean(axis=0), "-s", label="S_logSig (sim)")
    ax[1].loglog(k, sim["O_logp"][bi3].mean(axis=0), "-^", label="O_logp (obs+shot)")
    ax[1].set_xlabel("|k|"); ax[1].set_ylabel("band-power"); ax[1].legend(fontsize=8)
    ax[1].set_title("D03: band-powers at beta=3")
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "d03_logobs_analytic.png")
    fig.savefig(path, dpi=140); plt.close(fig)
    print(f"\nfigure: {path}\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

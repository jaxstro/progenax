r"""Focused end-to-end (beta, M) GRID posterior for gravoturb_fdf 2D inference.

The "does it actually recover beta" proof -- done the RIGHT way for a 2-parameter problem:
a direct grid evaluation of the log-posterior (exact, fast, no MCMC pathology). NUTS on this
2-D, sharply-curved, box-bounded posterior diverged catastrophically (R-hat~3, ESS~5, >1000
divergences from the hard -inf prior box); for 2 params a grid is exact and ~15 s. (NUTS, with a
proper logit reparameterization, is reserved for the higher-dim (M, alpha, beta) SBC later.)

Pipeline (all validated upstream):
  - mock "observed" cluster from the Option-A POINTWISE map ``smooth_copula_field`` (the
    generative model the analytic forward model is exact for).
  - FIXED, truth-INDEPENDENT fiducial mock covariance (computed ONCE at theta_fid).
  - forward model ``angular_bandpowers_2d_limber`` (exact BM19 density 2-pt -> Limber -> FFT),
    vmapped over a (M, beta) grid; + Poisson shot for the count observable.
  - log-posterior on the grid = Gaussian loglike (fixed mock precision) + flat (log-uniform) prior;
    marginals, posterior mean/sigma, truth coverage, (beta, M) correlation.
  - jaxstroviz figures: band-power fit + residual (plot_overlay_residual); 2-D posterior + marginals.

NO production-code edits beyond this scratch file. NO commits.
Run: PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync python -u \
     src/experimental/gravoturb_fdf/validation/_v2b_grid_posterior.py
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
    add_poisson_shot,
    angular_bandpowers_2d_limber,
    mock_covariance,
    mock_precision,
)
from gravoturb_fdf.inference.priors import BM19Prior
from gravoturb_fdf.validation.measure import (
    measure_angular_bandpowers_2d,
    project_counts_los,
    smooth_copula_field,
)
from jaxstroviz.experimental.plots.residual import plot_overlay_residual

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")

# ----------------------------- config -----------------------------
SHAPE = (64, 64, 64)
DEPTH = 64
B_FIXED = 0.4
M_TRUE = 8.0
BETA_TRUE = 3.0
ALPHA_TRUE = 2.5  # fixed (alpha depth-gated)
K_EDGES = np.linspace(1.0, 28.0, 11)  # signal band on the 64^3 grid (Nyquist 32)

N_STARS = 10**5
N_BAR_SKY = N_STARS / (SHAPE[0] ** 2)
N_BAR_3D = N_STARS / (SHAPE[0] ** 3)

M_FID, BETA_FID = 8.0, 3.0  # FIXED fiducial for the (truth-independent) covariance
N_REAL_COV = 64

PRIOR = BM19Prior()
M_LO, M_HI = PRIOR.m_range
BETA_LO, BETA_HI = PRIOR.beta_range

# Posterior grid (dense enough for smooth marginals): M log-spaced, beta linear.
N_GRID = 80
M_GRID = np.geomspace(M_LO, M_HI, N_GRID)
BETA_GRID = np.linspace(BETA_LO, BETA_HI, N_GRID)

SIGMA_BETA_FORECAST = 0.22


# ----------------------------- forward model (vmapped over the grid) -----------------------------
def _predict_clustering(M, beta):
    _kc, P, _nm = angular_bandpowers_2d_limber(SHAPE, beta, M, B_FIXED, ALPHA_TRUE, DEPTH, K_EDGES)
    return P


# vmap over a flat list of (M, beta) pairs; JIT the whole grid evaluation.
@jax.jit
def _grid_clustering(M_flat, beta_flat):
    return jax.vmap(_predict_clustering)(M_flat, beta_flat)  # (Ngrid^2, n_bins)


# ----------------------------- mock data (pointwise map) -----------------------------
def measure_density(g, M=M_TRUE):
    s = smooth_copula_field(g, M, B_FIXED, ALPHA_TRUE)
    return measure_angular_bandpowers_2d(np.exp(s).sum(axis=2), K_EDGES)


def measure_count(g, count_key, M=M_TRUE):
    s = smooth_copula_field(g, M, B_FIXED, ALPHA_TRUE)
    cnt = np.asarray(sample_cic_counts(jnp.asarray(s), N_BAR_3D, 1, count_key))
    return measure_angular_bandpowers_2d(project_counts_los(cnt, DEPTH, los_axis=2), K_EDGES)


def fiducial_covariance(observable, base_seed):
    rows = []
    for r in range(N_REAL_COV):
        g = gaussian_random_field(SHAPE, BETA_FID, jax.random.fold_in(jax.random.PRNGKey(base_seed), r))
        if observable == "density":
            rows.append(measure_density(g, M=M_FID))
        else:
            rows.append(measure_count(g, jax.random.fold_in(jax.random.PRNGKey(base_seed + 7), r), M=M_FID))
    rows = np.asarray(rows)
    return rows, mock_covariance(rows), mock_precision(rows)


# ----------------------------- grid posterior -----------------------------
def grid_posterior(name, data, precision, shot):
    """Evaluate the log-posterior on the (M, beta) grid; return marginal summaries + the grid."""
    MM, BB = np.meshgrid(M_GRID, BETA_GRID, indexing="ij")  # (NM, NB)
    M_flat = jnp.asarray(MM.ravel())
    beta_flat = jnp.asarray(BB.ravel())

    t0 = time.time()
    pred = np.asarray(_grid_clustering(M_flat, beta_flat))  # (Ngrid^2, n_bins)
    if shot:
        pred = np.asarray(add_poisson_shot(jnp.asarray(pred), N_BAR_SKY, DEPTH))
    wall = time.time() - t0

    resid = pred - data[None, :]                       # (Ngrid^2, n_bins)
    chi2 = np.einsum("gi,ij,gj->g", resid, precision, resid)
    logpost = (-0.5 * chi2).reshape(MM.shape)          # flat (log-uniform) prior in the box -> constant
    P = np.exp(logpost - logpost.max())
    P /= P.sum()

    # marginals + moments (grid is dense; trapezoid-free discrete moments on the normalized P)
    P_M = P.sum(axis=1)      # over beta -> P(M)
    P_B = P.sum(axis=0)      # over M    -> P(beta)
    M_mean = float((P_M * M_GRID).sum())
    M_std = float(np.sqrt((P_M * (M_GRID - M_mean) ** 2).sum()))
    B_mean = float((P_B * BETA_GRID).sum())
    B_std = float(np.sqrt((P_B * (BETA_GRID - B_mean) ** 2).sum()))
    # correlation
    EMB = float((P * MM * BB).sum())
    corr = (EMB - M_mean * B_mean) / (M_std * B_std)

    # MAP for the band-power overlay
    gi = np.unravel_index(np.argmax(logpost), logpost.shape)
    M_map, B_map = float(M_GRID[gi[0]]), float(BETA_GRID[gi[1]])
    pred_map = np.asarray(_predict_clustering(jnp.asarray(M_map), jnp.asarray(B_map)))
    if shot:
        pred_map = np.asarray(add_poisson_shot(jnp.asarray(pred_map), N_BAR_SKY, DEPTH))

    return {
        "name": name, "P": P, "P_M": P_M, "P_B": P_B,
        "M_mean": M_mean, "M_std": M_std, "B_mean": B_mean, "B_std": B_std,
        "corr": corr, "M_map": M_map, "B_map": B_map,
        "data": np.asarray(data), "pred_map": pred_map, "wall": wall,
    }


def cover(mean, std, truth):
    z = abs(mean - truth) / std
    return z, ("1sig" if z <= 1 else ("2sig" if z <= 2 else "OUT"))


def report(res):
    bz, bc = cover(res["B_mean"], res["B_std"], BETA_TRUE)
    mz, mc = cover(res["M_mean"], res["M_std"], M_TRUE)
    print(f"\n{'='*72}\n  OBSERVABLE: {res['name']}   (grid eval {res['wall']:.1f}s)\n{'='*72}")
    print(f"  beta = {res['B_mean']:.3f} +/- {res['B_std']:.3f}  (truth {BETA_TRUE}; |z|={bz:.2f} {bc})")
    print(f"  M    = {res['M_mean']:.3f} +/- {res['M_std']:.3f}  (truth {M_TRUE}; |z|={mz:.2f} {mc})")
    print(f"  recovered sigma(beta) = {res['B_std']:.3f}  vs ~{SIGMA_BETA_FORECAST} forecast "
          f"(ratio {res['B_std']/SIGMA_BETA_FORECAST:.2f}x)")
    print(f"  (beta, M) posterior correlation = {res['corr']:+.2f}")
    return {"name": res["name"], "B_mean": res["B_mean"], "B_std": res["B_std"], "B_z": bz, "B_c": bc,
            "M_mean": res["M_mean"], "M_std": res["M_std"], "M_z": mz, "M_c": mc, "corr": res["corr"]}


# ----------------------------- figures -----------------------------
def fig_posterior(kc, results):
    nobs = len(results)
    fig, axes = plt.subplots(2, nobs, figsize=(6.0 * nobs, 9.0), squeeze=False,
                             gridspec_kw={"height_ratios": [3, 2]})
    for j, res in enumerate(results):
        ax = axes[0, j]
        im = ax.pcolormesh(BETA_GRID, M_GRID, res["P"], cmap="viridis", shading="auto")
        fig.colorbar(im, ax=ax, label="posterior density")
        ax.axvline(BETA_TRUE, color="w", ls="--", lw=1.2)
        ax.axhline(M_TRUE, color="w", ls="--", lw=1.2)
        ax.plot(res["B_mean"], res["M_mean"], "r*", ms=14, label="post. mean")
        ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"$\mathcal{M}$")
        ax.set_title(f"{res['name']}: $\\beta={res['B_mean']:.2f}\\pm{res['B_std']:.2f}$ "
                     f"(truth {BETA_TRUE}); $\\sigma(\\beta)/0.22={res['B_std']/SIGMA_BETA_FORECAST:.2f}$; "
                     f"$r_{{\\beta\\mathcal{{M}}}}={res['corr']:+.2f}$")
        ax.legend(frameon=False, fontsize=9, loc="upper right")
        # beta marginal under the heatmap
        axb = axes[1, j]
        axb.plot(BETA_GRID, res["P_B"] / res["P_B"].max(), color="#1b3b6f", lw=2)
        axb.axvline(BETA_TRUE, color="k", ls="--", lw=1, label="truth")
        axb.axvspan(res["B_mean"] - res["B_std"], res["B_mean"] + res["B_std"], color="#c1543c", alpha=0.2,
                    label=r"$\pm1\sigma$")
        axb.set_xlabel(r"$\beta$"); axb.set_ylabel(r"$P(\beta)$ (norm.)")
        axb.legend(frameon=False, fontsize=9)
    fig.suptitle("v2b end-to-end GRID posterior (truth = dashed white/black)", y=0.99)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "v2b_posterior.png")
    fig.savefig(path, dpi=140); plt.close(fig)
    return path


def fig_bandpower_fit(kc, results):
    nobs = len(results)
    fig, axes = plt.subplots(2, nobs, figsize=(6.0 * nobs, 7.0), squeeze=False, sharex="col",
                             gridspec_kw={"height_ratios": [3, 1]})
    for j, res in enumerate(results):
        at, ab = axes[0, j], axes[1, j]
        d, m = res["data"], res["pred_map"]
        at.loglog(kc, d, "o-", color="#1b3b6f", label="measured (data)", ms=5)
        at.loglog(kc, m, "s--", color="#c1543c", label="model @ MAP", ms=4)
        at.set_ylabel(r"$P(k)$ band-power"); at.set_title(f"{res['name']} observable")
        at.legend(frameon=False, fontsize=9); at.grid(alpha=0.3, which="both")
        plot_overlay_residual(ab, kc, d, m, mode="frac")
        ab.set_xlabel(r"$|k|$"); ab.set_xscale("log"); ab.grid(alpha=0.3, which="both")
    fig.suptitle("v2b end-to-end: band-power fit (measured vs model @ MAP)", y=0.99)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "v2b_bandpower_fit.png")
    fig.savefig(path, dpi=140); plt.close(fig)
    return path


# ----------------------------- main -----------------------------
def main():
    print("v2b end-to-end (beta, M) GRID posterior")
    print(f"  shape={SHAPE} depth={DEPTH} truth=(M={M_TRUE}, beta={BETA_TRUE}) alpha={ALPHA_TRUE} (fixed)")
    print(f"  grid {N_GRID}x{N_GRID}  M in [{M_LO},{M_HI}] (log), beta in [{BETA_LO},{BETA_HI:.3f}] (lin)")
    print(f"  N_stars={N_STARS} -> n_bar_sky={N_BAR_SKY:.2f}; theta_fid=(M={M_FID},beta={BETA_FID}) [FIXED]")

    data_key = jax.random.PRNGKey(2024)
    g_star = gaussian_random_field(SHAPE, BETA_TRUE, data_key)
    d_density = measure_density(g_star, M=M_TRUE)
    d_count = measure_count(g_star, jax.random.PRNGKey(2025), M=M_TRUE)
    kc = np.asarray(angular_bandpowers_2d_limber(
        SHAPE, BETA_TRUE, M_TRUE, B_FIXED, ALPHA_TRUE, DEPTH, K_EDGES)[0])

    print(f"\nBuilding fixed fiducial covariances ({N_REAL_COV} realizations)...")
    t0 = time.time()
    _, _, prec_d = fiducial_covariance("density", base_seed=5000)
    _, _, prec_c = fiducial_covariance("count", base_seed=6000)
    print(f"  built in {time.time()-t0:.1f}s")

    res_d = grid_posterior("density", d_density, prec_d, shot=False)
    res_c = grid_posterior("count", d_count, prec_c, shot=True)
    s_d = report(res_d)
    s_c = report(res_c)

    print("\nMaking figures...")
    p1 = fig_posterior(kc, [res_d, res_c])
    p2 = fig_bandpower_fit(kc, [res_d, res_c])
    print(f"  {p1}\n  {p2}")

    print(f"\n{'#'*72}\n  FINAL SUMMARY (grid posterior; exact for 2 params)\n{'#'*72}")
    for s in (s_d, s_c):
        print(f"  [{s['name']:7s}] beta={s['B_mean']:.3f}+/-{s['B_std']:.3f} (truth 3.0, |z|={s['B_z']:.2f} {s['B_c']})"
              f"  M={s['M_mean']:.2f}+/-{s['M_std']:.2f} (truth 8.0, |z|={s['M_z']:.2f} {s['M_c']})"
              f"  sig(b)/0.22={s['B_std']/SIGMA_BETA_FORECAST:.2f}  r(b,M)={s['corr']:+.2f}")
    return s_d, s_c


if __name__ == "__main__":
    main()

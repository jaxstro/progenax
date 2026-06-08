r"""Gradient-based HMC (NUTS) for the (beta, M) 2D inference, FIXED with a logit reparameterization,
validated against the brute-force grid posterior (_v2b).

WHY: the first NUTS attempt used a hard ``-inf`` prior box (jnp.where(in_box, loglike, -inf)).
NUTS slammed into the beta=11/3 boundary -> 1000+ divergences, R-hat~3, ESS~5 (garbage). The fix is
NOT to abandon gradient-based inference (the project's whole point + the only thing that scales to the
real multi-param / hierarchical problem) -- it is to remove the hard boundary by sampling in a fully
UNCONSTRAINED space via a logit map, so the geometry is smooth everywhere.

REPARAMETERIZATION (per param with a log-uniform prior on [lo, hi], i.e. uniform on y=log x):
  y = log lo + (log hi - log lo) * sigmoid(z),   x = exp(y),   z in (-inf, inf).
  The target density in z is p(x)*|dx/dz| = sigmoid'(z)  (the 1/x log-uniform prior, the y-Jacobian
  L, and the exp-Jacobian all cancel to leave the logistic Jacobian). So
  log p(z) = loglike(theta(z)) + sum_i log sigmoid'(z_i),   log sigmoid'(z) = -softplus(-z)-softplus(z).
  Smooth, finite everywhere -- NO hard boundary, no boundary divergences.

VALIDATION: the logit-NUTS posterior (mean, sigma) is compared to the grid posterior (the exact
reference for 2 params). Agreement => working, validated, gradient-based inference that scales.

NO production-code edits beyond this scratch file. NO commits.
Run: PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync python -u \
     src/experimental/gravoturb_fdf/validation/_v2c_hmc_logit.py
"""
import os
import time

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from jax.nn import sigmoid, softplus

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
from jaxstroviz.experimental.plots.hmc import plot_hmc_rank, plot_hmc_trace

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")

# ----------------------------- config -----------------------------
SHAPE = (48, 48, 48)
DEPTH = 48
B_FIXED, M_TRUE, BETA_TRUE, ALPHA_TRUE = 0.4, 8.0, 3.0, 2.5
K_EDGES = np.linspace(1.0, 20.0, 9)
N_STARS = 10**5
N_BAR_SKY = N_STARS / (SHAPE[0] ** 2)
N_BAR_3D = N_STARS / (SHAPE[0] ** 3)
M_FID, BETA_FID, N_REAL_COV = 8.0, 3.0, 64

PRIOR = BM19Prior()
M_LO, M_HI = PRIOR.m_range
BETA_LO, BETA_HI = PRIOR.beta_range
LOGM_LO, LM = np.log(M_LO), np.log(M_HI) - np.log(M_LO)
LOGB_LO, LB = np.log(BETA_LO), np.log(BETA_HI) - np.log(BETA_LO)

N_WARMUP, N_SAMPLES, N_CHAINS, MAX_DOUBLINGS = 400, 800, 4, 7
N_GRID = 60  # grid reference resolution
SIGMA_BETA_FORECAST = 0.22


# ----------------------------- forward model -----------------------------
def predict_clustering(M, beta):
    _kc, P, _nm = angular_bandpowers_2d_limber(SHAPE, beta, M, B_FIXED, ALPHA_TRUE, DEPTH, K_EDGES)
    return P


# ----------------------------- logit reparam -----------------------------
def z_to_theta(z):
    """Unconstrained z=(zM, zB) -> (M, beta) via logit on the log-uniform boxes."""
    M = jnp.exp(LOGM_LO + LM * sigmoid(z[0]))
    beta = jnp.exp(LOGB_LO + LB * sigmoid(z[1]))
    return M, beta


def log_prior_jac(z):
    """sum log sigmoid'(z_i) = -softplus(-z)-softplus(z) (logistic Jacobian; see module docstring)."""
    return -(softplus(-z[0]) + softplus(z[0])) - (softplus(-z[1]) + softplus(z[1]))


def make_logdensity(data, precision, shot):
    data_j, prec_j = jnp.asarray(data), jnp.asarray(precision)

    def ld(z):
        M, beta = z_to_theta(z)
        pred = predict_clustering(M, beta)
        if shot:
            pred = add_poisson_shot(pred, N_BAR_SKY, DEPTH)
        resid = pred - data_j
        return -0.5 * resid @ (prec_j @ resid) + log_prior_jac(z)

    return ld


# ----------------------------- capped multi-chain NUTS -----------------------------
def local_nuts(ld, init, key, n_warmup, n_samples, n_chains, max_doublings):
    chain_keys = jax.random.split(key, n_chains)

    def run_one(ck):
        dk, wk, sk = jax.random.split(ck, 3)
        init0 = init + 0.7 * jax.random.normal(dk, jnp.shape(init))  # disperse in unconstrained z
        warmup = blackjax.window_adaptation(blackjax.nuts, ld, max_num_doublings=max_doublings)
        (state, params), _ = warmup.run(wk, init0, num_steps=n_warmup)
        kernel = blackjax.nuts(ld, **params)

        def step(state, k):
            state, info = kernel.step(k, state)
            return state, (state.position, info.is_divergent, info.num_trajectory_expansions)

        _, (pos, div, depth) = jax.lax.scan(step, state, jax.random.split(sk, n_samples))
        return pos, div, depth

    pos, div, depth = jax.vmap(run_one)(chain_keys)
    return {"positions": pos, "divergences": div, "depth": depth}


# ----------------------------- mock data + covariance -----------------------------
def measure_density(g, M=M_TRUE):
    s = smooth_copula_field(g, M, B_FIXED, ALPHA_TRUE)
    return measure_angular_bandpowers_2d(np.exp(s).sum(axis=2), K_EDGES)


def measure_count(g, key, M=M_TRUE):
    s = smooth_copula_field(g, M, B_FIXED, ALPHA_TRUE)
    cnt = np.asarray(sample_cic_counts(jnp.asarray(s), N_BAR_3D, 1, key))
    return measure_angular_bandpowers_2d(project_counts_los(cnt, DEPTH, los_axis=2), K_EDGES)


def fiducial_precision(observable, base_seed):
    rows = []
    for r in range(N_REAL_COV):
        g = gaussian_random_field(SHAPE, BETA_FID, jax.random.fold_in(jax.random.PRNGKey(base_seed), r))
        rows.append(measure_density(g, M=M_FID) if observable == "density"
                    else measure_count(g, jax.random.fold_in(jax.random.PRNGKey(base_seed + 7), r), M=M_FID))
    return mock_precision(np.asarray(rows))


# ----------------------------- grid reference -----------------------------
@jax.jit
def _grid_pred(M_flat, beta_flat):
    return jax.vmap(predict_clustering)(M_flat, beta_flat)


def grid_posterior(data, precision, shot):
    M_grid = np.geomspace(M_LO, M_HI, N_GRID)
    beta_grid = np.linspace(BETA_LO, BETA_HI, N_GRID)
    MM, BB = np.meshgrid(M_grid, beta_grid, indexing="ij")
    pred = np.asarray(_grid_pred(jnp.asarray(MM.ravel()), jnp.asarray(BB.ravel())))
    if shot:
        pred = np.asarray(add_poisson_shot(jnp.asarray(pred), N_BAR_SKY, DEPTH))
    resid = pred - data[None, :]
    logp = (-0.5 * np.einsum("gi,ij,gj->g", resid, precision, resid)).reshape(MM.shape)
    P = np.exp(logp - logp.max()); P /= P.sum()
    P_M, P_B = P.sum(axis=1), P.sum(axis=0)
    Bm = float((P_B * beta_grid).sum()); Bs = float(np.sqrt((P_B * (beta_grid - Bm) ** 2).sum()))
    Mm = float((P_M * M_grid).sum()); Ms = float(np.sqrt((P_M * (M_grid - Mm) ** 2).sum()))
    return {"M_grid": M_grid, "beta_grid": beta_grid, "P": P, "Bm": Bm, "Bs": Bs, "Mm": Mm, "Ms": Ms}


# ----------------------------- run one observable -----------------------------
def run_observable(name, data, precision, shot, key):
    ld = make_logdensity(data, precision, shot)
    t0 = time.time()
    out = local_nuts(ld, jnp.array([0.0, 0.0]), key, N_WARMUP, N_SAMPLES, N_CHAINS, MAX_DOUBLINGS)
    wall = time.time() - t0
    z = np.asarray(out["positions"])  # (nchains, nsamp, 2)
    # transform to (M, beta)
    M = np.exp(LOGM_LO + LM * (1 / (1 + np.exp(-z[:, :, 0]))))
    beta = np.exp(LOGB_LO + LB * (1 / (1 + np.exp(-z[:, :, 1]))))
    import arviz as az
    rb, eb = float(az.rhat(beta)), float(az.ess(beta))
    rm, em = float(az.rhat(M)), float(az.ess(M))
    ndiv = int(np.asarray(out["divergences"]).sum())
    max_depth = int(np.asarray(out["depth"]).max())
    grid = grid_posterior(data, precision, shot)
    return {
        "name": name, "M": M, "beta": beta,
        "Bm": float(beta.mean()), "Bs": float(beta.std(ddof=1)),
        "Mm": float(M.mean()), "Ms": float(M.std(ddof=1)),
        "corr": float(np.corrcoef(M.ravel(), beta.ravel())[0, 1]),
        "rb": rb, "eb": eb, "rm": rm, "em": em, "ndiv": ndiv, "max_depth": max_depth,
        "wall": wall, "grid": grid,
    }


def report(r):
    bz = abs(r["Bm"] - BETA_TRUE) / r["Bs"]
    mz = abs(r["Mm"] - M_TRUE) / r["Ms"]
    g = r["grid"]
    print(f"\n{'='*76}\n  OBSERVABLE: {r['name']}   (NUTS {r['wall']:.1f}s)\n{'='*76}")
    print(f"  HMC  beta = {r['Bm']:.3f} +/- {r['Bs']:.3f}  (truth {BETA_TRUE}; |z|={bz:.2f})")
    print(f"  GRID beta = {g['Bm']:.3f} +/- {g['Bs']:.3f}   <-- exact reference")
    print(f"       agreement: d(mean)={abs(r['Bm']-g['Bm']):.3f}  sigma ratio={r['Bs']/g['Bs']:.2f}")
    print(f"  HMC  M    = {r['Mm']:.3f} +/- {r['Ms']:.3f}  (truth {M_TRUE}; |z|={mz:.2f})")
    print(f"  GRID M    = {g['Mm']:.3f} +/- {g['Ms']:.3f}")
    print(f"  recovered sigma(beta)={r['Bs']:.3f} vs ~{SIGMA_BETA_FORECAST} forecast "
          f"(ratio {r['Bs']/SIGMA_BETA_FORECAST:.2f}x);  corr(b,M)={r['corr']:+.2f}")
    print(f"  CONVERGENCE: R-hat(beta,M)=({r['rb']:.3f},{r['rm']:.3f}) [want<1.01]  "
          f"ESS(beta,M)=({r['eb']:.0f},{r['em']:.0f})  div={r['ndiv']}  max_tree_depth={r['max_depth']}")


# ----------------------------- figures -----------------------------
def fig_diagnostics(results):
    nobs = len(results)
    fig, axes = plt.subplots(2, 2 * nobs, figsize=(6.0 * nobs, 6.0), squeeze=False)
    for j, r in enumerate(results):
        plot_hmc_trace(axes[0, 2 * j], r["beta"], param_name=r"$\beta$")
        axes[0, 2 * j].axhline(BETA_TRUE, color="k", ls="--", lw=1)
        axes[0, 2 * j].set_title(f"{r['name']}: $\\beta$ trace")
        plot_hmc_rank(axes[0, 2 * j + 1], r["beta"], param_name=r"$\beta$")
        plot_hmc_trace(axes[1, 2 * j], r["M"], param_name=r"$\mathcal{M}$")
        axes[1, 2 * j].axhline(M_TRUE, color="k", ls="--", lw=1)
        axes[1, 2 * j].set_title(f"{r['name']}: $\\mathcal{{M}}$ trace")
        plot_hmc_rank(axes[1, 2 * j + 1], r["M"], param_name=r"$\mathcal{M}$")
    fig.suptitle("v2c logit-NUTS diagnostics (traces should mix; rank hists flat+overlapping)", y=1.0)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "v2c_hmc_diagnostics.png")
    fig.savefig(path, dpi=140); plt.close(fig)
    return path


def fig_posterior_overlay(results):
    nobs = len(results)
    fig, axes = plt.subplots(1, nobs, figsize=(6.0 * nobs, 5.5), squeeze=False)
    for j, r in enumerate(results):
        ax = axes[0, j]; g = r["grid"]
        ax.pcolormesh(g["beta_grid"], g["M_grid"], g["P"], cmap="Greys", shading="auto")
        ax.plot(r["beta"].ravel()[::5], r["M"].ravel()[::5], ".", color="#c1543c", ms=1.5, alpha=0.3,
                label="HMC samples")
        ax.axvline(BETA_TRUE, color="b", ls="--", lw=1.2); ax.axhline(M_TRUE, color="b", ls="--", lw=1.2)
        ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"$\mathcal{M}$")
        ax.set_title(f"{r['name']}: HMC (pts) vs GRID (shade)\n"
                     f"$\\beta_{{HMC}}={r['Bm']:.2f}\\pm{r['Bs']:.2f}$ vs $\\beta_{{grid}}={g['Bm']:.2f}\\pm{g['Bs']:.2f}$")
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("v2c: gradient-based HMC validated against the exact grid posterior", y=1.0)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "v2c_hmc_vs_grid.png")
    fig.savefig(path, dpi=140); plt.close(fig)
    return path


def main():
    print("v2c logit-NUTS (gradient-based) vs grid reference")
    print(f"  shape={SHAPE} truth=(M={M_TRUE},beta={BETA_TRUE}) alpha={ALPHA_TRUE} (fixed); "
          f"n_bar_sky={N_BAR_SKY:.1f}")
    print(f"  NUTS: {N_CHAINS} chains x ({N_WARMUP} warmup + {N_SAMPLES}) cap 2^{MAX_DOUBLINGS}; "
          f"logit unconstrained z")
    data_key = jax.random.PRNGKey(2024)
    g_star = gaussian_random_field(SHAPE, BETA_TRUE, data_key)
    d_density = measure_density(g_star, M=M_TRUE)
    d_count = measure_count(g_star, jax.random.PRNGKey(2025), M=M_TRUE)

    print(f"\nBuilding fixed fiducial precisions ({N_REAL_COV} realizations)...")
    t0 = time.time()
    prec_d = fiducial_precision("density", 5000)
    prec_c = fiducial_precision("count", 6000)
    print(f"  built in {time.time()-t0:.1f}s")

    print("\nRunning logit-NUTS (density)...")
    r_d = run_observable("density", d_density, prec_d, False, jax.random.PRNGKey(11))
    report(r_d)
    print("\nRunning logit-NUTS (count)...")
    r_c = run_observable("count", d_count, prec_c, True, jax.random.PRNGKey(22))
    report(r_c)

    print("\nMaking figures...")
    p1 = fig_diagnostics([r_d, r_c])
    p2 = fig_posterior_overlay([r_d, r_c])
    print(f"  {p1}\n  {p2}")

    print(f"\n{'#'*76}\n  FINAL: gradient-based HMC (logit) vs exact grid\n{'#'*76}")
    for r in (r_d, r_c):
        g = r["grid"]
        ok = (r["rb"] < 1.01) and (r["rm"] < 1.01) and (r["ndiv"] == 0)
        print(f"  [{r['name']:7s}] HMC beta={r['Bm']:.3f}+/-{r['Bs']:.3f} | grid {g['Bm']:.3f}+/-{g['Bs']:.3f} "
              f"| Rhat=({r['rb']:.3f},{r['rm']:.3f}) ESS=({r['eb']:.0f},{r['em']:.0f}) div={r['ndiv']} "
              f"-> {'CLEAN' if ok else 'CHECK'}")


if __name__ == "__main__":
    main()

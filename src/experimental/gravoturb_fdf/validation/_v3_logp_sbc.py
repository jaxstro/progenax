r"""V3 — single-beta SBC of the ANALYTIC log+ forward model (Phase-2 gate).

Forward model: mu(beta) = interp(A_s_table; beta) * T_fid(k)  (beta-response purely analytic; the
table emulates the SMOOTH noise-free A_s, so the slope is preserved -- the v2h flaw avoided). Gaussian
likelihood with FIXED-fiducial Hartlap precision (truth-independent -> SBC-valid). logit-reparam NUTS.
beta-only: M, b, alpha fixed at fiducial in BOTH generation and inference.

SBC-validity contract: T_fid + precision computed ONCE at a fiducial theta independent of each trial's
truth; the generative statistic == the statistic the model predicts the mean of (log+ band-powers).

HEAD-TO-HEAD (Anna: keep/verify both): run the IDENTICAL pipeline with the rank-G observable. Because
rank-G's transfer is beta-DEPENDENT (Phase-0 D05: 39% CV) while log+'s is beta-stable (~5%), the
fixed-T analytic model should calibrate beta for log+ and MIS-calibrate for rank-G -- demonstrating
why log+ is the right observable.

EXPERIMENTAL scratch; no production edits, no commits.
Run: PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync python -u \
     src/experimental/gravoturb_fdf/validation/_v3_logp_sbc.py
"""
import os
import time

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from jax.nn import sigmoid, softplus
from scipy import special

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gravoturb_fdf.field.field import gaussian_random_field
from gravoturb_fdf.field.sampling import sample_cic_counts
from gravoturb_fdf.inference.covariance import mock_precision
from gravoturb_fdf.inference.projected_logp import (
    analytic_logdensity_bandpowers,
    calibrate_transfer,
    interp_logp_bandpowers,
    logp_loglike,
)
from gravoturb_fdf.validation.measure import (
    measure_angular_bandpowers_2d,
    project_counts_los,
    smooth_copula_field,
)
from jaxstroviz.experimental.analysis.sbc import compute_sbc_ecdf_diff, compute_sbc_rank_histogram
from jaxstroviz.experimental.plots.sbc import plot_sbc_ecdf_diff, plot_sbc_rank_histogram

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
SHAPE, DEPTH = (64, 64, 64), 64
B_FIXED, ALPHA, M_FID = 0.4, 2.5, 8.0
BETA_FID = 2.7
K_EDGES = np.linspace(2.0, 28.0, 10)          # drop the weak lowest-k bin (D05); 9 bins
NB = len(K_EDGES) - 1
N_STARS = 10**6                                # high-N first (D04: analytic model directly valid)
N_BAR_3D = N_STARS / (SHAPE[0] ** 3)
BETA_LO, BETA_HI = 2.0, 11.0 / 3.0
LOGB_LO, LB = np.log(BETA_LO), np.log(BETA_HI) - np.log(BETA_LO)
N_NODES, N_REAL_FID = 96, 64
N_MAX, N_QUAD = 14, 256
K_TRIALS, N_WARMUP, N_SAMPLES, N_CHAINS, MAX_DOUBLINGS, L_THIN = 128, 400, 600, 4, 8, 100


def rank_g(m):
    f = np.asarray(m, float).ravel(); r = np.argsort(np.argsort(f)); u = (r + 0.5) / f.size
    return (np.sqrt(2.0) * special.erfinv(2.0 * u - 1.0)).reshape(m.shape)


def log_plus(N, nb):
    d = np.asarray(N, float) / nb - 1.0
    return np.where(d > 0.0, np.log1p(np.where(d > 0.0, d, 0.0)), d)


def gen_count_map(beta, key):
    s = smooth_copula_field(gaussian_random_field(SHAPE, float(beta), key), M_FID, B_FIXED, ALPHA)
    cnt = np.asarray(sample_cic_counts(jnp.asarray(s), N_BAR_3D, 1, jax.random.fold_in(key, 1)))
    return project_counts_los(cnt, DEPTH, los_axis=2).astype(float)


def observable(name, cmap):
    if name == "logp":
        return measure_angular_bandpowers_2d(log_plus(cmap, cmap.mean()), K_EDGES)
    return measure_angular_bandpowers_2d(rank_g(cmap), K_EDGES)


def beta_of_z(z):
    return jnp.exp(LOGB_LO + LB * sigmoid(z))


def make_infer(beta_nodes, table, transfer, precision):
    prec = jnp.asarray(precision)
    T = jnp.asarray(transfer)

    def ld(z, data):
        beta = beta_of_z(z[0])
        mu = interp_logp_bandpowers(beta, beta_nodes, table, T)
        r = mu - data
        return -0.5 * r @ (prec @ r) - (softplus(-z[0]) + softplus(z[0]))

    def one_chain(ck, data):
        dk, wk, sk = jax.random.split(ck, 3)
        warm = blackjax.window_adaptation(blackjax.nuts, lambda z: ld(z, data),
                                          max_num_doublings=MAX_DOUBLINGS)
        (st, params), _ = warm.run(wk, 0.5 * jax.random.normal(dk, (1,)), num_steps=N_WARMUP)
        kernel = blackjax.nuts(lambda z: ld(z, data), **params)

        def step(s, k):
            s, info = kernel.step(k, s)
            return s, (s.position, info.is_divergent)

        _, (pos, div) = jax.lax.scan(step, st, jax.random.split(sk, N_SAMPLES))
        return pos, div

    @jax.jit
    def infer(data, key):
        return jax.vmap(lambda ck: one_chain(ck, data))(jax.random.split(key, N_CHAINS))

    return infer


def calibrate(name):
    """Fixed-fiducial transfer + Hartlap precision for observable `name` at (BETA_FID, M_FID)."""
    rows = np.array([observable(name, gen_count_map(BETA_FID, jax.random.fold_in(jax.random.PRNGKey(1000), r)))
                     for r in range(N_REAL_FID)])
    a_s_fid = np.asarray(analytic_logdensity_bandpowers(SHAPE, BETA_FID, M_FID, B_FIXED, ALPHA,
                                                        DEPTH, K_EDGES, N_MAX, N_QUAD))
    T = np.asarray(calibrate_transfer(jnp.asarray(rows), jnp.asarray(a_s_fid)))
    prec = mock_precision(rows)            # Hartlap-corrected (n_real=64 >> NB+2)
    return T, prec


def run_one(name, beta_nodes, table):
    T, prec = calibrate(name)
    infer = make_infer(beta_nodes, jnp.asarray(table), T, prec)

    # recovery sanity (one trial at the fiducial-ish truth)
    bt = 3.0
    data = observable(name, gen_count_map(bt, jax.random.fold_in(jax.random.PRNGKey(42), 0)))
    pos, div = infer(jnp.asarray(data), jax.random.PRNGKey(7))
    bpost = np.exp(LOGB_LO + LB / (1 + np.exp(-np.asarray(pos).reshape(-1))))
    print(f"  [{name}] recovery @ beta=3.0: post mean={bpost.mean():.3f} std={bpost.std():.3f} "
          f"(div={int(np.asarray(div).sum())})")

    # SBC
    truths = np.array([float(np.exp(LOGB_LO + LB * float(jax.random.uniform(jax.random.fold_in(
        jax.random.PRNGKey(777), k))))) for k in range(K_TRIALS)])
    post = np.zeros((K_TRIALS, L_THIN, 1)); ndiv = 0; t0 = time.time()
    for k in range(K_TRIALS):
        data = observable(name, gen_count_map(truths[k], jax.random.fold_in(jax.random.PRNGKey(9090), k)))
        pos, div = infer(jnp.asarray(data), jax.random.fold_in(jax.random.PRNGKey(55), k))
        ndiv += int(np.asarray(div).sum())
        z = np.asarray(pos).reshape(-1)
        b = np.exp(LOGB_LO + LB / (1 + np.exp(-z)))
        idx = np.linspace(0, len(b) - 1, L_THIN).astype(int)
        post[k, :, 0] = b[idx]
        if (k + 1) % 32 == 0:
            print(f"  [{name}] {k+1}/{K_TRIALS} ({time.time()-t0:.0f}s, div={ndiv})")
    rh = compute_sbc_rank_histogram(truths[:, None], post, param_names=["beta"])
    ed = compute_sbc_ecdf_diff(truths[:, None], post, param_names=["beta"])
    return {"name": name, "p": float(rh["p_value"][0]), "rh": rh, "ed": ed, "ndiv": ndiv}


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"V3 analytic-log+ SBC  shape={SHAPE} N_stars={N_STARS:.0e} K={K_TRIALS} bins={NB} "
          f"beta_fid={BETA_FID}")
    t0 = time.time()
    beta_nodes = jnp.linspace(BETA_LO, BETA_HI, N_NODES)
    table = np.array([np.asarray(analytic_logdensity_bandpowers(SHAPE, float(bn), M_FID, B_FIXED,
            ALPHA, DEPTH, K_EDGES, N_MAX, N_QUAD)) for bn in beta_nodes])
    print(f"[precompute] A_s table {table.shape} in {time.time()-t0:.0f}s")

    results = [run_one("logp", beta_nodes, table), run_one("rankg", beta_nodes, table)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)
    for i, res in enumerate(results):
        plot_sbc_rank_histogram(axes[i, 0], res["rh"]["ranks"][:, 0], n_draws=res["rh"]["n_draws"],
                                param_name="beta", n_bins=res["rh"]["n_bins"], n_trials=res["rh"]["n_trials"])
        axes[i, 0].set_title(f"{res['name']} beta: rank hist (p={res['p']:.3f})")
        plot_sbc_ecdf_diff(axes[i, 1], res["ed"]["eval_points"][0], res["ed"]["ecdf_diff"][0],
                           res["ed"]["band_lower"][0], res["ed"]["band_upper"][0], param_name="beta")
        axes[i, 1].set_title(f"{res['name']} beta: ECDF-diff")
    fig.suptitle(f"V3 analytic-log+ SBC (N_stars={N_STARS:.0e}): log+ should calibrate, rank-G should not")
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "v3_logp_sbc.png")
    fig.savefig(path, dpi=140); plt.close(fig)

    print(f"\n{'#'*64}\n  V3 ANALYTIC-LOG+ SBC VERDICT (N_stars={N_STARS:.0e}, K={K_TRIALS})\n{'#'*64}")
    for res in results:
        verdict = "CALIBRATED" if res["p"] > 0.05 else "miscalibrated"
        print(f"  [{res['name']:5s}] p(beta)={res['p']:.3f}  div={res['ndiv']}  -> {verdict}")
    print(f"  figure: {path}\n  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

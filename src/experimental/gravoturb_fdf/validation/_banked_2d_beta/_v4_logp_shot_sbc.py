r"""V4 — single-beta SBC of the ANALYTIC SHOT-TRANSFER forward model at LOW stellar density.

The fixed-transfer model (V3) passes at N_stars=1e6 but FAILS at 1e5 (shot makes the transfer
beta-dependent). Here the forward model is the FULLY ANALYTIC shot transfer
``predict_logp_bandpowers_shot`` = P_clust(beta) + W_shot (no fitted transfer), so the beta-response
stays analytic at any density. mu_shot(beta) is smooth+deterministic -> tabulate + interpolate for
fast NUTS (same noise-free-emulator argument as V3). Fixed-fiducial Hartlap precision (truth-
independent). beta-only; M,b,alpha fixed.

Acid test of the lognormal projected-density marginal (the one approximation): if single-beta SBC
PASSES at N_stars=1e5, the analytic method is N-agnostic.

EXPERIMENTAL scratch; no production edits, no commits.
Run (default N_stars=1e5; override with GFDF_NSTARS):
  PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync python -u \
    src/experimental/gravoturb_fdf/validation/_v4_logp_shot_sbc.py
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
from gravoturb_fdf.inference.covariance import mock_precision
from gravoturb_fdf.inference.projected_logp import interp_logp_bandpowers, predict_logp_bandpowers_shot
from gravoturb_fdf.validation.measure import (
    measure_angular_bandpowers_2d,
    project_counts_los,
    smooth_copula_field,
)
from jaxstroviz.experimental.analysis.sbc import compute_sbc_ecdf_diff, compute_sbc_rank_histogram
from jaxstroviz.experimental.plots.sbc import plot_sbc_ecdf_diff, plot_sbc_rank_histogram

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
SHAPE, DEPTH = (64, 64, 64), 64
B_FIXED, ALPHA, M_FID, BETA_FID = 0.4, 2.5, 8.0, 2.7
K_EDGES = np.linspace(2.0, 28.0, 10)
NB = len(K_EDGES) - 1
N_STARS = int(float(os.environ.get("GFDF_NSTARS", 10**5)))
N_BAR_3D = N_STARS / (SHAPE[0] ** 3)
BETA_LO, BETA_HI = 2.0, 11.0 / 3.0
LOGB_LO, LB = np.log(BETA_LO), np.log(BETA_HI) - np.log(BETA_LO)
N_NODES, N_REAL_FID = 96, 64
N_MAX, N_QUAD = 14, 256
N_COUNT_MAX = max(400, int(40 * N_BAR_3D * DEPTH))     # cover the Poisson tail at this density
K_TRIALS, N_WARMUP, N_SAMPLES, N_CHAINS, MAX_DOUBLINGS, L_THIN = 128, 400, 600, 4, 8, 100


def log_plus(N, nb):
    d = np.asarray(N, float) / nb - 1.0
    return np.where(d > 0.0, np.log1p(np.where(d > 0.0, d, 0.0)), d)


def gen_logp_data(beta, key):
    s = smooth_copula_field(gaussian_random_field(SHAPE, float(beta), key), M_FID, B_FIXED, ALPHA)
    cnt = np.asarray(sample_cic_counts(jnp.asarray(s), N_BAR_3D, 1, jax.random.fold_in(key, 1)))
    pc = project_counts_los(cnt, DEPTH, los_axis=2).astype(float)
    return measure_angular_bandpowers_2d(log_plus(pc, pc.mean()), K_EDGES)


def beta_of_z(z):
    return jnp.exp(LOGB_LO + LB * sigmoid(z))


def make_infer(beta_nodes, mu_table, precision):
    prec = jnp.asarray(precision)
    ones = jnp.ones(NB)

    def ld(z, data):
        mu = interp_logp_bandpowers(beta_of_z(z[0]), beta_nodes, mu_table, ones)
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


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"V4 analytic-shot SBC  shape={SHAPE} N_stars={N_STARS:.0e} n_bar_sky={N_BAR_3D*DEPTH:.1f} "
          f"n_count_max={N_COUNT_MAX} K={K_TRIALS} bins={NB}")
    t0 = time.time()
    beta_nodes = jnp.linspace(BETA_LO, BETA_HI, N_NODES)
    mu_table = np.array([np.asarray(predict_logp_bandpowers_shot(
        SHAPE, float(bn), M_FID, B_FIXED, ALPHA, DEPTH, K_EDGES, N_BAR_3D,
        n_max=N_MAX, n_quad=N_QUAD, n_count_max=N_COUNT_MAX)) for bn in beta_nodes])
    print(f"[precompute] mu_shot table {mu_table.shape} in {time.time()-t0:.0f}s")

    # fixed-fiducial Hartlap precision from sims at (BETA_FID, M_FID, this N_stars)
    rows = np.array([gen_logp_data(BETA_FID, jax.random.fold_in(jax.random.PRNGKey(1000), r))
                     for r in range(N_REAL_FID)])
    prec = mock_precision(rows)
    infer = make_infer(beta_nodes, jnp.asarray(mu_table), prec)

    # recovery sanity
    data = gen_logp_data(3.0, jax.random.fold_in(jax.random.PRNGKey(42), 0))
    pos, div = infer(jnp.asarray(data), jax.random.PRNGKey(7))
    bp = np.exp(LOGB_LO + LB / (1 + np.exp(-np.asarray(pos).reshape(-1))))
    print(f"  recovery @ beta=3.0: post mean={bp.mean():.3f} std={bp.std():.3f} (div={int(np.asarray(div).sum())})")

    truths = np.array([float(np.exp(LOGB_LO + LB * float(jax.random.uniform(jax.random.fold_in(
        jax.random.PRNGKey(777), k))))) for k in range(K_TRIALS)])
    post = np.zeros((K_TRIALS, L_THIN, 1)); ndiv = 0; t1 = time.time()
    for k in range(K_TRIALS):
        data = gen_logp_data(truths[k], jax.random.fold_in(jax.random.PRNGKey(9090), k))
        pos, div = infer(jnp.asarray(data), jax.random.fold_in(jax.random.PRNGKey(55), k))
        ndiv += int(np.asarray(div).sum())
        b = np.exp(LOGB_LO + LB / (1 + np.exp(-np.asarray(pos).reshape(-1))))
        idx = np.linspace(0, len(b) - 1, L_THIN).astype(int)
        post[k, :, 0] = b[idx]
        if (k + 1) % 32 == 0:
            print(f"  {k+1}/{K_TRIALS} ({time.time()-t1:.0f}s, div={ndiv})")

    rh = compute_sbc_rank_histogram(truths[:, None], post, param_names=["beta"])
    ed = compute_sbc_ecdf_diff(truths[:, None], post, param_names=["beta"])
    p = float(rh["p_value"][0])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plot_sbc_rank_histogram(ax[0], rh["ranks"][:, 0], n_draws=rh["n_draws"], param_name="beta",
                            n_bins=rh["n_bins"], n_trials=rh["n_trials"])
    ax[0].set_title(f"shot beta @ N={N_STARS:.0e}: rank hist (p={p:.3f})")
    plot_sbc_ecdf_diff(ax[1], ed["eval_points"][0], ed["ecdf_diff"][0], ed["band_lower"][0],
                       ed["band_upper"][0], param_name="beta")
    ax[1].set_title("shot beta: ECDF-diff")
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, f"v4_shot_sbc_{N_STARS:.0e}.png")
    fig.savefig(path, dpi=140); plt.close(fig)

    print(f"\n{'#'*64}\n  V4 ANALYTIC-SHOT SBC (N_stars={N_STARS:.0e}, K={K_TRIALS}, div={ndiv})\n{'#'*64}")
    print(f"  p(beta)={p:.3f}  -> {'CALIBRATED' if p > 0.05 else 'miscalibrated'}")
    print(f"  (vs fixed-transfer V3 @ 1e5: p=0.000)\n  figure: {path}\n  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

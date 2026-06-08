r"""V5 — flow-based NPE single-beta SBC at LOW stellar density (Path C; the analytic model's wall).

Train an amortized conditional normalizing flow q(z|s) (z=logit-beta, s=whitened log+ band-powers) on
(beta*~prior, s) pairs simulated at N_stars=1e5 -- the regime where the analytic shot model FAILS
(p=0.000) because the projected-density marginal has no simple analytic form. The flow LEARNS that
marginal implicitly. SBC over INDEPENDENT trials (flow fixed after training -> truth-independent).

Same observable as the analytic path (log+ band-powers) -> directly comparable ("both").

EXPERIMENTAL scratch; no production edits, no commits.
Run (default N_stars=1e5; override GFDF_NSTARS):
  PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync python -u \
    src/experimental/gravoturb_fdf/validation/_v5_flow_sbc.py
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
from gravoturb_fdf.inference.flow_npe import (
    beta_to_z,
    build_npe_flow,
    npe_posterior_z,
    train_npe,
    whiten,
    whiten_stats,
    z_to_beta,
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
K_EDGES = np.linspace(2.0, 28.0, 10)
N_STARS = int(float(os.environ.get("GFDF_NSTARS", 10**5)))
N_BAR_3D = N_STARS / (SHAPE[0] ** 3)
BETA_LO, BETA_HI = 2.0, 11.0 / 3.0
LOGB_LO, LB = np.log(BETA_LO), np.log(BETA_HI) - np.log(BETA_LO)
N_TRAIN = int(os.environ.get("GFDF_NTRAIN", 4000))
K_TRIALS, N_POST = 128, 400


def log_plus(N, nb):
    d = np.asarray(N, float) / nb - 1.0
    return np.where(d > 0.0, np.log1p(np.where(d > 0.0, d, 0.0)), d)


def gen_pair(seed):
    """(beta* ~ log-uniform prior, log+ band-power summary) at N_stars."""
    kb = jax.random.fold_in(jax.random.PRNGKey(20240), seed)
    beta = float(np.exp(LOGB_LO + LB * float(jax.random.uniform(kb))))
    kf = jax.random.fold_in(jax.random.PRNGKey(70707), seed)
    s = smooth_copula_field(gaussian_random_field(SHAPE, beta, kf), M_FID, B_FIXED, ALPHA)
    cnt = np.asarray(sample_cic_counts(jnp.asarray(s), N_BAR_3D, 1, jax.random.fold_in(kf, 1)))
    pc = project_counts_los(cnt, DEPTH, los_axis=2).astype(float)
    return beta, measure_angular_bandpowers_2d(log_plus(pc, pc.mean()), K_EDGES)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"V5 flow-NPE SBC  shape={SHAPE} N_stars={N_STARS:.0e} N_train={N_TRAIN} K={K_TRIALS}")
    t0 = time.time()

    # training set from the prior
    betas, summaries = [], []
    for i in range(N_TRAIN):
        b, s = gen_pair(i)
        betas.append(b); summaries.append(s)
        if (i + 1) % 1000 == 0:
            print(f"  [train sims] {i+1}/{N_TRAIN} ({time.time()-t0:.0f}s)")
    betas = np.array(betas); summaries = np.array(summaries)
    z = np.asarray(beta_to_z(jnp.asarray(betas), BETA_LO, BETA_HI))[:, None]
    mean, std = whiten_stats(jnp.asarray(summaries))
    s_white = whiten(jnp.asarray(summaries), mean, std)

    flow = build_npe_flow(jax.random.key(0), summary_dim=len(K_EDGES) - 1)
    flow = train_npe(jax.random.key(1), flow, jnp.asarray(z), s_white, max_epochs=300)
    print(f"[train] flow fitted ({time.time()-t0:.0f}s)")

    # SBC on INDEPENDENT trials
    truths = np.zeros(K_TRIALS); post = np.zeros((K_TRIALS, N_POST, 1))
    for k in range(K_TRIALS):
        b, s = gen_pair(10**6 + k)              # disjoint seeds from training
        truths[k] = b
        sw = whiten(jnp.asarray(s), mean, std)
        zk = npe_posterior_z(jax.random.key(1000 + k), flow, sw, n_samples=N_POST)
        post[k, :, 0] = np.asarray(z_to_beta(zk, BETA_LO, BETA_HI))
        if (k + 1) % 32 == 0:
            print(f"  [sbc] {k+1}/{K_TRIALS} ({time.time()-t0:.0f}s)")

    rh = compute_sbc_rank_histogram(truths[:, None], post, param_names=["beta"])
    ed = compute_sbc_ecdf_diff(truths[:, None], post, param_names=["beta"])
    p = float(rh["p_value"][0])
    # recovery scatter
    pm = post[:, :, 0].mean(axis=1)
    rms = float(np.sqrt(np.mean((pm - truths) ** 2)))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    plot_sbc_rank_histogram(ax[0], rh["ranks"][:, 0], n_draws=rh["n_draws"], param_name="beta",
                            n_bins=rh["n_bins"], n_trials=rh["n_trials"])
    ax[0].set_title(f"flow beta @ N={N_STARS:.0e}: rank hist (p={p:.3f})")
    plot_sbc_ecdf_diff(ax[1], ed["eval_points"][0], ed["ecdf_diff"][0], ed["band_lower"][0],
                       ed["band_upper"][0], param_name="beta")
    ax[1].set_title("flow beta: ECDF-diff")
    ax[2].plot(truths, pm, "o", ms=3); lim = [BETA_LO, BETA_HI]
    ax[2].plot(lim, lim, "k--"); ax[2].set_xlabel("beta truth"); ax[2].set_ylabel("posterior mean")
    ax[2].set_title(f"recovery (rms={rms:.3f})")
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, f"v5_flow_sbc_{N_STARS:.0e}.png")
    fig.savefig(path, dpi=140); plt.close(fig)

    print(f"\n{'#'*64}\n  V5 FLOW-NPE SBC (N_stars={N_STARS:.0e}, K={K_TRIALS}, N_train={N_TRAIN})\n{'#'*64}")
    print(f"  p(beta)={p:.3f}  recovery rms={rms:.3f}  -> {'CALIBRATED' if p > 0.05 else 'miscalibrated'}")
    print(f"  (vs analytic shot model @ 1e5: p=0.000)\n  figure: {path}\n  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

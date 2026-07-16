r"""SBC fix #2 (data-supported): rank-Gaussianized band-powers + Gaussian likelihood.

DIAGNOSIS (measured): raw count band-powers have skewness up to 12 / excess-kurtosis up to 167 ->
a Gaussian likelihood is fundamentally mis-specified (SBC p~0). A log transform of the band-powers
helps but is INSUFFICIENT (skew still ~3, exkurt ~18 at high k). RANK-GAUSSIANIZING THE MAP
(Neyrinck 2011 Eq.1) before measuring band-powers drives them to NEAR-GAUSSIAN (skew <~0.8,
exkurt <~1 at both fiducial and extreme theta) -> a Gaussian likelihood is now valid. rank-G also
maximizes beta information (V1a gain 0.86 vs 0.64) and is the established LSS approach.

PIPELINE (statistic IDENTICAL in generation + inference -> SBC-valid):
  count map -> rank_gaussianize_2d (Neyrinck Eq.1) -> band-powers.
Forward model + covariance are EMULATED from simulator ensembles on a coarse (beta, logM) grid
(rank-G is nonlinear -> no clean analytic band-power mean; emulation stays differentiable via
bilinear interpolation, and rank-G band-power moments vary mildly with theta -> coarse interp is OK,
unlike the raw band-powers whose logdet ranged over ~e^87). Gaussian likelihood with theta-dependent
mean mu_rg(theta), covariance C_rg(theta), and the log|C| term; logit-reparam NUTS.

NO core edits, NO commits. Run:
  PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync python -u \
    src/experimental/gravoturb_fdf/validation/_v2h_rankg_sbc.py
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
from gravoturb_fdf.inference.priors import BM19Prior
from gravoturb_fdf.validation.measure import (
    measure_angular_bandpowers_2d,
    project_counts_los,
    smooth_copula_field,
)
from jaxstroviz.experimental.analysis.sbc import compute_sbc_ecdf_diff, compute_sbc_rank_histogram
from jaxstroviz.experimental.plots.sbc import plot_sbc_ecdf_diff, plot_sbc_rank_histogram

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
SHAPE, DEPTH, B_FIXED, ALPHA_TRUE = (64, 64, 64), 64, 0.4, 2.5
K_EDGES = np.linspace(1.0, 28.0, 11)
N_STARS = 10**5
N_BAR_3D = N_STARS / (SHAPE[0] ** 3)
PRIOR = BM19Prior()
M_LO, M_HI = PRIOR.m_range
BETA_LO, BETA_HI = PRIOR.beta_range
LOGM_LO, LOGM_HI = np.log(M_LO), np.log(M_HI); LM = LOGM_HI - LOGM_LO
LOGB_LO, LB = np.log(BETA_LO), np.log(BETA_HI) - np.log(BETA_LO)
NB_C, NM_C, N_REAL_C = 7, 7, 48
RIDGE = 1e-3
K_TRIALS, N_WARMUP, N_SAMPLES, N_CHAINS, MAX_DOUBLINGS, L_THIN = 128, 400, 600, 4, 8, 100


# ----------------------------- rank-G statistic (Neyrinck Eq.1) -----------------------------
def rank_gaussianize_2d(map2d):
    """Neyrinck 2011 Eq.1: replace each pixel by the Gaussian quantile of its rank -> N(0,1) marginal."""
    f = np.asarray(map2d, float).ravel()
    ranks = np.argsort(np.argsort(f))
    u = (ranks + 0.5) / f.size
    return (np.sqrt(2.0) * special.erfinv(2.0 * u - 1.0)).reshape(map2d.shape)


def measure_rankg_count(beta, M, key):
    g = gaussian_random_field(SHAPE, beta, key)
    s = smooth_copula_field(g, M, B_FIXED, ALPHA_TRUE)
    cnt = np.asarray(sample_cic_counts(jnp.asarray(s), N_BAR_3D, 1, jax.random.fold_in(key, 1)))
    cmap = project_counts_los(cnt, DEPTH, los_axis=2)
    return measure_angular_bandpowers_2d(rank_gaussianize_2d(cmap), K_EDGES)


# ----------------------------- emulate mu_rg(theta), Cinv(theta), logdet(theta) -----------------
def build_grid():
    bnodes = np.linspace(BETA_LO, BETA_HI, NB_C)
    lmnodes = np.linspace(LOGM_LO, LOGM_HI, NM_C)
    nbins = len(K_EDGES) - 1
    mu = np.zeros((NB_C, NM_C, nbins))
    Cinv = np.zeros((NB_C, NM_C, nbins, nbins))
    logdet = np.zeros((NB_C, NM_C))
    for i, b in enumerate(bnodes):
        for j, lm in enumerate(lmnodes):
            M = float(np.exp(lm))
            rows = np.array([measure_rankg_count(float(b), M,
                            jax.random.fold_in(jax.random.PRNGKey(1234), (i * NM_C + j) * N_REAL_C + r))
                            for r in range(N_REAL_C)])
            mu[i, j] = rows.mean(axis=0)
            C = np.cov(rows, rowvar=False, ddof=1)
            C += RIDGE * np.mean(np.diag(C)) * np.eye(nbins)
            Cinv[i, j] = np.linalg.inv(C)
            logdet[i, j] = np.linalg.slogdet(C)[1]
    return jnp.asarray(mu), jnp.asarray(Cinv), jnp.asarray(logdet)


def _bilinear(table, beta, M):
    u = (beta - BETA_LO) / (BETA_HI - BETA_LO) * (NB_C - 1)
    v = (jnp.log(M) - LOGM_LO) / LM * (NM_C - 1)
    i0 = jnp.clip(jnp.floor(u), 0, NB_C - 2).astype(jnp.int32)
    j0 = jnp.clip(jnp.floor(v), 0, NM_C - 2).astype(jnp.int32)
    fu, fv = u - i0, v - j0
    t00, t10, t01, t11 = table[i0, j0], table[i0 + 1, j0], table[i0, j0 + 1], table[i0 + 1, j0 + 1]
    return (1 - fu) * (1 - fv) * t00 + fu * (1 - fv) * t10 + (1 - fu) * fv * t01 + fu * fv * t11


# ----------------------------- logit reparam + jitted NUTS -----------------------------
def z_to_beta_M(z):
    return jnp.exp(LOGB_LO + LB * sigmoid(z[0])), jnp.exp(LOGM_LO + LM * sigmoid(z[1]))


def log_prior_jac(z):
    return -(softplus(-z[0]) + softplus(z[0])) - (softplus(-z[1]) + softplus(z[1]))


def make_infer(mu_t, Cinv_t, logdet_t):
    def ld(z, data):
        beta, M = z_to_beta_M(z)
        r = _bilinear(mu_t, beta, M) - data
        Cinv = _bilinear(Cinv_t, beta, M)
        return -0.5 * r @ (Cinv @ r) - 0.5 * _bilinear(logdet_t, beta, M) + log_prior_jac(z)

    def one_chain(ck, data):
        dk, wk, sk = jax.random.split(ck, 3)
        warm = blackjax.window_adaptation(blackjax.nuts, lambda z: ld(z, data), max_num_doublings=MAX_DOUBLINGS)
        (st, params), _ = warm.run(wk, 0.7 * jax.random.normal(dk, (2,)), num_steps=N_WARMUP)
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
    print(f"v2h rank-G SBC  K={K_TRIALS}  (rank-G count band-powers + Gaussian likelihood)")
    t0 = time.time()
    mu_t, Cinv_t, logdet_t = build_grid()
    print(f"[grid] {NB_C}x{NM_C}x{N_REAL_C} in {time.time()-t0:.0f}s; "
          f"logdet range [{float(logdet_t.min()):.1f},{float(logdet_t.max()):.1f}] "
          f"(cf raw-bandpower v2g range [59.8,146.9] -> rank-G should be much milder)")
    infer = make_infer(mu_t, Cinv_t, logdet_t)

    truths = np.array([
        [float(np.exp(LOGB_LO + LB * float(jax.random.uniform(jax.random.fold_in(jax.random.PRNGKey(777), 2*k))))),
         float(np.exp(LOGM_LO + LM * float(jax.random.uniform(jax.random.fold_in(jax.random.PRNGKey(777), 2*k+1)))))]
        for k in range(K_TRIALS)])
    fields = [gaussian_random_field(SHAPE, float(truths[k, 0]), jax.random.fold_in(jax.random.PRNGKey(4242), k))
              for k in range(K_TRIALS)]
    posteriors = np.zeros((K_TRIALS, L_THIN, 2)); ndiv = 0
    t0 = time.time()
    for k in range(K_TRIALS):
        beta_t, M_t = truths[k]
        cnt = np.asarray(sample_cic_counts(jnp.asarray(smooth_copula_field(fields[k], M_t, B_FIXED, ALPHA_TRUE)),
                         N_BAR_3D, 1, jax.random.fold_in(jax.random.PRNGKey(9090), k)))
        data = measure_angular_bandpowers_2d(rank_gaussianize_2d(project_counts_los(cnt, DEPTH, 2)), K_EDGES)
        pos, div = infer(jnp.asarray(data), jax.random.fold_in(jax.random.PRNGKey(55), k))
        ndiv += int(np.asarray(div).sum())
        z = np.asarray(pos).reshape(-1, 2)
        beta = np.exp(LOGB_LO + LB / (1 + np.exp(-z[:, 0])))
        M = np.exp(LOGM_LO + LM / (1 + np.exp(-z[:, 1])))
        idx = np.linspace(0, len(beta) - 1, L_THIN).astype(int)
        posteriors[k, :, 0], posteriors[k, :, 1] = beta[idx], M[idx]
        if (k + 1) % 32 == 0:
            print(f"  {k+1}/{K_TRIALS} ({time.time()-t0:.0f}s, div={ndiv})")

    rh = compute_sbc_rank_histogram(truths, posteriors, param_names=["beta", "M"])
    ed = compute_sbc_ecdf_diff(truths, posteriors, param_names=["beta", "M"])
    pv = rh["p_value"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)
    for p, pn in enumerate(["beta", "M"]):
        plot_sbc_rank_histogram(axes[p, 0], rh["ranks"][:, p], n_draws=rh["n_draws"],
                                param_name=pn, n_bins=rh["n_bins"], n_trials=rh["n_trials"])
        axes[p, 0].set_title(f"rank-G {pn}: rank hist (p={pv[p]:.3f})")
        plot_sbc_ecdf_diff(axes[p, 1], ed["eval_points"][p], ed["ecdf_diff"][p],
                           ed["band_lower"][p], ed["band_upper"][p], param_name=pn)
        axes[p, 1].set_title(f"rank-G {pn}: ECDF-diff")
    fig.suptitle("v2h rank-G SBC-2D (count): calibrated => flat hist / inside band", y=1.0)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "v2h_rankg_sbc.png")
    fig.savefig(path, dpi=140); plt.close(fig)

    print(f"\n{'#'*64}\n  v2h RANK-G SBC VERDICT (count, K={K_TRIALS}, div={ndiv})\n{'#'*64}")
    print(f"  p(beta)={pv[0]:.3f}  p(M)={pv[1]:.3f}  (n_bins={rh['n_bins']}, L={rh['n_draws']})")
    print(f"  -> {'CALIBRATED' if (pv>0.05).all() else 'still miscalibrated (see which param/bins)'}")
    print(f"  (vs raw-bandpower v2e: p(beta)=0.001 p(M)=0.000)")
    print(f"  figure: {path}")


if __name__ == "__main__":
    main()

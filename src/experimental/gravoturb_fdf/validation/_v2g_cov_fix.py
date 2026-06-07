r"""SBC fix #1: theta-dependent (emulated) covariance C(theta) + log|C(theta)| term.

The SBC-2D failure (_v2e) was diagnosed (self-consistency PASSED, _v2f) as MODEL/COVARIANCE
adequacy: (1) the fixed-fiducial covariance is mis-specified across the prior -> under-dispersion
(U-shaped ranks); (2) a forward-model-vs-simulator mean bias -> M biased low.

FIX (1): emulate the covariance on a coarse (beta, logM) grid from simulator ensembles. Per node:
sample mean mu_sim, sample covariance C (+ small ridge for PD); store Cinv = inv(C) and logdet =
log|C|. At a query theta, BILINEAR-interpolate Cinv ELEMENTWISE (a convex combination of PD matrices
with non-negative weights -> still PD) and logdet (scalar) -> a smooth, differentiable, PD metric.
Likelihood: ld(theta) = -0.5 r^T Cinv(theta) r - 0.5 logdet(theta) + log_prior_jac, r = mu(theta)-data
(the log|C| term, which the fixed-C Gaussian likelihood drops, is REQUIRED for a theta-dependent C).

We test TWO means with the SAME C(theta):
  A: mu = analytic differentiable forward model (the d_n emulator)  -> thesis-preserving.
  B: mu = emulated SIMULATOR mean (from the same ensembles)         -> diagnostic for the M bias.
If A calibrates, the analytic model is adequate; if only B calibrates, the analytic mean's residual
bias is the remaining issue.

Observable: COUNT (the realistic star observable). NO core edits, NO commits.
Run: PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync python -u \
     src/experimental/gravoturb_fdf/validation/_v2g_cov_fix.py
"""
import time

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from jax.nn import sigmoid, softplus

from gravoturb_fdf.field.field import gaussian_random_field
from gravoturb_fdf.field.sampling import sample_cic_counts
from gravoturb_fdf.inference.covariance import add_poisson_shot, angular_bandpowers_2d_limber
from gravoturb_fdf.inference.priors import BM19Prior
from gravoturb_fdf.validation.measure import (
    measure_angular_bandpowers_2d,
    project_counts_los,
    smooth_copula_field,
)
from jaxstroviz.experimental.analysis.sbc import compute_sbc_rank_histogram

SHAPE, DEPTH, B_FIXED, ALPHA_TRUE = (64, 64, 64), 64, 0.4, 2.5
K_EDGES = np.linspace(1.0, 28.0, 11)
N_STARS = 10**5
N_BAR_SKY = N_STARS / (SHAPE[0] ** 2)
N_BAR_3D = N_STARS / (SHAPE[0] ** 3)
PRIOR = BM19Prior()
M_LO, M_HI = PRIOR.m_range
BETA_LO, BETA_HI = PRIOR.beta_range
LOGM_LO, LOGM_HI = np.log(M_LO), np.log(M_HI); LM = LOGM_HI - LOGM_LO
LOGB_LO, LB = np.log(BETA_LO), np.log(BETA_HI) - np.log(BETA_LO)

# analytic-mean emulator grid (fine) + covariance grid (coarse, ensembles per node)
NB_MU, NM_MU = 81, 81
NB_C, NM_C, N_REAL_C = 7, 7, 48
RIDGE = 1e-3  # covariance regularization (fraction of mean diagonal) for stable inv/logdet

K_TRIALS, N_WARMUP, N_SAMPLES, N_CHAINS, MAX_DOUBLINGS, L_THIN = 128, 400, 600, 4, 8, 100


# ----------------------------- forward model + analytic-mean emulator -----------------------------
def predict_direct(M, beta):
    _kc, P, _nm = angular_bandpowers_2d_limber(SHAPE, beta, M, B_FIXED, ALPHA_TRUE, DEPTH, K_EDGES)
    return P


def build_mu_emulator():
    bnodes = jnp.linspace(BETA_LO, BETA_HI, NB_MU)
    lmnodes = jnp.linspace(LOGM_LO, LOGM_HI, NM_MU)
    BB, LM_ = jnp.meshgrid(bnodes, lmnodes, indexing="ij")
    params = jnp.stack([jnp.exp(LM_.ravel()), BB.ravel()], axis=1)
    table = jax.lax.map(lambda p: add_poisson_shot(predict_direct(p[0], p[1]), N_BAR_SKY, DEPTH), params)
    return jnp.asarray(table).reshape(NB_MU, NM_MU, table.shape[1])


def _bilinear(table, beta, M, nb, nm):
    """Bilinear interp of `table` (nb, nm, ...) at (beta, logM); differentiable in (beta, M)."""
    u = (beta - BETA_LO) / (BETA_HI - BETA_LO) * (nb - 1)
    v = (jnp.log(M) - LOGM_LO) / LM * (nm - 1)
    i0 = jnp.clip(jnp.floor(u), 0, nb - 2).astype(jnp.int32)
    j0 = jnp.clip(jnp.floor(v), 0, nm - 2).astype(jnp.int32)
    fu, fv = u - i0, v - j0
    t00, t10, t01, t11 = table[i0, j0], table[i0 + 1, j0], table[i0, j0 + 1], table[i0 + 1, j0 + 1]
    return ((1 - fu) * (1 - fv) * t00 + fu * (1 - fv) * t10
            + (1 - fu) * fv * t01 + fu * fv * t11)


# ----------------------------- simulator ensembles -> mu_sim, Cinv, logdet on the coarse grid -----
def measure_count_real(beta, M, key):
    g = gaussian_random_field(SHAPE, beta, key)
    s = smooth_copula_field(g, M, B_FIXED, ALPHA_TRUE)
    cnt = np.asarray(sample_cic_counts(jnp.asarray(s), N_BAR_3D, 1, jax.random.fold_in(key, 1)))
    return measure_angular_bandpowers_2d(project_counts_los(cnt, DEPTH, los_axis=2), K_EDGES)


def build_cov_grid():
    """Per coarse node: simulator mean mu_sim, Cinv = inv(C+ridge), logdet = log|C+ridge|."""
    bnodes = np.linspace(BETA_LO, BETA_HI, NB_C)
    lmnodes = np.linspace(LOGM_LO, LOGM_HI, NM_C)
    nbins = len(K_EDGES) - 1
    mu = np.zeros((NB_C, NM_C, nbins))
    Cinv = np.zeros((NB_C, NM_C, nbins, nbins))
    logdet = np.zeros((NB_C, NM_C))
    for i, b in enumerate(bnodes):
        for j, lm in enumerate(lmnodes):
            M = float(np.exp(lm))
            rows = np.array([measure_count_real(float(b), M,
                            jax.random.fold_in(jax.random.PRNGKey(1234), (i * NM_C + j) * N_REAL_C + r))
                            for r in range(N_REAL_C)])
            mu[i, j] = rows.mean(axis=0)
            C = np.cov(rows, rowvar=False, ddof=1)
            C += RIDGE * np.mean(np.diag(C)) * np.eye(nbins)  # PD regularization
            Cinv[i, j] = np.linalg.inv(C)
            logdet[i, j] = np.linalg.slogdet(C)[1]
    return (jnp.asarray(mu), jnp.asarray(Cinv), jnp.asarray(logdet))


# ----------------------------- logit reparam + NUTS with C(theta) -----------------------------
def z_to_beta_M(z):
    return jnp.exp(LOGB_LO + LB * sigmoid(z[0])), jnp.exp(LOGM_LO + LM * sigmoid(z[1]))


def log_prior_jac(z):
    return -(softplus(-z[0]) + softplus(z[0])) - (softplus(-z[1]) + softplus(z[1]))


def make_infer(mu_fn, mu_table, Cinv_table, logdet_table):
    def ld(z, data):
        beta, M = z_to_beta_M(z)
        mu = mu_fn(mu_table, beta, M)
        Cinv = _bilinear(Cinv_table, beta, M, NB_C, NM_C)
        logdet = _bilinear(logdet_table, beta, M, NB_C, NM_C)
        r = mu - data
        return -0.5 * r @ (Cinv @ r) - 0.5 * logdet + log_prior_jac(z)

    def one_chain(ck, data):
        dk, wk, sk = jax.random.split(ck, 3)
        warm = blackjax.window_adaptation(blackjax.nuts, lambda z: ld(z, data), max_num_doublings=MAX_DOUBLINGS)
        (state, params), _ = warm.run(wk, 0.7 * jax.random.normal(dk, (2,)), num_steps=N_WARMUP)
        kernel = blackjax.nuts(lambda z: ld(z, data), **params)

        def step(s, k):
            s, info = kernel.step(k, s)
            return s, (s.position, info.is_divergent)

        _, (pos, div) = jax.lax.scan(step, state, jax.random.split(sk, N_SAMPLES))
        return pos, div

    @jax.jit
    def infer(data, key):
        return jax.vmap(lambda ck: one_chain(ck, data))(jax.random.split(key, N_CHAINS))

    return infer


# mu functions: A = analytic emulator (fine grid); B = simulator-mean (coarse grid)
def mu_analytic(table, beta, M):
    return _bilinear(table, beta, M, NB_MU, NM_MU)


def mu_sim(table, beta, M):
    return _bilinear(table, beta, M, NB_C, NM_C)


def run_sbc(name, infer, truths, fields, count_keys, infer_keys):
    posteriors = np.zeros((K_TRIALS, L_THIN, 2)); ndiv = 0
    t0 = time.time()
    for k in range(K_TRIALS):
        beta_t, M_t = truths[k]
        g = fields[k]
        s = smooth_copula_field(g, M_t, B_FIXED, ALPHA_TRUE)
        cnt = np.asarray(sample_cic_counts(jnp.asarray(s), N_BAR_3D, 1, count_keys[k]))
        data = measure_angular_bandpowers_2d(project_counts_los(cnt, DEPTH, los_axis=2), K_EDGES)
        pos, div = infer(jnp.asarray(data), infer_keys[k])
        ndiv += int(np.asarray(div).sum())
        z = np.asarray(pos).reshape(-1, 2)
        beta = np.exp(LOGB_LO + LB / (1 + np.exp(-z[:, 0])))
        M = np.exp(LOGM_LO + LM / (1 + np.exp(-z[:, 1])))
        idx = np.linspace(0, len(beta) - 1, L_THIN).astype(int)
        posteriors[k, :, 0], posteriors[k, :, 1] = beta[idx], M[idx]
        if (k + 1) % 32 == 0:
            print(f"    [{name}] {k+1}/{K_TRIALS} ({time.time()-t0:.0f}s, div={ndiv})")
    rh = compute_sbc_rank_histogram(truths, posteriors, param_names=["beta", "M"])
    pv = rh["p_value"]
    print(f"  [{name}] p(beta)={pv[0]:.3f}  p(M)={pv[1]:.3f}  div={ndiv}  "
          f"-> {'CALIBRATED' if (pv>0.05).all() else 'still miscalibrated'}")
    return pv


def main():
    print(f"v2g covariance fix: theta-dependent C(theta) + log|C|; COUNT; K={K_TRIALS}")
    t0 = time.time()
    mu_table = build_mu_emulator(); mu_table.block_until_ready()
    print(f"[mu emulator] {tuple(mu_table.shape)} in {time.time()-t0:.1f}s")
    t0 = time.time()
    mu_sim_table, Cinv_table, logdet_table = build_cov_grid()
    print(f"[cov grid] {NB_C}x{NM_C} nodes x {N_REAL_C} reals in {time.time()-t0:.1f}s; "
          f"logdet range [{float(logdet_table.min()):.1f},{float(logdet_table.max()):.1f}]")

    truths = np.array([
        [float(np.exp(LOGB_LO + LB * float(jax.random.uniform(jax.random.fold_in(jax.random.PRNGKey(777), 2*k))))),
         float(np.exp(LOGM_LO + LM * float(jax.random.uniform(jax.random.fold_in(jax.random.PRNGKey(777), 2*k+1)))))]
        for k in range(K_TRIALS)])
    fields = [gaussian_random_field(SHAPE, float(truths[k, 0]), jax.random.fold_in(jax.random.PRNGKey(4242), k))
              for k in range(K_TRIALS)]
    count_keys = [jax.random.fold_in(jax.random.PRNGKey(9090), k) for k in range(K_TRIALS)]
    infer_keys = [jax.random.fold_in(jax.random.PRNGKey(55), k) for k in range(K_TRIALS)]

    print("\n[A] analytic-mean + C(theta):")
    infer_A = make_infer(mu_analytic, mu_table, Cinv_table, logdet_table)
    pv_A = run_sbc("A:analytic-mu", infer_A, truths, fields, count_keys, infer_keys)

    print("\n[B] simulator-mean + C(theta):")
    infer_B = make_infer(mu_sim, mu_sim_table, Cinv_table, logdet_table)
    pv_B = run_sbc("B:sim-mu", infer_B, truths, fields, count_keys, infer_keys)

    print(f"\n{'#'*64}\n  v2g COVARIANCE-FIX SBC VERDICT (count)\n{'#'*64}")
    print(f"  A analytic-mean + C(theta): p(beta)={pv_A[0]:.3f} p(M)={pv_A[1]:.3f}")
    print(f"  B sim-mean      + C(theta): p(beta)={pv_B[0]:.3f} p(M)={pv_B[1]:.3f}")
    print("  (vs _v2e fixed-C: p(beta)=0.001 p(M)=0.000)")


if __name__ == "__main__":
    main()

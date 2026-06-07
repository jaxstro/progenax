r"""SBC self-consistency diagnostic: isolate harness/sampler from model/covariance adequacy.

The real SBC (_v2e) FAILED (U-shaped ranks => under-dispersed; M biased low). Two candidate causes:
(A) forward-model bias vs the simulator, (B) fixed-fiducial covariance mis-specified across the prior.
Both are MODEL/COVARIANCE adequacy issues -- NOT the sampler. This diagnostic removes both by
generating the data FROM THE MODEL ITSELF:

    data_k = emulate(theta*_k) + L_chol(C) @ N(0, I),     fit: emulate + precision = inv(C).

Gen and fit use the IDENTICAL model + covariance, so by construction there is no forward-model bias
and no covariance mismatch. If SBC is now UNIFORM (p>0.05) => the sampler + logit reparam + rank/thin
harness are correct, and the _v2e failure is purely (A)/(B) model adequacy (the thing to fix next).
If it still FAILS => there is a harness/sampler bug to fix first.

NO production-code edits beyond this scratch file. NO commits.
Run: PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync python -u \
     src/experimental/gravoturb_fdf/validation/_v2f_sbc_selfcheck.py
"""
import time

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from jax.nn import sigmoid, softplus

from gravoturb_fdf.field.field import gaussian_random_field
from gravoturb_fdf.inference.covariance import angular_bandpowers_2d_limber, mock_covariance
from gravoturb_fdf.inference.priors import BM19Prior
from gravoturb_fdf.validation.measure import measure_angular_bandpowers_2d, smooth_copula_field
from jaxstroviz.experimental.analysis.sbc import compute_sbc_rank_histogram

SHAPE, DEPTH, B_FIXED, ALPHA_TRUE = (64, 64, 64), 64, 0.4, 2.5
K_EDGES = np.linspace(1.0, 28.0, 11)
M_FID, BETA_FID, N_REAL_COV = 8.0, 3.0, 64
PRIOR = BM19Prior()
M_LO, M_HI = PRIOR.m_range
BETA_LO, BETA_HI = PRIOR.beta_range
LOGM_LO, LOGM_HI = np.log(M_LO), np.log(M_HI); LM = LOGM_HI - LOGM_LO
LOGB_LO, LB = np.log(BETA_LO), np.log(BETA_HI) - np.log(BETA_LO)
NB_EMU, NM_EMU = 81, 81
BETA_NODES = jnp.linspace(BETA_LO, BETA_HI, NB_EMU)
LOGM_NODES = jnp.linspace(LOGM_LO, LOGM_HI, NM_EMU)
K_TRIALS, N_WARMUP, N_SAMPLES, N_CHAINS, MAX_DOUBLINGS, L_THIN = 128, 400, 600, 4, 8, 100


def predict_direct(M, beta):
    _kc, P, _nm = angular_bandpowers_2d_limber(SHAPE, beta, M, B_FIXED, ALPHA_TRUE, DEPTH, K_EDGES)
    return P


def build_emulator():
    BB, LM_ = jnp.meshgrid(BETA_NODES, LOGM_NODES, indexing="ij")
    params = jnp.stack([jnp.exp(LM_.ravel()), BB.ravel()], axis=1)
    table = jax.lax.map(lambda p: predict_direct(p[0], p[1]), params)
    return jnp.asarray(table).reshape(NB_EMU, NM_EMU, table.shape[1])


def make_emulate(table):
    nb, nm = NB_EMU, NM_EMU

    def emulate(M, beta):
        u = (beta - BETA_LO) / (BETA_HI - BETA_LO) * (nb - 1)
        v = (jnp.log(M) - LOGM_LO) / LM * (nm - 1)
        i0 = jnp.clip(jnp.floor(u), 0, nb - 2).astype(jnp.int32)
        j0 = jnp.clip(jnp.floor(v), 0, nm - 2).astype(jnp.int32)
        fu, fv = u - i0, v - j0
        b00, b10, b01, b11 = table[i0, j0], table[i0 + 1, j0], table[i0, j0 + 1], table[i0 + 1, j0 + 1]
        return (1 - fu) * (1 - fv) * b00 + fu * (1 - fv) * b10 + (1 - fu) * fv * b01 + fu * fv * b11

    return emulate


def z_to_beta_M(z):
    return jnp.exp(LOGB_LO + LB * sigmoid(z[0])), jnp.exp(LOGM_LO + LM * sigmoid(z[1]))


def log_prior_jac(z):
    return -(softplus(-z[0]) + softplus(z[0])) - (softplus(-z[1]) + softplus(z[1]))


def make_infer(emulate, precision):
    prec_j = jnp.asarray(precision)

    def ld(z, data):
        beta, M = z_to_beta_M(z)
        r = emulate(M, beta) - data
        return -0.5 * r @ (prec_j @ r) + log_prior_jac(z)

    def one_chain(ck, data):
        dk, wk, sk = jax.random.split(ck, 3)
        warm = blackjax.window_adaptation(blackjax.nuts, lambda z: ld(z, data), max_num_doublings=MAX_DOUBLINGS)
        (state, params), _ = warm.run(wk, 0.7 * jax.random.normal(dk, (2,)), num_steps=N_WARMUP)
        kernel = blackjax.nuts(lambda z: ld(z, data), **params)

        def step(s, k):
            s, info = kernel.step(k, s)
            return s, s.position

        _, pos = jax.lax.scan(step, state, jax.random.split(sk, N_SAMPLES))
        return pos

    @jax.jit
    def infer(data, key):
        return jax.vmap(lambda ck: one_chain(ck, data))(jax.random.split(key, N_CHAINS))

    return infer


def main():
    print(f"v2f SBC self-consistency diagnostic  K={K_TRIALS} (data ~ model(theta*) + N(0,C))")
    t0 = time.time()
    table = build_emulator(); table.block_until_ready()
    emulate = make_emulate(table)
    print(f"[precompute] emulator in {time.time()-t0:.1f}s")

    # fiducial covariance (count observable) -> exact precision (NO Hartlap, for clean self-consistency)
    rows = []
    for r in range(N_REAL_COV):
        g = gaussian_random_field(SHAPE, BETA_FID, jax.random.fold_in(jax.random.PRNGKey(6000), r))
        s = smooth_copula_field(g, M_FID, B_FIXED, ALPHA_TRUE)
        # count-like? use density band-powers for the cov shape; self-consistency is map-agnostic.
        rows.append(measure_angular_bandpowers_2d(np.exp(s).sum(axis=2), K_EDGES))
    C = mock_covariance(np.asarray(rows))
    precision = np.linalg.inv(C)
    Lchol = np.linalg.cholesky(C)
    print(f"[covariance] C ({C.shape}) built; cond(C)={np.linalg.cond(C):.1e}")

    infer = make_infer(emulate, precision)

    truths = np.array([
        [float(np.exp(LOGB_LO + LB * float(jax.random.uniform(jax.random.fold_in(jax.random.PRNGKey(777), 2 * k))))),
         float(np.exp(LOGM_LO + LM * float(jax.random.uniform(jax.random.fold_in(jax.random.PRNGKey(777), 2 * k + 1)))))]
        for k in range(K_TRIALS)])

    posteriors = np.zeros((K_TRIALS, L_THIN, 2))
    rng = np.random.default_rng(0)
    t0 = time.time()
    for k in range(K_TRIALS):
        beta_t, M_t = truths[k]
        model_mean = np.asarray(emulate(jnp.asarray(M_t), jnp.asarray(beta_t)))
        data = model_mean + Lchol @ rng.standard_normal(model_mean.shape)  # data ~ N(model, C)
        pos = np.asarray(infer(jnp.asarray(data), jax.random.fold_in(jax.random.PRNGKey(55), k)))
        z = pos.reshape(-1, 2)
        beta = np.exp(LOGB_LO + LB / (1 + np.exp(-z[:, 0])))
        M = np.exp(LOGM_LO + LM / (1 + np.exp(-z[:, 1])))
        idx = np.linspace(0, len(beta) - 1, L_THIN).astype(int)
        posteriors[k, :, 0], posteriors[k, :, 1] = beta[idx], M[idx]
        if (k + 1) % 32 == 0:
            print(f"  {k+1}/{K_TRIALS} ({time.time()-t0:.0f}s)")

    rh = compute_sbc_rank_histogram(truths, posteriors, param_names=["beta", "M"])
    pv = rh["p_value"]
    print(f"\n{'#'*64}")
    print(f"  SELF-CONSISTENCY SBC: p(beta)={pv[0]:.3f}  p(M)={pv[1]:.3f}")
    verdict = ("HARNESS+SAMPLER OK -> _v2e failure is MODEL/COVARIANCE adequacy"
               if (pv > 0.05).all() else
               "STILL FAILS -> harness/sampler bug to fix first")
    print(f"  -> {verdict}")
    print(f"{'#'*64}")


if __name__ == "__main__":
    main()

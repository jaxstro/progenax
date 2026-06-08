r"""SBC-2D: simulation-based calibration of the (beta, M) gradient-based inference.

The trustworthiness payoff. For K trials: draw truth theta*=(beta*, M*) from the prior, generate a
mock from the Option-A POINTWISE map (gen = inference model), fit with the emulator logit-NUTS using
the FIXED, truth-INDEPENDENT fiducial covariance, and record the Talts rank of theta* among the
posterior draws. Calibrated <=> ranks uniform (integer-aware chi^2 + Sailynoja ECDF bands).

EFFICIENCY: the emulator (1064x per-eval) makes per-trial HMC ~1-2 s; and the inference is wrapped in
ONE jax.jit with the per-trial DATA as a traced argument, so the NUTS graph compiles ONCE and is
reused across all K trials (no per-trial recompile). This is what makes SBC feasible.

HONEST RISK: the fixed-fiducial covariance assumes C(theta) ~ C_fid over the prior. If C varies, SBC
will show miscalibration -- exactly the wall the earlier 3-D attempt hit. SBC is the honest test:
pass => calibrated; fail => the covariance is the next thing to fix. Not something to tune away.

NO production-code edits beyond this scratch file. NO commits.
Run: PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync python -u \
     src/experimental/gravoturb_fdf/validation/_v2e_sbc.py
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
    mock_precision,
)
from gravoturb_fdf.inference.priors import BM19Prior
from gravoturb_fdf.validation.measure import (
    measure_angular_bandpowers_2d,
    project_counts_los,
    smooth_copula_field,
)
from jaxstroviz.experimental.analysis.sbc import (
    compute_sbc_ecdf_diff,
    compute_sbc_rank_histogram,
)
from jaxstroviz.experimental.plots.sbc import plot_sbc_ecdf_diff, plot_sbc_rank_histogram

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")

# ----------------------------- config -----------------------------
SHAPE = (64, 64, 64)
DEPTH = 64
B_FIXED, ALPHA_TRUE = 0.4, 2.5
K_EDGES = np.linspace(1.0, 28.0, 11)
N_STARS = 10**5
N_BAR_SKY = N_STARS / (SHAPE[0] ** 2)
N_BAR_3D = N_STARS / (SHAPE[0] ** 3)
M_FID, BETA_FID, N_REAL_COV = 8.0, 3.0, 64

PRIOR = BM19Prior()
M_LO, M_HI = PRIOR.m_range
BETA_LO, BETA_HI = PRIOR.beta_range
LOGM_LO, LOGM_HI = np.log(M_LO), np.log(M_HI)
LM = LOGM_HI - LOGM_LO
LOGB_LO, LB = np.log(BETA_LO), np.log(BETA_HI) - np.log(BETA_LO)

NB_EMU, NM_EMU = 81, 81
BETA_NODES = jnp.linspace(BETA_LO, BETA_HI, NB_EMU)
LOGM_NODES = jnp.linspace(LOGM_LO, LOGM_HI, NM_EMU)

K_TRIALS = 128
N_WARMUP, N_SAMPLES, N_CHAINS, MAX_DOUBLINGS = 400, 600, 4, 8
L_THIN = 100  # thinned independent posterior draws per trial (ranks in {0..L})


# ----------------------------- emulator -----------------------------
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
        b00, b10 = table[i0, j0], table[i0 + 1, j0]
        b01, b11 = table[i0, j0 + 1], table[i0 + 1, j0 + 1]
        return ((1 - fu) * (1 - fv) * b00 + fu * (1 - fv) * b10
                + (1 - fu) * fv * b01 + fu * fv * b11)

    return emulate


# ----------------------------- logit reparam + NUTS (data traced) -----------------------------
def z_to_beta_M(z):
    beta = jnp.exp(LOGB_LO + LB * sigmoid(z[0]))
    M = jnp.exp(LOGM_LO + LM * sigmoid(z[1]))
    return beta, M


def log_prior_jac(z):
    return -(softplus(-z[0]) + softplus(z[0])) - (softplus(-z[1]) + softplus(z[1]))


def make_infer(emulate, precision, shot):
    """One JIT-compiled inference: data is a TRACED arg -> compiles once, reused all trials."""
    prec_j = jnp.asarray(precision)

    def ld(z, data):
        beta, M = z_to_beta_M(z)
        pred = emulate(M, beta)
        if shot:
            pred = add_poisson_shot(pred, N_BAR_SKY, DEPTH)
        r = pred - data
        return -0.5 * r @ (prec_j @ r) + log_prior_jac(z)

    def one_chain(ck, data):
        dk, wk, sk = jax.random.split(ck, 3)
        init0 = 0.7 * jax.random.normal(dk, (2,))
        warm = blackjax.window_adaptation(blackjax.nuts, lambda z: ld(z, data),
                                          max_num_doublings=MAX_DOUBLINGS)
        (state, params), _ = warm.run(wk, init0, num_steps=N_WARMUP)
        kernel = blackjax.nuts(lambda z: ld(z, data), **params)

        def step(s, k):
            s, info = kernel.step(k, s)
            return s, (s.position, info.is_divergent)

        _, (pos, div) = jax.lax.scan(step, state, jax.random.split(sk, N_SAMPLES))
        return pos, div

    @jax.jit
    def infer(data, key):
        pos, div = jax.vmap(lambda ck: one_chain(ck, data))(jax.random.split(key, N_CHAINS))
        return pos, div  # pos (n_chains, n_samples, 2), div (n_chains, n_samples)

    return infer


# ----------------------------- data + covariance -----------------------------
def measure_density(g, M):
    s = smooth_copula_field(g, M, B_FIXED, ALPHA_TRUE)
    return measure_angular_bandpowers_2d(np.exp(s).sum(axis=2), K_EDGES)


def measure_count(g, key, M):
    s = smooth_copula_field(g, M, B_FIXED, ALPHA_TRUE)
    cnt = np.asarray(sample_cic_counts(jnp.asarray(s), N_BAR_3D, 1, key))
    return measure_angular_bandpowers_2d(project_counts_los(cnt, DEPTH, los_axis=2), K_EDGES)


def fiducial_precision(observable, base_seed):
    rows = []
    for r in range(N_REAL_COV):
        g = gaussian_random_field(SHAPE, BETA_FID, jax.random.fold_in(jax.random.PRNGKey(base_seed), r))
        rows.append(measure_density(g, M_FID) if observable == "density"
                    else measure_count(g, jax.random.fold_in(jax.random.PRNGKey(base_seed + 7), r), M_FID))
    return mock_precision(np.asarray(rows))


def draw_prior(key):
    """theta* = (beta*, M*) ~ prior (log-uniform on both)."""
    kb, km = jax.random.split(key)
    beta = float(np.exp(LOGB_LO + LB * float(jax.random.uniform(kb))))
    M = float(np.exp(LOGM_LO + LM * float(jax.random.uniform(km))))
    return beta, M


# ----------------------------- SBC for one observable -----------------------------
def run_sbc(name, emulate, precision, shot, truths, fields, count_keys, infer_keys):
    infer = make_infer(emulate, precision, shot)
    posteriors = np.zeros((K_TRIALS, L_THIN, 2))
    t0 = time.time()
    ndiv_tot = 0
    for k in range(K_TRIALS):
        beta_t, M_t = truths[k]
        if shot:
            data = measure_count(fields[k], count_keys[k], M_t)
        else:
            data = measure_density(fields[k], M_t)
        pos, div = infer(jnp.asarray(data), infer_keys[k])
        ndiv_tot += int(np.asarray(div).sum())
        z = np.asarray(pos).reshape(-1, 2)  # pool chains
        beta = np.exp(LOGB_LO + LB / (1 + np.exp(-z[:, 0])))
        M = np.exp(LOGM_LO + LM / (1 + np.exp(-z[:, 1])))
        idx = np.linspace(0, len(beta) - 1, L_THIN).astype(int)  # thin to L independent-ish draws
        posteriors[k, :, 0] = beta[idx]
        posteriors[k, :, 1] = M[idx]
        if (k + 1) % 32 == 0:
            print(f"    [{name}] {k+1}/{K_TRIALS} trials  ({time.time()-t0:.0f}s, div_tot={ndiv_tot})")
    wall = time.time() - t0

    rh = compute_sbc_rank_histogram(truths, posteriors, param_names=["beta", "M"])
    ed = compute_sbc_ecdf_diff(truths, posteriors, param_names=["beta", "M"])
    print(f"  [{name}] DONE {wall:.0f}s  div_tot={ndiv_tot}  "
          f"p_uniform: beta={rh['p_value'][0]:.3f}  M={rh['p_value'][1]:.3f}  "
          f"(n_bins={rh['n_bins']}, L={rh['n_draws']})")
    return {"name": name, "rh": rh, "ed": ed}


def fig_sbc(results):
    nobs = len(results)
    # rank histograms: rows = (beta, M), cols = observables
    fig, axes = plt.subplots(2, 2 * nobs, figsize=(6.0 * nobs, 7.0), squeeze=False)
    for j, res in enumerate(results):
        rh, ed = res["rh"], res["ed"]
        for p, pname in enumerate(["beta", "M"]):
            ax = axes[p, 2 * j]
            plot_sbc_rank_histogram(ax, rh["ranks"][:, p], n_draws=rh["n_draws"],
                                    param_name=f"{pname} ({res['name']})", n_bins=rh["n_bins"],
                                    n_trials=rh["n_trials"])
            ax.set_title(f"{res['name']} {pname}: rank hist (p={rh['p_value'][p]:.3f})")
            axe = axes[p, 2 * j + 1]
            plot_sbc_ecdf_diff(axe, ed["eval_points"][p], ed["ecdf_diff"][p],
                               ed["band_lower"][p], ed["band_upper"][p], param_name=f"{pname}")
            axe.set_title(f"{res['name']} {pname}: ECDF-diff")
    fig.suptitle("v2e SBC-2D: rank uniformity + ECDF bands (calibrated => flat / inside band)", y=1.0)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "v2e_sbc.png")
    fig.savefig(path, dpi=140); plt.close(fig)
    return path


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"v2e SBC-2D  K={K_TRIALS} trials, L={L_THIN} draws/trial, shape={SHAPE}")

    t0 = time.time()
    table = build_emulator(); table.block_until_ready()
    emulate = make_emulate(table)
    print(f"[precompute] emulator {tuple(table.shape)} in {time.time()-t0:.1f}s")

    # K truths from the prior + K fields (shared by density & count) + per-trial keys
    truths = np.array([draw_prior(jax.random.fold_in(jax.random.PRNGKey(777), k)) for k in range(K_TRIALS)])
    fields = [gaussian_random_field(SHAPE, float(truths[k, 0]), jax.random.fold_in(jax.random.PRNGKey(4242), k))
              for k in range(K_TRIALS)]
    count_keys = [jax.random.fold_in(jax.random.PRNGKey(9090), k) for k in range(K_TRIALS)]
    infer_keys = [jax.random.fold_in(jax.random.PRNGKey(55), k) for k in range(K_TRIALS)]
    print(f"  drew {K_TRIALS} truths + fields; beta* in [{truths[:,0].min():.2f},{truths[:,0].max():.2f}], "
          f"M* in [{truths[:,1].min():.1f},{truths[:,1].max():.1f}]")

    print("Building fixed fiducial precisions...")
    prec_d = fiducial_precision("density", 5000)
    prec_c = fiducial_precision("count", 6000)

    print("\nRunning SBC (count = realistic observable)...")
    r_c = run_sbc("count", emulate, prec_c, True, truths, fields, count_keys, infer_keys)
    print("Running SBC (density)...")
    r_d = run_sbc("density", emulate, prec_d, False, truths, fields, count_keys, infer_keys)

    p = fig_sbc([r_c, r_d])
    print(f"\nFigure: {p}")

    print(f"\n{'#'*72}\n  SBC-2D VERDICT (uniform if p>0.05)\n{'#'*72}")
    for r in (r_c, r_d):
        pv = r["rh"]["p_value"]
        verdict = "CALIBRATED" if (pv > 0.05).all() else "MISCALIBRATED (see which param)"
        print(f"  [{r['name']:7s}] p(beta)={pv[0]:.3f}  p(M)={pv[1]:.3f}  -> {verdict}")


if __name__ == "__main__":
    main()

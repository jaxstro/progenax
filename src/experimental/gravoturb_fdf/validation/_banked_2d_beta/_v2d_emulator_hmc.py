r"""SBC-scale optimization: precompute the forward model on a (beta, M) grid + a DIFFERENTIABLE
bilinear interpolator ("emulator"), so per-eval drops from ~7 ms (full O(n^3) forward model) to
~microseconds, while staying JAX-native + differentiable (gradient-based HMC unchanged).

WHY: for the low-D free space (beta, M; alpha depth-gated, b fixed) the forward model is a smooth
deterministic map theta -> 8 band-powers. Precomputing it ONCE on a dense grid and bilinearly
interpolating is exact up to grid density, ~1000x faster per eval, reused across ALL SBC trials, and
differentiable (so logit-NUTS still works). The O(n^2) radial Limber path is the alternative per-eval
optimization but is an approximation and only matters at higher dimensionality; precompute+interp is
strictly better for <=3 params.

Checks (verify-first):
  1. ACCURACY: emulator vs direct forward model at random (beta, M) -> max rel err (want <~1%).
  2. SPEED: emulator eval vs direct -> the speedup factor.
  3. logit-NUTS with the emulator (density + count) -> FAST + still clean (R-hat<1.01, div=0) and
     agreeing with the grid posterior (now also emulator-based, so consistent + fast).

NO production-code edits beyond this scratch file. NO commits.
Run: PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync python -u \
     src/experimental/gravoturb_fdf/validation/_v2d_emulator_hmc.py
"""
import os
import time

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from jax.nn import sigmoid, softplus

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

# ----------------------------- config -----------------------------
SHAPE = (64, 64, 64)
DEPTH = 64
B_FIXED, M_TRUE, BETA_TRUE, ALPHA_TRUE = 0.4, 8.0, 3.0, 2.5
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

# Emulator grid (dense; beta linear, log M uniform -> matches the log-uniform prior geometry).
NB_EMU, NM_EMU = 81, 81
BETA_NODES = jnp.linspace(BETA_LO, BETA_HI, NB_EMU)
LOGM_NODES = jnp.linspace(LOGM_LO, LOGM_HI, NM_EMU)

N_WARMUP, N_SAMPLES, N_CHAINS, MAX_DOUBLINGS = 400, 800, 4, 8
N_GRID = 80
SIGMA_BETA_FORECAST = 0.22


# ----------------------------- direct forward model -----------------------------
def predict_direct(M, beta):
    _kc, P, _nm = angular_bandpowers_2d_limber(SHAPE, beta, M, B_FIXED, ALPHA_TRUE, DEPTH, K_EDGES)
    return P


# ----------------------------- precompute + emulator -----------------------------
def build_emulator():
    """Precompute band-powers on the (beta, logM) node grid -> (NB, NM, n_bins) table (one-time).

    Uses ``jax.lax.map`` (SEQUENTIAL) NOT a single big ``vmap``: vmapping the full 64^3 forward
    model over all NB*NM nodes would materialize NB*NM x 64^3 intermediates (memory blowup). lax.map
    evaluates one node at a time (low, constant memory) -- a one-time ~NB*NM x per-eval cost.
    """
    BB, LM_ = jnp.meshgrid(BETA_NODES, LOGM_NODES, indexing="ij")  # (NB, NM)
    params = jnp.stack([jnp.exp(LM_.ravel()), BB.ravel()], axis=1)  # (N, 2) = (M, beta)
    table = jax.lax.map(lambda p: predict_direct(p[0], p[1]), params)  # (N, n_bins), sequential
    return jnp.asarray(table).reshape(NB_EMU, NM_EMU, table.shape[1])


def make_emulate(table):
    """Differentiable bilinear interpolation of the precomputed table at (M, beta)."""
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


# ----------------------------- logit reparam + NUTS -----------------------------
def z_to_theta(z):
    M = jnp.exp(LOGM_LO + LM * sigmoid(z[1]))
    beta = jnp.exp(LOGB_LO + LB * sigmoid(z[0]))
    return M, beta


def log_prior_jac(z):
    return -(softplus(-z[0]) + softplus(z[0])) - (softplus(-z[1]) + softplus(z[1]))


def make_logdensity(emulate, data, precision, shot):
    data_j, prec_j = jnp.asarray(data), jnp.asarray(precision)

    def ld(z):
        M, beta = z_to_theta(z)
        pred = emulate(M, beta)
        if shot:
            pred = add_poisson_shot(pred, N_BAR_SKY, DEPTH)
        r = pred - data_j
        return -0.5 * r @ (prec_j @ r) + log_prior_jac(z)

    return ld


def local_nuts(ld, init, key, n_warmup, n_samples, n_chains, max_doublings):
    def run_one(ck):
        dk, wk, sk = jax.random.split(ck, 3)
        init0 = init + 0.7 * jax.random.normal(dk, jnp.shape(init))
        warmup = blackjax.window_adaptation(blackjax.nuts, ld, max_num_doublings=max_doublings)
        (state, params), _ = warmup.run(wk, init0, num_steps=n_warmup)
        kernel = blackjax.nuts(ld, **params)

        def step(s, k):
            s, info = kernel.step(k, s)
            return s, (s.position, info.is_divergent, info.num_trajectory_expansions)

        _, (pos, div, depth) = jax.lax.scan(step, state, jax.random.split(sk, n_samples))
        return pos, div, depth

    pos, div, depth = jax.vmap(run_one)(jax.random.split(key, n_chains))
    return {"positions": pos, "divergences": div, "depth": depth}


# ----------------------------- data + covariance -----------------------------
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


# ----------------------------- emulator grid posterior (reference, now fast) -----------------------------
def grid_posterior(emulate, data, precision, shot):
    M_grid = np.geomspace(M_LO, M_HI, N_GRID)
    beta_grid = np.linspace(BETA_LO, BETA_HI, N_GRID)
    MM, BB = np.meshgrid(M_grid, beta_grid, indexing="ij")
    pred = np.asarray(jax.jit(lambda Mf, bf: jax.vmap(emulate)(Mf, bf))(
        jnp.asarray(MM.ravel()), jnp.asarray(BB.ravel())))
    if shot:
        pred = np.asarray(add_poisson_shot(jnp.asarray(pred), N_BAR_SKY, DEPTH))
    resid = pred - data[None, :]
    logp = (-0.5 * np.einsum("gi,ij,gj->g", resid, precision, resid)).reshape(MM.shape)
    P = np.exp(logp - logp.max()); P /= P.sum()
    P_M, P_B = P.sum(axis=1), P.sum(axis=0)
    Bm = float((P_B * beta_grid).sum()); Bs = float(np.sqrt((P_B * (beta_grid - Bm) ** 2).sum()))
    Mm = float((P_M * M_grid).sum()); Ms = float(np.sqrt((P_M * (M_grid - Mm) ** 2).sum()))
    return {"Bm": Bm, "Bs": Bs, "Mm": Mm, "Ms": Ms}


def run_hmc(emulate, name, data, precision, shot, key):
    ld = make_logdensity(emulate, data, precision, shot)
    t0 = time.time()
    out = local_nuts(ld, jnp.array([0.0, 0.0]), key, N_WARMUP, N_SAMPLES, N_CHAINS, MAX_DOUBLINGS)
    z = np.asarray(out["positions"])  # force
    wall = time.time() - t0
    beta = np.exp(LOGB_LO + LB / (1 + np.exp(-z[:, :, 0])))
    M = np.exp(LOGM_LO + LM / (1 + np.exp(-z[:, :, 1])))
    import arviz as az
    g = grid_posterior(emulate, data, precision, shot)
    return {
        "name": name, "beta": beta, "M": M,
        "Bm": float(beta.mean()), "Bs": float(beta.std(ddof=1)),
        "Mm": float(M.mean()), "Ms": float(M.std(ddof=1)),
        "rb": float(az.rhat(beta)), "eb": float(az.ess(beta)),
        "rm": float(az.rhat(M)), "em": float(az.ess(M)),
        "ndiv": int(np.asarray(out["divergences"]).sum()),
        "max_depth": int(np.asarray(out["depth"]).max()), "wall": wall, "grid": g,
    }


def main():
    print("v2d emulator (precompute + differentiable bilinear interp) + logit-NUTS")
    print(f"  shape={SHAPE} truth=(M={M_TRUE},beta={BETA_TRUE}) emulator grid {NB_EMU}x{NM_EMU}")

    # ---- precompute ----
    t0 = time.time()
    table = build_emulator()
    table.block_until_ready()
    t_pre = time.time() - t0
    emulate = make_emulate(table)
    print(f"\n[precompute] table {tuple(table.shape)} built in {t_pre:.1f}s (one-time, reused all trials)")

    # ---- accuracy vs direct ----
    key = jax.random.PRNGKey(7)
    n_test = 60
    bt = np.random.default_rng(0).uniform(BETA_LO + 0.05, BETA_HI - 0.05, n_test)
    mt = np.exp(np.random.default_rng(1).uniform(LOGM_LO + 0.05, LOGM_HI - 0.05, n_test))
    emu = np.asarray(jax.vmap(emulate)(jnp.asarray(mt), jnp.asarray(bt)))  # emulate is tiny -> vmap ok
    pp = jnp.stack([jnp.asarray(mt), jnp.asarray(bt)], axis=1)
    ddirect = np.asarray(jax.lax.map(lambda p: predict_direct(p[0], p[1]), pp))  # sequential (low mem)
    rel = np.abs(emu - ddirect) / np.abs(ddirect)
    print(f"[accuracy] emulator vs direct over {n_test} random (beta,M): "
          f"max rel err={rel.max()*100:.3f}%  median={np.median(rel)*100:.3f}%")

    # ---- speed ----
    import timeit
    ej = jax.jit(emulate); ej(8.0, 3.0).block_until_ready()
    dj = jax.jit(predict_direct); dj(8.0, 3.0).block_until_ready()
    t_emu = timeit.timeit(lambda: ej(8.0, 3.0).block_until_ready(), number=200) / 200
    t_dir = timeit.timeit(lambda: dj(8.0, 3.0).block_until_ready(), number=20) / 20
    print(f"[speed] emulator={t_emu*1e6:.1f}us  direct={t_dir*1e3:.2f}ms  speedup={t_dir/t_emu:.0f}x")

    # ---- data + covariance ----
    g_star = gaussian_random_field(SHAPE, BETA_TRUE, jax.random.PRNGKey(2024))
    d_density = measure_density(g_star, M=M_TRUE)
    d_count = measure_count(g_star, jax.random.PRNGKey(2025), M=M_TRUE)
    print(f"\nBuilding fixed fiducial precisions ({N_REAL_COV} realizations)...")
    t0 = time.time()
    prec_d = fiducial_precision("density", 5000)
    prec_c = fiducial_precision("count", 6000)
    print(f"  built in {time.time()-t0:.1f}s")

    # ---- HMC with emulator (fast) ----
    print("\nRunning emulator logit-NUTS (density + count)...")
    r_d = run_hmc(emulate, "density", d_density, prec_d, False, jax.random.PRNGKey(11))
    r_c = run_hmc(emulate, "count", d_count, prec_c, True, jax.random.PRNGKey(22))

    print(f"\n{'#'*78}\n  EMULATOR HMC RESULTS (per-observable NUTS wall incl. compile)\n{'#'*78}")
    for r in (r_d, r_c):
        g = r["grid"]
        bz = abs(r["Bm"] - BETA_TRUE) / r["Bs"]
        ok = (r["rb"] < 1.01) and (r["rm"] < 1.01) and (r["ndiv"] == 0)
        print(f"  [{r['name']:7s}] HMC beta={r['Bm']:.3f}+/-{r['Bs']:.3f} (truth 3.0,|z|={bz:.2f}) "
              f"| grid {g['Bm']:.3f}+/-{g['Bs']:.3f} (dmean={abs(r['Bm']-g['Bm']):.3f}) "
              f"| M={r['Mm']:.2f}+/-{r['Ms']:.2f}")
        print(f"            Rhat=({r['rb']:.3f},{r['rm']:.3f}) ESS=({r['eb']:.0f},{r['em']:.0f}) "
              f"div={r['ndiv']} max_depth={r['max_depth']} NUTSwall={r['wall']:.1f}s -> {'CLEAN' if ok else 'CHECK'}")
    return r_d, r_c


if __name__ == "__main__":
    main()

r"""D06 — WHERE does sigma(beta) come from? Decompose the per-cluster beta error into its physical
contributors, to answer: is it cosmic variance, the lognormal density transform, or the star shot?

For each "space" along the generative chain we measure, over N_real independent fields at a fixed beta,
the band-power log-log slope; sigma(beta) = std_realizations(slope) / |d slope/d beta|. The gain is
estimated from the mean slope at two bracketing betas. Comparing spaces isolates each effect:

  log-density (proj)      s.sum(LOS)            -> COSMIC-VARIANCE FLOOR (beta best preserved, gain~1)
  density (proj)          exp(s).sum(LOS)       -> + lognormal e^s compression (smaller gain -> bigger sigma)
  log+ counts @ N=1e7     ~no shot               -> + projection only (shot negligible)
  log+ counts @ N=1e5     realistic             -> + Poisson shot (star placement)

This is a DIAGNOSTIC (understanding), not an inference build.
Run: PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync python -u \
     src/experimental/gravoturb_fdf/validation/_d06_sigma_beta_decomposition.py
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy import special

from gravoturb_fdf.field.field import gaussian_random_field
from gravoturb_fdf.field.sampling import sample_cic_counts
from gravoturb_fdf.validation.measure import (
    measure_angular_bandpowers_2d,
    project_counts_los,
    smooth_copula_field,
)

SHAPE, DEPTH = (64, 64, 64), 64
B_FIXED, ALPHA, M_FID = 0.4, 2.5, 8.0
K_EDGES = np.linspace(2.0, 28.0, 10)
BETAS = np.array([2.5, 3.0, 3.5])
N_REAL = 32


def log_plus(N, nb):
    d = np.asarray(N, float) / nb - 1.0
    return np.where(d > 0.0, np.log1p(np.where(d > 0.0, d, 0.0)), d)


def slope(P):
    k = 0.5 * (K_EDGES[:-1] + K_EDGES[1:]); P = np.asarray(P)
    return np.polyfit(np.log(k), np.log(P), 1)[0]


def main():
    k = 0.5 * (K_EDGES[:-1] + K_EDGES[1:]); t0 = time.time()
    spaces = ["logdens_proj", "dens_proj", "logp_1e7", "logp_1e5"]
    sl = {s: np.zeros((len(BETAS), N_REAL)) for s in spaces}
    for bi, beta in enumerate(BETAS):
        for r in range(N_REAL):
            key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(606), bi), r)
            s = smooth_copula_field(gaussian_random_field(SHAPE, float(beta), key), M_FID, B_FIXED, ALPHA)
            sl["logdens_proj"][bi, r] = slope(measure_angular_bandpowers_2d(s.sum(axis=2), K_EDGES))
            sl["dens_proj"][bi, r] = slope(measure_angular_bandpowers_2d(np.exp(s).sum(axis=2), K_EDGES))
            for tag, ns in (("logp_1e7", 1e7), ("logp_1e5", 1e5)):
                nb3 = ns / SHAPE[0] ** 3
                cnt = np.asarray(sample_cic_counts(jnp.asarray(s), nb3, 1, jax.random.fold_in(key, int(ns) % 91 + 1)))
                pc = project_counts_los(cnt, DEPTH, 2).astype(float)
                sl[tag][bi, r] = slope(measure_angular_bandpowers_2d(log_plus(pc, pc.mean()), K_EDGES))
        print(f"  beta={beta:.2f} ({time.time()-t0:.0f}s)")

    bi3 = int(np.argmin(np.abs(BETAS - 3.0)))
    print("\n  WHERE sigma(beta) COMES FROM (per cluster, 64^3 box):")
    print(f"  {'space':<16} {'gain |dslope/dbeta|':>19} {'std(slope)@b=3':>15} {'sigma(beta)':>12}")
    prev = None
    for s in spaces:
        gain = abs(np.polyfit(BETAS, sl[s].mean(axis=1), 1)[0])
        sst = sl[s][bi3].std(ddof=1)
        sigb = sst / gain
        delta = "" if prev is None else f"   (+{sigb-prev:+.3f} vs above)"
        print(f"  {s:<16} {gain:19.3f} {sst:15.3f} {sigb:12.3f}{delta}")
        prev = sigb
    print("\n  Reading: row1 = cosmic-variance floor; row2-row1 = lognormal e^s compression;")
    print("  row4-row3 = Poisson shot (star placement); row3 ~ row1 if projection alone is benign.")
    print(f"  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

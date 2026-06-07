"""Does grid resolution matter for the beta inference?

At fixed beta=3.0, M=8, under the Option-A pointwise generative map (smooth_copula), sweep
n in {48,64,96,128} and report, over a FIXED physical k-band [1,20] (cycles/box):
  (1) forward-model predicted slope vs measured slope  -> predictor accuracy vs n
  (2) per-realization std of the measured slope         -> implied per-cluster sigma(beta) vs n
  (3) amplitude ratio predicted/measured                -> finite-grid residual vs n

Hypotheses: (a) predictor accuracy improves with n (finer grid -> better continuum); (b) sigma(beta)
does NOT improve much with n, because a fixed k-band [1,20] holds ~the same number of modes
regardless of n (resolution adds only HIGH-k, shot-dominated modes) -> beta is box-SIZE
(cosmic-variance) limited, not resolution limited; (c) amplitude residual shrinks with n.

Run: PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync \
       python src/experimental/gravoturb_fdf/validation/_v1c_resolution_sweep.py
"""
import jax
import jax.numpy as jnp
import numpy as np

from gravoturb_fdf.field.field import gaussian_random_field
from gravoturb_fdf.inference.covariance import angular_bandpowers_2d_limber
from gravoturb_fdf.validation.measure import (
    measure_angular_bandpowers_2d,
    smooth_copula_field,
)

B, ALPHA, MACH, BETA = 0.4, 2.5, 8.0, 3.0
N_REAL = 20
SIG_LO, SIG_HI = 1.0, 20.0


def _slope(kc, bp):
    m = (kc >= SIG_LO) & (kc <= SIG_HI) & (bp > 0) & np.isfinite(bp)
    return float(np.polyfit(np.log(kc[m]), np.log(bp[m]), 1)[0])


def _kc_2d(n, edges):
    ky = np.fft.fftfreq(n) * n
    kmag = np.sqrt(ky[:, None] ** 2 + ky[None, :] ** 2)
    return np.array([
        kmag[(kmag >= lo) & (kmag < hi)].mean() if ((kmag >= lo) & (kmag < hi)).any() else np.nan
        for lo, hi in zip(edges[:-1], edges[1:])
    ])


def main():
    key0 = jax.random.PRNGKey(20260607)
    print("=" * 84)
    print(f"  RESOLUTION SWEEP  (beta={BETA}, M={MACH}, pointwise map, n_real={N_REAL}, k-band [1,20])")
    print("=" * 84)
    print(f"  {'n':>4} | {'slope_meas':>11} {'slope_pred':>11} {'|d|':>7} | "
          f"{'sigma_slope':>11} {'~sigma(beta)':>12} | {'amp ratio':>9} | {'n_modes[1,20]':>13}")
    print("-" * 84)
    GAIN = 0.86  # observable-space transfer gain (V1a) -> sigma(beta) = sigma_slope / gain
    for n in (48, 64, 96, 128):
        shape = (n, n, n)
        edges = np.linspace(1.0, n / 2.0, 13)
        kc = _kc_2d(n, edges)
        # count modes in the fit band
        ky = np.fft.fftfreq(n) * n
        kmag2 = np.sqrt(ky[:, None] ** 2 + ky[None, :] ** 2)
        n_modes = int(((kmag2 >= SIG_LO) & (kmag2 <= SIG_HI)).sum())

        slopes = []
        rows = []
        for r in range(N_REAL):
            g = gaussian_random_field(shape, BETA, jax.random.fold_in(key0, r))
            s = smooth_copula_field(g, MACH, B, ALPHA)
            bp = measure_angular_bandpowers_2d(np.exp(s).sum(axis=2), edges)
            rows.append(bp)
            slopes.append(_slope(kc, bp))
        meas = np.mean(rows, axis=0)
        sl_meas, sl_std = float(np.mean(slopes)), float(np.std(slopes, ddof=1))

        _, bp_pred, _ = angular_bandpowers_2d_limber(
            shape, jnp.asarray(BETA), jnp.asarray(MACH), jnp.asarray(B),
            jnp.asarray(ALPHA), jnp.asarray(float(n)), jnp.asarray(edges), 14, 256,
        )
        bp_pred = np.asarray(bp_pred)
        sl_pred = _slope(kc, bp_pred)
        band = (kc >= SIG_LO) & (kc <= SIG_HI) & np.isfinite(bp_pred) & (meas > 0)
        amp = float(np.nanmedian(bp_pred[band] / meas[band]))

        print(f"  {n:>4} | {sl_meas:>11.3f} {sl_pred:>11.3f} {abs(sl_pred-sl_meas):>7.3f} | "
              f"{sl_std:>11.3f} {sl_std/GAIN:>12.3f} | {amp:>9.3f} | {n_modes:>13}")
    print("=" * 84)
    print("  sigma(beta) ~ flat vs n => beta is BOX-SIZE (cosmic-variance) limited, not resolution.")


if __name__ == "__main__":
    main()

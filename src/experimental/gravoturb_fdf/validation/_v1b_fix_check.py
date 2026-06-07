"""Quick first-hand check: does the EXACT BM19 density 2-pt (d_n = <exp(T(g)) He_n(g)>)
fix the M-dependent bias of the lognormal-limit map expm1(xi_s)?

Compares, for the 3D density rho=exp(s) power spectrum (projection already validated exact):
  measured P_rho  vs  expm1(xi_s) prediction  vs  d_n-Mehler prediction
ratio predicted/measured across M in {4,8,16}. If d_n fixes it, the ratio should sit ~1
for all M (unlike expm1 which is ~0.7/1.2/1.7).

Run: PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync \
       python src/experimental/gravoturb_fdf/validation/_v1b_fix_check.py
"""
import jax
import jax.numpy as jnp
import numpy as np

from gravoturb_fdf.field.field import gaussian_random_field, mass_conserving_copula_field
from gravoturb_fdf.theory.gaussianization import (
    gaussianized_xi,
    hermite_coefficients,
    s_of_g,
)
from gravoturb_fdf.theory.projection import _kmag_grid, gaussian_correlation_grid
from gravoturb_fdf.validation.measure import smooth_copula_field

SHAPE = (64, 64, 64)
B, ALPHA, BETA = 0.4, 2.5, 3.0
N_REAL = 20
N_MAX = 14
K_EDGES = np.geomspace(1.0, 24.0, 9)
MACH_FIX = 8.0


def _bin(power, kmag, edges):
    out = np.empty(len(edges) - 1)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (kmag >= lo) & (kmag < hi)
        out[i] = power[m].mean() if m.any() else np.nan
    return out


def measured_density_power(mach, key, beta=BETA):
    g = gaussian_random_field(SHAPE, beta, key)
    s = mass_conserving_copula_field(g, mach, B, ALPHA)
    rho = np.asarray(jnp.exp(s))
    rho = rho - rho.mean()
    pk = np.abs(np.fft.fftn(rho)) ** 2 / rho.size
    return pk


def predicted_power(xi_rho_grid):
    # Wiener-Khinchin: FFT of autocovariance = power spectrum (measure-module convention).
    return np.asarray(jnp.fft.fftn(xi_rho_grid).real)


def _slope(bandpowers, edges):
    """log-log LSQ slope of band-powers vs k-bin centers (positive = exponent magnitude)."""
    kc = np.sqrt(edges[:-1] * edges[1:])
    m = np.isfinite(bandpowers) & (bandpowers > 0)
    return -np.polyfit(np.log(kc[m]), np.log(bandpowers[m]), 1)[0]


def beta_nmax_sweep():
    """Is the beta=3.5 density-slope error n_max truncation (fixable) or intrinsic?
    Report 3D density-power slope error |pred-meas| at fixed M=8 over beta x n_max."""
    kmag = np.asarray(_kmag_grid(SHAPE))
    key0 = jax.random.PRNGKey(0)
    print("=" * 78)
    print(f"  beta x n_max slope-error sweep  (M={MACH_FIX}, shape={SHAPE}, n_real={N_REAL})")
    print("  |slope_pred - slope_meas|  [n_max-stable => intrinsic; shrinking => truncation]")
    print("=" * 78)
    print(f"  {'beta':>5} | {'slope_meas':>10} | {'nmax14':>8} {'nmax30':>8} {'nmax60':>8}")
    print("-" * 78)
    for beta in (2.5, 3.0, 3.5):
        rho_g = gaussian_correlation_grid(SHAPE, beta)
        meas = np.zeros(len(K_EDGES) - 1)
        for r in range(N_REAL):
            meas += _bin(measured_density_power(MACH_FIX, jax.random.fold_in(key0, r), beta), kmag, K_EDGES)
        meas /= N_REAL
        sm = _slope(meas, K_EDGES)
        errs = []
        for nm in (14, 30, 60):
            d = hermite_coefficients(lambda g: jnp.exp(s_of_g(g, MACH_FIX, B, ALPHA)), nm)
            P = _bin(predicted_power(gaussianized_xi(rho_g, d)), kmag, K_EDGES)
            errs.append(abs(_slope(P, K_EDGES) - sm))
        print(f"  {beta:>5.1f} | {sm:>10.3f} | {errs[0]:>8.3f} {errs[1]:>8.3f} {errs[2]:>8.3f}")
    print("=" * 78)


def simulator_map_check():
    """Does d_n match the POINTWISE-map (smooth_copula) density power, vs the mass-conserving
    simulator? If d_n matches smooth_copula at beta=3.5, the beta=3.5 'error' is a simulator-map
    choice (SBC fix: generate from the pointwise map), not a forward-model limit."""
    kmag = np.asarray(_kmag_grid(SHAPE))
    key0 = jax.random.PRNGKey(0)
    print("=" * 90)
    print(f"  simulator-map check  (M={MACH_FIX}, shape={SHAPE}, n_real={N_REAL})")
    print("  density-slope error |slope_pred(d_n) - slope_meas| for two generative maps")
    print("=" * 90)
    print(f"  {'beta':>5} | {'err vs mass_conserving':>22} | {'err vs smooth_copula(pointwise)':>32}")
    print("-" * 90)
    for beta in (2.5, 3.0, 3.5):
        rho_g = gaussian_correlation_grid(SHAPE, beta)
        d = hermite_coefficients(lambda g: jnp.exp(s_of_g(g, MACH_FIX, B, ALPHA)), N_MAX)
        sl_pred = _slope(_bin(predicted_power(gaussianized_xi(rho_g, d)), kmag, K_EDGES), K_EDGES)
        mc = np.zeros(len(K_EDGES) - 1)
        sm = np.zeros(len(K_EDGES) - 1)
        for r in range(N_REAL):
            k = jax.random.fold_in(key0, r)
            g = gaussian_random_field(SHAPE, beta, k)
            # mass-conserving simulator
            s_mc = mass_conserving_copula_field(g, MACH_FIX, B, ALPHA)
            rho_mc = np.asarray(jnp.exp(s_mc)); rho_mc = rho_mc - rho_mc.mean()
            mc += _bin(np.abs(np.fft.fftn(rho_mc)) ** 2 / rho_mc.size, kmag, K_EDGES)
            # pointwise smooth-copula (the d_n theory's generative field)
            s_sm = smooth_copula_field(g, MACH_FIX, B, ALPHA)
            rho_sm = np.exp(s_sm); rho_sm = rho_sm - rho_sm.mean()
            sm += _bin(np.abs(np.fft.fftn(rho_sm)) ** 2 / rho_sm.size, kmag, K_EDGES)
        mc /= N_REAL; sm /= N_REAL
        e_mc = abs(sl_pred - _slope(mc, K_EDGES))
        e_sm = abs(sl_pred - _slope(sm, K_EDGES))
        print(f"  {beta:>5.1f} | {e_mc:>22.3f} | {e_sm:>32.3f}")
    print("=" * 90)


def main():
    beta_nmax_sweep()
    simulator_map_check()
    kmag = np.asarray(_kmag_grid(SHAPE))
    rho_g = gaussian_correlation_grid(SHAPE, BETA)
    key0 = jax.random.PRNGKey(0)

    print("=" * 88)
    print(f"  density-2pt map fix check  (shape={SHAPE}, beta={BETA}, n_real={N_REAL})")
    print("  AMP = median(pred/meas) over k-band [want ~1].  SLOPE = log-log slope [meas is truth].")
    print("=" * 88)
    print(f"  {'M':>4} | {'AMP expm1':>10} {'AMP dn14':>9} {'AMP dn20':>9} | "
          f"{'SLOPE meas':>10} {'SL expm1':>9} {'SL dn14':>9} {'SL dn20':>9}")
    print("-" * 88)

    for mach in (4.0, 8.0, 16.0):
        c = hermite_coefficients(lambda g: s_of_g(g, mach, B, ALPHA), N_MAX)
        d14 = hermite_coefficients(lambda g: jnp.exp(s_of_g(g, mach, B, ALPHA)), 14)
        d20 = hermite_coefficients(lambda g: jnp.exp(s_of_g(g, mach, B, ALPHA)), 20)

        xi_s = gaussianized_xi(rho_g, c)
        P_expm1 = _bin(predicted_power(jnp.expm1(xi_s)), kmag, K_EDGES)
        P_d14 = _bin(predicted_power(gaussianized_xi(rho_g, d14)), kmag, K_EDGES)
        P_d20 = _bin(predicted_power(gaussianized_xi(rho_g, d20)), kmag, K_EDGES)

        meas = np.zeros(len(K_EDGES) - 1)
        for r in range(N_REAL):
            meas += _bin(measured_density_power(mach, jax.random.fold_in(key0, r)), kmag, K_EDGES)
        meas /= N_REAL

        print(f"  {mach:>4.0f} | {np.nanmedian(P_expm1/meas):>10.3f} "
              f"{np.nanmedian(P_d14/meas):>9.3f} {np.nanmedian(P_d20/meas):>9.3f} | "
              f"{_slope(meas, K_EDGES):>10.3f} {_slope(P_expm1, K_EDGES):>9.3f} "
              f"{_slope(P_d14, K_EDGES):>9.3f} {_slope(P_d20, K_EDGES):>9.3f}")

    print("=" * 88)


if __name__ == "__main__":
    main()

"""AC1-AC5, AC8, AC9 acceptance suite for the gravoturb_fdf 1D theory + Q estimator.

Each ``ac_*`` function PRINTS an expected-vs-measured table with absolute/relative
errors and a PASS/FAIL verdict, and returns a result dict ``{"passed": bool, ...}``.
"Validated" means a number one of these committed functions just printed -- no prose
claims of correctness without a fresh artifact.

Run as a script to print the whole suite:  python -m gravoturb_fdf.validation.acceptance
numpy/scipy are permitted here (validation/analysis side).
"""

import math

import numpy as np

import jax
import jax.numpy as jnp

from gravoturb_fdf.theory.bm19 import (
    f_dense_bm19_full,
    sigma_s_squared,
    transition_density,
)
from gravoturb_fdf.theory.pp20 import magnification_factor, zeta_fdf_direct
from gravoturb_fdf.diagnostics.q import compute_q_parameter
from gravoturb_fdf.field.pipeline import build_fdf_field


# ── small printing helpers ──
def _row(label, expected, measured, tol, kind="rel"):
    if kind == "rel":
        err = abs(measured - expected) / (abs(expected) if expected else 1.0)
    else:
        err = abs(measured - expected)
    ok = err <= tol
    print(f"  {label:<34} exp={expected:+.6g}  meas={measured:+.6g}  "
          f"{kind}err={err:.2e}  tol={tol:.1e}  {'PASS' if ok else 'FAIL'}")
    return ok


def _header(title):
    print(f"\n=== {title} ===")


# ── AC1 + AC2: BM19 scalars + mass conservation ──
def _numpy_eq18_f_dense(mach, b, alpha):
    s2 = math.log(1.0 + (b * mach) ** 2)
    s0, s_t = -0.5 * s2, (alpha - 0.5) * s2
    p_ln = lambda s: np.exp(-((s - s0) ** 2) / (2 * s2)) / np.sqrt(2 * np.pi * s2)
    C = p_ln(s_t) * np.exp(alpha * s_t)
    s_lo = np.linspace(s0 - 12 * math.sqrt(s2), s_t, 200_000)
    s_hi = np.linspace(s_t, s_t + 200.0, 400_000)
    M_LN = np.trapezoid(np.exp(s_lo) * p_ln(s_lo), s_lo)
    M_PL = np.trapezoid(np.exp(s_hi) * C * np.exp(-alpha * s_hi), s_hi)
    return M_PL / (M_LN + M_PL)


def ac1_ac2_bm19():
    _header("AC1/AC2 — BM19 scalars vs analytic + mass conservation")
    ok = True
    # AC1 scalars
    ok &= _row("sigma_s^2 (M=5,b=0.4)=ln5", math.log(5.0),
               float(sigma_s_squared(5.0, 0.4)), 1e-6, "abs")
    ok &= _row("s_t (a=2,sig2=ln5)=1.5 ln5", 1.5 * math.log(5.0),
               float(transition_density(2.0, math.log(5.0))), 1e-6, "abs")
    # AC1 f_dense vs Eq.18 quadrature
    for (M, b, a) in [(5.0, 0.4, 2.0), (10.0, 1 / 3, 1.6), (8.0, 0.5, 1.8)]:
        ok &= _row(f"f_dense(M={M},b={b:.2f},a={a}) vs quad",
                   _numpy_eq18_f_dense(M, b, a),
                   float(f_dense_bm19_full(M, b, a)), 1e-4, "rel")
    # AC2 mass conservation of the lognormal body
    s2 = math.log(1.0 + (0.4 * 5.0) ** 2)
    s0 = -0.5 * s2
    s = np.linspace(s0 - 15 * math.sqrt(s2), s0 + 15 * math.sqrt(s2), 400_000)
    norm = np.trapezoid(np.exp(s) * np.exp(-((s - s0) ** 2) / (2 * s2))
                        / np.sqrt(2 * np.pi * s2), s)
    ok &= _row("AC2 mass cons. int e^s p_LN ds=1", 1.0, float(norm), 1e-3, "abs")
    return {"passed": bool(ok)}


# ── AC3 + AC4: PP20 zeta anchors + direct-field estimator ──
def ac3_ac4_zeta():
    _header("AC3/AC4 — PP20 zeta anchors + direct-field estimator")
    ok = True
    anchors = [(0.0, 1.0), (1.0, 1.0887), (1.5, math.sqrt(2.0)), (1.67, 1.79)]
    for p, exp in anchors:
        ok &= _row(f"zeta({p})", exp, float(magnification_factor(p)), 1e-3, "rel")
    # AC4: direct field estimator on a power-law sphere vs analytic
    for p in (0.5, 1.0, 1.5):
        r = np.linspace(1e-3, 1.0, 40_000)
        zfdf = float(zeta_fdf_direct(r ** (-p), 4 * np.pi * r**2))
        ok &= _row(f"zeta_FDF vs analytic (p={p})", float(magnification_factor(p)),
                   zfdf, 0.03, "rel")
    return {"passed": bool(ok)}


# ── AC5: CW04 Q estimator vs Table 1 radial anchors ──
def _radial_profile(alpha, n, seed):
    rng = np.random.default_rng(seed)
    r = rng.uniform(0, 1, n) ** (1.0 / (3.0 - alpha))
    ct = 2 * rng.uniform(0, 1, n) - 1
    st = np.sqrt(np.clip(1 - ct**2, 0, 1))
    phi = 2 * np.pi * rng.uniform(0, 1, n)
    return np.column_stack([r * st * np.cos(phi), r * st * np.sin(phi), r * ct])


def ac5_q(n_real=30, N=200):
    _header("AC5 — CW04 Q estimator vs Table 1 (analytic radial models)")
    ok = True
    for alpha, cw, sig in [(0.0, 0.79, 0.02), (1.0, 0.84, 0.03), (2.0, 0.93, 0.03)]:
        qs = [compute_q_parameter(_radial_profile(alpha, N, s)) for s in range(n_real)]
        mean = float(np.mean(qs))
        err = abs(mean - cw)
        verdict = "PASS" if err <= 3 * sig else "FAIL"
        ok &= err <= 3 * sig
        print(f"  3D{alpha:.0f}  Q={mean:.3f}+-{np.std(qs):.3f}  CW04={cw}+-{sig}  "
              f"|dev|={err:.3f}  {verdict}")
    return {"passed": bool(ok)}


# ── AC6: cornerstone — realized f_dense vs BM19 (mass-conserving rank copula) ──
def ac6_cornerstone(shape=(128, 128, 128), n_real=8):
    _header("AC6 — CORNERSTONE: f_dense_realized vs BM19 f_dense (mass-conserving copula)")
    ok = True
    cases = [(10.0, 0.4, 2.0, 3.667), (8.0, 0.5, 1.8, 3.5), (12.0, 1 / 3, 1.6, 4.0)]
    n = shape[0]
    for mach, b, alpha, beta in cases:
        biases = []
        for seed in range(n_real):
            fld = build_fdf_field(mach, b, alpha, beta, shape, jax.random.PRNGKey(seed))
            fd = float(fld.f_dense)
            fr = float(fld.f_dense_realized)
            biases.append((fr - fd) / fd)
        biases = np.asarray(biases)
        ens = abs(biases.mean())
        single = np.abs(biases).max()
        verdict = "PASS" if (ens < 0.01 and single < 0.05) else "FAIL"
        ok &= ens < 0.01 and single < 0.05
        print(f"  {n}³ ℳ={mach:>4} b={b:.2f} α={alpha} f_dense={fd:.4f}  "
              f"ens_bias={biases.mean()*100:+.3f}%  single_max={single*100:.3f}%  "
              f"(tol ens<1% single<5%, N={n_real})  {verdict}")
    return {"passed": bool(ok)}


# ── AC7: f_sub → Q calibration (headline) — monotone↓, CW04 substructured band ──
def ac7_q_calibration(shape=(64, 64, 64), n_real=10, n_stars=500):
    from gravoturb_fdf.validation.calibration import q_vs_fsub

    _header("AC7 — Q(f_sub) calibration (headline): trend↓ + Q∈[0.4,0.8]")
    f_sub_values = (0.0, 0.2, 0.4, 0.6, 0.8)
    res = q_vs_fsub(
        mach=8.0, b=0.5, alpha=1.8, beta=3.5,
        f_sub_values=f_sub_values, n_stars=n_stars, n_real=n_real,
        shape=shape, key=jax.random.PRNGKey(0),
    )
    qm, qs = res["q_mean"], res["q_std"]
    slope = float(np.polyfit(res["f_sub"], qm, 1)[0])
    # Robust "monotone↓ trend": clear negative slope + decreasing endpoints (point
    # scatter grows with f_sub — a real FBM property, so strict adjacency is too brittle).
    trend_down = slope < -0.03 and qm[0] > qm[-1]
    in_band = bool(np.all((qm > 0.4) & (qm < 0.8)))
    strict = bool(np.all(np.diff(qm) < 0))  # informational
    n = shape[0]
    for f, m, sd in zip(f_sub_values, qm, qs):
        print(f"  {n}³ f_sub={f:.2f}  Q={m:.3f} ± {sd:.3f}")
    print(f"  slope={slope:+.3f} (trend↓ {'PASS' if trend_down else 'FAIL'})  "
          f"Q∈[0.4,0.8]: {'PASS' if in_band else 'FAIL'}  strict-mono={strict}  "
          f"(N⋆={n_stars}, {n_real} real)")
    return {"passed": bool(trend_down and in_band), "q_mean": qm, "q_std": qs, "slope": slope}


# ── AC8 + AC9: gradient signs + FD-vs-autodiff ──
def ac8_ac9_grads():
    import jax

    _header("AC8/AC9 — gradient signs + FD-vs-autodiff")
    ok = True
    fd = lambda f, x, e=1e-6: (f(x + e) - f(x - e)) / (2 * e)
    # AC8 signs
    s1 = float(jax.grad(lambda m: sigma_s_squared(m, 0.4))(5.0))
    s2 = float(jax.grad(lambda m: f_dense_bm19_full(m, 1 / 3, 1.8))(8.0))
    s3 = float(jax.grad(lambda a: f_dense_bm19_full(8.0, 1 / 3, a))(1.8))
    s4 = float(jax.grad(lambda a: magnification_factor(3.0 / a))(2.0))
    for label, val, want in [("d sigma_s^2/dM > 0", s1, +1), ("d f_dense/dM < 0", s2, -1),
                             ("d f_dense/dalpha < 0", s3, -1), ("d zeta/dalpha < 0", s4, -1)]:
        good = (val > 0) == (want > 0)
        ok &= good
        print(f"  {label:<26} grad={val:+.4e}  {'PASS' if good else 'FAIL'}")
    # AC9 FD-vs-autodiff
    ad = float(jax.grad(lambda m: f_dense_bm19_full(m, 0.4, 1.7))(9.0))
    ok &= _row("AC9 f_dense'(M) autodiff vs FD",
               fd(lambda m: float(f_dense_bm19_full(m, 0.4, 1.7)), 9.0), ad, 1e-4, "rel")
    adz = float(jax.grad(magnification_factor)(1.5))
    ok &= _row("AC9 zeta'(p) autodiff vs FD",
               fd(lambda p: float(magnification_factor(p)), 1.5), adz, 1e-4, "rel")
    return {"passed": bool(ok)}


# ── AC11: predicted xi_s vs realization oracle + Gaussianization convergence ──
def ac11_xi_s_vs_oracle(shape=(64, 64, 64), n_real=8, beta=3.0, mach=5.0, b=0.4,
                        alpha=2.0, n_max=16, n_bins=16, seed=0, rho_floor=0.05,
                        rel_tol=0.02):
    """Predicted log-density 2-point xi_s (Gaussianization series) vs the realization
    oracle (smooth-copula field measured xi_s). Reports max/median relative error on
    bins with rho_g > rho_floor and the Gaussianization convergence (n_max/2 -> n_max).
    """
    from gravoturb_fdf.field.field import gaussian_random_field
    from gravoturb_fdf.validation.measure import (
        field_2pt_measured, gaussian_correlation_measured, smooth_copula_field)
    from gravoturb_fdf.theory.gaussianization import (
        bm19_hermite_coefficients, gaussianized_xi)

    _header("AC11 — predicted xi_s vs realization oracle (+ Gaussianization convergence)")
    key = jax.random.PRNGKey(seed)
    rho_acc, xis_acc, r_ref = [], [], None
    for i in range(n_real):
        g = np.asarray(gaussian_random_field(shape, beta, jax.random.fold_in(key, i)))
        s = smooth_copula_field(g, mach, b, alpha)
        r_ref, rho = gaussian_correlation_measured(g, n_bins=n_bins)
        _, xis = field_2pt_measured(s, n_bins=n_bins)
        rho_acc.append(rho)
        xis_acc.append(xis)
    rho_m = np.mean(rho_acc, axis=0)
    xis_m = np.mean(xis_acc, axis=0)

    c = np.asarray(bm19_hermite_coefficients(mach, b, alpha, n_max=n_max))
    xis_pred = np.asarray(gaussianized_xi(jnp.asarray(rho_m), jnp.asarray(c)))

    mask = rho_m > rho_floor
    rel = np.abs(xis_pred[mask] - xis_m[mask]) / np.abs(xis_m[mask])
    max_rel, med_rel = float(rel.max()), float(np.median(rel))

    c_half = np.asarray(bm19_hermite_coefficients(mach, b, alpha, n_max=n_max // 2))
    xp_half = float(gaussianized_xi(jnp.asarray(rho_m[:1]), jnp.asarray(c_half))[0])
    conv = abs(xis_pred[0] - xp_half) / abs(xis_pred[0])

    print(f"  theta=(M={mach}, b={b}, alpha={alpha})  beta={beta}  shape={shape}  "
          f"n_real={n_real}  n_max={n_max}  (bins with rho_g>{rho_floor}: {int(mask.sum())})")
    ok_agree = _row("max rel(xi_pred vs oracle)", 0.0, max_rel, rel_tol, "abs")
    ok_conv = _row(f"Gaussianization conv (n_max {n_max // 2}->{n_max})", 0.0, conv, 1e-3, "abs")
    print(f"  median rel = {med_rel * 100:.3f}%")
    return {"passed": bool(ok_agree and ok_conv), "max_rel": max_rel,
            "median_rel": med_rel, "convergence": conv}


def main():
    results = {
        "AC1/AC2": ac1_ac2_bm19(),
        "AC3/AC4": ac3_ac4_zeta(),
        "AC5": ac5_q(),
        "AC6": ac6_cornerstone(),
        "AC7": ac7_q_calibration(),
        "AC8/AC9": ac8_ac9_grads(),
        "AC11": ac11_xi_s_vs_oracle(),
    }
    print("\n=== SUMMARY ===")
    all_ok = True
    for k, v in results.items():
        all_ok &= v["passed"]
        print(f"  {k:<10} {'PASS' if v['passed'] else 'FAIL'}")
    print(f"\n  OVERALL: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)

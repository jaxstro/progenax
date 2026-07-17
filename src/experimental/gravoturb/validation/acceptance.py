"""AC1-AC5, AC8, AC9 acceptance suite for the gravoturb 1D theory + Q estimator.

Each ``ac_*`` function PRINTS an expected-vs-measured table with absolute/relative
errors and a PASS/FAIL verdict, and returns a result dict ``{"passed": bool, ...}``.
"Validated" means a number one of these committed functions just printed -- no prose
claims of correctness without a fresh artifact.

Run as a script to print the whole suite:  python -m gravoturb.validation.acceptance
numpy/scipy are permitted here (validation/analysis side).

**CLOSED TO NEW ACCEPTANCE SECTIONS (2026-07-16 architecture review):** this file is
at 2× the file-size target and accretes one section per AC. New gates open new
scripts — cluster-IC gates live in cluster_acceptance.py; Phase-3 coupling gates go
in a new coupling_acceptance.py.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from gravoturb.diagnostics.q import compute_q_parameter
from gravoturb.realization.pipeline import build_turbulent_field
from gravoturb.theory.dense_gas_sfr import magnification_factor, zeta_from_field
from gravoturb.theory.density_pdf import (
    dense_mass_fraction,
    sigma_s_squared,
    transition_density,
)


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
                   float(dense_mass_fraction(M, b, a)), 1e-4, "rel")
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
        zfdf = float(zeta_from_field(r ** (-p), 4 * np.pi * r**2))
        ok &= _row(f"zeta (direct field) vs analytic (p={p})", float(magnification_factor(p)),
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
            fld = build_turbulent_field(mach, b, alpha, beta, shape, jax.random.PRNGKey(seed))
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
    from gravoturb.validation.calibration import q_vs_fsub

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
    s2 = float(jax.grad(lambda m: dense_mass_fraction(m, 1 / 3, 1.8))(8.0))
    s3 = float(jax.grad(lambda a: dense_mass_fraction(8.0, 1 / 3, a))(1.8))
    s4 = float(jax.grad(lambda a: magnification_factor(3.0 / a))(2.0))
    for label, val, want in [("d sigma_s^2/dM > 0", s1, +1), ("d f_dense/dM < 0", s2, -1),
                             ("d f_dense/dalpha < 0", s3, -1), ("d zeta/dalpha < 0", s4, -1)]:
        good = (val > 0) == (want > 0)
        ok &= good
        print(f"  {label:<26} grad={val:+.4e}  {'PASS' if good else 'FAIL'}")
    # AC9 FD-vs-autodiff
    ad = float(jax.grad(lambda m: dense_mass_fraction(m, 0.4, 1.7))(9.0))
    ok &= _row("AC9 f_dense'(M) autodiff vs FD",
               fd(lambda m: float(dense_mass_fraction(m, 0.4, 1.7)), 9.0), ad, 1e-4, "rel")
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
    from gravoturb.diagnostics.measure import (
        field_2pt_measured,
        gaussian_correlation_measured,
        smooth_copula_field,
    )
    from gravoturb.realization.gaussian_field import gaussian_random_field
    from gravoturb.theory.log_correlations import (
        gaussianized_xi,
        log_density_hermite_coefficients,
    )

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

    c = np.asarray(log_density_hermite_coefficients(mach, b, alpha, n_max=n_max))
    xis_pred = np.asarray(gaussianized_xi(jnp.asarray(rho_m), jnp.asarray(c)))

    mask = rho_m > rho_floor
    rel = np.abs(xis_pred[mask] - xis_m[mask]) / np.abs(xis_m[mask])
    max_rel, med_rel = float(rel.max()), float(np.median(rel))

    c_half = np.asarray(log_density_hermite_coefficients(mach, b, alpha, n_max=n_max // 2))
    xp_half = float(gaussianized_xi(jnp.asarray(rho_m[:1]), jnp.asarray(c_half))[0])
    conv = abs(xis_pred[0] - xp_half) / abs(xis_pred[0])

    print(f"  theta=(M={mach}, b={b}, alpha={alpha})  beta={beta}  shape={shape}  "
          f"n_real={n_real}  n_max={n_max}  (bins with rho_g>{rho_floor}: {int(mask.sum())})")
    ok_agree = _row("max rel(xi_pred vs oracle)", 0.0, max_rel, rel_tol, "abs")
    ok_conv = _row(f"Gaussianization conv (n_max {n_max // 2}->{n_max})", 0.0, conv, 1e-3, "abs")
    print(f"  median rel = {med_rel * 100:.3f}%")
    return {"passed": bool(ok_agree and ok_conv), "max_rel": max_rel,
            "median_rel": med_rel, "convergence": conv}


# ── AC11b: physical rank/mass-conserving copula xi_s vs analytic prediction ──
def ac11b_rank_copula_equivalence(shape=(64, 64, 64), n_real=6, beta=3.0, mach=5.0,
                                  b=0.4, alpha=2.0, n_max=16, n_bins=14, seed=0,
                                  rho_floor=0.05, rel_tol=0.03):
    """Map-mismatch check: the analytic xi_s prediction (point-map T=F^{-1} o Phi) also
    describes the PHYSICAL simulator fields, which use the empirical-rank copulas
    (rank_copula_field, mass_conserving_copula_field). Both should match the prediction
    to ~<1% on rho_g>floor bins -- the empirical CDF -> Phi, and the mass-averaging
    perturbs the marginal (1-pt) but barely the log-density 2-pt. (The discrepancy is
    noise-limited, not a systematic rank-vs-Phi bias.)
    """
    from gravoturb.diagnostics.measure import (
        field_2pt_measured,
        gaussian_correlation_measured,
    )
    from gravoturb.realization.copula import (
        mass_conserving_copula_field,
        rank_copula_field,
    )
    from gravoturb.realization.gaussian_field import gaussian_random_field
    from gravoturb.theory.log_correlations import (
        gaussianized_xi,
        log_density_hermite_coefficients,
    )

    _header("AC11b — rank/mass-conserving copula xi_s vs analytic prediction (map-mismatch)")
    key = jax.random.PRNGKey(seed)
    rho_a, xr_a, xm_a = [], [], []
    for i in range(n_real):
        g = gaussian_random_field(shape, beta, jax.random.fold_in(key, i))
        s_rank = np.asarray(rank_copula_field(g, mach, b, alpha)).reshape(shape)
        s_mass = np.asarray(mass_conserving_copula_field(g, mach, b, alpha)).reshape(shape)
        _, rho = gaussian_correlation_measured(np.asarray(g), n_bins=n_bins)
        _, xr = field_2pt_measured(s_rank, n_bins=n_bins)
        _, xm = field_2pt_measured(s_mass, n_bins=n_bins)
        rho_a.append(rho)
        xr_a.append(xr)
        xm_a.append(xm)
    rho_m = np.mean(rho_a, axis=0)
    xr_m, xm_m = np.mean(xr_a, axis=0), np.mean(xm_a, axis=0)
    c = np.asarray(log_density_hermite_coefficients(mach, b, alpha, n_max=n_max))
    xpred = np.asarray(gaussianized_xi(jnp.asarray(rho_m), jnp.asarray(c)))

    mask = rho_m > rho_floor
    rr = np.abs(xr_m[mask] - xpred[mask]) / np.abs(xpred[mask])
    rm = np.abs(xm_m[mask] - xpred[mask]) / np.abs(xpred[mask])
    print(f"  theta=(M={mach}, b={b}, alpha={alpha})  beta={beta}  shape={shape}  "
          f"n_real={n_real}  (bins rho_g>{rho_floor}: {int(mask.sum())})")
    ok1 = _row("rank_copula xi_s vs pred (max rel)", 0.0, float(rr.max()), rel_tol, "abs")
    ok2 = _row("mass_conserving xi_s vs pred (max rel)", 0.0, float(rm.max()), rel_tol, "abs")
    print(f"  medians: rank {np.median(rr) * 100:.3f}%  mass {np.median(rm) * 100:.3f}%")
    return {"passed": bool(ok1 and ok2), "rank_max_rel": float(rr.max()),
            "mass_max_rel": float(rm.max())}


# ── AC12: Limber-projected analytic 2-pt vs realization oracle ──
def ac12_limber_projection_vs_oracle(shape=(48, 48, 48), n_real=48, beta=3.0, n_bins=12,
                                     seed=0, w_floor=0.1, abs_tol=0.03):
    """The analytic 3-D correlation rho_g, Limber-projected along the LOS, reproduces the
    measured 2-D correlation of the column-projected realization (normalized to w(0)=1).

    Criterion: max ABSOLUTE error ``max|w_pred - w_meas|`` over the signal bins
    (w_pred>w_floor). The agreement is flat-in-absolute (~0.007 of the peak at n_real=48), so
    absolute error is the robust metric -- a relative error divides that flat residual by
    w->w_floor at the outer bin and spuriously explodes (the diagnosis Anna prompted
    2026-06-05). The residual is cosmic-variance-limited (LOS projection discards modes;
    steep red beta=3 -> few low-k modes coherently tilt the normalized curve): seed=0 n_real=48
    -> 0.008, worst of seeds 0-4 -> 0.026, so abs_tol=0.03 is robust and deterministic at the
    fiducial seed=0. main() runs that 48^3 x 48 ensemble."""
    from gravoturb.diagnostics.measure import autocovariance_3d, radial_average
    from gravoturb.realization.gaussian_field import gaussian_random_field
    from gravoturb.theory.projection import (
        gaussian_correlation_grid,
        limber_project_grid,
    )

    _header("AC12 — Limber-projected analytic 2-pt vs realization oracle")
    proj = np.asarray(limber_project_grid(gaussian_correlation_grid(shape, beta), los_axis=2))
    _, w_pred = radial_average(proj / proj[0, 0], n_bins=n_bins)

    key = jax.random.PRNGKey(seed)
    xi_sum = None
    for i in range(n_real):
        g = np.asarray(gaussian_random_field(shape, beta, jax.random.fold_in(key, i)))
        xi_col = autocovariance_3d(g.sum(axis=2))
        xi_sum = xi_col if xi_sum is None else xi_sum + xi_col
    xi_mean = xi_sum / n_real  # average autocov FIRST, then normalize (unbiased)
    _, w_meas = radial_average(xi_mean / xi_mean[0, 0], n_bins=n_bins)

    mask = w_pred > w_floor
    absd = np.abs(w_pred[mask] - w_meas[mask])
    rel = absd / np.abs(w_pred[mask])
    print(f"  beta={beta}  shape={shape}  n_real={n_real}  (bins w_pred>{w_floor}: {int(mask.sum())})")
    ok = _row("Limber proj 2-pt vs oracle (max |dw|)", 0.0, float(absd.max()), abs_tol, "abs")
    print(f"  median |dw| = {np.median(absd):.4f}  (outer-bin max rel = {rel.max() * 100:.1f}%, "
          f"a small-denominator artifact -- abs residual is flat)")
    return {"passed": bool(ok), "max_abs": float(absd.max()),
            "median_abs": float(np.median(absd)), "max_rel": float(rel.max())}


def ac13_cic_vs_oracle(shape=(48, 48, 48), n_real=24, c=4, beta=3.0, mach=5.0, b=0.4,
                       alpha=3.0, n_bar=40, seed=0, n_max=16,
                       cox_tol=0.04, theory_tol=0.08, l1_tol=0.08):
    """Counts-in-cells: the Cox / compound-Poisson predictions vs mock star counts.

    Mock = smooth-copula log-density field (the AC11/AC12 oracle; rank/mass-copula
    equivalence is AC11b) -> Cox star sampling (cloud_to_stars, f_sub=0, intensity ~ rho)
    -> counts in CUBIC cells of side c (box window). Validates: (1) the CIC variance formula
    sigma^2_N = N_bar + N_bar^2 xi_bar reproduces Var(N) using the field's OWN cell variance
    (sampler+formula, tight); (2) the differentiable Route-A prediction xi_bar_rho(box c) vs
    the measured clustering; (3) the compound-Poisson P(N) vs the mock count histogram.
    The linear-rho moment is genuinely tail-sensitive (the old realization pipeline scattered
    ~90%); Route A reaches a few % at n_real=24 (cosmic-variance-limited at small n_real)."""
    from gravoturb.diagnostics.measure import smooth_copula_field
    from gravoturb.realization.gaussian_field import gaussian_random_field
    from gravoturb.realization.pipeline import TurbulentField, cloud_to_stars
    from gravoturb.theory.counts_in_cells import (
        cell_averaged_xi_rho,
        cic_variance,
        count_distribution,
    )
    from gravoturb.theory.projection import box_window_sq_grid

    _header("AC13 — counts-in-cells sigma^2_N and P(N) vs mock star counts")
    n = shape[0]
    M = n // c
    ncell = M**3
    n_stars = int(n_bar * ncell)
    Nbar = n_stars / ncell
    w2 = box_window_sq_grid(shape, c)
    s_t = jnp.asarray(transition_density(alpha, sigma_s_squared(mach, b)))
    xi_pred = float(cell_averaged_xi_rho(shape, beta, float(c), mach, b, alpha,
                                         n_max=n_max, w2=w2))

    n_hist = int(Nbar * 10) + 50
    hist = np.zeros(n_hist)
    key = jax.random.PRNGKey(seed)
    var_list, xibar_list = [], []
    for i in range(n_real):
        g = gaussian_random_field(shape, beta, jax.random.fold_in(key, i))
        s = jnp.asarray(smooth_copula_field(g, mach, b, alpha))
        field = TurbulentField(s=s, s_t=s_t, f_dense=jnp.asarray(0.0),
                         f_dense_realized=jnp.asarray(0.0), low_resolution=False)
        pos = np.asarray(cloud_to_stars(field, 0.0, n_stars, jax.random.fold_in(key, 1000 + i)))
        ijk = np.floor(pos * M).astype(int) % M
        flat = (ijk[:, 0] * M + ijk[:, 1]) * M + ijk[:, 2]
        cnt = np.bincount(flat, minlength=ncell)
        var_list.append(cnt.var())
        hist += np.bincount(cnt, minlength=n_hist)[:n_hist]
        rt = np.asarray(jnp.exp(s) / jnp.mean(jnp.exp(s))).reshape(
            M, c, M, c, M, c).mean(axis=(1, 3, 5))
        xibar_list.append(rt.var())

    varN = float(np.mean(var_list))
    xibar_orc = float(np.mean(xibar_list))
    sig2_pred = float(cic_variance(Nbar, xi_pred))
    Pmock = hist / hist.sum()
    Nvals = jnp.arange(0, n_hist)
    Ppred = np.asarray(count_distribution(Nvals, Nbar, shape, beta, float(c), mach, b, alpha,
                                          n_max=n_max, w2=w2, n_s=2048, s_max=40.0))
    l1 = float(np.sum(np.abs(Pmock - Ppred)))

    print(f"  shape={shape} c={c} (M={M}, {ncell} cells) n_stars={n_stars} "
          f"N_bar={Nbar:.1f} n_real={n_real}")
    ok_cox = _row("Cox: Var(N) vs N_bar+N_bar^2 xi_orc", Nbar + Nbar**2 * xibar_orc, varN, cox_tol)
    ok_thy = _row("Route-A xi_bar_rho(box) vs measured", xibar_orc, xi_pred, theory_tol)
    ok_sig = _row("predicted sigma^2_N vs Var(N)", varN, sig2_pred, theory_tol)
    ok_pn = _row("P(N) total-variation L1", 0.0, l1, l1_tol, "abs")
    print(f"  (P(N) mean: mock={float(np.sum(np.arange(n_hist) * Pmock)):.2f} "
          f"pred={float(np.sum(np.asarray(Nvals) * Ppred)):.2f}; sum Ppred={Ppred.sum():.3f})")
    ok = ok_cox and ok_thy and ok_sig and ok_pn
    return {"passed": bool(ok), "sigma2_rel": abs(varN - sig2_pred) / varN,
            "xi_rel": abs(xi_pred - xibar_orc) / xibar_orc, "l1": l1}


def ac14_grad_validation(shape=(24, 24, 24), R=2.0, mach=5.0, b=0.4, alpha=2.5, beta=3.0,
                         n_real_crn=48, eps_crn=0.1, seed=0, fd_tol=1e-4, beta_rel_tol=0.20):
    """AC14 — gradient validation of the differentiable predicted-statistics layer.

    (1) Autodiff vs CENTRAL finite-difference for xi_bar_rho(R) in each of (mach,b,alpha,beta)
        -- the analytic gradients the Fisher/HMC phases ride on (rel < fd_tol; the unit tests
        sweep h to ~1e-5). This is the rigorous gradient-correctness check.
    (2) The analytic BETA path (Decision #3, no soft-sort) vs the SIMULATOR's beta-response, on
        the LOG-density variance sigma_s^2(R) -- the beta-carrier, finite-variance for any alpha
        (the linear xi_bar_rho derivative is tail-heavy and a poor estimator). A paired-CRN
        finite difference (one white-noise field realized at beta+/-eps; common random numbers
        cancel the realization structure) measures d sigma_s^2/d beta from the actual mock.
        The residual (~8% on a 24^3, R=2 grid) is the FORWARD-model discretization bias (the
        same theory-vs-simulator fidelity AC13 drives to ~2.4% at the production grid), NOT a
        gradient error -- part (1) proves the gradient math to ~1e-5. Reported as rel agreement."""
    from gravoturb.diagnostics.measure import (
        smooth_copula_field,
        smoothed_linear_variance,
    )
    from gravoturb.realization.gaussian_field import gaussian_random_field
    from gravoturb.theory.counts_in_cells import (
        cell_averaged_xi_rho,
        smoothed_log_variance,
    )
    from gravoturb.theory.projection import top_hat_window

    _header("AC14 — gradient validation (autodiff vs FD; analytic beta vs simulator CRN)")

    def xib(mach_, b_, alpha_, beta_):
        return cell_averaged_xi_rho(shape, beta_, R, mach_, b_, alpha_, n_max=16)

    base = dict(mach_=mach, b_=b, alpha_=alpha, beta_=beta)
    ok = True
    for name, key_, x0, h in [("mach", "mach_", mach, 1e-4), ("b", "b_", b, 1e-5),
                              ("alpha", "alpha_", alpha, 1e-5), ("beta", "beta_", beta, 1e-4)]:
        ad = float(jax.grad(lambda v: xib(**{**base, key_: v}))(x0))
        fd = float((xib(**{**base, key_: x0 + h}) - xib(**{**base, key_: x0 - h})) / (2 * h))
        ok &= _row(f"d xi_bar_rho / d {name} (AD vs FD)", fd, ad, fd_tol)

    # (2) analytic beta-gradient of the LOG-density sigma_s^2(R) vs the simulator (paired CRN)
    ad_beta = float(jax.grad(
        lambda be: smoothed_log_variance(shape, be, R, mach, b, alpha, n_max=16))(beta))
    key = jax.random.PRNGKey(seed)
    derivs = []
    for i in range(n_real_crn):
        k = jax.random.fold_in(key, i)
        sp = np.asarray(smooth_copula_field(gaussian_random_field(shape, beta + eps_crn, k),
                                            mach, b, alpha))
        sm = np.asarray(smooth_copula_field(gaussian_random_field(shape, beta - eps_crn, k),
                                            mach, b, alpha))
        derivs.append((smoothed_linear_variance(sp, R, top_hat_window)
                       - smoothed_linear_variance(sm, R, top_hat_window)) / (2 * eps_crn))
    sim = float(np.mean(derivs))
    se = float(np.std(derivs) / np.sqrt(len(derivs)))
    rel = abs(ad_beta - sim) / (abs(sim) + 1e-30)
    print(f"  analytic d sigma_s^2/d beta = {ad_beta:+.5f}   CRN-simulator = {sim:+.5f} +/- "
          f"{se:.5f}  (n_real={n_real_crn}, eps={eps_crn}; nsig={abs(ad_beta - sim) / (se + 1e-30):.1f})")
    ok_beta = _row("analytic-vs-simulator beta-grad", 0.0, rel, beta_rel_tol, "abs")
    print("  (residual = forward-model discretization bias, shrinks with grid -- cf AC13; "
          "the gradient MATH is rel~1e-5 above)")
    ok = ok and ok_beta
    return {"passed": bool(ok), "beta_grad_rel": rel}


def ac15_fisher_forecast(shape=(32, 32, 32), n_real=150, c=4, beta=3.0, mach=5.0, b=0.4,
                         alpha=2.5, n_bar=30, seed=0, n_max=14):
    """AC15 -- Fisher forecast: a FIELD-LEVEL INFORMATION UPPER BOUND on (mach, alpha, beta).

    Builds the mock covariance of d = [P_s(k_i), sigma^2_N(c)] from n_real realization mocks
    (Hartlap-corrected precision), the analytic Jacobian J = d d/d theta, and F = J^T Cinv J.
    Reports marginal sigma(mach,alpha,beta) (b fixed -- the data constrains (mach,b) only via
    sigma_s^2=ln(1+(b mach)^2), so the full 4-param F is rank-3 singular), the tightening with
    survey volume (sigma ~ 1/sqrt(V)), and the mach-b degeneracy.

    *** SCOPING (Anna 2026-06-05) -- these are OPTIMISTIC, an UPPER BOUND, because: ***
    (1) the band-power block is measured from the CONTINUOUS log-density field (no shot noise),
        i.e. it assumes the density is perfectly observed; real star counts add shot noise and
        log(counts) is ill-defined where counts->0, so the realistic sigma(beta) is weaker;
    (2) N_bar~30/cell here is a ~10^4-star box, far richer than a real ~10^2-10^3-member cluster
        -- the per-box sigmas are not per-cluster; the science is POPULATION-stacked (the 1/sqrt(V)).
    The ROBUST results are the hierarchy (beta best-constrained, alpha weakest -- alpha is a
    PDF-TAIL slope needing the high-N tail of P(N)), the 1/sqrt(V) scaling, and the mach-b
    degeneracy. The realistic, shot-noise-consistent forecast (with alpha rescued by the
    compound-Poisson P(N) likelihood) is the Phase-6 deliverable (AC16, inherently star-level)."""
    from gravoturb.diagnostics.measure import smooth_copula_field
    from gravoturb.inference.covariance import (
        hartlap_factor,
        measured_bandpowers,
        mock_covariance,
    )
    from gravoturb.inference.fisher import fisher_matrix, marginal_errors
    from gravoturb.realization.gaussian_field import gaussian_random_field

    _header("AC15 -- Fisher forecast (FIELD-LEVEL UPPER BOUND; band-powers + CIC variance)")
    print("  NOTE: optimistic upper bound -- band-powers are field-level (no star shot noise) and")
    print("        the box is ~10^4 stars (not a real ~10^2-10^3 cluster). Robust: the hierarchy,")
    print("        1/sqrt(V) scaling, mach-b degeneracy. Realistic forecast = Phase 6 (P(N)+HMC).")
    k_edges = jnp.linspace(2.0, 11.0, 5)  # 4 band-power bins
    n = shape[0]
    M = n // c
    ncell = M**3
    n_stars = int(n_bar * ncell)
    theta = jnp.array([mach, b, alpha, beta])
    cfg = dict(shape=shape, k_edges=k_edges, cell_sizes=(c,), n_bar=float(n_stars / ncell),
               n_max=n_max)

    key = jax.random.PRNGKey(seed)
    rows = []
    for i in range(n_real):
        g = gaussian_random_field(shape, beta, jax.random.fold_in(key, i))
        s = jnp.asarray(smooth_copula_field(g, mach, b, alpha))
        bp = measured_bandpowers(np.asarray(s), shape, k_edges)
        rho = jnp.exp(s)
        p = (rho / jnp.sum(rho)).ravel()
        idx = jax.random.choice(jax.random.fold_in(key, 9000 + i), rho.size, (n_stars,),
                                replace=True, p=p)
        ijk = jnp.stack(jnp.unravel_index(idx, shape), axis=-1) // c
        flat = np.asarray(ijk[:, 0] * M * M + ijk[:, 1] * M + ijk[:, 2])
        var_n = float(np.bincount(flat, minlength=ncell).var())
        rows.append(np.concatenate([bp, [var_n]]))
    rows = np.array(rows)
    C = mock_covariance(rows)
    cinv = hartlap_factor(n_real, rows.shape[1]) * np.linalg.inv(C)
    cinv = jnp.asarray(cinv)

    free = (0, 2, 3)  # (mach, alpha, beta); b fixed
    names = ["mach", "alpha", "beta"]
    F = fisher_matrix(theta, cinv, free=free, **cfg)
    sig = np.asarray(marginal_errors(F))
    sig4 = np.asarray(marginal_errors(fisher_matrix(theta, 4.0 * cinv, free=free, **cfg)))
    fid = np.array([mach, alpha, beta])

    print(f"  shape={shape} c={c} (N_bar={n_stars / ncell:.0f}) n_real={n_real} "
          f"k-bins={len(k_edges) - 1}  fiducial=(M={mach},b={b},a={alpha},beta={beta})")
    for nm, s_, s4_, f_ in zip(names, sig, sig4, fid):
        print(f"    sigma({nm:<5}) = {s_:.4f}  ({100 * s_ / f_:5.1f}% of fid)   "
              f"4x-volume = {s4_:.4f}  ({s_ / s4_:.2f}x tighter)")

    ok_pd = bool(np.all(np.linalg.eigvalsh(np.asarray(F)) > 0))
    ok_finite = bool(np.all(np.isfinite(sig)) and np.all(sig > 0))
    ok_vol = bool(np.allclose(sig4, sig / 2.0, rtol=1e-4))  # sigma ~ 1/sqrt(V)
    F4 = np.asarray(fisher_matrix(theta, cinv, free=(0, 1, 2, 3), **cfg))
    ev = np.linalg.eigvalsh(F4)
    degenerate = bool(ev[0] / ev[-1] < 1e-6)
    print(f"    4-param Fisher cond: lambda_min/lambda_max = {ev[0] / ev[-1]:.1e} "
          f"-> mach-b degeneracy {'DETECTED' if degenerate else 'ABSENT'} "
          f"(data constrains sigma_s^2=ln(1+(b M)^2))")
    print(f"  Fisher PD={ok_pd}  errors finite={ok_finite}  sigma~1/sqrt(V)={ok_vol}  "
          f"degeneracy={degenerate}  {'PASS' if (ok_pd and ok_finite and ok_vol and degenerate) else 'FAIL'}")
    ok = ok_pd and ok_finite and ok_vol and degenerate
    return {"passed": bool(ok), "sigma": {n_: float(s_) for n_, s_ in zip(names, sig)}}


def ac16_hmc_recovery(shape=(24, 24, 24), density_shape=(128, 128, 128), cell_sizes=(2, 4),
                      beta=3.0, mach=5.0, b=0.4, alpha=2.5, n_stars=18000, seed=0,
                      n_warmup=250, n_samples=400, n_max=10, n_s=400,
                      s_thr_margin=0.75, n_exc_bins=12, cover_nsigma=3.0):
    """AC16 -- joint (mach, alpha, beta) HMC recovery on injected-theta mocks (POT alpha block).

    Stellar counts-in-cells (the CLEAN inhomogeneous-Poisson sampler, multiple scales, on the SAME
    grid the CIC model FFTs on -> forward-bias-matched) -> (mach, beta). The gas-density tail is
    reduced to threshold EXCEEDANCES above ``s_thr = s_t(theta_true) + s_thr_margin`` and fit with
    the POT truncated-exponential block (:func:`tail_exceedance_loglike`) -> alpha: geometry-free
    (no cross-grid bias) and decoupled from sigma_s^2 (the lognormal norm cancels), which is the fix
    for the finite-field tail truncation that biased the old full-PDF fit high. The faithful
    ``rank_copula_field`` supplies BOTH mocks (NOT smooth/mass-conserving copula). b is fixed
    (mach-b degeneracy). A POT-validity soft barrier keeps the chain where ``s_t(theta) <= s_thr``.

    PASS requires: posterior covers theta_true within ``cover_nsigma``; alpha posterior width within
    [0.5, 2]x the truncation-corrected Fisher sigma(alpha) (not "covers only by being too wide");
    and small |corr(mach, alpha)| (the POT block breaks the old mach-alpha degeneracy). This is an
    INJECTION-RECOVERY test of the inference machinery (mock drawn from the same BM19 model); the
    transferable science result is AC17's sigma(alpha)-vs-N_tail forecast."""
    from gravoturb.diagnostics.measure import (
        estimate_log_count_variance_var,
        measure_exceedances,
        measure_log_count_variance,
    )
    from gravoturb.inference.covariance import measured_bandpowers, mock_precision
    from gravoturb.inference.fisher import sigma_alpha
    from gravoturb.inference.hmc import run_nuts, to_constrained, to_unconstrained
    from gravoturb.realization.copula import rank_copula_field
    from gravoturb.realization.gaussian_field import (
        expected_cells_above_transition,
        gaussian_random_field,
    )
    from gravoturb.realization.placement import sample_cic_counts

    _header("AC16 -- joint (mach,alpha,beta) HMC recovery (stellar CIC -> M,beta; POT tail -> alpha)")
    key = jax.random.PRNGKey(seed)

    # --- gas-density map (faithful rank copula) -> threshold exceedances -> alpha (POT) ---
    g_hi = gaussian_random_field(density_shape, beta, jax.random.fold_in(key, 7))
    s_hi = np.asarray(rank_copula_field(g_hi, mach, b, alpha))
    s_t_true = float(transition_density(alpha, sigma_s_squared(mach, b)))
    s_thr = s_t_true + s_thr_margin
    exc_counts_np, exc_edges_np, s_max, n_tail = measure_exceedances(s_hi, s_thr, n_bins=n_exc_bins)
    exc_counts, exc_edges = jnp.asarray(exc_counts_np), jnp.asarray(exc_edges_np)
    n_tail_exp = float(expected_cells_above_transition(int(np.prod(density_shape)), mach, b, alpha))

    # --- stellar CIC counts on the SAME grid the model FFTs on (forward-bias-matched) -> M,beta ---
    # Tail-robust log-count-variance statistic (Task 7), mirroring sbc.py::_build_mock EXACTLY:
    # measured Var_cells[log_plus(N)] per cell + a FIXED-FIDUCIAL estimator variance var_v
    # (truth-INDEPENDENT: computed at (M,alpha,beta)=(_MACH_FID,_ALPHA_FID,_BETA_FID), NOT at the
    # injected mach/alpha/beta -- a truth-keyed var_v would be exactly the SBC artifact the old POT
    # barrier was). Replaces the tail-sensitive bincount count-histogram block (the AC18 M-bias).
    from gravoturb.inference.model import K_EDGES as _K_EDGES
    from gravoturb.inference.sbc import (
        _ALPHA_FID,
        _BETA_FID,
        _MACH_FID,
        _N_REAL_BP,
        _N_REAL_VAR_V,
    )
    s_lo = rank_copula_field(gaussian_random_field(shape, beta, jax.random.fold_in(key, 1)),
                             mach, b, alpha)
    k_var = jax.random.fold_in(key, 2**31)  # disjoint stream for the fixed-fiducial var_v
    log_count_vars, var_vs, nbars = [], [], []
    for c in cell_sizes:
        nb = n_stars / (shape[0] // c) ** 3
        cnt = np.asarray(sample_cic_counts(s_lo, nb, c, jax.random.fold_in(key, 100 + c)))
        log_count_vars.append(measure_log_count_variance(cnt, nb))
        var_vs.append(estimate_log_count_variance_var(
            mach=_MACH_FID, b=b, alpha=_ALPHA_FID, beta=_BETA_FID, shape=shape,
            cell_size=c, n_bar=nb, n_real=_N_REAL_VAR_V, key=jax.random.fold_in(k_var, c)))
        nbars.append(nb)

    # --- field-level 2-pt band powers (the beta channel) -> beta, mirroring sbc.py EXACTLY ---
    # measured band powers of the SAME latent stellar field s_lo on _K_EDGES; plus a FIXED-FIDUCIAL
    # Hartlap precision bp_precision computed at (_MACH_FID,_ALPHA_FID,_BETA_FID) (truth-independent,
    # disjoint 2**30 stream) -- the design-intended beta carrier the scalar log-count variance lacks.
    band_powers = measured_bandpowers(np.asarray(s_lo), shape, _K_EDGES)
    k_bp = jax.random.fold_in(key, 2**30)
    bp_rows = [measured_bandpowers(np.asarray(rank_copula_field(
        gaussian_random_field(shape, _BETA_FID, jax.random.fold_in(k_bp, i)),
        _MACH_FID, b, _ALPHA_FID)), shape, _K_EDGES) for i in range(_N_REAL_BP)]
    bp_precision = mock_precision(bp_rows)

    # Shared prior-aware log-density factory (single source of truth with the SBC driver,
    # Task 6). AC16 uses a weakly-informative BM19Prior whose log-uniform/uniform boxes
    # comfortably contain the injected (mach, alpha, beta), so the prior is nearly flat over
    # the posterior bulk and the recovery is unchanged from the old flat-in-theta closure.
    from gravoturb.inference.model import build_logdensity
    from gravoturb.inference.priors import BM19Prior
    data = {"exc_counts": exc_counts, "exc_edges": exc_edges,
            "log_count_vars": tuple(log_count_vars), "var_vs": tuple(var_vs),
            "n_bars": tuple(nbars), "band_powers": band_powers}
    logdensity = build_logdensity(
        BM19Prior(), data, b=b, s_thr=s_thr, s_max=s_max, shape=shape,
        cell_sizes=cell_sizes, bp_precision=bp_precision, n_max=n_max, n_s=n_s)

    z0 = to_unconstrained(jnp.array([mach, alpha, beta]))
    sz = run_nuts(logdensity, z0, jax.random.fold_in(key, 2), n_warmup, n_samples)
    sc = np.asarray(jax.vmap(to_constrained)(sz))
    means, stds = sc.mean(0), sc.std(0)
    truth = np.array([mach, alpha, beta])
    cover = np.abs(means - truth) < cover_nsigma * stds

    # alpha-specific diagnostics: posterior width vs truncation-corrected Fisher + mach-alpha decoupling
    L = s_max - s_thr
    sig_fisher_alpha = float(sigma_alpha(alpha, L, float(n_tail)))
    width_ratio = float(stds[1] / sig_fisher_alpha)
    width_ok = 0.5 <= width_ratio <= 2.0
    corr_ma = float(np.corrcoef(sc[:, 0], sc[:, 1])[0, 1])
    corr_ok = abs(corr_ma) < 0.6

    print(f"  CIC shape={shape} cells={cell_sizes} n_stars={n_stars} (matched grid) | "
          f"gas map={density_shape} | n_warmup={n_warmup} n_samples={n_samples}")
    print(f"  inject (M={mach}, a={alpha}, beta={beta}; b={b} fixed)  s_t={s_t_true:.3f} "
          f"s_thr={s_thr:.3f} s_max={s_max:.3f} L={L:.3f}  N_tail={n_tail} "
          f"(E[>s_t]={n_tail_exp:.0f})")
    for nm, tr, mu, sd, cv in zip(("mach", "alpha", "beta"), truth, means, stds, cover):
        print(f"    {nm:<5} post={mu:+.3f} +/- {sd:.3f}  truth={tr:+.2f}  "
              f"{abs(mu - tr) / sd:.2f}sigma  {'COVER' if cv else 'MISS'}")
    print(f"    alpha width: post={stds[1]:.3f} vs Fisher={sig_fisher_alpha:.3f} "
          f"(ratio={width_ratio:.2f}) {'OK' if width_ok else 'BAD'};  "
          f"corr(M,alpha)={corr_ma:+.2f} {'OK' if corr_ok else 'BAD'}")
    ok = bool(np.all(cover) and np.all(np.isfinite(stds)) and np.all(stds > 0)
              and width_ok and corr_ok)
    print(f"  recovery {'PASS' if ok else 'FAIL'} (covers theta_true within {cover_nsigma} sigma; "
          f"alpha width sane; mach-alpha decoupled)")
    return {"passed": ok, "means": means.tolist(), "stds": stds.tolist(),
            "n_tail": int(n_tail), "L": L, "sigma_alpha_fisher": sig_fisher_alpha,
            "corr_mach_alpha": corr_ma}


def ac17_alpha_forecast(grids=((64, 64, 64), (96, 96, 96), (128, 128, 128)), n_iid=400,
                        n_field=50, caveat_grid=(96, 96, 96), beta=3.0, mach=5.0, b=0.4,
                        alpha=2.5, s_thr_margin=0.5, n_exc_bins=12, seed=0, sigma_tol=0.25,
                        slope_tol=0.15, fdense_tol=0.03, corr_lo=1.0, corr_hi=3.0):
    """AC17 -- sigma(alpha) vs N_tail forecast: the truncation-corrected Fisher, validated + caveated.

    The transferable science result ("how many independent tail elements N a gas map needs to
    measure the natal density-PDF slope alpha"), complementing AC16's single-resolution recovery.

    METHOD NOTE (a real discovery): the rank copula has a DETERMINISTIC marginal -- ``s_i =
    F^{-1}((rank_i+0.5)/N)``, the exact order statistics -- so repeated rank-copula realizations give
    ZERO 1-pt scatter (only the spatial arrangement varies). The plan's "K rank-copula mocks ->
    std(alpha_hat)" therefore cannot validate the forecast. Instead (Anna-approved):

    PRIMARY (asserted): use one rank-copula field per grid only to read off the representative
    ``(N_tail, L)`` at that resolution, then validate the forecast with ``n_iid`` genuine
    truncated-exponential draws of ``N_tail`` exceedances -- the definition of N INDEPENDENT tail
    elements. Assert ``|sigma_emp/sigma_fisher - 1| < sigma_tol`` per grid and the sqrt(N) law
    (``log sigma`` vs ``log N_tail`` slope within ``slope_tol`` of -0.5 AND of the Fisher slope).

    CAVEAT (reported, loose sanity bound): a REALISTIC correlated field (``smooth_copula_field``,
    ``n_field`` mocks at ``caveat_grid``) scatters wider than the i.i.d. bound by a correlation
    factor ``c = sigma_field/sigma_fisher`` (the red-spectrum tail's cells are not independent;
    N_eff ~ N_tail/c^2). Reported as the honest "field-level upper bound" caveat (cf. AC15); only
    bounded to ``[corr_lo, corr_hi]`` as a sanity check, not pinned.

    Also (Option B) the robust f_dense cross-check: a MASS-conserving realization's dense-mass
    fraction matches ``dense_mass_fraction`` (convergent, truncation-robust). numpy MLE (validation)."""
    from gravoturb.diagnostics.measure import measure_exceedances, smooth_copula_field
    from gravoturb.inference.fisher import sigma_alpha
    from gravoturb.realization.copula import (
        mass_conserving_copula_field,
        rank_copula_field,
    )
    from gravoturb.realization.gaussian_field import gaussian_random_field

    _header("AC17 -- sigma(alpha) vs N_tail forecast (iid-validated Fisher + correlation caveat)")
    key = jax.random.PRNGKey(seed)
    s_t = float(transition_density(alpha, sigma_s_squared(mach, b)))
    s_thr = s_t + s_thr_margin
    a_grid = np.arange(1.2, 5.0, 0.005)
    rng = np.random.default_rng(seed)

    def mle_alpha(counts, edges, s_max):
        """1-D MLE of alpha = argmax of the binned truncated-exponential loglike (vectorized)."""
        x = edges - s_thr
        L = s_max - s_thr
        x_lo, dx = x[:-1][None, :], np.diff(x)[None, :]
        a = a_grid[:, None]
        logp = (-a * x_lo) + np.log(-np.expm1(-a * dx)) - np.log(-np.expm1(-a * L))
        return float(a_grid[int(np.argmax((counts[None, :] * logp).sum(1)))])

    # --- PRIMARY: i.i.d. validation of the truncation-corrected Fisher across the N_tail ladder ---
    rows = []
    for gi, grid in enumerate(grids):
        s = np.asarray(rank_copula_field(
            gaussian_random_field(grid, beta, jax.random.fold_in(key, gi)), mach, b, alpha))
        _c, _e, s_max, n_tail = measure_exceedances(s, s_thr, n_bins=n_exc_bins)  # representative (N,L)
        L = s_max - s_thr
        edges = np.linspace(s_thr, s_max, n_exc_bins + 1)
        Z = 1.0 - np.exp(-alpha * L)
        a_hats = np.empty(n_iid)
        for j in range(n_iid):
            x = -np.log1p(-rng.random(n_tail) * Z) / alpha          # truncated-exp draws on [0, L]
            counts, _ = np.histogram(x, edges - s_thr)
            a_hats[j] = mle_alpha(counts.astype(float), edges, s_max)
        sigma_emp, sigma_fish = float(a_hats.std()), float(sigma_alpha(alpha, L, n_tail))
        rel = abs(sigma_emp / sigma_fish - 1.0)
        rows.append((n_tail, sigma_emp, sigma_fish))
        print(f"  grid={grid[0]:>3}^3  N_tail={n_tail:6d}  L={L:.2f}  sigma_emp={sigma_emp:.3f}  "
              f"sigma_fish={sigma_fish:.3f}  rel={rel:.2f}  {'OK' if rel < sigma_tol else 'BAD'}")

    rows = np.asarray(rows)
    nt, se, sf = rows[:, 0], rows[:, 1], rows[:, 2]
    per_grid_ok = bool(np.all(np.abs(se / sf - 1.0) < sigma_tol))
    slope_emp = float(np.polyfit(np.log(nt), np.log(se), 1)[0])
    slope_fish = float(np.polyfit(np.log(nt), np.log(sf), 1)[0])
    slope_ok = bool(abs(slope_emp + 0.5) < slope_tol and abs(slope_emp - slope_fish) < slope_tol)

    # --- CAVEAT: a realistic correlated field scatters wider than the iid bound (report c) ---
    a_field, nts_c, Ls_c = [], [], []
    for k in range(n_field):
        s = smooth_copula_field(
            gaussian_random_field(caveat_grid, beta, jax.random.fold_in(key, 5000 + k)), mach, b, alpha)
        c_, e_, smax_, nt_ = measure_exceedances(s, s_thr, n_bins=n_exc_bins)
        if nt_ < 10:
            continue
        a_field.append(mle_alpha(c_, e_, smax_))
        nts_c.append(nt_)
        Ls_c.append(smax_ - s_thr)
    sig_field = float(np.std(a_field))
    sig_fish_c = float(sigma_alpha(alpha, float(np.mean(Ls_c)), float(np.mean(nts_c))))
    corr_factor = sig_field / sig_fish_c
    corr_ok = bool(corr_lo <= corr_factor <= corr_hi)

    # --- Option B: robust f_dense cross-check (MASS-conserving realization vs analytic) ---
    s_mc = np.asarray(mass_conserving_copula_field(
        gaussian_random_field((96, 96, 96), beta, jax.random.fold_in(key, 99)), mach, b, alpha))
    rho = np.exp(s_mc)
    f_dense_real, f_dense_an = float(rho[s_mc > s_t].sum() / rho.sum()), float(dense_mass_fraction(mach, b, alpha))
    fd_rel = abs(f_dense_real / f_dense_an - 1.0)
    fd_ok = fd_rel < fdense_tol

    print(f"  sqrt(N) law: slope_emp={slope_emp:+.3f} (Fisher {slope_fish:+.3f}; ideal -0.5) "
          f"{'OK' if slope_ok else 'BAD'}")
    print(f"  correlation caveat: realistic field sigma={sig_field:.3f} vs iid Fisher {sig_fish_c:.3f}"
          f" -> c={corr_factor:.2f} (N_eff ~ N_tail/{corr_factor**2:.1f})  {'OK' if corr_ok else 'BAD'}")
    print(f"  f_dense cross-check (Option B): realized={f_dense_real:.4f} analytic={f_dense_an:.4f}"
          f"  rel={fd_rel:.3f}  {'OK' if fd_ok else 'BAD'}")
    ok = per_grid_ok and slope_ok and corr_ok and fd_ok
    print(f"  forecast {'PASS' if ok else 'FAIL'} (iid sigma_emp ~ corrected Fisher; sqrt(N); "
          f"correlation caveat; f_dense robust)")
    return {"passed": bool(ok), "n_tail": nt.tolist(), "sigma_emp": se.tolist(),
            "sigma_fisher": sf.tolist(), "slope_emp": slope_emp, "slope_fisher": slope_fish,
            "corr_factor": corr_factor, "f_dense_rel": fd_rel}


def ac18_sbc_rank_uniformity(n_trials, b=0.4, s_thr_margin=0.75, shape=(24,) * 3,
                             density_shape=(64,) * 3, n_warmup=120, n_samples=200, n_thin=4,
                             cell_sizes=(2, 4), n_stars=4.0e4, key=None, alpha_level=0.05):
    """AC18 -- SBC rank-uniformity: the EMPIRICAL calibration check of the inference engine.

    Runs the Simulation-Based Calibration loop (Talts et al. 2018) via
    :func:`gravoturb.inference.sbc.sbc_ranks` -- each trial draws ``theta* ~ BM19Prior``,
    builds a BM19 mock from theta*, runs NUTS, and ranks theta* among the thinned posterior
    draws -- then tests that the per-parameter rank statistics are DiscreteUniform over
    ``{0, ..., L}`` (L = number of thinned draws). A calibrated engine yields uniform ranks;
    deviations diagnose miscalibration (U/inverted-U = over/under-dispersion, slope = bias).

    This is the empirical proof that Task 6's drop of the truth-keyed POT-validity barrier
    produced a *calibrated* engine. The uniformity test uses the jaxstroviz integer-aware
    chi^2 helpers (the "C1" fix): ranks live in the DISCRETE set ``{0..L}``, so a naive flat
    ``K / n_bins`` expectation falsely rejects calibrated engines; ``sbc_expected_counts``
    weights each bin by how many integer rank values it contains.

    *** CAVEAT (under-the-model only): *** this is calibration UNDER the BM19 generative model
    (mock drawn from the same prior+likelihood the posterior assumes). It certifies the
    sampler/reparametrization/prior are self-consistent; it does NOT validate the BM19 model
    against real (mis-specified) data. The transferable science result is AC17's forecast.

    Returns ``{"passed", "p_value": [pM, pAlpha, pBeta], "n_trials", "n_draws" (=L),
    "ranks", "param_names"}``. PASS iff every per-param chi^2 p-value exceeds ``alpha_level``.
    """
    import matplotlib  # noqa: E402  (lazy: keep acceptance.py import light for other ACs)
    matplotlib.use("Agg")
    from jaxstroviz.experimental.analysis.sbc import (  # noqa: E402
        resolve_sbc_n_bins,
        sbc_bin_assignment,
        sbc_expected_counts,
    )
    from scipy import stats  # noqa: E402  (validation/analysis side)

    from gravoturb.inference.priors import BM19Prior
    from gravoturb.inference.sbc import sbc_ranks

    _header("AC18 -- SBC rank-uniformity (calibration UNDER the BM19 model; not vs real data)")
    print("  CAVEAT: under-model calibration only (mock ~ same prior+likelihood the posterior")
    print("          assumes); certifies sampler/reparam/prior self-consistency, NOT BM19-vs-data.")

    prior = BM19Prior()
    if key is None:
        key = jax.random.PRNGKey(0)

    out = sbc_ranks(prior, key, n_trials, b, s_thr_margin, shape, density_shape,
                    n_warmup, n_samples, n_thin, cell_sizes, n_stars)
    ranks = out["ranks"]                       # (n_trials, 3) int in {0..L}
    L = int(out["n_draws"])
    n_possible_ranks = L + 1                    # m = L + 1 distinct rank values
    n_bins = resolve_sbc_n_bins(n_trials, n_possible_ranks, target_per_bin=20)
    expected, _n_per_bin = sbc_expected_counts(n_trials, n_possible_ranks, n_bins)

    p_value = []
    for p, name in enumerate(out["param_names"]):
        bins = sbc_bin_assignment(ranks[:, p], n_possible_ranks, n_bins)
        observed = np.bincount(bins, minlength=n_bins)[:n_bins].astype(float)
        pv = float(stats.chisquare(observed, f_exp=expected).pvalue)
        p_value.append(pv)
        print(f"    {name:<5} chi^2 uniformity p={pv:.4f}  "
              f"{'PASS' if pv > alpha_level else 'FAIL'}  "
              f"(ranks: min={int(ranks[:, p].min())} max={int(ranks[:, p].max())})")

    passed = all(pv > alpha_level for pv in p_value)
    print(f"  n_trials={n_trials}  L={L} (m={n_possible_ranks} ranks)  n_bins={n_bins}  "
          f"(target ~20 trials/bin, integer-aware chi^2)")
    print(f"  SBC rank-uniformity {'PASS' if passed else 'FAIL'} "
          f"(all per-param p > {alpha_level})")
    return {"passed": bool(passed), "p_value": p_value, "n_trials": int(n_trials),
            "n_draws": L, "ranks": ranks, "param_names": list(out["param_names"])}


def ac20_log_count_variance_oracle(
    shape=(64, 64, 64), c=4, n_bar=5.0, b=0.4, alpha=2.5, beta=3.0,
    # ℳ set spans the RESTRICTED (ℳ≥4) calibrated prior, incl. the high edge ℳ=20.
    # The count/ℳ channel is calibrated only for ℳ≥4 (below that the field is transonic /
    # shot-noise-dominated, outside the supersonic GMC range); measured low-edge residual at
    # ℳ=4 is ~+1.5%, ℳ≥5 <1%. n_real=6 makes the ℳ=4-edge gate robust.
    machs=(4.0, 6.0, 8.0, 12.0, 16.0, 20.0), n_real=6, n_s=1024, rel_tol=0.06,
    n_count_max=None,
):
    r"""AC20 -- the DECISIVE count-model gate: tail-robust log-count variance across the M prior.

    The quantitative replacement for the design-doc Sec.1 over-prediction table (the old linear
    counts-in-cells M-channel over-predicted +9% -> +36% growing with Mach, the fat-tail signature
    that broke AC18 SBC). For each ``mach`` in the prior: generate a ``shape`` rank-copula gas field
    (``gaussian_random_field`` -> ``rank_copula_field``), Poisson-sample CIC counts (``sample_cic_counts``,
    cubic cell ``c``, mean ``n_bar``) over ``n_real`` realizations, MEASURE ``Var_cells[log_plus(N)]``
    (the FINITE-field oracle, truncated at the densest realized cell), and compare to the analytic
    :func:`~gravoturb.theory.counts_in_cells.predict_log_count_variance` (same ``log_plus`` transform, ``w2``
    = the cubic-cell window). Reports the SIGNED relative residual per mach plus its slope vs mach:
    the cure is proven by (a) ``|rel| < rel_tol`` at EVERY mach and (b) a FLAT (non-monotone-positive)
    residual -- the old bug's signature was an all-positive residual GROWING with mach. ``log_plus``
    compresses the tail so this converges and is insensitive to ``n_count_max`` (the count-grid
    extent); pass an explicit ``n_count_max`` to probe that (a no-op if truly tail-robust). numpy
    oracle (validation/non-differentiable); the prediction is the differentiable theory side.
    """
    from gravoturb.diagnostics.measure import measure_log_count_variance
    from gravoturb.realization.copula import rank_copula_field
    from gravoturb.realization.gaussian_field import gaussian_random_field
    from gravoturb.realization.placement import sample_cic_counts
    from gravoturb.theory.counts_in_cells import predict_log_count_variance
    from gravoturb.theory.projection import box_window_sq_grid

    _header("AC20 -- tail-robust log-count variance: predicted vs finite-field oracle across Mach")
    w2 = box_window_sq_grid(shape, c)
    ncm_kw = {} if n_count_max is None else {"n_count_max": n_count_max}
    ncm_label = "default(int(80*nbar)+50)" if n_count_max is None else str(n_count_max)
    print(f"  shape={shape} cell={c} n_bar={n_bar} (b={b}, alpha={alpha}, beta={beta})  "
          f"n_real={n_real}  n_s={n_s}  n_count_max={ncm_label}")
    print(f"  {'mach':>5} {'predicted':>11} {'oracle-meas':>12} {'maxN':>5} "
          f"{'signed-rel':>11}  verdict")

    rels = []
    ok = True
    for mach in machs:
        meas, max_n = [], 0
        for r in range(n_real):
            k = jax.random.PRNGKey(100 * r + int(mach))
            s = rank_copula_field(gaussian_random_field(shape, beta, k), mach, b, alpha)
            cnt = np.asarray(sample_cic_counts(s, n_bar, c, jax.random.fold_in(k, 1)))
            meas.append(measure_log_count_variance(cnt, n_bar))
            max_n = max(max_n, int(cnt.max()))
        measured = float(np.mean(meas))
        pred = float(predict_log_count_variance(
            n_bar, shape, beta, float(c), mach, b, alpha, w2=w2, n_s=n_s, **ncm_kw))
        rel = (pred - measured) / measured  # SIGNED: see coherent bias direction
        rels.append(rel)
        good = abs(rel) < rel_tol
        ok &= good
        print(f"  {mach:>5} {pred:>11.5f} {measured:>12.5f} {max_n:>5} "
              f"{rel:>+10.2%}  {'PASS' if good else 'FAIL'}")

    rels = np.asarray(rels)
    slope = float(np.polyfit(np.asarray(machs, dtype=float), rels, 1)[0])
    # The bug's signature was a coherent all-positive residual growing with mach. "Flat" here
    # means no such high-mach monotone-positive bias (a small low-mach offset is benign).
    flat = bool(slope <= 2e-3)  # not POSITIVE-sloped (the bug direction)
    print(f"  signed-rel: min={rels.min():+.2%} max={rels.max():+.2%} "
          f"|max|={np.abs(rels).max():.2%}  d(rel)/d(mach)={slope:+.2e} "
          f"({'FLAT (no high-M bias)' if flat else 'POSITIVE-SLOPED (bug signature!)'})")
    print(f"  AC20 {'PASS' if (ok and flat) else 'FAIL'} "
          f"(all |rel| < {rel_tol:.0%} AND residual not positively sloped in M)")
    return {"passed": bool(ok and flat), "machs": list(machs),
            "rel": rels.tolist(), "max_abs_rel": float(np.abs(rels).max()),
            "slope": slope}


def main():
    results = {
        "AC1/AC2": ac1_ac2_bm19(),
        "AC3/AC4": ac3_ac4_zeta(),
        "AC5": ac5_q(),
        "AC6": ac6_cornerstone(),
        "AC7": ac7_q_calibration(),
        "AC8/AC9": ac8_ac9_grads(),
        "AC11": ac11_xi_s_vs_oracle(),
        "AC11b": ac11b_rank_copula_equivalence(),
        "AC12": ac12_limber_projection_vs_oracle(),
        "AC13": ac13_cic_vs_oracle(),
        "AC14": ac14_grad_validation(),
        "AC15": ac15_fisher_forecast(),
        # AC16 (joint mach,alpha,beta HMC recovery) -- the POT truncated-exponential tail block
        # makes alpha recoverable; production run uses a 160^3 gas map (N_tail ~ 500) + long chains.
        "AC16": ac16_hmc_recovery(density_shape=(160, 160, 160), n_warmup=500, n_samples=1000),
        # AC17 (sigma(alpha) vs N_tail forecast) -- iid-validated truncation-corrected Fisher +
        # the realistic-field correlation caveat. Production ladder up to 128^3.
        "AC17": ac17_alpha_forecast(grids=((64,)*3, (96,)*3, (128,)*3), n_field=60),
        # AC18 (SBC rank-uniformity) -- the empirical calibration check (under the BM19 model)
        # that Task 6's POT-barrier drop worked. Full config: 128 trials, 96^3 gas map, long
        # chains thinned to ~independence. Slow.
        "AC18": ac18_sbc_rank_uniformity(
            n_trials=128, density_shape=(96,)*3, n_warmup=300, n_samples=600, n_thin=4),
        # AC20 (tail-robust log-count variance) -- the DECISIVE count-model gate across the
        # RESTRICTED ℳ≥4 calibrated prior. Predicted vs finite-field Var[log_plus(N)] oracle.
        "AC20": ac20_log_count_variance_oracle(),
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

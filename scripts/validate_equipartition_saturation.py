#!/usr/bin/env python
"""Derived-m_eq equipartition-saturation validation (Bianchini et al. 2016) -- FIT-FREE.

ZERO new released-core code and ZERO fitting: this script only exercises the
existing Engine-A multimass machinery (`find_alpha_for_masses` +
`lowered_exponential`) and compares it against EXACT closed-form statements.
We have the exact sigma(m) curve, so nothing is fitted -- every gate is an
exact identity or an exact-shape property evaluated analytically (autodiff).

Physics narrative (why sigma(m) saturates, and what m_eq is):
  (i)   LOW-MASS FLATTENING is an escape-speed ceiling, not equipartition
        failure per se: light components sit at W_j = mu_j^(2 delta) W0 -> 0,
        i.e. maximally tidally truncated, so their dispersion is pinned near
        the (mass-independent) central escape speed (Merritt 1981).
  (ii)  HIGH-MASS sigma proportional to m^(-1/2) is the deep-well thermalized
        regime: W_j >> 1 components are effectively untruncated isothermal
        spheres with m s_j^2 = const -- true equipartition.
  (iii) m_eq IS THE CROSSOVER between (i) and (ii). It is DERIVED, not fitted:
        m_eq = m-bar (g + 5/2)(g + 7/2) / phi0-hat with phi0-hat = W0
        (Bianchini App. A, A2<->A3 linear terms; verified against the held
        PDF, design doc 2026-06-11). Note m_eq/m-bar proportional to 1/W0:
        more concentrated clusters reach equipartition down to lighter stars.
  (iv)  THE SATURATION IS SHARPER THAN EXPONENTIAL: the exact quadratic Taylor
        coefficient of ln sigma(m) is negative (Bianchini A2's printed
        coefficient, verified), opposite the exponential's positive one.
        Hence Bianchini eq-3 exponential fits over finite mass windows are
        biased LOW: over mu = m/m-bar in [0.14, 1.7] (the 20-bin [0.1, 1] Msun
        model here) such fits recover ~0.5x the derived m_eq (measured
        0.50-0.53, 2026-06-11). Recorded as a DOCUMENTATION NOTE for
        literature comparisons -- it is NOT a gate, and no fitting code exists
        in this script. The figure's dashed tangent exponential
        sigma0 exp(-m / 2 m_eq) illustrates the bias visually.

Model: 20 log-spaced mass bins over [0.1, 1.0] Msun (Bianchini Fig. 9 style)
with EQUAL target mass fractions M_j = 1/20, delta = 0.5, g = 1.5,
W0 in {5, 7, 9}. alpha_j converge via `find_alpha_for_masses`; the
central-density-weighted m-bar = sum_j m_j alpha_j (GZ15 eq 26).

Per-component CENTRAL 1-D dispersion, two independent routes (free cross-check):
  - quadrature oracle (tests/validation/test_multimass_equilibrium_physics.py):
      sigma_j0 = s_j sqrt[ I4/I2/3 ],  I_n = int_0^sqrt(2 W_j) u^n E_g(g, W_j-u^2/2) du,
    W_j = mu_j^(2 delta) W0, s_j = mu_j^(-delta) (s = 1: only the SHAPE matters);
  - Bianchini eq A1 closed form (the paper omits the ^(1/2); restored here --
    confirmed by the paper's own A2 expansion):
      sigma_j0 = mu_j^(-delta) [ E_g(g+5/2; W_j) / E_g(g+3/2; W_j) ]^(1/2),
    E_g(a, x) = exp(x) gammainc(a, x) (regularized lower incomplete gamma).
The two agree analytically (Laplace convolution: int_0^W t^(b-1) E_g(a, W-t) dt
= Gamma(b) E_g(a+b, W)); the measured max relative difference is gated.

Gates (all EXACT statements; MEASURED FIRST 2026-06-11, then frozen):
  1. solver (quadrature oracle) vs exact closed form: max rel diff <= 1e-6
     (measured 1.5e-7..1.7e-7, pure n_u=400 quadrature resolution);
  2. HEADLINE mu -> 0 identity (zero new parameters): the local saturation
     mass m/(2 eta(m)), eta = -dln sigma/dln m by AUTODIFF of the closed form,
     evaluated at mu = 1e-3, equals the derived m_eq to <= 0.5%
     (measured 0.11-0.19% across W0 in {5, 7, 9});
  3. exact shape, eta(m) evaluated ANALYTICALLY (autodiff -- nothing fitted):
     eta(m_min) small (flat escape-limited end; measured 0.026-0.039,
     gate 0.10); eta strictly monotone increasing across the 20 bins;
     |eta - 1/2| at mu = 20 (deep-well end; eta -> 1/2 only asymptotically --
     the gate freezes the measured value honestly with headroom, and the
     measured number is printed).

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_equipartition_saturation.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from progenax.profiles.limepy import lowered_exponential
from progenax.profiles.limepy_multimass import find_alpha_for_masses

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plotstyle import OI, apply_pub_style, panel_label, save_fig

apply_pub_style()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "validation", "plots")

# Model grid (Bianchini Fig. 9 style; delta=0.5 is the GZ15/Bianchini default).
N_BINS = 20
M_RANGE = (0.1, 1.0)  # Msun
DELTA = 0.5
G_TRUNC = 1.5
W0_GRID = (5.0, 7.0, 9.0)

# GATES -- MEASURED FIRST (2026-06-11, this script's tables), then frozen.
# All are exact statements about the closed form / solver; NOTHING is fitted.
# 1. closed-vs-quadrature max rel diff measured 1.5e-7..1.7e-7 (pure n_u=400
#    quadrature resolution) -> gate 1e-6.
# 2. headline mu->0 identity: m/(2 eta) at mu=1e-3 vs derived m_eq, deviations
#    measured 0.11-0.19% across W0 in {5,7,9} -> gate 0.5% (~3x worst).
# 3. exact shape (eta by autodiff of the closed form): eta(m_min) measured
#    0.026-0.039 (flat, escape-limited) -> gate 0.10; eta strictly monotone
#    increasing; |eta - 1/2| at mu=20 measured 0..1.1e-16 (the e^{-W_j}
#    truncation correction underflows for W_j = 100-180, leaving float64
#    autodiff noise) -> gate 1e-12 (generous headroom over noise; the measured
#    value is printed). NEVER loosen any of these to pass.
GATE_XCHECK = 1e-6
GATE_ASYMPTOTIC_REL = 0.005   # m/(2 eta) at MU_ASYMP vs derived m_eq
MU_ASYMP = 1e-3
GATE_ETA_LOW = 0.10
MU_HIGH = 20.0
GATE_ETA_HALF = 1e-12         # |eta(mu=20) - 1/2|; measured ~3e-15..7e-15
MU_SWEEP = (0.5, 0.2, 0.1, 0.05, 0.01, 0.001)  # convergence evidence table


def central_sigma_quadrature(W_j, w_j, g, n_u=400):
    """Quadrature-oracle central 1-D dispersion (s=1), one component.

    The tests/validation/test_multimass_equilibrium_physics.py:70-86 recipe at
    r -> 0: sigma_j0 = w_j sqrt(I4/I2/3) with I_n = int u^n E_g(g, W_j - u^2/2) du.
    """
    u = jnp.linspace(0.0, jnp.sqrt(2.0 * W_j), n_u)
    E = lowered_exponential(g, W_j - u**2 / 2.0)
    return w_j * jnp.sqrt(
        jnp.trapezoid(u**4 * E, u) / jnp.trapezoid(u**2 * E, u) / 3.0)


def central_sigma_closed_form(W_j, w_j, g):
    """Bianchini eq A1 closed form, one component (s = 1):

        sigma_j0 = mu^(-delta) [E_g(g+5/2; W_j) / E_g(g+3/2; W_j)]^(1/2),

    with w_j = mu^(-delta). The paper omits the ^(1/2); it is restored here,
    as confirmed by the paper's own A2 expansion.
    """
    return w_j * jnp.sqrt(lowered_exponential(g + 2.5, W_j)
                          / lowered_exponential(g + 1.5, W_j))


def ln_sigma_exact(lnm, bar_m, W0, g, delta=DELTA):
    """ln of the EXACT closed-form sigma-hat at mass m = exp(lnm) (s = 1)."""
    mu = jnp.exp(lnm) / bar_m
    W = mu ** (2.0 * delta) * W0
    return (-delta * jnp.log(mu)
            + 0.5 * (jnp.log(lowered_exponential(g + 2.5, W))
                     - jnp.log(lowered_exponential(g + 1.5, W))))


def eta_analytic(m, bar_m, W0, g):
    """eta(m) = -dln sigma/dln m by AUTODIFF of the exact closed form."""
    return -jax.grad(ln_sigma_exact)(jnp.log(m), bar_m, W0, g)


def local_saturation_mass(mu, bar_m, W0, g):
    """m/(2 eta(m)) at mu = m/m-bar -- the exact local saturation mass. Its
    mu -> 0 limit is the derived m_eq (Bianchini App. A2<->A3); the finite-mu
    values quantify the leading-order truncation of the App.-A expansion."""
    m = mu * bar_m
    return float(m / (2.0 * eta_analytic(m, bar_m, W0, g)))


def run_w0(W0):
    """Build the converged multimass model at one W0; return everything measured."""
    m_j = jnp.logspace(jnp.log10(M_RANGE[0]), jnp.log10(M_RANGE[1]), N_BINS)
    M_j = jnp.full(N_BINS, 1.0 / N_BINS)  # equal target mass fractions
    alpha_j, residual = find_alpha_for_masses(m_j, M_j, W0, G_TRUNC, DELTA)

    bar_m = float(jnp.sum(m_j * alpha_j))  # GZ15 eq 26 central-weighted
    mu_j = m_j / bar_m
    W_j = mu_j ** (2.0 * DELTA) * W0      # rescale_j W0
    w_j = mu_j ** (-DELTA)                # s_j with s = 1

    sig_quad = jax.vmap(lambda W, w: central_sigma_quadrature(W, w, G_TRUNC))(W_j, w_j)
    sig_clsd = jax.vmap(lambda W, w: central_sigma_closed_form(W, w, G_TRUNC))(W_j, w_j)
    xcheck = float(jnp.max(jnp.abs(sig_quad - sig_clsd) / sig_clsd))

    meq_derived = bar_m * (G_TRUNC + 2.5) * (G_TRUNC + 3.5) / W0

    # Headline identity + convergence evidence: m/(2 eta) across mu (autodiff).
    meq_local = [local_saturation_mass(mu, bar_m, W0, G_TRUNC) for mu in MU_SWEEP]
    meq_asymp = local_saturation_mass(MU_ASYMP, bar_m, W0, G_TRUNC)

    # Exact shape: eta(m) ANALYTICALLY (autodiff of the closed form) at the
    # 20 bin masses, plus the deep-well evaluation at mu = MU_HIGH.
    eta = jax.vmap(lambda m: eta_analytic(m, bar_m, W0, G_TRUNC))(m_j)
    eta_high = float(eta_analytic(jnp.asarray(MU_HIGH * bar_m), bar_m, W0, G_TRUNC))

    return dict(
        W0=W0, m_j=m_j, alpha_j=alpha_j, residual=float(residual), bar_m=bar_m,
        sig_quad=sig_quad, xcheck=xcheck, meq_derived=meq_derived,
        meq_local=meq_local,
        asymp_rel=abs(meq_asymp - meq_derived) / meq_derived,
        eta=eta, eta_low=float(eta[0]),
        eta_half_dev=abs(eta_high - 0.5),
    )


def make_figure(results):
    """Per W0: the EXACT sigma(m) curve, the model's 20 solved points on it,
    the derived-m_eq vertical line, the m^(-1/2) asymptote, and -- clearly
    labeled, ILLUSTRATION only (no gate) -- the dashed tangent exponential
    sigma0 exp(-m / 2 m_eq) with the derived m_eq, showing visually why
    exponential fits of finite mass windows are biased low (the exact curve
    falls BELOW its tangent exponential: sharper-than-exponential saturation).
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(results), figsize=(9.6, 3.4), sharey=False,
        gridspec_kw=dict(wspace=0.30))
    for k, res in enumerate(results):
        ax = axes[k]
        W0, bar_m, meq = res["W0"], res["bar_m"], res["meq_derived"]
        m_j = np.asarray(res["m_j"])
        sig = np.asarray(res["sig_quad"])
        mm = np.logspace(np.log10(0.5 * M_RANGE[0]), np.log10(4.0 * meq), 400)

        # Exact closed-form curve (no free parameters once the model is solved):
        sig_exact = np.exp(np.asarray(jax.vmap(
            lambda lm: ln_sigma_exact(lm, bar_m, W0, G_TRUNC))(jnp.log(mm))))
        ax.plot(mm, sig_exact, color=OI["black"], lw=1.6,
                label=r"exact $\hat\sigma(m)$")
        # High-mass thermalized asymptote sigma = (m/m-bar)^(-1/2) (exact limit):
        ax.plot(mm, (mm / bar_m) ** -0.5, color=OI["green"], lw=1.2, ls="-.",
                label=r"$\propto m^{-1/2}$ (deep well)")
        # Tangent exponential -- ILLUSTRATION ONLY (no gate): sigma0 e^(-m/2 m_eq)
        # with the DERIVED m_eq and the exact m -> 0 intercept
        # sigma0 = sqrt(W0/(g+5/2)). The exact curve falls below it (sharper-
        # than-exponential), which is why finite-window eq-3 fits bias m_eq low.
        sigma0 = np.sqrt(W0 / (G_TRUNC + 2.5))
        ax.plot(mm, sigma0 * np.exp(-mm / (2.0 * meq)), color=OI["blue"],
                lw=1.3, ls="--",
                label=r"$\sigma_0 e^{-m/2m_{\rm eq}}$ (illustration)")
        # Derived m_eq marker:
        ax.axvline(meq, color=OI["vermilion"], lw=1.2, ls=":",
                   label=rf"derived $m_{{\rm eq}}={meq:.3f}$")
        # The model's 20 solved points (quadrature oracle) lie ON the exact curve:
        ax.plot(m_j, sig, "o", ms=3.6, mfc="white", mec=OI["black"], mew=0.9,
                label=r"model $\hat\sigma_{1d,j0}$ (20 bins)", zorder=5)

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$m\ [{\rm M_\odot}]$")
        ax.set_ylabel(r"$\hat\sigma_{1d,0}(m)$" if k == 0 else None)
        ax.legend(fontsize=6.4, loc="lower left")
        panel_label(ax, f"({chr(97 + k)})", loc="upper right")
        ax.text(0.96, 0.72, rf"$\hat W_0={W0:.0f}$",
                transform=ax.transAxes, ha="right", fontsize=9)
    save_fig(fig, OUTPUT_DIR, "equipartition_saturation")


def main():
    print("\n" + "=" * 88)
    print("DERIVED-m_eq EQUIPARTITION SATURATION (Bianchini et al. 2016 App. A)"
          " -- FIT-FREE")
    print(f"  {N_BINS} log bins on [{M_RANGE[0]}, {M_RANGE[1]}] Msun, equal M_j; "
          f"delta={DELTA}, g={G_TRUNC}; m_eq = m-bar (g+5/2)(g+7/2)/W0")
    print("  All gates are exact statements (closed form + autodiff);"
          " nothing is fitted.")
    print("=" * 88)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = [run_w0(W0) for W0 in W0_GRID]

    all_ok = True
    print("\n  [1] Solver vs exact closed form (Bianchini A1, ^(1/2) restored):"
          " per-component\n      central sigma, quadrature oracle vs"
          " mu^(-delta)[E_g(g+5/2;W)/E_g(g+3/2;W)]^(1/2)")
    print(f"  {'W0':>4} {'quantity':<38} {'measured':>10} {'gate':>9} {'status':>7}")
    print("  " + "-" * 74)
    for res in results:
        checks = [
            ("closed-form vs quadrature max rel diff", res["xcheck"], GATE_XCHECK),
            ("alpha-iteration residual", res["residual"], 1e-5),
        ]
        for label, val, gate in checks:
            ok = val <= gate
            all_ok &= ok
            print(f"  {res['W0']:>4.0f} {label:<38} {val:>10.2e} {gate:>9.0e} "
                  f"{'PASS' if ok else 'FAIL':>7}")

    print("\n  [2] HEADLINE (mu -> 0 identity, zero new params): "
          f"m/(2 eta) at mu={MU_ASYMP:g} vs derived m_eq,\n      eta by autodiff"
          " of the closed form")
    print(f"  {'W0':>4} {'m-bar':>7} {'derived m_eq':>13} {'m/(2 eta)':>10} "
          f"{'rel dev':>8} {'gate':>6} {'status':>7}")
    print("  " + "-" * 62)
    for res in results:
        ok = res["asymp_rel"] <= GATE_ASYMPTOTIC_REL
        all_ok &= ok
        print(f"  {res['W0']:>4.0f} {res['bar_m']:>7.4f} "
              f"{res['meq_derived']:>13.4f} "
              f"{res['meq_local'][MU_SWEEP.index(MU_ASYMP)]:>10.4f} "
              f"{res['asymp_rel']:>8.2%} {GATE_ASYMPTOTIC_REL:>6.1%} "
              f"{'PASS' if ok else 'FAIL':>7}")

    print("\n      Convergence evidence: local saturation mass m/(2 eta) /"
          " derived m_eq vs mu = m/m-bar")
    print("      (-> 1 as mu -> 0: App. A2<->A3 is the exact leading order."
          " The finite-mu deficit is\n       why finite-window eq-3 exponential"
          " fits recover ~0.5x the derived m_eq -- a\n       documentation note"
          " for literature comparisons, NOT a gate; no fitting here.)")
    print(f"  {'W0':>4}" + "".join(f"  mu={mu:<6g}" for mu in MU_SWEEP))
    print("  " + "-" * (6 + 11 * len(MU_SWEEP)))
    for res in results:
        print(f"  {res['W0']:>4.0f}" + "".join(
            f"  {loc / res['meq_derived']:<9.4f}" for loc in res["meq_local"]))

    print("\n  [3] Exact shape gates (eta(m) = -dln sigma/dln m, ANALYTIC via"
          " autodiff -- no fits):")
    print(f"  {'W0':>4} {'quantity':<38} {'measured':>10} {'gate':>9} {'status':>7}")
    print("  " + "-" * 74)
    for res in results:
        checks = [
            ("eta(m_min) (flat escape-limited end)", res["eta_low"], GATE_ETA_LOW),
            (f"|eta - 1/2| at mu={MU_HIGH:g} (deep well)", res["eta_half_dev"],
             GATE_ETA_HALF),
        ]
        for label, val, gate in checks:
            ok = val <= gate
            all_ok &= ok
            print(f"  {res['W0']:>4.0f} {label:<38} {val:>10.2e} {gate:>9.0e} "
                  f"{'PASS' if ok else 'FAIL':>7}")
        # eta must rise strictly monotonically toward saturation:
        mono = bool(jnp.all(jnp.diff(res["eta"]) > 0.0))
        all_ok &= mono
        print(f"  {res['W0']:>4.0f} {'eta(m) strictly monotone increasing':<38} "
              f"{'yes' if mono else 'NO':>10} {'':>9} {'PASS' if mono else 'FAIL':>7}")

    make_figure(results)
    print("=" * 88)
    print(f"  figure: {OUTPUT_DIR}/equipartition_saturation.{{png,pdf}}")
    print("  EQUIPARTITION SATURATION PASS" if all_ok
          else "  EQUIPARTITION SATURATION FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

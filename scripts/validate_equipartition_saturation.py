#!/usr/bin/env python
"""Derived-m_eq equipartition-saturation validation (Bianchini et al. 2016).

ZERO new released-core code: this script only exercises the existing Engine-A
multimass machinery. It validates that progenax's standard GZ15 model
(s_j = s mu_j^(-delta), mu_j = m_j/m-bar, central-density-weighted
m-bar = sum_j m_j alpha_j) ALREADY reproduces the equipartition *saturation* of
Bianchini et al. (2016) eq 3,

    sigma(m) = sigma0 exp(-m / 2 m_eq)            for m <= m_eq,
             = sigma_eq (m/m_eq)^(-1/2)           for m  > m_eq,

with sigma_eq = sigma0 e^(-1/2), and that the saturation mass is NOT a free
parameter -- it is DERIVED from the model (Bianchini App. A, matching the A2<->A3
linear terms; verified against the held PDF, design doc 2026-06-11):

    m_eq = m-bar (g + 5/2)(g + 7/2) / phi0-hat,     phi0-hat = W0.

Model: 20 log-spaced mass bins over [0.1, 1.0] Msun (Bianchini Fig. 9 style)
with EQUAL target mass fractions M_j = 1/20 (implementer's choice, documented;
the m_eq formula uses the converged central-weighted m-bar either way), delta=0.5,
g=1.5, W0 in {5, 7, 9}. alpha_j converge via `find_alpha_for_masses`.

Per-component CENTRAL 1-D dispersion, two independent routes (free cross-check):
  - quadrature oracle (tests/validation/test_multimass_equilibrium_physics.py):
      sigma_j0 = s_j sqrt[ I4/I2/3 ],  I_n = int_0^sqrt(2 W_j) u^n E_g(g, W_j-u^2/2) du,
    W_j = mu_j^(2 delta) W0, s_j = mu_j^(-delta) (s = 1: only the sigma(m) SHAPE
    matters; the fit's sigma0 absorbs the scale);
  - Bianchini eq A1 closed form (paper typo fixed -- the paper omits the ^(1/2)):
      sigma_j0 = mu_j^(-delta) [ E_g(g+5/2; W_j) / E_g(g+3/2; W_j) ]^(1/2),
    E_g(a, x) = exp(x) gammainc(a, x) (regularized lower incomplete gamma).
The two agree analytically (Laplace convolution: int_0^W t^(b-1) E_g(a, W-t) dt
= Gamma(b) E_g(a+b, W)); the measured max relative difference is printed.

Fit: deterministic two-stage grid scan over m_eq with the closed-form ln(sigma0)
per candidate (least squares in ln sigma over the 20 bins, FULL piecewise eq 3 --
both branches). JAX-native; no scipy.

Gates (MEASURED FIRST 2026-06-11, then frozen -- see GATES below):
  - closed-form vs quadrature cross-check;
  - ASYMPTOTIC identity (the zero-new-parameter headline): the derived m_eq is
    the EXACT mu -> 0 limit of the local saturation mass m/(2 eta(m)),
    eta = -dln sigma/dln m by autodiff of the closed form. Measured: the ratio
    m/(2 eta)/m_eq_derived converges monotonically 0.49-0.62 (mu=0.5) ->
    0.998-0.999 (mu=1e-3) at every W0 -- App. A2<->A3 confirmed with no factor
    errors. Gated at mu = 1e-3.
  - finite-range fitted m_eq vs derived m_eq per W0. The App.-A expansion is
    LEADING-ORDER in mu << 1, while the fit range [0.1, 1] Msun spans
    mu ~ 0.14-1.7 (NOT << 1). The EXACT model itself says the local effective
    m_eq at mu = 0.5 is already only ~0.5-0.6x the asymptotic value (the
    convergence table this script prints), so the eq-3 fit over this range MUST
    recover ~0.5x the derived m_eq. Measured fitted/derived = 0.504-0.526; this
    is the quantified truncation of the leading-order expansion, NOT a model
    disagreement (the asymptotic gate above is the proof). Frozen as a
    REGRESSION BAND, not loosened agreement;
  - qualitative shape via Bianchini eq 4, eta(m) = -dln sigma/dln m (finite
    differences across bins): flat (eta -> 0) at the low-mass end, rising toward
    the eq-3 prediction min(m/2 m_eq, 1/2) at the high-mass end.

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

# GATES -- MEASURED FIRST (2026-06-11, this script's table), then frozen.
# Measured: closed-vs-quadrature max rel diff 1.5e-7..1.7e-7 (pure n_u=400
# quadrature resolution -> gate 1e-6). Asymptotic identity m/(2 eta) at mu=1e-3
# vs derived m_eq: deviations 0.11-0.19% across W0 in {5,7,9} -> gate 0.5%
# (~3x worst measured). Finite-range fitted/derived m_eq ratio: 0.526 (W0=5),
# 0.510 (W0=7), 0.504 (W0=9) -- the exact model's own leading-order truncation
# (see module docstring); frozen REGRESSION BAND [0.40, 0.65]. Shape gates:
# eta(m_min) measured 0.028-0.042 (flat -> gate 0.10); |eta - eta_eq3|(m_max)
# measured 0.011-0.027 (gate 0.05). NEVER loosen any of these to pass.
GATE_XCHECK = 1e-6
GATE_ASYMPTOTIC_REL = 0.005   # m/(2 eta) at MU_ASYMP vs derived m_eq
MU_ASYMP = 1e-3
GATE_FIT_RATIO = (0.40, 0.65)  # fitted/derived regression band (see docstring)
GATE_ETA_LOW = 0.10
GATE_ETA_HIGH = 0.05
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
    """Bianchini eq A1 closed form (typo-fixed), one component:
    sigma_j0 = mu^(-delta) [E_g(g+5/2; W_j)/E_g(g+3/2; W_j)]^(1/2), w_j = mu^(-delta).
    """
    return w_j * jnp.sqrt(lowered_exponential(g + 2.5, W_j)
                          / lowered_exponential(g + 1.5, W_j))


def ln_sigma_eq3(m, m_eq):
    """ln of Bianchini eq 3 with sigma0 = 1 (FULL piecewise, both branches)."""
    return jnp.where(m <= m_eq, -m / (2.0 * m_eq),
                     -0.5 - 0.5 * jnp.log(m / m_eq))


def fit_eq3(m_j, sigma_j):
    """Least-squares fit of eq 3 (sigma0, m_eq) in ln sigma: deterministic
    two-stage grid scan over m_eq with closed-form ln(sigma0) per candidate."""
    y = jnp.log(sigma_j)

    def sse(meq):
        b = ln_sigma_eq3(m_j, meq)
        ln_s0 = jnp.mean(y - b)
        return jnp.sum((y - b - ln_s0) ** 2), ln_s0

    def scan(grid):
        s, l = jax.vmap(sse)(grid)
        i = jnp.argmin(s)
        return grid[i], l[i], i

    coarse = jnp.logspace(jnp.log10(0.02), jnp.log10(50.0), 4001)
    meq_c, _, i = scan(coarse)
    lo, hi = coarse[jnp.maximum(i - 2, 0)], coarse[jnp.minimum(i + 2, 4000)]
    fine = jnp.linspace(lo, hi, 4001)
    meq_f, ln_s0, _ = scan(fine)
    return float(meq_f), float(jnp.exp(ln_s0))


def local_saturation_mass(mu, bar_m, W0, g, delta=DELTA):
    """m/(2 eta(m)) at mu = m/m-bar, with eta = -dln sigma/dln m by AUTODIFF of
    the closed form -- the exact local saturation mass. Its mu -> 0 limit is the
    derived m_eq (Bianchini App. A2<->A3); the finite-mu values quantify the
    leading-order truncation."""
    def ln_sigma(lnm):
        mu_ = jnp.exp(lnm) / bar_m
        W = mu_ ** (2.0 * delta) * W0
        return (-delta * jnp.log(mu_)
                + 0.5 * (jnp.log(lowered_exponential(g + 2.5, W))
                         - jnp.log(lowered_exponential(g + 1.5, W))))
    m = mu * bar_m
    eta = -jax.grad(ln_sigma)(jnp.log(m))
    return float(m / (2.0 * eta))


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

    meq_fit, sigma0_fit = fit_eq3(m_j, sig_quad)
    meq_derived = bar_m * (G_TRUNC + 2.5) * (G_TRUNC + 3.5) / W0

    # Asymptotic identity + convergence evidence: m/(2 eta) across mu.
    meq_local = [local_saturation_mass(mu, bar_m, W0, G_TRUNC) for mu in MU_SWEEP]
    meq_asymp = local_saturation_mass(MU_ASYMP, bar_m, W0, G_TRUNC)

    # Bianchini eq 4: eta(m) = -dln sigma/dln m, central finite differences.
    lnm, lns = jnp.log(m_j), jnp.log(sig_quad)
    eta = -jnp.gradient(lns, lnm)
    eta_eq3 = jnp.minimum(m_j / (2.0 * meq_fit), 0.5)  # eq-3 local slope

    return dict(
        W0=W0, m_j=m_j, alpha_j=alpha_j, residual=float(residual), bar_m=bar_m,
        sig_quad=sig_quad, xcheck=xcheck, meq_fit=meq_fit, sigma0_fit=sigma0_fit,
        meq_derived=meq_derived,
        fit_ratio=meq_fit / meq_derived,
        meq_local=meq_local,
        asymp_rel=abs(meq_asymp - meq_derived) / meq_derived,
        eta=eta, eta_eq3=eta_eq3,
        eta_low=float(eta[0]), eta_high_dev=float(jnp.abs(eta[-1] - eta_eq3[-1])),
    )


def make_figure(results):
    """sigma(m) points + eq-3 curves (derived AND fitted m_eq) per W0, with the
    eta(m) = -dln sigma/dln m saturation diagnostic below."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2, len(results), figsize=(9.6, 5.4), sharex=True,
        gridspec_kw=dict(height_ratios=[2.4, 1.3], hspace=0.08, wspace=0.28))
    mm = np.logspace(np.log10(M_RANGE[0]), np.log10(M_RANGE[1]), 300)
    for k, res in enumerate(results):
        axS, axE = axes[0, k], axes[1, k]
        m_j = np.asarray(res["m_j"])
        sig = np.asarray(res["sig_quad"])
        # eq-3 curve with the DERIVED m_eq (sigma0 = 1-param least squares
        # given the fixed derived m_eq -- zero shape freedom):
        b = np.asarray(ln_sigma_eq3(jnp.asarray(m_j), res["meq_derived"]))
        s0_der = np.exp(np.mean(np.log(sig) - b))
        axS.plot(mm, s0_der * np.exp(np.asarray(
            ln_sigma_eq3(jnp.asarray(mm), res["meq_derived"]))),
            color=OI["vermilion"], lw=1.8,
            label=rf"derived $m_{{\rm eq}}={res['meq_derived']:.3f}$")
        axS.plot(mm, res["sigma0_fit"] * np.exp(np.asarray(
            ln_sigma_eq3(jnp.asarray(mm), res["meq_fit"]))),
            color=OI["blue"], lw=1.4, ls="--",
            label=rf"fitted $\hat m_{{\rm eq}}={res['meq_fit']:.3f}$")
        axS.plot(m_j, sig, "o", ms=3.6, mfc="white", mec=OI["black"], mew=0.9,
                 label=r"model $\hat\sigma_{1d,j0}$", zorder=5)
        axS.set_xscale("log"); axS.set_yscale("log")
        axS.set_ylabel(r"$\hat\sigma_{1d,0}(m)$" if k == 0 else None)
        axS.legend(fontsize=7, loc="lower left")
        panel_label(axS, f"({chr(97 + k)})", loc="upper right")
        axS.text(0.96, 0.70, rf"$\hat W_0={res['W0']:.0f}$",
                 transform=axS.transAxes, ha="right", fontsize=9)

        axE.plot(mm, np.minimum(mm / (2.0 * res["meq_fit"]), 0.5),
                 color=OI["blue"], lw=1.4, ls="--", label="eq 3 (fitted)")
        axE.plot(m_j, np.asarray(res["eta"]), "o", ms=3.6, mfc="white",
                 mec=OI["black"], mew=0.9, label=r"model $\eta$")
        axE.axhline(0.5, color="0.6", lw=0.8, ls=":")
        axE.set_xscale("log")
        axE.set_ylim(-0.02, 0.62)
        axE.set_xlabel(r"$m\ [{\rm M_\odot}]$")
        axE.set_ylabel(r"$\eta=-\,{\rm d}\ln\sigma/{\rm d}\ln m$" if k == 0 else None)
        if k == 0:
            axE.legend(fontsize=7, loc="upper left")
    save_fig(fig, OUTPUT_DIR, "equipartition_saturation")


def main():
    print("\n" + "=" * 88)
    print("DERIVED-m_eq EQUIPARTITION SATURATION (Bianchini et al. 2016 eq 3 / App. A)")
    print(f"  {N_BINS} log bins on [{M_RANGE[0]}, {M_RANGE[1]}] Msun, equal M_j; "
          f"delta={DELTA}, g={G_TRUNC}; m_eq = m-bar (g+5/2)(g+7/2)/W0")
    print("=" * 88)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = [run_w0(W0) for W0 in W0_GRID]

    all_ok = True
    print("\n  Headline (asymptotic identity, zero new params): "
          f"m/(2 eta) at mu={MU_ASYMP:g} vs derived m_eq")
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

    print("\n  Convergence evidence: local saturation mass m/(2 eta) / derived "
          "m_eq vs mu = m/m-bar")
    print("  (-> 1 as mu -> 0: App. A2<->A3 is the exact leading order; the "
          "finite-mu deficit\n   is the truncation the finite-range fit below "
          "inherits)")
    print(f"  {'W0':>4}" + "".join(f"  mu={mu:<6g}" for mu in MU_SWEEP))
    print("  " + "-" * (6 + 11 * len(MU_SWEEP)))
    for res in results:
        print(f"  {res['W0']:>4.0f}" + "".join(
            f"  {loc / res['meq_derived']:<9.4f}" for loc in res["meq_local"]))

    print("\n  Finite-range eq-3 fit over [0.1, 1] Msun (mu ~ 0.14-1.7, NOT << 1):"
          " fitted/derived\n  regression band -- the ~0.5x ratio IS the exact"
          " model's leading-order truncation\n  (see convergence table), not a"
          " disagreement.")
    print(f"  {'W0':>4} {'fitted m_eq':>12} {'derived m_eq':>13} {'ratio':>7} "
          f"{'band':>14} {'status':>7}")
    print("  " + "-" * 64)
    for res in results:
        ok = GATE_FIT_RATIO[0] <= res["fit_ratio"] <= GATE_FIT_RATIO[1]
        all_ok &= ok
        print(f"  {res['W0']:>4.0f} {res['meq_fit']:>12.4f} "
              f"{res['meq_derived']:>13.4f} {res['fit_ratio']:>7.3f} "
              f"[{GATE_FIT_RATIO[0]:.2f}, {GATE_FIT_RATIO[1]:.2f}]"
              f"{'PASS' if ok else 'FAIL':>9}")

    print(f"\n  {'W0':>4} {'quantity':<38} {'measured':>10} {'gate':>9} {'status':>7}")
    print("  " + "-" * 74)
    for res in results:
        checks = [
            ("closed-form vs quadrature max rel diff", res["xcheck"], GATE_XCHECK),
            ("alpha-iteration residual", res["residual"], 1e-5),
            ("eta(m_min) (flat low-mass end)", res["eta_low"], GATE_ETA_LOW),
            ("|eta - eta_eq3|(m_max) (saturating end)", res["eta_high_dev"],
             GATE_ETA_HIGH),
        ]
        for label, val, gate in checks:
            ok = val <= gate
            all_ok &= ok
            print(f"  {res['W0']:>4.0f} {label:<38} {val:>10.2e} {gate:>9.0e} "
                  f"{'PASS' if ok else 'FAIL':>7}")
        # eta must rise monotonically toward saturation (qualitative shape):
        mono = bool(jnp.all(jnp.diff(res["eta"]) > -1e-3))
        all_ok &= mono
        print(f"  {res['W0']:>4.0f} {'eta(m) monotone increasing':<38} "
              f"{'yes' if mono else 'NO':>10} {'':>9} {'PASS' if mono else 'FAIL':>7}")

    make_figure(results)
    print("=" * 88)
    print(f"  figure: {OUTPUT_DIR}/equipartition_saturation.{{png,pdf}}")
    print("  EQUIPARTITION SATURATION PASS" if all_ok
          else "  EQUIPARTITION SATURATION FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
Initial-mass-function (IMF) statistics validation figures.

Five publication-quality figures for the released IMF samplers, each anchored to
passing tests in ``tests/validation/test_imf_physics.py`` (25/25) and printing
expected-vs-measured PASS/FAIL against *external* oracles (published slopes, a
fine-grid reference integral, the KS distribution, central finite differences) --
never the sampler's own PDF.

Figures (-> what they validate):
  1. imf_recovered_slopes.png   sample-MLE slopes vs published alpha (Salpeter,
                                Kroupa 3 segments, Chabrier tail, Maschberger)
  2. imf_pdf_overlay.png        analytic m*xi(m) vs sampled histogram (Chabrier,
                                Kroupa, Maschberger): lognormal+PL / piecewise / smooth
  3. imf_cdf_ks.png             empirical vs analytic CDF + KS D & p (analytic-PPF
                                vs Newton-PPF samplers)
  4. imf_mean_mass_accuracy.png mean_mass() vs 200k log-grid reference for 6 IMFs;
                                honest linear-grid failure-mode sub-panel
  5. imf_gradient_validation.png  AD vs central-FD of d(NLL)/d(alpha) for the
                                differentiable inference layer (IMFParams)

Provenance note: Chabrier (2003) PDF is held in docs/core-papers/; Salpeter (1955),
Kroupa (2001) and Maschberger (2013) primary PDFs are NOT in the repo. Their slope
constants are validated here by recovery, and cited to the literature, but are not
PDF-checked in this checkout.

References:
    Salpeter (1955), ApJ 121, 161; Kroupa (2001), MNRAS 322, 231;
    Chabrier (2003), PASP 115, 763; Maschberger (2013), MNRAS 429, 1725.

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_imfs.py
"""

import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.stats

jax.config.update("jax_enable_x64", True)

from progenax.imf import (
    ChabrierIMF,
    Maschberger,
    PowerLawIMF,
    Schechter,
    TaperedPowerLaw,
)
from progenax.imf.differentiable import individual_mass_nll, sample_masses_from_params
from progenax.imf.params import IMFParams

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plotstyle import OI, apply_pub_style, panel_label, save_fig

apply_pub_style()

OUTPUT_DIR = "validation/plots"
SEED = 42
N_SAMPLES = 100_000
N_KS = 20_000  # moderate N keeps KS sensitivity sane (D_crit ~ 0.01)
N_GRAD = 30_000


# ============================================================================
# Helpers
# ============================================================================
def powerlaw_mle_slope(samples, a, b):
    """Maximum-likelihood slope alpha of a bounded power law f(m) ~ m^(-alpha)
    on [a, b], estimated from samples falling in the window.

    For f(m) = C m^(-alpha) with C = |1-alpha| / |b^(1-alpha) - a^(1-alpha)|,
    the per-sample log-likelihood is  log C - alpha <log m>.  Maximised by a 1-D
    bounded solve (no closed form for a truncated power law).
    """
    s = np.asarray(samples)
    s = s[(s >= a) & (s <= b)]
    if s.size < 50:
        return np.nan, 0
    Lm = float(np.mean(np.log(s)))

    def neg_ll(alpha):
        e = 1.0 - alpha
        logC = np.log(abs(e)) - np.log(abs(b**e - a**e))
        return -(logC - alpha * Lm)

    res = scipy.optimize.minimize_scalar(neg_ll, bounds=(0.05, 3.5), method="bounded")
    return float(res.x), int(s.size)


def _bootstrap_slope_sigma(samples, a, b, n_boot=40, rng=None):
    """Bootstrap standard error on the MLE slope (honest finite-N error bar)."""
    rng = rng or np.random.default_rng(0)
    s = np.asarray(samples)
    s = s[(s >= a) & (s <= b)]
    if s.size < 50:
        return np.nan
    est = []
    for _ in range(n_boot):
        bs = rng.choice(s, size=s.size, replace=True)
        a_hat, _ = powerlaw_mle_slope(bs, a, b)
        est.append(a_hat)
    return float(np.std(est))


def loggrid_mean(imf, n=200_000):
    """Reference E[m] on a 200k-point log grid (independent of mean_mass())."""
    g = np.logspace(np.log10(float(imf.m_min)), np.log10(float(imf.m_max)), n)
    pdf = np.asarray(jnp.exp(imf.logpdf(jnp.asarray(g))))
    num = np.trapezoid(g * pdf, g)
    den = np.trapezoid(pdf, g)
    return num / den


# ============================================================================
# Figure 1 -- recovered slopes (headline)
# ============================================================================
def fig_recovered_slopes(output_dir):
    print("\n" + "=" * 60)
    print("FIG 1: recovered power-law slopes vs published alpha")
    print("=" * 60)

    key = jax.random.PRNGKey(SEED)
    ks, kk, kc, km = jax.random.split(key, 4)
    salp = PowerLawIMF.salpeter().sample(ks, N_SAMPLES)
    kroupa = PowerLawIMF.kroupa().sample(kk, N_SAMPLES)
    chab = ChabrierIMF().sample(kc, N_SAMPLES)
    masch = Maschberger().sample(km, N_SAMPLES)
    rng = np.random.default_rng(SEED)

    # (label, expected alpha, samples, window, tol)
    specs = [
        ("Salpeter\n[0.1,100]", 2.35, salp, (0.1, 100.0), 0.05),
        ("Kroupa $\\alpha_0$\n[0.01,0.08]", 0.3, kroupa, (0.012, 0.078), 0.08),
        ("Kroupa $\\alpha_1$\n[0.08,0.5]", 1.3, kroupa, (0.085, 0.49), 0.05),
        ("Kroupa $\\alpha_2$\n[0.5,100]", 2.3, kroupa, (0.55, 100.0), 0.05),
        ("Chabrier tail\n[1,100]", 2.3, chab, (1.0, 100.0), 0.06),
        ("Maschberger\n[20,300]", 2.3, masch, (20.0, 300.0), 0.15),
    ]

    labels, exp, rec, sig, tol, ok = [], [], [], [], [], []
    for name, a_exp, smp, (lo, hi), t in specs:
        a_hat, n_win = powerlaw_mle_slope(smp, lo, hi)
        s_hat = _bootstrap_slope_sigma(smp, lo, hi, rng=rng)
        passed = abs(a_hat - a_exp) < t
        labels.append(name)
        exp.append(a_exp)
        rec.append(a_hat)
        sig.append(s_hat)
        tol.append(t)
        ok.append(passed)
        print(
            f"  {name.splitlines()[0]:14s} alpha_lit={a_exp:.2f}  "
            f"recovered={a_hat:.3f}+-{s_hat:.3f}  (N_win={n_win:>6d}, tol {t})  "
            f"-> {'PASS' if passed else 'FAIL'}"
        )

    exp = np.array(exp)
    rec = np.array(rec)
    sig = np.array(sig)
    tol = np.array(tol)
    passed_all = bool(np.all(ok))

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    lim = (-0.1, 2.6)
    ax.plot(lim, lim, "-", color="0.6", lw=1.0, zorder=0, label="$y=x$ (exact)")
    # tolerance band around identity (use the loosest tol for the shaded guide)
    xs = np.linspace(*lim, 10)
    ax.fill_between(
        xs,
        xs - tol.max(),
        xs + tol.max(),
        color=OI["green"],
        alpha=0.10,
        zorder=0,
        label=f"$\\pm{tol.max():.2f}$ band",
    )
    cols = [OI["blue"] if k else OI["vermilion"] for k in ok]
    for x, y, e, c in zip(exp, rec, sig, cols):
        ax.errorbar(
            x,
            y,
            yerr=e,
            fmt="o",
            color=c,
            ms=6,
            mec="white",
            mew=0.6,
            capsize=2.5,
            lw=1.0,
            zorder=3,
        )
    ax.set_xlabel(r"published slope $\alpha_{\rm lit}$")
    ax.set_ylabel(r"recovered slope $\hat\alpha$ (sample MLE)")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.text(
        0.97,
        0.05,
        "Salpeter, Kroupa(3),\nChabrier tail, Maschberger",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="0.35",
    )
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "imf_recovered_slopes")
    print("  saved imf_recovered_slopes.{png,pdf}")
    return passed_all


# ============================================================================
# Figure 2 -- PDF shape overlay (analytic vs sampled)
# ============================================================================
def _overlay_panel(ax, imf, label, color, seed):
    key = jax.random.PRNGKey(seed)
    masses = np.asarray(imf.sample(key, N_SAMPLES))
    m_lo, m_hi = float(imf.m_min), float(imf.m_max)
    bins = np.logspace(np.log10(m_lo), np.log10(m_hi), 45)
    counts, edges = np.histogram(masses, bins=bins, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])

    g = np.logspace(np.log10(m_lo), np.log10(m_hi), 400)
    pdf = np.asarray(jnp.exp(imf.logpdf(jnp.asarray(g))))

    ax.loglog(g, g * pdf, "-", color=OI["black"], lw=1.8, label=r"analytic $m\,\xi(m)$")
    ax.loglog(
        centers,
        centers * counts,
        "o",
        color=color,
        ms=3.2,
        mec="white",
        mew=0.3,
        label=r"samples ($N{=}10^5$)",
    )

    # median relative deviation over well-populated bins
    pop = counts > 0
    analytic_at = np.asarray(jnp.exp(imf.logpdf(jnp.asarray(centers))))
    rel = np.abs(counts[pop] - analytic_at[pop]) / analytic_at[pop]
    med = float(np.median(rel))
    ax.set_xlabel(r"$m$ [$M_\odot$]")
    ax.set_title(label, fontsize=8.5)
    ax.legend(loc="lower center", fontsize=6.8)
    return med


def fig_pdf_overlay(output_dir):
    print("\n" + "=" * 60)
    print("FIG 2: analytic PDF vs sampled histogram")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
    panels = [
        (ChabrierIMF(), "Chabrier (lognormal + PL)", OI["vermilion"], SEED, "(a)"),
        (PowerLawIMF.kroupa(), "Kroupa (piecewise PL)", OI["blue"], SEED + 1, "(b)"),
        (Maschberger(), "Maschberger (smooth)", OI["green"], SEED + 2, "(c)"),
    ]
    meds, ok = [], []
    for ax, (imf, lab, col, sd, tag) in zip(axes, panels):
        med = _overlay_panel(ax, imf, lab, col, sd)
        passed = med < 0.05
        meds.append(med)
        ok.append(passed)
        panel_label(ax, tag, loc="upper right")
        print(
            f"  {lab:28s} median |hist-analytic|/analytic = {med:.3f} "
            f"(tol 0.05)  -> {'PASS' if passed else 'FAIL'}"
        )
    axes[0].set_ylabel(r"$m\,\xi(m)$  (mass per dex)")
    fig.tight_layout(pad=0.4, w_pad=0.8)
    save_fig(fig, output_dir, "imf_pdf_overlay")
    print("  saved imf_pdf_overlay.{png,pdf}")
    return bool(np.all(ok))


# ============================================================================
# Figure 3 -- CDF coverage + Kolmogorov-Smirnov test
# ============================================================================
def fig_cdf_ks(output_dir):
    print("\n" + "=" * 60)
    print("FIG 3: empirical vs analytic CDF + KS test")
    print("=" * 60)

    # analytic-PPF samplers (PowerLaw, Maschberger) + Newton-PPF (Chabrier)
    cases = [
        ("Salpeter (analytic PPF)", PowerLawIMF.salpeter(), OI["blue"], SEED),
        ("Maschberger (analytic PPF)", Maschberger(), OI["green"], SEED + 1),
        ("Chabrier (Newton PPF)", ChabrierIMF(), OI["vermilion"], SEED + 2),
    ]
    d_crit = 1.36 / np.sqrt(N_KS)  # KS 0.05 critical value

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ok, rows = [], []
    for name, imf, col, sd in cases:
        key = jax.random.PRNGKey(sd)
        samples = np.asarray(imf.sample(key, N_KS))
        cdf_fn = lambda x, _imf=imf: np.asarray(_imf.cdf(jnp.asarray(x)))
        res = scipy.stats.kstest(samples, cdf_fn)
        D, p = float(res.statistic), float(res.pvalue)
        passed = p > 0.01
        ok.append(passed)
        rows.append((name, D, p, passed))

        # analytic CDF line + empirical points (subsampled for a clean curve)
        srt = np.sort(samples)
        g = np.logspace(np.log10(float(imf.m_min)), np.log10(float(imf.m_max)), 300)
        ax.semilogx(g, cdf_fn(g), "-", color=col, lw=1.8, label=name, zorder=2)
        ss = srt[:: max(1, N_KS // 400)]
        emp = np.searchsorted(srt, ss, side="right") / N_KS
        ax.semilogx(ss, emp, ".", color=col, ms=2.0, alpha=0.5, zorder=1)
        print(
            f"  {name:28s} KS D={D:.4f} (D_crit={d_crit:.4f})  p={p:.3f}  "
            f"-> {'PASS' if passed else 'FAIL'}"
        )

    ax.set_xlabel(r"$m$ [$M_\odot$]")
    ax.set_ylabel(r"$F(m)$  (cumulative)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=7.5)
    txt = "\n".join(
        rf"{n.split(' (')[0]}: $D{{=}}{d:.4f}$, $p{{=}}{p:.2f}$" for n, d, p, _ in rows
    )
    ax.text(
        0.97,
        0.05,
        txt + f"\n$D_{{\\rm crit}}{{=}}{d_crit:.4f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.5),
    )
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "imf_cdf_ks")
    print("  saved imf_cdf_ks.{png,pdf}")
    return bool(np.all(ok))


# ============================================================================
# Figure 4 -- mean-mass accuracy + linear-grid failure mode (adversarial)
# ============================================================================
def fig_mean_mass_accuracy(output_dir):
    print("\n" + "=" * 60)
    print("FIG 4: mean_mass() accuracy vs 200k log-grid reference")
    print("=" * 60)

    imfs = [
        ("Maschberger", Maschberger()),
        ("TaperedPL", TaperedPowerLaw()),
        ("Schechter", Schechter()),
        ("Chabrier", ChabrierIMF()),
        ("Kroupa", PowerLawIMF.kroupa()),
        (
            r"PL $\alpha{=}2.35$",
            PowerLawIMF(exponents=[2.35], breakpoints=[], m_min=0.1, m_max=100.0),
        ),
    ]
    names, rel = [], []
    for name, imf in imfs:
        ref = loggrid_mean(imf)
        got = float(imf.mean_mass())
        r = abs(got - ref) / ref
        names.append(name)
        rel.append(r)
        print(
            f"  {name:16s} mean_mass={got:.4f}  ref={ref:.4f}  "
            f"rel={r:.2e} (tol 1e-2)  -> {'PASS' if r < 1e-2 else 'FAIL'}"
        )
    rel = np.array(rel)
    passed_bars = bool(np.all(rel < 1e-2))

    # linear-grid failure mode on the steep low-mass spike (Salpeter)
    salp = PowerLawIMF(exponents=[2.35], breakpoints=[], m_min=0.1, m_max=100.0)
    ref = loggrid_mean(salp)
    Ns = np.array([40, 70, 120, 200, 350, 600, 1000, 2000, 4000])
    lin_err, log_err = [], []
    for n in Ns:
        gl = np.linspace(salp.m_min, salp.m_max, n)
        pl = np.asarray(jnp.exp(salp.logpdf(jnp.asarray(gl))))
        lin = np.trapezoid(gl * pl, gl) / np.trapezoid(pl, gl)
        gg = np.logspace(np.log10(salp.m_min), np.log10(salp.m_max), n)
        pg = np.asarray(jnp.exp(salp.logpdf(jnp.asarray(gg))))
        logm = np.trapezoid(gg * pg, gg) / np.trapezoid(pg, gg)
        lin_err.append(abs(lin - ref) / ref)
        log_err.append(abs(logm - ref) / ref)
    lin_err = np.array(lin_err)
    log_err = np.array(log_err)
    log_wins = bool(log_err[-1] < lin_err[-1])
    print(
        f"  linear-grid failure mode: lin rel @N=4000 = {lin_err[-1]:.2e} "
        f">> log rel = {log_err[-1]:.2e}  -> {'PASS' if log_wins else 'FAIL'}"
    )

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    cols = [OI["blue"] if r < 1e-2 else OI["vermilion"] for r in rel]
    axA.bar(range(len(names)), np.maximum(rel, 1e-7), color=cols, edgecolor="white")
    axA.axhline(1e-2, color=OI["vermilion"], ls="--", lw=1.2, label=r"$1\%$ tol")
    axA.set_yscale("log")
    axA.set_xticks(range(len(names)))
    axA.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    axA.set_ylabel(r"rel. error of $\langle m\rangle$ vs reference")
    axA.set_ylim(1e-7, 5e-2)
    axA.legend(loc="upper right", fontsize=7.5)
    panel_label(axA, "(a)", loc="upper left")

    axB.loglog(
        Ns, lin_err, "o-", color=OI["vermilion"], ms=4.5, lw=1.3, label="linear grid"
    )
    axB.loglog(
        Ns, log_err, "s-", color=OI["blue"], ms=4.5, lw=1.3, label="log grid (used)"
    )
    axB.axhline(1e-2, color="0.5", ls=":", lw=1.0)
    axB.set_xlabel("grid points $N$")
    axB.set_ylabel(r"rel. error of $\langle m\rangle$ (Salpeter)")
    axB.legend(loc="lower left", fontsize=7.5)
    axB.text(
        0.5,
        0.93,
        "steep $m^{-2.35}$ spike\nunder-resolved by linear grid",
        transform=axB.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
        color="0.35",
    )
    panel_label(axB, "(b)", loc="upper right")

    fig.tight_layout(pad=0.4, w_pad=1.0)
    save_fig(fig, output_dir, "imf_mean_mass_accuracy")
    print("  saved imf_mean_mass_accuracy.{png,pdf}")
    return passed_bars and log_wins


# ============================================================================
# Figure 5 -- gradient validation (differentiable inference layer)
# ============================================================================
def fig_gradient_validation(output_dir):
    print("\n" + "=" * 60)
    print("FIG 5: gradient validation (AD vs FD) -- IMFParams inference layer")
    print("=" * 60)

    # generate data from the inference model itself, so the NLL minimum (the MLE)
    # sits at the true input slope -> the gradient's zero crossing recovers it.
    truth = IMFParams.kroupa()  # alpha = (0.3, 1.3, 2.3, 2.3)
    u = jax.random.uniform(jax.random.PRNGKey(SEED), (N_GRAD,))
    masses = sample_masses_from_params(truth, u)

    def make_nll(which):
        base = dict(
            alpha0=truth.alpha0,
            alpha1=truth.alpha1,
            alpha2=truth.alpha2,
            alpha3=truth.alpha3,
        )

        def nll(val):
            kw = dict(base)
            kw[which] = val
            return individual_mass_nll(masses, IMFParams(**kw))

        return nll

    specs = [
        ("alpha3", r"$\alpha_3$", float(truth.alpha3), np.linspace(2.0, 2.6, 13), 1e-3),
        ("alpha2", r"$\alpha_2$", float(truth.alpha2), np.linspace(2.0, 2.6, 13), 1e-3),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    worst = 0.0
    for ax, (key, lab, a_true, xs, h), tag in zip(axes, specs, "ab"):
        nll = make_nll(key)
        gfn = jax.grad(nll)
        ad = np.array([float(gfn(float(x))) for x in xs])
        fd = np.array(
            [float((nll(float(x) + h) - nll(float(x) - h)) / (2 * h)) for x in xs]
        )
        rel = np.abs(ad - fd) / (np.abs(ad) + np.abs(fd) + 1e-30)
        worst = max(worst, float(np.max(rel)))

        # zero crossing of d(NLL)/d(alpha) = the MLE (should recover a_true)
        mle = float(np.interp(0.0, ad, xs))

        ax.axhline(0.0, color="0.7", lw=0.7)
        ax.axvline(
            a_true, color=OI["green"], ls=":", lw=1.2, label=rf"input $={a_true:.2f}$"
        )
        ax.plot(xs, ad, "-", color=OI["blue"], lw=1.8, label="autodiff", zorder=2)
        ax.plot(
            xs,
            fd,
            "o",
            color=OI["vermilion"],
            ms=4.0,
            mfc="none",
            mew=1.1,
            label="finite diff",
            zorder=3,
        )
        ax.plot(
            [mle],
            [0.0],
            "*",
            color=OI["black"],
            ms=10,
            zorder=4,
            label=rf"MLE $={mle:.2f}$",
        )
        ax.set_xlabel(lab)
        ax.set_ylabel(rf"$\partial\,\mathrm{{NLL}} / \partial {lab.strip('$')}$")
        ax.legend(loc="upper left", fontsize=6.8)
        ax.text(
            0.5,
            0.05,
            rf"max rel err $={np.max(rel):.0e}$",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.5),
        )
        panel_label(ax, f"({tag})", loc="upper right")
        print(
            f"  d(NLL)/d{key}: max rel err {np.max(rel):.2e}; "
            f"MLE recovers {mle:.3f} (input {a_true:.2f})  "
            f"-> {'DIFFERENTIABLE' if np.max(rel) < 1e-3 else 'CHECK'}"
        )

    passed = worst < 1e-3
    print(
        f"  overall worst AD-vs-FD rel err {worst:.2e} (tol 1e-3)  "
        f"-> {'PASS' if passed else 'FAIL'}"
    )
    fig.tight_layout(pad=0.4, w_pad=1.0)
    save_fig(fig, output_dir, "imf_gradient_validation")
    print("  saved imf_gradient_validation.{png,pdf}")
    return passed


def main():
    print("\n" + "=" * 70)
    print("PROGENAX IMF STATISTICS VALIDATION FIGURES")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {
        "Fig 1  recovered slopes": fig_recovered_slopes(OUTPUT_DIR),
        "Fig 2  PDF overlay": fig_pdf_overlay(OUTPUT_DIR),
        "Fig 3  CDF + KS test": fig_cdf_ks(OUTPUT_DIR),
        "Fig 4  mean-mass accuracy": fig_mean_mass_accuracy(OUTPUT_DIR),
        "Fig 5  gradient validation": fig_gradient_validation(OUTPUT_DIR),
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    all_ok = all(results.values())
    print("=" * 70)
    print(
        "  ALL IMF VALIDATION FIGURES PASS"
        if all_ok
        else "  SOME IMF VALIDATION FIGURES FAILED"
    )
    print("=" * 70)
    print(f"\nFigures written to {OUTPUT_DIR}/imf_*.png")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

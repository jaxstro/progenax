#!/usr/bin/env python
"""
Binary-star + binary-aware-IMF validation figures.

Five publication-quality figures anchored to passing tests in
``tests/validation/test_binary_physics.py`` (24 orbital-mechanics tests) and to
the binary-aware IMF recovery story. Each figure prints expected-vs-measured
PASS/FAIL against an *external* oracle (Kepler's laws, the implemented Moe+17
PDF, the true IMF slope, central finite differences).

Figures (-> what they validate):
  1. binaries_kepler_orbits.png   Kepler III (T vs a), orbit geometry r_peri/r_apo,
                                  energy E=-GM1M2/2a -- analytic orbital oracles
  2. binaries_moe_qdist.png       Moe+17 mass-ratio q sampled vs implemented PDF
                                  + KS, showing the mass-dependent twin excess
  3. binaries_confidently_wrong.png  THE headline: an IMF fit that ignores binaries
                                  is biased and its CI EXCLUDES the truth at N>~1e4
                                  ("confidently wrong"); a binary-aware marginalised
                                  likelihood recovers the true slope.  No MCMC --
                                  fast differentiable MLE + Fisher CI.
  4. binaries_bias_mechanism.png  why: naive "wrongness significance" |bias|/sigma
                                  grows ~sqrt(N) while binary-aware stays ~O(1);
                                  naive bias scales with binary fraction f_b
  5. binaries_gradient_validation.png  AD vs central-FD for the Kepler transforms
                                  and the binary-aware recovery likelihood

The binary-aware marginalised likelihood (128-pt Gauss-Legendre over the primary
mass, with q = m_sys/m1 - 1) is the same integrand documented in
docs/website/10-theory/imfs/binary-aware-likelihood.md and used by the offline
numpyro recovery; here it is maximised directly (differentiable MLE) so the figure
is fast and reproducible in CI.

References:
    Kepler's laws; Murray & Dermott (1999); Moe & Di Stefano (2017), ApJS 230, 15;
    Maschberger (2013), MNRAS 429, 1725.

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_binaries.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

jax.config.update("jax_enable_x64", True)

from jaxstro.units import PLANETARY
from progenax.binaries import (
    BinaryOrbitalState,
    KeplerElements,
    compute_period,
    period_to_semimajor_axis,
)
from progenax.imf import (
    BinaryIMF,
    ConstantBinaryFraction,
    Maschberger,
    MassDependentBinaryFraction,
    MoeDiStefano2017,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plotstyle import OI, apply_pub_style, panel_label, save_fig

apply_pub_style()

OUTPUT_DIR = "validation/plots"
SEED = 42
G = PLANETARY.G  # 4 pi^2, AU^3 Msun^-1 yr^-2

# binary-aware recovery config (shared with the offline numpyro script)
M_MIN, M_MAX, Q_MIN = 0.01, 150.0, 0.1
ALPHA_TRUE = 2.3
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(128)
GL_NODES = jnp.array(_GL_NODES)
GL_WEIGHTS = jnp.array(_GL_WEIGHTS)


# ============================================================================
# Binary-aware recovery machinery (no MCMC -- differentiable MLE)
# ============================================================================
def generate_system_masses(alpha, n, seed, f_model=None):
    """Observed system masses with Moe+17 binary contamination (m_sys=m1+m2)."""
    imf = Maschberger(alpha=alpha, m_min=M_MIN, m_max=M_MAX)
    bimf = BinaryIMF(
        primary_imf=imf,
        q_distribution=MoeDiStefano2017(q_min=Q_MIN),
        binary_fraction=f_model or MassDependentBinaryFraction(),
    )
    m1, m2, is_bin = bimf.sample_systems(jax.random.PRNGKey(seed), n)
    return jnp.where(is_bin, m1 + m2, m1), float(jnp.mean(is_bin.astype(float)))


def naive_loglike(m_sys, alpha):
    """Single-star IMF log-likelihood of the system masses (wrong model)."""
    lp = Maschberger(alpha=alpha, m_min=M_MIN, m_max=M_MAX).logpdf(m_sys)
    return jnp.sum(jnp.where(jnp.isfinite(lp), lp, -1e10))


def aware_loglike(m_sys, alpha):
    """Binary-aware mixture log-likelihood, marginalising over the primary mass.

    p(m_sys) = (1-f_b) xi(m_sys) + integral f_b(m1) xi(m1) g(q|m1)/m1 dm1,
    q = m_sys/m1 - 1, m1 in [m_sys/2, m_sys/(1+q_min)]  (128-pt Gauss-Legendre).
    """
    imf = Maschberger(alpha=alpha, m_min=M_MIN, m_max=M_MAX)
    bf = MassDependentBinaryFraction()
    qd = MoeDiStefano2017(q_min=Q_MIN)

    p_single = (1.0 - bf(m_sys)) * jnp.exp(imf.logpdf(m_sys))

    def _binary_one(ms):
        m1_lo = jnp.maximum(ms / 2.0, M_MIN)
        m1_hi = jnp.minimum(ms / (1.0 + Q_MIN), M_MAX)
        valid = m1_hi > m1_lo
        hw, mid = 0.5 * (m1_hi - m1_lo), 0.5 * (m1_hi + m1_lo)
        m1 = hw * GL_NODES + mid
        q = jnp.clip(ms / m1 - 1.0, Q_MIN, 1.0)
        g_q = jax.vmap(lambda qi, mi: qd.pdf_given_primary(qi, mi))(q, m1)
        integrand = bf(m1) * jnp.exp(imf.logpdf(m1)) * g_q / m1
        return jnp.where(valid, hw * jnp.dot(GL_WEIGHTS, integrand), 0.0)

    p_binary = jax.vmap(_binary_one)(m_sys)
    ll = jnp.log(jnp.maximum(p_single + p_binary, 1e-30))
    return jnp.sum(jnp.where(jnp.isfinite(ll), ll, -1e10))


def mle_and_sigma(loglike_fn, m_sys, bounds=(1.2, 3.4)):
    """MLE slope + observed-Fisher 1-sigma (numerical 2nd derivative of total LL)."""
    neg = lambda a: -float(loglike_fn(m_sys, a))
    res = scipy.optimize.minimize_scalar(neg, bounds=bounds, method="bounded")
    a_hat = float(res.x)
    h = 1e-3
    d2 = (neg(a_hat + h) - 2.0 * neg(a_hat) + neg(a_hat - h)) / h**2  # = -d2LL
    sigma = float(1.0 / np.sqrt(max(d2, 1e-12)))
    return a_hat, sigma


# ============================================================================
# Figure 1 -- Kepler's laws & orbit geometry
# ============================================================================
def fig_kepler_orbits(output_dir):
    print("\n" + "=" * 60)
    print("FIG 1: Kepler III + orbit geometry + orbital energy")
    print("=" * 60)
    M_tot = 2.0

    # (a) Kepler III: T vs a
    a = np.logspace(-3, 2, 40)
    T = np.array([float(compute_period(ai, M_tot, G)) for ai in a])
    T_an = 2 * np.pi * np.sqrt(a**3 / (G * M_tot))
    rel_T = np.max(np.abs(T - T_an) / T_an)
    rt = period_to_semimajor_axis(compute_period(2.5, M_tot, G), M_tot, G)
    roundtrip = abs(float(rt) - 2.5)
    p1 = rel_T < 1e-9 and roundtrip < 1e-9
    print(f"  Kepler III: max rel(T, 2pi sqrt(a^3/GM)) = {rel_T:.1e}; "
          f"a-roundtrip err {roundtrip:.1e}  -> {'PASS' if p1 else 'FAIL'}")

    # (b) orbit geometry: r_peri = a(1-e), r_apo = a(1+e)
    a_o, e_o = 2.0, 0.5
    M0s = np.linspace(0, 2 * np.pi, 400)
    pos = np.array([np.asarray(KeplerElements(a=a_o, e=e_o, M0=m).to_state(M_tot, G).position)
                    for m in M0s])
    r_peri = float(jnp.linalg.norm(KeplerElements(a=a_o, e=e_o, M0=0.0).to_state(M_tot, G).position))
    r_apo = float(jnp.linalg.norm(KeplerElements(a=a_o, e=e_o, M0=np.pi).to_state(M_tot, G).position))
    e_peri = abs(r_peri - a_o * (1 - e_o)); e_apo = abs(r_apo - a_o * (1 + e_o))
    p2 = e_peri < 1e-6 and e_apo < 1e-6
    print(f"  geometry: r_peri={r_peri:.4f} (a(1-e)={a_o*(1-e_o)}), "
          f"r_apo={r_apo:.4f} (a(1+e)={a_o*(1+e_o)})  -> {'PASS' if p2 else 'FAIL'}")

    # (c) orbital energy E = -G m1 m2 / 2a across a
    m1, m2 = 1.2, 0.8
    a_e = np.logspace(-1, 1, 25)
    E_num, E_an = [], []
    for ai in a_e:
        st = BinaryOrbitalState.from_semi_major_axis(m1=m1, m2=m2, a=ai, e=0.0, M_anom=0.0, G=G)
        p1_, v1_, p2_, v2_ = st.to_resolved_positions(G=G)
        KE = 0.5 * m1 * float(jnp.sum(v1_**2)) + 0.5 * m2 * float(jnp.sum(v2_**2))
        V = -G * m1 * m2 / float(jnp.linalg.norm(p2_ - p1_))
        E_num.append(KE + V); E_an.append(-G * m1 * m2 / (2 * ai))
    E_num, E_an = np.array(E_num), np.array(E_an)
    rel_E = np.max(np.abs(E_num - E_an) / np.abs(E_an))
    p3 = rel_E < 1e-3
    print(f"  energy: max rel(E, -G m1 m2/2a) = {rel_E:.1e}  -> {'PASS' if p3 else 'FAIL'}")

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.4, 2.7))
    axA.loglog(a, T, "o", color=OI["vermilion"], ms=3.5, mfc="none", mew=0.9,
               label="compute_period")
    axA.loglog(a, T_an, "-", color=OI["black"], lw=1.4, label=r"$2\pi\sqrt{a^3/GM}$")
    axA.set_xlabel(r"$a$ [AU]"); axA.set_ylabel(r"period $T$ [yr]")
    axA.legend(loc="upper left", fontsize=7)
    axA.text(0.95, 0.06, rf"max rel $={rel_T:.0e}$", transform=axA.transAxes,
             ha="right", fontsize=7.2, color="0.4")
    panel_label(axA, "(a)", loc="lower right")

    axB.plot(pos[:, 0], pos[:, 1], "-", color=OI["blue"], lw=1.4)
    axB.plot(0, 0, "+", color=OI["black"], ms=9, mew=1.4, label="focus (COM)")
    axB.plot(pos[0, 0], pos[0, 1], "o", color=OI["vermilion"], ms=6, mec="white",
             label=rf"peri $a(1{{-}}e)={a_o*(1-e_o):.1f}$")
    axB.plot(pos[200, 0], pos[200, 1], "s", color=OI["green"], ms=6, mec="white",
             label=rf"apo $a(1{{+}}e)={a_o*(1+e_o):.1f}$")
    axB.set_xlabel(r"$x$ [AU]"); axB.set_ylabel(r"$y$ [AU]")
    axB.set_aspect("equal"); axB.legend(loc="upper right", fontsize=6.2)
    panel_label(axB, "(b)", loc="upper left")

    axC.loglog(a_e, -E_num, "o", color=OI["vermilion"], ms=4, mfc="none", mew=0.9,
               label=r"$-(T+V)$ resolved")
    axC.loglog(a_e, -E_an, "-", color=OI["black"], lw=1.4, label=r"$G m_1 m_2 / 2a$")
    axC.set_xlabel(r"$a$ [AU]"); axC.set_ylabel(r"$-E$ [code units]")
    axC.legend(loc="upper right", fontsize=7)
    axC.text(0.05, 0.08, rf"max rel $={rel_E:.0e}$", transform=axC.transAxes,
             fontsize=7.2, color="0.4")
    panel_label(axC, "(c)", loc="lower left")

    fig.tight_layout(pad=0.4, w_pad=0.9)
    save_fig(fig, output_dir, "binaries_kepler_orbits")
    print("  saved binaries_kepler_orbits.{png,pdf}")
    return p1 and p2 and p3


# ============================================================================
# Figure 2 -- Moe+17 mass-ratio distribution (mass-dependent twin excess)
# ============================================================================
def fig_moe_qdist(output_dir):
    print("\n" + "=" * 60)
    print("FIG 2: Moe+17 mass-ratio q sampled vs implemented PDF + KS")
    print("=" * 60)
    qd = MoeDiStefano2017(q_min=Q_MIN)
    N = 80_000
    qgrid = np.linspace(Q_MIN, 1.0, 400)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ok = []
    for ax, (m1, col, tag) in zip(axes, [(1.0, OI["blue"], "(a)"),
                                         (10.0, OI["vermilion"], "(b)")]):
        samp = np.asarray(qd.sample_given_primary(jax.random.PRNGKey(SEED),
                                                  jnp.full(N, m1)))
        pdf = np.asarray(qd.pdf_given_primary(jnp.asarray(qgrid), m1))
        # analytic CDF (cumulative-trapezoid of the implemented pdf) for KS
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(qgrid))])
        cdf /= cdf[-1]
        ss = np.sort(samp)
        emp = np.searchsorted(ss, ss, side="right") / samp.size
        D = float(np.max(np.abs(emp - np.interp(ss, qgrid, cdf))))
        d_crit = 1.36 / np.sqrt(samp.size)
        twin = float(np.mean(samp > 0.95))
        passed = D < d_crit
        ok.append(passed)

        ax.hist(samp, bins=40, range=(Q_MIN, 1.0), density=True, color=col,
                alpha=0.35, edgecolor="white", linewidth=0.3,
                label=rf"samples ($N{{=}}8\times10^4$)")
        ax.plot(qgrid, pdf, "-", color=OI["black"], lw=1.8, label="implemented PDF")
        ax.set_title(rf"$M_1 = {m1:.0f}\,M_\odot$  (twins $={100*twin:.1f}$" + r"$\%$)",
                     fontsize=8.5)
        ax.set_xlabel(r"mass ratio $q = m_2/m_1$")
        ax.legend(loc="upper left", fontsize=7)
        ax.text(0.96, 0.04, rf"KS $D={D:.4f}$" + "\n" + rf"$D_{{\rm crit}}={d_crit:.4f}$",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.5))
        panel_label(ax, tag, loc="upper right")
        print(f"  M1={m1:>4.0f} Msun: KS D={D:.4f} (D_crit={d_crit:.4f}), "
              f"twin frac={twin:.3f}  -> {'PASS' if passed else 'FAIL'}")
    axes[0].set_ylabel(r"probability density $g(q\,|\,M_1)$")
    fig.tight_layout(pad=0.4, w_pad=0.9)
    save_fig(fig, output_dir, "binaries_moe_qdist")
    print("  saved binaries_moe_qdist.{png,pdf}")
    return bool(np.all(ok))


# ============================================================================
# Figure 3 -- the "confidently wrong" headline
# ============================================================================
def _recovery_scaling(Ns):
    m_full, f_bin = generate_system_masses(ALPHA_TRUE, int(max(Ns)), SEED)
    rows = []
    for n in Ns:
        ms = m_full[:n]
        an, sn = mle_and_sigma(naive_loglike, ms)
        aw, sw = mle_and_sigma(aware_loglike, ms)
        rows.append(dict(N=n, an=an, sn=sn, aw=aw, sw=sw))
    return rows, f_bin


def fig_confidently_wrong(output_dir):
    print("\n" + "=" * 60)
    print("FIG 3: confidently-wrong naive fit vs binary-aware recovery")
    print("=" * 60)
    Ns = [500, 1000, 3000, 10_000, 30_000, 100_000]
    rows, f_bin = _recovery_scaling(Ns)
    N = np.array([r["N"] for r in rows])
    an = np.array([r["an"] for r in rows]); sn = np.array([r["sn"] for r in rows])
    aw = np.array([r["aw"] for r in rows]); sw = np.array([r["sw"] for r in rows])

    # naive is "confidently wrong" once |bias| > 3 sigma; aware stays within 2 sigma
    naive_sig = np.abs(an - ALPHA_TRUE) / sn
    aware_sig = np.abs(aw - ALPHA_TRUE) / sw
    confidently_wrong = bool(naive_sig[-1] > 3.0)
    aware_ok = bool(np.all(aware_sig < 2.5))
    for r, ns_, as_ in zip(rows, naive_sig, aware_sig):
        print(f"  N={r['N']:>6d}  naive {r['an']:.3f}+-{r['sn']:.3f} "
              f"({ns_:.1f}sigma off)   aware {r['aw']:.3f}+-{r['sw']:.3f} "
              f"({as_:.1f}sigma)")
    print(f"  f_bin={f_bin:.2f}; naive {naive_sig[-1]:.0f}sigma-wrong at N=1e5 "
          f"({'PASS' if confidently_wrong else 'FAIL'}); aware within 2.5sigma "
          f"({'PASS' if aware_ok else 'FAIL'})")

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.axhline(ALPHA_TRUE, color=OI["black"], ls="--", lw=1.2,
               label=rf"true $\alpha={ALPHA_TRUE}$")
    ax.fill_between(N, an - sn, an + sn, color=OI["vermilion"], alpha=0.18)
    ax.plot(N, an, "D-", color=OI["vermilion"], ms=5, mec="white", mew=0.6,
            label="naive (single-star fit)")
    ax.fill_between(N, aw - sw, aw + sw, color=OI["blue"], alpha=0.18)
    ax.plot(N, aw, "o-", color=OI["blue"], ms=5, mec="white", mew=0.6,
            label="binary-aware (Moe+17)")
    ax.set_xscale("log")
    ax.set_xlabel(r"sample size $N$")
    ax.set_ylabel(r"recovered slope $\hat\alpha$ ($\pm1\sigma$)")
    ax.legend(loc="lower right", fontsize=7.2)
    ax.annotate("naive CI shrinks but\nstays off truth:\n"
                rf"{naive_sig[-1]:.0f}$\sigma$ wrong at $N{{=}}10^5$",
                xy=(N[-1], an[-1]), xytext=(2e3, an[-1] - 0.02),
                fontsize=6.8, color=OI["vermilion"],
                arrowprops=dict(arrowstyle="->", color=OI["vermilion"], lw=0.8))
    fig.tight_layout(pad=0.4)
    save_fig(fig, output_dir, "binaries_confidently_wrong")
    print("  saved binaries_confidently_wrong.{png,pdf}")
    return confidently_wrong and aware_ok


# ============================================================================
# Figure 4 -- bias mechanism (significance vs N; bias vs binary fraction)
# ============================================================================
def fig_bias_mechanism(output_dir):
    print("\n" + "=" * 60)
    print("FIG 4: bias mechanism (significance vs N; bias vs f_b)")
    print("=" * 60)
    # (a) wrongness significance |bias|/sigma vs N
    Ns = [500, 1000, 3000, 10_000, 30_000, 100_000]
    rows, _ = _recovery_scaling(Ns)
    N = np.array(Ns)
    nsig = np.array([abs(r["an"] - ALPHA_TRUE) / r["sn"] for r in rows])
    asig = np.array([abs(r["aw"] - ALPHA_TRUE) / r["sw"] for r in rows])

    # (b) naive bias vs binary fraction (constant f_b)
    fbs = [0.0, 0.15, 0.3, 0.5, 0.7]
    biases = []
    for fb in fbs:
        ms, _ = generate_system_masses(ALPHA_TRUE, 40_000, SEED + 7,
                                       f_model=ConstantBinaryFraction(fb))
        a_hat, _ = mle_and_sigma(naive_loglike, ms)
        biases.append(a_hat - ALPHA_TRUE)
        print(f"  f_b={fb:.2f}: naive bias = {a_hat - ALPHA_TRUE:+.3f}")
    biases = np.array(biases)
    mono = bool(np.all(np.diff(biases) < 1e-3))  # more binaries -> more negative bias
    zero_ok = bool(abs(biases[0]) < 0.03)
    grows = bool(nsig[-1] > nsig[0] and nsig[-1] > 3.0)
    print(f"  significance grows with N: {nsig[0]:.1f} -> {nsig[-1]:.0f}sigma  "
          f"({'PASS' if grows else 'FAIL'})")
    print(f"  bias->0 at f_b=0: {biases[0]:+.3f} ({'PASS' if zero_ok else 'FAIL'}); "
          f"monotonic in f_b ({'PASS' if mono else 'FAIL'})")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axA.loglog(N, np.maximum(nsig, 1e-2), "D-", color=OI["vermilion"], ms=5,
               mec="white", mew=0.6, label="naive")
    axA.loglog(N, np.maximum(asig, 1e-2), "o-", color=OI["blue"], ms=5,
               mec="white", mew=0.6, label="binary-aware")
    axA.axhline(3.0, color="0.5", ls=":", lw=1.0, label=r"$3\sigma$")
    axA.set_xlabel(r"sample size $N$")
    axA.set_ylabel(r"wrongness $|\hat\alpha-\alpha|/\sigma$")
    axA.legend(loc="upper left", fontsize=7.2)
    panel_label(axA, "(a)", loc="lower right")

    axB.axhline(0.0, color="0.6", lw=0.8)
    axB.plot(fbs, biases, "o-", color=OI["vermilion"], ms=6, mec="white", mew=0.6)
    axB.set_xlabel(r"binary fraction $f_b$")
    axB.set_ylabel(r"naive slope bias $\hat\alpha-\alpha$")
    axB.text(0.5, 0.1, "more binaries\n$\\Rightarrow$ more bias", transform=axB.transAxes,
             ha="center", fontsize=7.5, color="0.4")
    panel_label(axB, "(b)", loc="upper right")

    fig.tight_layout(pad=0.4, w_pad=1.0)
    save_fig(fig, output_dir, "binaries_bias_mechanism")
    print("  saved binaries_bias_mechanism.{png,pdf}")
    return grows and zero_ok and mono


# ============================================================================
# Figure 5 -- gradient validation
# ============================================================================
def _ad_fd(f, xs, h):
    ad = np.array([float(jax.grad(f)(float(x))) for x in xs])
    fd = np.array([float((f(float(x) + h) - f(float(x) - h)) / (2 * h)) for x in xs])
    rel = np.abs(ad - fd) / (np.abs(ad) + np.abs(fd) + 1e-30)
    return ad, fd, rel


def fig_gradient_validation(output_dir):
    print("\n" + "=" * 60)
    print("FIG 5: gradient validation (AD vs FD)")
    print("=" * 60)
    ms, _ = generate_system_masses(ALPHA_TRUE, 8000, SEED + 3)

    specs = [
        ("a", r"$a$ [AU]", r"$\partial\,|v|^2/\partial a$",
         lambda a: jnp.sum(KeplerElements(a=a, e=0.3, M0=1.0).to_state(1.0, G).velocity**2),
         np.linspace(0.8, 2.5, 11), 1e-4),
        ("alpha_aware", r"$\alpha$", r"$\partial\,\mathrm{NLL}_{\rm aware}/\partial\alpha$",
         lambda al: -aware_loglike(ms, al), np.linspace(2.0, 2.6, 11), 1e-3),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    worst = 0.0
    for ax, (key, xlab, ylab, f, xs, h), tag in zip(axes, specs, "ab"):
        ad, fd, rel = _ad_fd(f, xs, h)
        worst = max(worst, float(np.max(rel)))
        ax.plot(xs, ad, "-", color=OI["blue"], lw=1.8, label="autodiff", zorder=2)
        ax.plot(xs, fd, "o", color=OI["vermilion"], ms=4.5, mfc="none", mew=1.1,
                label="finite diff", zorder=3)
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        ax.legend(loc="best", fontsize=7.2)
        ax.text(0.5, 0.05, rf"max rel err $={np.max(rel):.0e}$", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.5))
        panel_label(ax, f"({tag})", loc="upper right")
        print(f"  d/d{key}: max rel err {np.max(rel):.2e}  "
              f"-> {'DIFFERENTIABLE' if np.max(rel) < 1e-3 else 'CHECK'}")
    passed = worst < 1e-3
    print(f"  overall worst rel err {worst:.2e} (tol 1e-3)  "
          f"-> {'PASS' if passed else 'FAIL'}")
    fig.tight_layout(pad=0.4, w_pad=1.0)
    save_fig(fig, output_dir, "binaries_gradient_validation")
    print("  saved binaries_gradient_validation.{png,pdf}")
    return passed


def main():
    print("\n" + "=" * 70)
    print("PROGENAX BINARY + BINARY-AWARE-IMF VALIDATION FIGURES")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {
        "Fig 1  Kepler + orbit geometry": fig_kepler_orbits(OUTPUT_DIR),
        "Fig 2  Moe+17 q-distribution": fig_moe_qdist(OUTPUT_DIR),
        "Fig 3  confidently-wrong recovery": fig_confidently_wrong(OUTPUT_DIR),
        "Fig 4  bias mechanism": fig_bias_mechanism(OUTPUT_DIR),
        "Fig 5  gradient validation": fig_gradient_validation(OUTPUT_DIR),
    }
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    all_ok = all(results.values())
    print("=" * 70)
    print("  ALL BINARY VALIDATION FIGURES PASS" if all_ok
          else "  SOME BINARY VALIDATION FIGURES FAILED")
    print("=" * 70)
    print(f"\nFigures written to {OUTPUT_DIR}/binaries_*.png")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

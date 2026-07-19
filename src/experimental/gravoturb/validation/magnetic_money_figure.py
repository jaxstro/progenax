"""Magnetized-turbulence money figure (ADR-0060..0063), 3 panels:

(a) the magnetized natal cloud — gas column density + the realized divergence-free vector B
    field (the RMHD-seeding payload);
(b) magnetic regulation of star formation — collapse-eligible fraction vs Alfven Mach number,
    ideal MHD (flux-frozen: SF ceases below trans-Alfvenic) vs + ambipolar diffusion (flux loss
    rescues SF);
(c) the two channels behind (b), from the REAL realized field (128^3, stacked seeds) on a log
    axis so the BM19 power-law tail is visible: magnetic support narrows the density PDF, and the
    s_crit channel restores the collapse threshold to ~hydro (width-only would drop it and
    over-produce).

Publication settings (300 dpi, colorblind-safe seaborn palette, PNG + vector PDF). numpy is
permitted here (validation/analysis side); scipy is avoided (math.erfc + bisection).

Run:
    PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync \
        python -m gravoturb.validation.magnetic_money_figure
"""

import math
import os

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from jaxstro.units import STELLAR  # noqa: E402

from gravoturb.cluster import build_cluster_ic  # noqa: E402
from gravoturb.realization.magnetic import (  # noqa: E402
    beta_from_mass_to_flux,
    magnetothermal_threshold_shift,
)
from gravoturb.realization.pipeline import build_turbulent_field  # noqa: E402
from gravoturb.specs import (  # noqa: E402
    CloudSpec,
    CompositionSpec,
    GasSpec,
    GeometrySpec,
    MagneticSpec,
    VelocitySpec,
)
from progenax import PlummerProfile  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "plots", "feasibility")
os.makedirs(OUT, exist_ok=True)

G = STELLAR.G
MASSES = jnp.linspace(0.3, 8.0, 300)
BOX, N = 4.0, 48
MACH, BDR, ALPHA, SFE, RH, CS = 8.0, 0.5, 1.8, 0.02, 1.2, 1.0


def _kw():
    return dict(
        cloud=CloudSpec(mach=MACH, b=BDR, alpha=ALPHA, beta=3.0),
        geometry=GeometrySpec(profile=PlummerProfile(r_h=RH), box_size=BOX, shape=(N, N, N)),
        velocity=VelocitySpec(beta_v=4.0, mode="physical", c_s=CS),
        composition=CompositionSpec(), gas=GasSpec(sfe=SFE),
        G=G, units=STELLAR, key=jax.random.PRNGKey(0),
    )


# --- analytic collapse-eligible mass fraction (the model's own theory) -------------------- #
_M_HALF = float(jnp.sum(MASSES)) / (2 * SFE)
_CS_INT = CS / STELLAR.velocity_scale_km_s
_SIG2_HY = math.log(1 + (BDR * MACH) ** 2)


def _beta0(mu):
    return float(beta_from_mass_to_flux(mu, mach=MACH, c_s=_CS_INT, m_half=_M_HALF, r_h=RH, G=G))


def _sig2(beta0):
    return math.log(1 + (BDR * MACH) ** 2 * beta0 / (beta0 + 1))


def _f_above(thr, sig2):
    """Mass-weighted lognormal fraction with s > thr."""
    return 0.5 * math.erfc((thr - 0.5 * sig2) / math.sqrt(2 * sig2))


def _f_ideal(mu):
    b0 = _beta0(mu); s2 = _sig2(b0)
    return _f_above((ALPHA - 0.5) * s2 + float(magnetothermal_threshold_shift(b0)), s2)


def _f_ambi(mu, s_ad=1.0, kap=4.0):
    """Ambipolar: per-cell threshold thr(s)=s_t+Δs·σ(κ(s_ad−s)); eligible above the crossover."""
    b0 = _beta0(mu); s2 = _sig2(b0); d0 = float(magnetothermal_threshold_shift(b0))
    st0 = (ALPHA - 0.5) * s2

    def thr(s):
        return st0 + d0 / (1 + math.exp(-kap * (s_ad - s)))

    lo, hi = st0 - 1.0, st0 + d0 + 5.0            # bisection on g(s)=s−thr(s) (monotone ↑)
    if (lo - thr(lo)) * (hi - thr(hi)) >= 0:
        return _f_above(st0 + d0, s2)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if (lo - thr(lo)) * (mid - thr(mid)) <= 0:
            hi = mid
        else:
            lo = mid
    return _f_above(0.5 * (lo + hi), s2)


def build_figure():
    sns.set_theme(context="paper", style="whitegrid", font="serif")
    cb = sns.color_palette("colorblind")
    c_ideal, c_ambi = cb[0], cb[2]
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 11.5, "axes.titlesize": 11.5,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9,
        "mathtext.fontset": "cm", "savefig.dpi": 300,
    })

    # (a) magnetized cloud + vector B
    ic = build_cluster_ic(MASSES, **_kw(), magnetic=MagneticSpec(
        mu_phi=4.0, realize="field", anisotropy="fixed", anisotropy_value=1.5))
    col = np.asarray(ic.gas.rho_cloud).sum(2) * float(ic.gas.cell_volume) / (BOX / N) ** 2
    lc = np.log10(np.maximum(col, col[col > 0].min()))
    B = np.asarray(ic.magnetic.B_field)
    Bx, By = B[0].mean(2), B[1].mean(2)
    xs = np.linspace(0, BOX, N)

    # (b) SF-regulation curves
    mus = np.geomspace(0.6, 40, 40)
    mA = np.array([MACH * math.sqrt(_beta0(m) / 2) for m in mus])
    f_hy = _f_above((ALPHA - 0.5) * _SIG2_HY, _SIG2_HY)
    yi = np.array([_f_ideal(m) for m in mus]) / f_hy
    ya = np.minimum(np.array([_f_ambi(m) for m in mus]) / f_hy, 1.0)

    # (c) REAL realized-field PDFs (hydro vs magnetic b_eff), stacked seeds @128^3
    mu_c = 3.0; b0c = _beta0(mu_c); d0 = float(magnetothermal_threshold_shift(b0c))
    b_eff = BDR * math.sqrt(b0c / (b0c + 1.0))

    def real_s(b_drive):
        out, f = [], None
        for sd in range(3):
            f = build_turbulent_field(MACH, float(b_drive), ALPHA, 3.0, (128, 128, 128),
                                      jax.random.PRNGKey(sd))
            out.append(np.asarray(f.s).ravel())
        return np.concatenate(out), float(f.s_t)

    s_hy, st_hy = real_s(BDR)
    s_mag, st_mag = real_s(b_eff)
    s_crit_mag = st_mag + d0

    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(15.5, 4.35),
                                        gridspec_kw={"width_ratios": [1.0, 1.05, 1.0]})

    # --- (a)
    axa.set_facecolor("#0d0d12"); axa.grid(False)
    im = axa.imshow(lc.T, origin="lower", extent=[0, BOX, 0, BOX], cmap="magma",
                    vmax=np.percentile(lc, 99.5))
    axa.streamplot(xs, xs, Bx.T, By.T, color="#e3ecff", density=1.15, linewidth=0.7, arrowsize=0.6)
    axa.set_title(r"(a) Magnetized natal cloud $+$ divergence-free $\mathbf{B}$")
    axa.set_xlabel("$x$ [pc]"); axa.set_ylabel("$y$ [pc]")
    cbar = fig.colorbar(im, ax=axa, fraction=0.046, pad=0.02)
    cbar.set_label(r"$\log_{10}\,\Sigma_{\rm gas}\ [M_\odot\,{\rm pc}^{-2}]$", fontsize=10)

    # --- (b)
    axb.axhline(1.0, color="0.6", lw=0.9, ls=":")
    axb.plot(mA, yi, color=c_ideal, lw=2.4, label="ideal MHD (flux-frozen)")
    axb.plot(mA, ya, color=c_ambi, lw=2.4, ls="--", label="+ ambipolar diffusion")
    axb.fill_between(mA, yi, ya, color=c_ambi, alpha=0.12)
    axb.set_xscale("log"); axb.set_xlim(mA.min(), mA.max()); axb.set_ylim(0, 1.13)
    axb.axvline(1.0, color="0.5", lw=0.8)
    axb.text(1.0, 1.055, "trans-Alfvénic", ha="center", va="bottom", fontsize=8, color="0.45")
    axb.annotate("SF ceases\n(flux-frozen)", xy=(mA.min() * 1.25, 0.03),
                 xytext=(mA.min() * 2.4, 0.30), fontsize=8.5, color=c_ideal, ha="left",
                 arrowprops=dict(arrowstyle="->", color=c_ideal, lw=1.0))
    axb.annotate("flux loss\nrescues SF", xy=(1.55, 0.62), xytext=(3.2, 0.45),
                 fontsize=8.5, color=c_ambi, ha="left",
                 arrowprops=dict(arrowstyle="->", color=c_ambi, lw=1.0))
    axb.set_xlabel(r"Alfvén Mach number $\mathcal{M}_A$  ($\leftarrow$ stronger field)")
    axb.set_ylabel(r"collapse-eligible fraction / hydro")
    axb.set_title("(b) Magnetic regulation of star formation")
    axb.legend(frameon=False, loc="lower right", borderaxespad=0.8)
    sns.despine(ax=axb)

    # --- (c)
    edges = np.linspace(-10, 12, 120); ctr = 0.5 * (edges[:-1] + edges[1:])
    h_hy = np.histogram(s_hy, bins=edges, density=True)[0]
    h_mag = np.histogram(s_mag, bins=edges, density=True)[0]
    axc.plot(ctr, np.where(h_hy > 0, h_hy, np.nan), color="0.55", lw=2.0, label="hydro")
    axc.plot(ctr, np.where(h_mag > 0, h_mag, np.nan), color=c_ideal, lw=2.4,
             label=r"magnetic ($\mu_\Phi=3$)")
    axc.axvline(st_mag, color="#d1495b", lw=1.5, ls="--")
    axc.axvline(st_hy, color="0.5", lw=1.4, ls=":")
    axc.axvline(s_crit_mag, color=c_ideal, lw=1.5, ls=":")
    axc.annotate("width-only $s_t$\n(would over-produce)", xy=(st_mag, 3.5e-3),
                 xytext=(st_mag - 0.6, 3.5e-3), fontsize=8, ha="right", va="center",
                 color="#d1495b", arrowprops=dict(arrowstyle="->", color="#d1495b", lw=1.0))
    axc.annotate(r"$s_{\rm crit}$ restores threshold" + "\nto $\\approx$ hydro $s_t$",
                 xy=((st_hy + s_crit_mag) / 2, 3.0e-2), xytext=(5.0, 8e-2), fontsize=8,
                 ha="left", va="center", color="0.2",
                 arrowprops=dict(arrowstyle="->", color="0.2", lw=1.0))
    axc.annotate("width $\\downarrow$", xy=(-1.6, 0.34), xytext=(-9.4, 0.05), fontsize=9.5,
                 color=c_ideal, arrowprops=dict(arrowstyle="->", color=c_ideal, lw=1.1))
    axc.annotate("power-law tail", xy=(6.0, 2.0e-3), xytext=(7.5, 1.5e-2), fontsize=7.5,
                 color="0.45", ha="center", arrowprops=dict(arrowstyle="->", color="0.55", lw=0.9))
    axc.set_yscale("log"); axc.set_ylim(1e-3, 0.6); axc.set_xlim(-10, 12)
    axc.set_xlabel(r"log density  $s=\ln(\rho/\rho_0)$")
    axc.set_ylabel(r"volume PDF  $p(s)$")
    axc.set_title("(c) Two channels: narrower PDF, threshold restored")
    axc.legend(frameon=False, loc="upper left", fontsize=8.5)
    sns.despine(ax=axc)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"gravoturb_magnetic.{ext}"), facecolor="white",
                    bbox_inches="tight")
    print(f"[magnetic] wrote {OUT}/gravoturb_magnetic.png (+ .pdf)")
    print(f"    hydro collapse-eligible fraction = {f_hy:.4f};  "
          f"M_A range [{mA.min():.2f}, {mA.max():.2f}]")
    print(f"    (c) thresholds: width-only s_t={st_mag:.2f}, s_crit={s_crit_mag:.2f}, "
          f"hydro s_t={st_hy:.2f}")


if __name__ == "__main__":
    build_figure()

"""ICViz IMF figures.

F6 ``imf-classic-slopes`` — the classic-IMF comparison + local-slope panel:
the figure behind the house rule that Maschberger is the preferred production
default because it is SMOOTH. Panel (a) overlays the four classic IMFs (with
sampled histograms proving sampler == pdf); panel (b) plots the local slope
S(m) = -d ln xi / d ln m computed BY AUTODIFF — Kroupa's breakpoint jumps and
Chabrier's 1-Msun kink are discontinuities in exactly the quantity
gradient-based inference differentiates through, while Maschberger's S(m) is
one smooth curve.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from progenax.imf import ChabrierIMF, Maschberger, PowerLawIMF

from .style import polish_axes, setup_style

SEED = 42
_N_SAMPLES = 200_000

# family -> (instance, palette hue, label). Maschberger gets the signature teal.
_FAMILIES = [
    ("Salpeter (1955)", PowerLawIMF.salpeter(), "#355C7D"),
    ("Kroupa (2001)", PowerLawIMF.kroupa(), "#E76F51"),
    ("Chabrier (2003)", ChabrierIMF(), "#E9C46A"),
    ("Maschberger (2013)", Maschberger(), "#2A9D8F"),
]


def _dn_dlnm(imf, m: np.ndarray) -> np.ndarray:
    """m * xi(m) = dN/d ln m for a PDF-normalized IMF, on its own support."""
    lp = np.array(imf.logpdf(jnp.asarray(m)))
    out = m * np.exp(lp)
    out[(m < imf.m_min) | (m > imf.m_max)] = np.nan
    return out


def _local_slope(imf, m: np.ndarray) -> np.ndarray:
    """S(m) = -d ln xi / d ln m via jax.grad (the quantity inference sees)."""

    def neg_dlnxi_dlnm(log_m):
        return -jax.grad(lambda lm: imf.logpdf(jnp.exp(lm)))(log_m)

    s = np.array(jax.vmap(neg_dlnxi_dlnm)(jnp.log(jnp.asarray(m))))
    s[(m < imf.m_min * 1.001) | (m > imf.m_max * 0.999)] = np.nan
    return s


def build_imf_classic_slopes() -> plt.Figure:
    setup_style()
    fig, (ax_pdf, ax_slope) = plt.subplots(
        1, 2, figsize=(7.4, 3.1), constrained_layout=True
    )
    key = jax.random.PRNGKey(SEED)

    m_grid = np.geomspace(0.01, 150.0, 600)

    # --- (a) dN/dln m: analytic curves + sampled histograms -----------------
    for (label, imf, color), k in zip(
        _FAMILIES, jax.random.split(key, len(_FAMILIES))
    ):
        samples = np.asarray(imf.sample(k, _N_SAMPLES))
        bins = np.geomspace(imf.m_min, imf.m_max, 70)
        hist, edges = np.histogram(samples, bins=bins, density=True)
        centers = np.sqrt(edges[1:] * edges[:-1])
        ax_pdf.stairs(
            hist * centers, edges, color=color, alpha=0.35, lw=0.9, baseline=None
        )
        ax_pdf.plot(m_grid, _dn_dlnm(imf, m_grid), color=color, label=label)

    ax_pdf.set_xscale("log")
    ax_pdf.set_yscale("log")
    ax_pdf.set_xlim(0.01, 150)
    ax_pdf.set_ylim(3e-5, 4.0)
    ax_pdf.set_xlabel(r"$m\ \, [\mathrm{M_\odot}]$")
    ax_pdf.set_ylabel(r"$m\,\xi(m) = \mathrm{d}N/\mathrm{d}\ln m$")
    ax_pdf.legend(frameon=False, loc="lower left", handlelength=1.6)
    ax_pdf.text(
        0.03, 0.97, "(a)", transform=ax_pdf.transAxes, va="top", fontweight="bold"
    )
    polish_axes(ax_pdf)

    # --- (b) the autodiff local slope: kinks vs smooth -----------------------
    for label, imf, color in _FAMILIES:
        ax_slope.plot(m_grid, _local_slope(imf, m_grid), color=color, label=label)

    # Annotate the smoothness story: Kroupa's breakpoint jumps, Chabrier's
    # 1-Msun kink, Maschberger's single smooth curve.
    ax_slope.annotate(
        "Kroupa breakpoint\njumps",
        xy=(0.5, 1.82), xytext=(0.32, 2.6), color="#E76F51", fontsize=7.0,
        arrowprops=dict(arrowstyle="-", color="#E76F51", lw=0.7, alpha=0.8,
                        shrinkB=2),
        ha="center",
    )
    ax_slope.annotate(
        "",
        xy=(0.083, 0.9), xytext=(0.19, 2.42),
        arrowprops=dict(arrowstyle="-", color="#E76F51", lw=0.7, alpha=0.8),
    )
    ax_slope.annotate(
        "Chabrier kink\nat $1\,\mathrm{M_\odot}$",
        xy=(1.0, 2.12), xytext=(4.5, 1.35), color="#C79A2E", fontsize=7.0,
        arrowprops=dict(arrowstyle="-", color="#C79A2E", lw=0.7, alpha=0.8),
        ha="center",
    )
    ax_slope.text(
        3.4, 0.55, "Maschberger:\none smooth curve", color="#2A9D8F",
        fontsize=7.0, ha="center",
    )

    ax_slope.set_xscale("log")
    ax_slope.set_xlim(0.01, 150)
    ax_slope.set_ylim(0.0, 2.9)
    ax_slope.set_xlabel(r"$m\ \, [\mathrm{M_\odot}]$")
    ax_slope.set_ylabel(
        r"local slope $S(m) = -\,\mathrm{d}\ln\xi/\mathrm{d}\ln m$ (autodiff)"
    )
    ax_slope.text(
        0.03, 0.97, "(b)", transform=ax_slope.transAxes, va="top", fontweight="bold"
    )
    polish_axes(ax_slope)

    return fig

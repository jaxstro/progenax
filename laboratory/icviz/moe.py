"""ICViz Moe & Di Stefano (2017) P–q–e figure.

F7 ``moe-pqe`` — the non-separable binary statistics: (a) the mass-ratio
distribution's dependence on primary mass AND period (twin excess at short
periods), (b) the (P, e) plane against the Roche eccentricity ceiling
e_max(P) = 1 - (P / 2 d)^(-2/3).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxstro.units import STELLAR

from progenax.binaries import MoeCompanions
from progenax.binaries.kepler_period import compute_period

from .style import polish_axes, setup_style

SEED = 41
_G = STELLAR.G
_DAY = 86400.0 / STELLAR.time_scale_cgs
_N = 400_000


def _sample_moe(key, m1_value: float):
    m1 = jnp.full(_N, m1_value)
    is_b, elems = MoeCompanions().sample(key, m1, G=_G, day_in_time_units=_DAY)
    is_b = np.array(is_b)
    q = np.array(elems.m2 / m1)[is_b]
    P_days = np.array(compute_period(elems.a, m1 + elems.m2, _G))[is_b] / _DAY
    e = np.array(elems.e)[is_b]
    return q, np.log10(P_days), e


def build_moe_pqe() -> plt.Figure:
    import seaborn as sns
    from matplotlib.colors import LogNorm

    setup_style()
    fig, (ax_q, ax_pe) = plt.subplots(
        1, 2, figsize=(7.4, 3.2), constrained_layout=True
    )
    k1, k2 = jax.random.split(jax.random.PRNGKey(SEED))

    # --- (a) g(q): the twin excess is a SHORT-PERIOD, mass-dependent feature
    q_sol, logP_sol, _ = _sample_moe(k1, 1.0)
    q_B, logP_B, e_B = _sample_moe(k2, 10.0)

    bins = np.linspace(0.1, 1.0, 37)
    slices = [
        (q_sol[logP_sol < 3], r"$M_1 = 1\,\mathrm{M_\odot}$, $\log P < 3$", "#E9C46A"),
        (q_B[logP_B < 3], r"$M_1 = 10\,\mathrm{M_\odot}$, $\log P < 3$", "#2A9D8F"),
        (q_B[logP_B > 4], r"$M_1 = 10\,\mathrm{M_\odot}$, $\log P > 4$", "#355C7D"),
    ]
    for q_slice, label, color in slices:
        hist, edges = np.histogram(q_slice, bins=bins, density=True)
        ax_q.stairs(hist, edges, color=color, lw=1.3, label=label)
    ax_q.annotate(
        "twin excess\n($q > 0.95$, short $P$)",
        xy=(0.975, 1.55), xytext=(0.62, 2.0), fontsize=7.0, color="#555555",
        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6), ha="center",
    )
    ax_q.set_xlabel(r"mass ratio $q = m_2/m_1$")
    ax_q.set_ylabel(r"$g(q)$  (density)")
    ax_q.set_xlim(0.1, 1.0)
    ax_q.legend(frameon=False, fontsize=6.6, loc="upper left")
    ax_q.set_title("the $q$ distribution is NOT separable", fontsize=8.4)
    polish_axes(ax_q)

    # --- (b) the (log P, e) plane under the Roche ceiling -------------------
    cmap = sns.color_palette("mako_r", as_cmap=True)
    hb = ax_pe.hexbin(
        logP_B, e_B, gridsize=70, norm=LogNorm(vmin=1), cmap=cmap, mincnt=1,
        linewidths=0.1, extent=(0.0, 8.0, 0.0, 1.0),
    )
    logP_line = np.linspace(np.log10(2.0) + 1e-3, 8.0, 400)
    e_max = 1.0 - (10.0**logP_line / 2.0) ** (-2.0 / 3.0)
    ax_pe.plot(logP_line, e_max, color="#E76F51", lw=1.3,
               label=r"$e_{\max}(P) = 1 - (P/2\,\mathrm{d})^{-2/3}$")
    ax_pe.set_xlabel(r"$\log_{10} (P / \mathrm{d})$")
    ax_pe.set_ylabel(r"eccentricity $e$")
    ax_pe.set_ylim(0, 1.0)
    ax_pe.legend(frameon=False, fontsize=6.6, loc="upper left")
    ax_pe.set_title(r"$(P, e)$ under the Roche ceiling ($M_1 = 10\,\mathrm{M_\odot}$)",
                    fontsize=8.4)
    cb = fig.colorbar(hb, ax=ax_pe, pad=0.02, aspect=30)
    cb.set_label("binaries per hex (log)", fontsize=6.8)
    cb.ax.tick_params(labelsize=6.2)
    polish_axes(ax_pe)
    return fig

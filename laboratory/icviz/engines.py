"""ICViz multi-component engine figures.

F8 ``engine-a-segregation`` — mass segregation as a genuine equilibrium:
    per-component densities of a 3-mass Engine-A model at delta = 1/2 vs the
    delta = 0 single-mass degeneracy (every component collapses onto one
    profile — the cleanest oracle).
F9 ``engine-b-mix`` — the density-defined route: a Plummer halo + EFF core
    mix in ONE shared potential; sampled per-component densities on the
    prescribed curves, per-component dispersions from the shared-Psi
    Eddington DFs, and the theory-oracle Q_j = 1/2.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxstro.units import STELLAR

from progenax import EFFProfile, MultiComponentCluster, PlummerProfile

from .style import polish_axes, setup_style

SEED = 31
_G = STELLAR.G
_N = 300_000

_COMP_COLORS = ["#E9C46A", "#2A9D8F", "#355C7D"]


def _binned_number_density(radii: np.ndarray, bins: np.ndarray, n_total: int):
    counts, edges = np.histogram(radii, bins=bins)
    centers = np.sqrt(edges[1:] * edges[:-1])
    shell = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    dens = counts / shell / n_total
    good = counts >= 10
    err = dens / np.sqrt(np.maximum(counts, 1))
    return centers[good], dens[good], err[good]


def build_engine_a_segregation() -> plt.Figure:
    setup_style()
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(7.4, 3.3), constrained_layout=True, sharey=True
    )
    m_j = [0.3, 1.0, 3.0]
    key = jax.random.PRNGKey(SEED)

    for ax, delta, title in ((ax0, 0.0, r"$\delta = 0$ — no segregation"),
                             (ax1, 0.5, r"$\delta = \frac{1}{2}$ — equipartition ansatz")):
        model = MultiComponentCluster.from_mass_segregation(
            alpha_j=jnp.array([1 / 3, 1 / 3, 1 / 3]),
            m_j=jnp.array(m_j),
            W0=7.0,
            g=1.0,
            delta=delta,
        )
        key, k = jax.random.split(key)
        ic = model.sample_cluster(k, n_stars=_N, G=_G)
        radii = np.array(jnp.linalg.norm(ic.positions, axis=1))
        comp = np.array(ic.component_id)
        bins = np.geomspace(0.05, float(model.r_t), 34)
        for j, (m, color) in enumerate(zip(m_j, _COMP_COLORS)):
            sel = comp == j
            c, d, e = _binned_number_density(radii[sel], bins, int(sel.sum()))
            ax.errorbar(c, d, yerr=e, fmt="o", ms=2.4, lw=0.7, color=color,
                        alpha=0.7, label=rf"$m_j = {m:g}\,\mathrm{{M_\odot}}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=8.4)
        ax.set_xlabel(r"$r / r_c$")
        polish_axes(ax)

    ax0.set_ylabel(r"component number density (per-comp. norm)")
    ax0.legend(frameon=False, fontsize=6.8, loc="lower left")
    ax1.annotate(
        "heavier components sink:\na DEEPER effective well,\nnot a reshuffle",
        xy=(0.5, 3e-4), xytext=(2.6, 6e-3), fontsize=7.0, color="#555555",
        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6),
    )
    return fig


def build_engine_b_mix() -> plt.Figure:
    setup_style()
    fig, (ax_rho, ax_sig) = plt.subplots(
        1, 2, figsize=(7.4, 3.3), constrained_layout=True
    )
    profiles = [PlummerProfile(r_h=2.0), EFFProfile(a=0.8, gamma=5.0, r_t=9.0)]
    fractions = jnp.array([0.6, 0.4])
    labels = ["Plummer halo (60%)", r"EFF $\gamma{=}5$ core (40%)"]
    colors = ["#355C7D", "#2A9D8F"]

    model = MultiComponentCluster.from_density_profiles(
        profiles, fractions, m_j=jnp.array([1.0, 1.0])
    )
    ic = model.sample_cluster(jax.random.PRNGKey(SEED + 1), n_stars=_N, G=_G)
    pos = np.array(ic.positions)
    vel = np.array(ic.velocities)
    radii = np.linalg.norm(pos, axis=1)
    v_r = np.einsum("ij,ij->i", vel, pos / radii[:, None])
    comp = np.array(ic.component_id)
    r_t = float(model.r_t)

    # (a) prescribed curves vs sampled per-component densities --------------
    r_line = np.geomspace(0.03, r_t, 400)
    for j, (prof, frac, label, color) in enumerate(
        zip(profiles, np.array(fractions), labels, colors)
    ):
        rho = np.array(prof.density(jnp.asarray(r_line)))
        # normalize the prescribed shape to a unit-mass PDF on (0, r_t]
        r_n = np.geomspace(1e-4, r_t, 4000)
        rho_n = np.array(prof.density(jnp.asarray(r_n)))
        m_shape = np.trapezoid(4 * np.pi * rho_n * r_n**2, r_n)
        ax_rho.plot(r_line, rho / m_shape, color=color, label=f"{label} — prescribed")
        sel = comp == j
        bins = np.geomspace(0.05, r_t, 32)
        c, d, e = _binned_number_density(radii[sel], bins, int(sel.sum()))
        ax_rho.errorbar(c, d, yerr=e, fmt="o", ms=2.4, lw=0.7, color=color, alpha=0.6)
    ax_rho.set_xscale("log")
    ax_rho.set_yscale("log")
    ax_rho.set_xlabel(r"$r$  [pc]")
    ax_rho.set_ylabel(r"$\rho_j(r)$  (unit-mass norm)")
    ax_rho.legend(frameon=False, fontsize=6.6, loc="lower left")
    ax_rho.set_title("prescribed densities, one shared potential", fontsize=8.4)
    polish_axes(ax_rho)

    # (b) per-component sigma_r(r) from the shared-Psi Eddington DFs ---------
    bins = np.geomspace(0.08, r_t * 0.9, 22)
    centers = np.sqrt(bins[1:] * bins[:-1])
    for j, (label, color) in enumerate(zip(labels, colors)):
        sel = comp == j
        idx = np.digitize(radii[sel], bins) - 1
        vr_j = v_r[sel]
        sig = np.full(len(centers), np.nan)
        err = np.full(len(centers), np.nan)
        for i in range(len(centers)):
            s = idx == i
            n = int(s.sum())
            if n >= 80:
                sig[i] = np.std(vr_j[s])
                err[i] = sig[i] / np.sqrt(2 * (n - 1))
        good = ~np.isnan(sig)
        ax_sig.errorbar(centers[good], sig[good], yerr=err[good], fmt="o", ms=2.6,
                        lw=0.7, color=color, alpha=0.75, label=label)

    q_j = np.array(model.component_virial_ratios())
    ax_sig.text(
        0.97, 0.95,
        rf"theory oracle: $Q_1 = {q_j.ravel()[0]:.3f}$, $Q_2 = {q_j.ravel()[1]:.3f}$",
        transform=ax_sig.transAxes, ha="right", va="top", fontsize=7.2,
    )
    ax_sig.set_xscale("log")
    ax_sig.set_xlabel(r"$r$  [pc]")
    ax_sig.set_ylabel(r"$\sigma_r(r)$  [pc/Myr]")
    ax_sig.legend(frameon=False, fontsize=6.6, loc="lower left")
    ax_sig.set_title("per-component dispersions in the shared well", fontsize=8.4)
    polish_axes(ax_sig)
    return fig

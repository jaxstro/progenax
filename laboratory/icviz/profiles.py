"""ICViz spatial-profile figures.

F1 ``profile-family-portrait`` — every released density family on one
half-mass-normalized log-log panel (model selection at a glance).

F2 ``profile-density-residuals`` — the correctness proof: sampled radial
densities against the analytic curves with a residual subpanel per family
(Plummer closed form, King ODE density, EFF closed form), N = 2x10^5 each,
Poisson error bands.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from progenax import EFFProfile, KingProfile, MichieProfile, PlummerProfile

from .style import polish_axes, setup_style

SEED = 7
_N_SAMPLES = 200_000


def _half_mass_radius(profile, r_max: float) -> float:
    """r_h from the profile's own density by cumulative quadrature."""
    r = np.geomspace(1e-4, r_max, 4000)
    rho = np.array(profile.density(jnp.asarray(r)))
    integ = rho * r**3  # d(ln r) quadrature: 4 pi rho r^2 * r
    m = np.concatenate([[0.0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(np.log(r)))])
    m /= m[-1]
    return float(np.interp(0.5, m, r))


def build_family_portrait() -> plt.Figure:
    setup_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)

    king_w0 = [(3.0, "#E9C46A"), (7.0, "#F4A261"), (12.0, "#E76F51")]
    members: list[tuple[str, object, float, str, str]] = [
        ("Plummer", PlummerProfile(r_h=1.0), 30.0, "#355C7D", "-"),
    ]
    for w0, color in king_w0:
        prof = KingProfile.from_W0_rc(W0=w0, r_c=1.0)
        members.append((f"King $W_0={w0:g}$", prof, float(prof.r_t), color, "-"))
    eff = EFFProfile(a=1.0, gamma=3.5, r_t=60.0)
    members.append((r"EFF $\gamma=3.5$", eff, 60.0, "#2A9D8F", "-"))
    michie = MichieProfile.from_W0_rc(W0=7.0, r_c=1.0, r_a=8.0)
    members.append(
        (r"Michie $W_0=7,\ r_a=8r_c$", michie, float(michie.r_t), "#6C5B7B", (0, (4, 2)))
    )

    for label, prof, r_max, color, ls in members:
        r_h = _half_mass_radius(prof, r_max)
        r = np.geomspace(1e-3 * r_h, r_max, 700)
        rho = np.array(prof.density(jnp.asarray(r)))
        rho0 = np.array(prof.density(jnp.asarray(np.full(1, 1e-3 * r_h))))[0]
        ax.plot(r / r_h, rho / rho0, color=color, ls=ls, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 60)
    ax.set_ylim(1e-9, 2.0)
    ax.set_xlabel(r"$r / r_h$")
    ax.set_ylabel(r"$\rho(r)\, /\, \rho_0$")
    ax.legend(frameon=False, loc="lower left", handlelength=1.7)
    polish_axes(ax)
    return fig


_F2_FAMILIES = [
    ("Plummer", PlummerProfile(r_h=1.0), 25.0, "#355C7D"),
    ("King $W_0=7$", KingProfile.from_W0_rc(W0=7.0, r_c=1.0), None, "#E76F51"),
    (r"EFF $\gamma=4$", EFFProfile(a=1.0, gamma=4.0, r_t=30.0), 30.0, "#2A9D8F"),
]


def build_density_residuals() -> plt.Figure:
    setup_style()
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.6, 3.6),
        constrained_layout=True,
        sharex="col",
        height_ratios=[2.6, 1.0],
    )
    key = jax.random.PRNGKey(SEED)

    for (label, prof, r_max, color), k, (ax, ax_res) in zip(
        _F2_FAMILIES, jax.random.split(key, 3), zip(axes[0], axes[1])
    ):
        r_edge = float(getattr(prof, "r_t", r_max) or r_max)
        masses = jnp.ones(_N_SAMPLES)
        pos = prof.sample_positions(masses, k)
        radii = np.array(jnp.linalg.norm(pos, axis=1))

        bins = np.geomspace(max(1e-2, radii.min()), r_edge, 40)
        counts, edges = np.histogram(radii, bins=bins)
        centers = np.sqrt(edges[1:] * edges[:-1])
        shell = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
        rho_hat = counts / shell / _N_SAMPLES  # number density / N (unit total mass)

        rho_true = np.array(prof.density(jnp.asarray(centers)))
        # profile.density is normalized to total mass; sampled rho_hat integrates
        # to 1 by construction -> normalize the analytic curve the same way.
        norm = np.array(prof.density(jnp.asarray(bins)))
        mass_int = np.trapezoid(4 * np.pi * bins**2 * norm, bins)
        rho_true = rho_true / mass_int

        # Drop near-empty bins (counts < 10): their +/- 1/sqrt(N) spikes span
        # the panel and carry no information.
        good = counts >= 10
        rel_err = 1.0 / np.sqrt(np.maximum(counts, 1))

        ax.plot(centers, rho_true, color=color, lw=1.4, label="analytic")
        ax.errorbar(
            centers[good],
            rho_hat[good],
            yerr=rho_hat[good] * rel_err[good],
            fmt="o",
            ms=2.4,
            lw=0.7,
            color=color,
            alpha=0.55,
            label=rf"sampled ($N=2{{\times}}10^5$)",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(label, fontsize=8.6)
        ax.set_ylabel(r"$\rho(r)\,/\,M_{\rm tot}$" if label.startswith("Plummer") else "")
        polish_axes(ax)

        resid = rho_hat[good] / rho_true[good] - 1.0
        ax_res.axhline(0.0, color="#9A9A9A", lw=0.6)
        ax_res.fill_between(
            centers[good], -rel_err[good], rel_err[good], color=color, alpha=0.15, lw=0
        )
        ax_res.plot(centers[good], resid, "o", ms=2.2, color=color, alpha=0.75)
        ax_res.set_xscale("log")
        ax_res.set_ylim(-0.12, 0.12)
        ax_res.set_xlabel(r"$r$  [pc]")
        ax_res.set_ylabel("resid." if label.startswith("Plummer") else "")
        polish_axes(ax_res, grid_axis="y")

    axes[0][0].legend(frameon=False, loc="lower left", fontsize=6.6)
    return fig

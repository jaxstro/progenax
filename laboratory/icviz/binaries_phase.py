"""ICViz binary-population phase-space figure.

F12 ``binary-phase-space`` — (r, v_r) phase space of populations WITH
primordial binaries, resolved vs unresolved, young vs old:

* rows: an EFF young massive cluster (Moe & Di Stefano companions — high
  mass-dependent multiplicity, short-period massive binaries) and a King
  old globular (low constant f_b, DM91 periods, thermal eccentricities).
* columns: RESOLVED (every component star — binary orbital motion punctures
  the single-star escape envelope) vs UNRESOLVED (each binary photometrically
  blended: COM position, FLUX-weighted velocity via the Tout ZAMS
  luminosities — the punctures vanish but the measured dispersion stays
  silently inflated).

Each panel annotates the measured 1-D dispersion sigma(v_z); the
singles-only value is the truth reference in the caption.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxstro.units import STELLAR

from progenax import (
    EFFProfile,
    EFFVelocityDF,
    KingProfile,
    KingVelocityDF,
    Maschberger,
    Systems,
    build_binary_cluster,
)
from progenax.binaries import (
    IndependentCompanions,
    LogNormalPeriod,
    MoeCompanions,
    ThermalEccentricity,
)
from progenax.imf.binary import ConstantBinaryFraction, FlatMassRatio
from progenax.stellar import zams_luminosity

from .dfs import _psi_from_density
from .style import polish_axes, setup_style

SEED = 21
_G = STELLAR.G
_N_SYS = 40_000


def _signed_vr(pos: np.ndarray, vel: np.ndarray):
    r = np.linalg.norm(pos, axis=1)
    v_r = np.einsum("ij,ij->i", vel, pos / r[:, None])
    return r, v_r


def _blend_unresolved(ic) -> tuple[np.ndarray, np.ndarray]:
    """Photometric blends: per primordial system, COM position and
    FLUX-weighted velocity (L = Tout ZAMS L(m)). Singles pass through."""
    pos = np.array(ic.positions)
    vel = np.array(ic.velocities)
    m = np.array(ic.masses)
    sys_id = np.array(ic.primordial_system_id)
    lum = np.array(zams_luminosity(jnp.asarray(m)))

    order = np.argsort(sys_id, kind="stable")
    pos, vel, m, lum, sys_id = (
        pos[order], vel[order], m[order], lum[order], sys_id[order]
    )
    uniq, start, counts = np.unique(sys_id, return_index=True, return_counts=True)
    out_pos = np.empty((len(uniq), 3))
    out_vel = np.empty((len(uniq), 3))
    # vectorized pair/single reduction via segment sums
    w_m = m
    w_l = lum
    sum_m = np.add.reduceat(w_m, start)
    out_pos = np.add.reduceat(pos * w_m[:, None], start) / sum_m[:, None]  # COM
    sum_l = np.add.reduceat(w_l, start)
    out_vel = np.add.reduceat(vel * w_l[:, None], start) / sum_l[:, None]  # flux-wtd
    return out_pos, out_vel


def build_binary_phase_space() -> plt.Figure:
    import seaborn as sns
    from matplotlib.colors import LogNorm

    setup_style()
    fig, axes = plt.subplots(
        2, 2, figsize=(7.6, 5.4), constrained_layout=True, sharey="row"
    )
    cmap = sns.color_palette("mako_r", as_cmap=True)
    hex_norm = LogNorm(vmin=1, vmax=1e3)
    key = jax.random.PRNGKey(SEED)
    k_eff, k_king = jax.random.split(key)

    populations = [
        (
            "EFF young massive cluster + Moe binaries",
            EFFProfile(a=1.0, gamma=4.0, r_t=15.0),
            EFFVelocityDF(a=1.0, gamma=4.0, r_t=15.0),
            MoeCompanions(),
            k_eff,
            15.0,
        ),
        (
            r"King $W_0=7$ old globular + $f_b = 0.15$ binaries",
            KingProfile.from_W0_rc(W0=7.0, r_c=1.0),
            KingVelocityDF(W0=7.0, r_c=1.0),
            IndependentCompanions(
                binary_fraction=ConstantBinaryFraction(0.15),
                q_distribution=FlatMassRatio(q_min=0.1),
                period_distribution=LogNormalPeriod(),
                eccentricity_distribution=ThermalEccentricity(),
            ),
            k_king,
            None,
        ),
    ]

    for row, (title, prof, df, companions, k, r_max_in) in enumerate(populations):
        ic = build_binary_cluster(
            prof, df, Maschberger(), companions, Systems(_N_SYS), k, units=STELLAR
        )
        M_tot = float(jnp.sum(ic.masses))
        r_max = float(getattr(prof, "r_t", r_max_in) or r_max_in)

        # single-star escape envelope in THIS population's realized mass
        r_grid = np.linspace(1e-4, r_max, 6000)
        psi = _psi_from_density(prof, r_grid, M_tot)
        env = np.sqrt(2.0 * np.clip(psi, 0.0, None))

        is_pair = np.bincount(np.array(ic.primordial_system_id)) > 1
        in_binary = is_pair[np.array(ic.primordial_system_id)]

        # --- resolved -----------------------------------------------------
        ax = axes[row][0]
        pos, vel = np.array(ic.positions), np.array(ic.velocities)
        r, v_r = _signed_vr(pos, vel)
        # scatter UNDER the hexbin: the punctures live outside the hex extent,
        # so they stay visible while the cluster lens stays readable on top.
        ax.plot(
            r[in_binary], v_r[in_binary], ".", ms=1.1, color="#E76F51", alpha=0.22,
            rasterized=True, zorder=1,
        )
        ax.hexbin(
            r[~in_binary], v_r[~in_binary], gridsize=95, norm=hex_norm, cmap=cmap,
            mincnt=1, linewidths=0.1, extent=(0.0, r_max, -3.2, 3.2), zorder=2,
        )
        ax.plot(r_grid, env, color="#3A3A3A", lw=1.0)
        ax.plot(r_grid, -env, color="#3A3A3A", lw=1.0)
        sigma = float(np.std(vel[:, 2]))
        sigma_single = float(np.std(vel[~in_binary, 2]))
        ax.text(
            0.97, 0.06,
            rf"$\sigma_{{1\mathrm{{d}}}} = {sigma:.2f}$"
            "\n"
            rf"(singles: ${sigma_single:.2f}$)",
            transform=ax.transAxes, ha="right", fontsize=7.0,
        )
        ax.set_title(f"resolved — {title}", fontsize=7.9)
        ax.set_ylabel(r"$v_r$  [pc/Myr]")
        if row == 1:
            ax.set_xlabel(r"$r$  [pc]")
        polish_axes(ax)

        # --- unresolved ----------------------------------------------------
        ax = axes[row][1]
        b_pos, b_vel = _blend_unresolved(ic)
        r_b, v_rb = _signed_vr(b_pos, b_vel)
        ax.hexbin(
            r_b, v_rb, gridsize=95, norm=hex_norm, cmap=cmap, mincnt=1,
            linewidths=0.1, extent=(0.0, r_max, -3.2, 3.2),
        )
        ax.plot(r_grid, env, color="#3A3A3A", lw=1.0)
        ax.plot(r_grid, -env, color="#3A3A3A", lw=1.0)
        sigma_b = float(np.std(b_vel[:, 2]))
        ax.text(
            0.97, 0.06, rf"$\sigma_{{1\mathrm{{d}}}} = {sigma_b:.2f}$",
            transform=ax.transAxes, ha="right", fontsize=7.0,
        )
        ax.set_title("unresolved (flux-weighted blends)", fontsize=7.9)
        if row == 1:
            ax.set_xlabel(r"$r$  [pc]")
        polish_axes(ax)

        for a in axes[row]:
            a.set_yscale("symlog", linthresh=3.0)
            a.set_ylim(-1500, 1500)
            for y in (3.0, -3.0):
                a.axhline(y, color="#CFCFCF", lw=0.4, ls=(0, (1, 2)), zorder=0)

    # shared legend proxies on the first panel
    axes[0][0].plot([], [], ".", ms=4, color="#E76F51", label="binary components")
    axes[0][0].plot([], [], color="#3A3A3A", lw=1.0, label=r"single-star $\pm v_{\rm esc}$")
    axes[0][0].legend(frameon=False, fontsize=6.4, loc="upper right")
    return fig

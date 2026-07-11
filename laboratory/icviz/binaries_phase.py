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
        2, 2, figsize=(7.8, 5.2), constrained_layout=True,
        sharey=True, sharex="row",
    )
    cmap = sns.color_palette("mako_r", as_cmap=True)
    hex_norm = LogNorm(vmin=1, vmax=1e3)
    env_color = "#E76F51"   # same coral as the single-population phase plot
    binary_color = "#8E5A7F"  # plum (palette NEGATIVE) — distinct from teal/coral
    key = jax.random.PRNGKey(SEED)
    k_eff, k_king = jax.random.split(key)

    populations = [
        (
            "EFF young massive cluster (Moe binaries)",
            EFFProfile(a=1.0, gamma=4.0, r_t=15.0),
            EFFVelocityDF(a=1.0, gamma=4.0, r_t=15.0),
            MoeCompanions(),
            k_eff,
            15.0,
        ),
        (
            r"King $W_0{=}7$ old globular ($f_b = 0.15$)",
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

    hb = None
    for row, (title, prof, df, companions, k, r_max_in) in enumerate(populations):
        ic = build_binary_cluster(
            prof, df, Maschberger(), companions, Systems(_N_SYS), k, units=STELLAR
        )
        M_tot = float(jnp.sum(ic.masses))
        r_t = float(getattr(prof, "r_t", r_max_in) or r_max_in)
        r_max = r_t * 1.06

        # single-star escape envelope in THIS population's realized mass
        r_grid = np.linspace(1e-4, r_t, 6000)
        psi = _psi_from_density(prof, r_grid, M_tot)
        env = np.sqrt(2.0 * np.clip(psi, 0.0, None))

        is_pair = np.bincount(np.array(ic.primordial_system_id)) > 1
        in_binary = is_pair[np.array(ic.primordial_system_id)]

        pos, vel = np.array(ic.positions), np.array(ic.velocities)
        r, v_r = _signed_vr(pos, vel)
        b_pos, b_vel = _blend_unresolved(ic)
        r_b, v_rb = _signed_vr(b_pos, b_vel)

        sigma = float(np.std(vel[:, 2]))
        sigma_single = float(np.std(vel[~in_binary, 2]))
        sigma_b = float(np.std(b_vel[:, 2]))

        # Envelope violators: |v_r| above the local single-star v_esc — the
        # stars that CANNOT be bound singles (binary-orbital contaminants).
        # Everything else (singles AND envelope-respecting binary components)
        # belongs to the population hexbin; the linear zone spans the full
        # envelope so no bound star is silently clipped.
        env_at = np.interp(r, r_grid, env)
        violator = np.abs(v_r) > env_at * 1.02
        n_viol = int(violator.sum())

        panels = [
            (axes[row][0], f"resolved — {title}",
             dict(hex_xy=(r[~violator], v_r[~violator]),
                  scatter=(r[violator], v_r[violator]),
                  sigma_text=(rf"$\sigma_{{1\mathrm{{d}}}} = {sigma:.2f}$"
                              "\n" + rf"singles: ${sigma_single:.2f}$"
                              "\n" + rf"{n_viol:,} violators"))),
            (axes[row][1], "unresolved — flux-weighted blends",
             dict(hex_xy=(r_b, v_rb), scatter=None,
                  sigma_text=rf"$\sigma_{{1\mathrm{{d}}}} = {sigma_b:.2f}$")),
        ]
        for ax, ptitle, d in panels:
            if d["scatter"] is not None:
                ax.plot(*d["scatter"], ".", ms=1.4, color=binary_color,
                        alpha=0.35, rasterized=True, zorder=1)
            hb = ax.hexbin(
                *d["hex_xy"], gridsize=(90, 55), norm=hex_norm, cmap=cmap,
                mincnt=1, linewidths=0.1, extent=(0.0, r_max, -11.0, 11.0),
                zorder=2,
            )
            ax.plot(r_grid, env, color=env_color, lw=1.2, zorder=3)
            ax.plot(r_grid, -env, color=env_color, lw=1.2, zorder=3)
            ax.axvline(r_t, color="#9A9A9A", lw=0.7, ls=(0, (4, 2)), alpha=0.8)
            ax.annotate(r"$r_t$", xy=(r_t, 400), xytext=(r_t * 0.86, 430),
                        fontsize=7.4, color="#7A7A7A")
            ax.text(0.975, 0.04, d["sigma_text"], transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7.0)
            ax.set_title(ptitle, fontsize=8.0)
            ax.set_yscale("symlog", linthresh=11.0, linscale=2.2)
            ax.set_ylim(-1100, 1100)
            ax.set_xlim(-r_max * 0.02, r_max)
            for y in (11.0, -11.0):
                ax.axhline(y, color="#D8D8D8", lw=0.4, ls=(0, (1, 2)), zorder=0)
            polish_axes(ax)
        axes[row][0].set_ylabel(r"$v_r$  [pc/Myr]")

    for ax in axes[1]:
        ax.set_xlabel(r"$r$  [pc]")

    # legend proxies (first panel), colorbar (shared, like the single-pop figure)
    axes[0][0].plot([], [], ".", ms=4, color=binary_color,
                    label="envelope violators (binary orbits)")
    axes[0][0].plot([], [], color=env_color, lw=1.2,
                    label=r"single-star $\pm v_{\rm esc}$")
    axes[0][0].legend(frameon=False, fontsize=6.4, loc="lower left")
    cb = fig.colorbar(hb, ax=axes, pad=0.012, aspect=34)
    cb.set_label("stars per hex (log)", fontsize=7.2)
    cb.ax.tick_params(labelsize=6.4)
    return fig

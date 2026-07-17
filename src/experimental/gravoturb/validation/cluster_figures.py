"""Figure gallery for the gravoturbulent cluster-IC acceptance suite.

Split out of ``cluster_acceptance`` (which stays the numbers/PASS-FAIL side);
``cluster_acceptance.main()`` imports and calls these after the ac_* checks run.
numpy/matplotlib are permitted here (validation/analysis side).
"""

import os

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gravoturb.realization.envelope import apply_spherical_envelope
from gravoturb.realization.pipeline import build_turbulent_field
from gravoturb.realization.placement import sample_positions
from gravoturb.validation.cluster_acceptance import (
    ALPHA,
    BOX,
    MACH,
    PLOTS,
    SHAPE,
    B,
    _ic,
)
from progenax import PlummerProfile


def _fig_scatter(seed=0):
    ic = _ic(n=4000, seed=seed)
    pos = np.asarray(ic.stars.positions)
    r = np.linalg.norm(pos, axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (a, bx, lbl) in zip(
        axes, [(0, 1, "x–y"), (0, 2, "x–z"), (1, 2, "y–z")]
    ):
        ax.scatter(pos[:, a], pos[:, bx], s=3, c=r, cmap="viridis", alpha=0.5)
        ax.set_xlim(-BOX / 2, BOX / 2); ax.set_ylim(-BOX / 2, BOX / 2)
        ax.set_aspect("equal"); ax.set_title(f"{lbl}  (colour = radius)")
        ax.set_xlabel("pc")
    fig.suptitle("Gravoturbulent cluster IC — spherical envelope + turbulent substructure "
                 f"(ℳ={MACH}, β=3.0, r_h=0.5 pc, N=4000)")
    fig.savefig(os.path.join(PLOTS, "cluster_scatter.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


_RP_EDGES = np.linspace(0, 2.0, 24)
_RP_CEN = 0.5 * (_RP_EDGES[:-1] + _RP_EDGES[1:])
_RP_SHELL = 4 / 3 * np.pi * (_RP_EDGES[1:] ** 3 - _RP_EDGES[:-1] ** 3)


def _radial_profile_sampled(r_h, mach, s_turb_zero, seeds, n=6000):
    """Mean sampled number-density profile ρ(r). ``s_turb_zero`` → pure-envelope control."""
    prof = PlummerProfile(r_h=r_h)
    stack = []
    for sd in seeds:
        fld = build_turbulent_field(mach, B, ALPHA, 3.0, SHAPE, jax.random.PRNGKey(sd))
        s_turb = jnp.zeros(SHAPE) if s_turb_zero else fld.s
        s_tot = apply_spherical_envelope(s_turb, prof, BOX)
        pos = np.asarray(sample_positions(s_turb, fld.s_t, 8.0, 0.3, n,
                                          jax.random.PRNGKey(sd + 50), box_size=BOX,
                                          s_density=s_tot)) - BOX / 2
        cnt, _ = np.histogram(np.linalg.norm(pos, axis=1), bins=_RP_EDGES)
        stack.append(cnt / _RP_SHELL)
    return np.mean(stack, axis=0)


def _fig_radial_profile(seeds=(0, 1, 2)):
    # normalise at r = r_h (a RESOLVED radius, ~4 cells) — NOT the sub-cell innermost bin
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    # panel A: fiducial (ℳ=8) sampled vs analytic, 3 envelopes
    ax = axes[0]
    for r_h, col in zip([0.3, 0.5, 0.8], ["C0", "C1", "C2"]):
        iref = int(np.argmin(np.abs(_RP_CEN - r_h)))
        dens = _radial_profile_sampled(r_h, MACH, False, seeds)
        ax.plot(_RP_CEN, dens / dens[iref], "o-", color=col, label=f"sampled, r_h={r_h}")
        pl = np.asarray(PlummerProfile(r_h=r_h).density(jnp.asarray(_RP_CEN)))
        ax.plot(_RP_CEN, pl / pl[iref], "--", color=col, alpha=0.7)
    ax.set_yscale("log"); ax.set_xlabel("r (pc)"); ax.set_ylabel("ρ(r)/ρ(r_h)")
    ax.set_title("Fiducial (ℳ=8) sampled (points) vs analytic Plummer (dashed)\n"
                 "normalised at r=r_h; turbulent BM19 tail broadens the wings")
    ax.legend()
    # panel B: control — turbulence on/off at r_h=0.5 proves envelope fidelity
    ax = axes[1]
    iref = int(np.argmin(np.abs(_RP_CEN - 0.5)))
    pl = np.asarray(PlummerProfile(r_h=0.5).density(jnp.asarray(_RP_CEN)))
    pe = _radial_profile_sampled(0.5, MACH, True, seeds)    # pure envelope (turbulence off)
    m8 = _radial_profile_sampled(0.5, MACH, False, seeds)   # fiducial (turbulence on)
    ax.plot(_RP_CEN, pl / pl[iref], "k--", lw=2, label="analytic Plummer")
    ax.plot(_RP_CEN, pe / pe[iref], "s-", color="C2", label="sampled, turbulence OFF")
    ax.plot(_RP_CEN, m8 / m8[iref], "o-", color="C3", label="sampled, ℳ=8 (turbulence ON)")
    ax.set_yscale("log"); ax.set_xlabel("r (pc)"); ax.set_ylabel("ρ(r)/ρ(r_h)")
    ax.set_title("Control (r_h=0.5): envelope sampling = analytic (few %);\n"
                 "turbulence is the large-r EXCESS; central cusp grid-under-resolved")
    ax.legend()
    fig.savefig(os.path.join(PLOTS, "cluster_radial_profile.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def _fig_beta_recovery(rec):
    rows = np.array(rec["rows"])  # (β, slope_s, σ_s, slope_dens, σ_dens, err)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    lo, hi = rows[:, 0].min() - 0.2, rows[:, 0].max() + 0.2
    ax.plot([lo, hi], [lo, hi], "k:", label="1:1 (perfect recovery)")
    ax.errorbar(rows[:, 0], rows[:, 1], yerr=rows[:, 2], fmt="o-", color="C0", capsize=3,
                label="log-density slope (≈ input β)")
    ax.errorbar(rows[:, 0], rows[:, 3], yerr=rows[:, 4], fmt="s-", color="C1", capsize=3,
                label="density e^s slope (compressed)")
    ax.set_xlabel("input β"); ax.set_ylabel("measured P(k) slope")
    ax.set_title(f"β recovery: log-density slope tracks input β to "
                 f"max |err|={rec['max_err']:.3f}\n(density slope is compressed by the BM19 tail)")
    ax.legend()
    fig.savefig(os.path.join(PLOTS, "cluster_beta_recovery.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def _fig_substructure_plane(sub):
    beta_rows = np.array(sub["beta_rows"]); conc_rows = np.array(sub["conc_rows"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # (m̄, s̄) plane with the two trajectories
    ax = axes[0]
    ax.plot(beta_rows[:, 2], beta_rows[:, 3], "o-", color="C3", label="β-sweep (r_h=0.5)")
    for b, _, m, s in beta_rows:
        ax.annotate(f"{b:.1f}", (m, s), fontsize=7, color="C3")
    ax.plot(conc_rows[:, 2], conc_rows[:, 3], "s-", color="C0", label="concentration (β=3.0)")
    for rh, _, m, s in conc_rows:
        ax.annotate(f"{rh:.1f}", (m, s), fontsize=7, color="C0")
    ax.set_xlabel("m̄  (normalised MST edge — concentration axis)")
    ax.set_ylabel("s̄  (normalised mean separation)")
    ax.set_title("CW04 (m̄, s̄) plane\nβ and concentration trace independent directions")
    ax.legend()
    # Q vs β and Q vs r_h (the conflation Q alone can't resolve)
    axes[1].plot(beta_rows[:, 0], beta_rows[:, 1], "o-", color="C3")
    axes[1].set_xlabel("β"); axes[1].set_ylabel("CW04 Q")
    axes[1].set_title("Q ↓ with β (substructure) — at fixed envelope")
    axes[2].plot(conc_rows[:, 0], conc_rows[:, 1], "s-", color="C0")
    axes[2].set_xlabel("envelope r_h (pc)"); axes[2].set_ylabel("CW04 Q")
    axes[2].set_title("Q ↓ with r_h (less concentrated) — at fixed β\n→ Q alone conflates the two")
    fig.savefig(os.path.join(PLOTS, "cluster_substructure_plane.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def _fig_velocity(vc, seed=0):
    ic = _ic(n=1200, seed=seed)
    pos = np.asarray(ic.stars.positions); vel = np.asarray(ic.stars.velocities)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sp = np.linalg.norm(vel, axis=1)
    axes[0].quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1], sp,
                   cmap="coolwarm", scale_units="xy", angles="xy", width=0.003)
    axes[0].set_aspect("equal"); axes[0].set_xlabel("x (pc)"); axes[0].set_ylabel("y (pc)")
    axes[0].set_title("Stellar velocity field (x–y)\ncoherent — nearby stars move together")
    # alignment vs separation
    sep, cos = vc["sep"], vc["cos"]
    edges = np.linspace(0, sep.max(), 18); cen = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(sep, edges) - 1, 0, len(cen) - 1)
    prof = np.array([cos[idx == i].mean() if np.any(idx == i) else np.nan for i in range(len(cen))])
    axes[1].plot(cen, prof, "o-")
    axes[1].axhline(0, color="k", lw=0.8, ls=":")
    axes[1].set_xlabel("pair separation (pc)"); axes[1].set_ylabel("mean velocity alignment cosθ")
    axes[1].set_title(f"Velocity coherence decays with separation\n"
                      f"near {vc['align_near']:+.2f} → far {vc['align_far']:+.2f}")
    fig.savefig(os.path.join(PLOTS, "cluster_velocity_coherence.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

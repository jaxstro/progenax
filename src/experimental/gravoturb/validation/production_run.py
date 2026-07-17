"""CAREER production run (Phase 5, Anna-directed 2026-07-16): ONE seed + figures.

Configuration (ratified + pilot-informed):
- fiducial production IC: N=2000 × 1 M⊙, Helmholtz-coupled (β = β_v − 2, χ_F10),
  multi-freefall placement at 64³ (the ≥64³-at-ℳ≥8 caveat), field-first physical
  velocities, residual gas carried (SFE=0.2) for the IC figure panel;
- dynamics (Anna-directed): TIME-SYMMETRIC Hermite (Kokubo P(EC)³, order 4 — the
  order-6 snap path is point-mass-only in gravax), fixed dt = 2.5e-4 Myr, with
  ε = Δx/4 = 0.0156 pc: sub-cell pair structure is PLACEMENT NOISE below the grid
  resolution (pilot: min pair sep 4.6e-4 pc ≪ Δx). Pilot ladder (measured):
  unsoftened adaptive → dt~1e-9 Myr (impossible); unsoftened fixed-dt symmetric →
  energy EXPLOSION (unresolved 1/r pericenters); order-4 symmetric + ε=Δx/4 →
  |ΔE/E| = 1.5e-4 per 0.1 Myr at dt=5e-4, dt⁴ scaling ⇒ ~1e-5 at the production dt;
- span 10 t_cross (t_cross = 0.25 Myr measured on the realized cluster), 50 snapshots;
- NO IAS15 cross-check (Anna's call 2026-07-16).

Outputs (validation/plots/production/): snapshots.npz + three production figures
(PNG 300 dpi + PDF): (1) the stars+gas natal state, (2) the cold-collapse evolution,
(3) diagnostic time series incl. the CW04 substructure-memory decay.
"""

import os
import time

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jaxstro.units import STELLAR

from gravoturb.specs import (
    CloudSpec,
    CompositionSpec,
    GasSpec,
    GeometrySpec,
    VelocitySpec,
)
from gravoturb.cluster import build_cluster_ic
from progenax import PlummerProfile
from progenax.diagnostics import compute_q_parameter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "plots", "production")
os.makedirs(OUT, exist_ok=True)

N, BOX, NGRID = 2000, 4.0, 64
DX = BOX / NGRID
EPS = DX / 4.0
DT = 2.5e-4  # Myr (order-4 symmetric; pilot ladder in the module docstring)
SEED = 0
N_SNAP = 50
G = STELLAR.G

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
    "axes.titlesize": 12, "legend.fontsize": 9, "figure.dpi": 120,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def build_ic():
    return build_cluster_ic(
        jnp.ones(N),
        cloud=CloudSpec(mach=8.0, b=0.5, alpha=1.8, beta=None, coupling="helmholtz"),
        geometry=GeometrySpec(profile=PlummerProfile(r_h=0.5), box_size=BOX,
                              shape=(NGRID,) * 3),
        velocity=VelocitySpec(beta_v=4.0, mode="physical", c_s=0.2),
        composition=CompositionSpec(),  # multi_freefall default
        G=G, units=STELLAR, key=jax.random.PRNGKey(SEED),
        gas=GasSpec(sfe=0.2),
    )


def run(ic):
    from gravax import ParticleSystem, SymmetricHermiteIntegrator

    r_h = float(np.median(np.linalg.norm(np.asarray(ic.stars.positions), axis=1)))
    m, v = ic.stars.masses, ic.stars.velocities
    sig = float(jnp.sqrt(jnp.sum(m * jnp.sum(v**2, axis=1)) / jnp.sum(m)))
    t_cross = 2.0 * r_h / sig
    t_end = 10.0 * t_cross
    print(f"[run] r_h={r_h:.3f} pc, sigma={sig:.3f} pc/Myr, t_cross={t_cross:.3f} Myr "
          f"-> t_end={t_end:.2f} Myr, eps={EPS:.4f} pc, {N_SNAP} snapshots")
    system = ParticleSystem.from_velocities(
        positions=ic.stars.positions, velocities=ic.stars.velocities,
        masses=ic.stars.masses, units=STELLAR, softening=EPS,
    )
    t0 = time.time()
    systems, times = SymmetricHermiteIntegrator(dt=DT, n_iter=3, order=4).simulate(
        system, t_end=t_end, n_snapshots=N_SNAP)
    jax.block_until_ready(systems.positions)
    print(f"[run] integration wall time: {(time.time() - t0)/60:.1f} min")
    return system, systems, np.asarray(times), t_cross


def diagnostics(system0, systems, times):
    e0 = float(system0.total_energy)
    rows = []
    for i in range(len(times)):
        pos = np.asarray(systems.positions[i])
        vel = np.asarray(systems.velocities[i])
        mass = np.asarray(systems.masses[i])
        com = (pos * mass[:, None]).sum(0) / mass.sum()
        r = np.linalg.norm(pos - com, axis=1)
        r_h = np.median(r)
        T = 0.5 * float((mass * (vel**2).sum(1)).sum())
        # robust energy: direct softened T + V on the snapshot arrays (matches the
        # integrator's softened Hamiltonian; O(N^2) x 50 snapshots is cheap)
        dxm = pos[:, None, :] - pos[None, :, :]
        r2 = (dxm**2).sum(-1) + EPS**2
        iu = np.triu_indices(len(mass), k=1)
        V = -G * float((mass[:, None] * mass[None, :] / np.sqrt(r2))[iu].sum())
        q_sub = float(compute_q_parameter(pos - com))
        rows.append((float(times[i]), r_h, q_sub, T, T + V))
    rows = np.array(rows)
    dE = np.abs((rows[:, 4] - rows[0, 4]) / rows[0, 4])
    return rows, dE, e0


def figures(ic, system0, systems, times, t_cross, rows, dE):
    # ── Fig 1: the natal stars+gas state ──
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    rho = np.asarray(ic.gas.rho_cloud)
    col = rho.sum(axis=2) * float(ic.gas.cell_volume) / DX**2  # column density M⊙/pc²
    origin = np.asarray(ic.ledger.frame.origin)
    ext = [-origin[0], BOX - origin[0], -origin[1], BOX - origin[1]]
    im = axes[0].imshow(np.log10(col).T, origin="lower", extent=ext, cmap="magma")
    plt.colorbar(im, ax=axes[0], label=r"$\log_{10}\,\Sigma_{\rm cl}$ [M$_\odot$ pc$^{-2}$]")
    pos = np.asarray(ic.stars.positions)
    axes[0].scatter(pos[:, 0], pos[:, 1], s=1.5, c="cyan", alpha=0.5, lw=0)
    axes[0].set(xlabel="x [pc]", ylabel="y [pc]",
                title=f"Parent cloud + stars (ℳ=8, SFE=0.2, χ=χ$_{{F10}}$)")
    colg = np.asarray(ic.gas.rho_residual).sum(axis=2) * float(ic.gas.cell_volume) / DX**2
    im = axes[1].imshow(np.log10(np.maximum(colg, colg[colg > 0].min())).T,
                        origin="lower", extent=ext, cmap="viridis")
    plt.colorbar(im, ax=axes[1], label=r"$\log_{10}\,\Sigma_{g,0}$ [M$_\odot$ pc$^{-2}$]")
    axes[1].set(xlabel="x [pc]", title="Residual gas (ε$_⋆$ partition)")
    vel = np.asarray(ic.stars.velocities)
    sc = axes[2].scatter(pos[:, 0], pos[:, 1], s=3, c=vel[:, 2], cmap="RdBu_r",
                         vmin=-2, vmax=2, lw=0)
    plt.colorbar(sc, ax=axes[2], label=r"$v_z$ [pc Myr$^{-1}$]")
    axes[2].set(xlabel="x [pc]", title="Stellar kinematics (Helmholtz-coupled)")
    for ax in axes:
        ax.set_aspect("equal")
    fig.suptitle("Gravoturbulent natal state — TurbulentCloudIC "
                 f"(N={N}, M$_{{cl}}$={float(ic.ledger.M_cl):.0f} M$_\\odot$, "
                 f"Q$_0$={float(ic.ledger.Q_virial):.3f})", y=1.03)
    for ext_ in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig1_natal_state.{ext_}"))
    plt.close(fig)

    # ── Fig 2: cold-collapse evolution (4 epochs) ──
    idx = [0, max(1, N_SNAP // 10), N_SNAP // 2, N_SNAP - 1]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    for ax, i in zip(axes, idx):
        p = np.asarray(systems.positions[i])
        mass = np.asarray(systems.masses[i])
        com = (p * mass[:, None]).sum(0) / mass.sum()
        p = p - com
        ax.scatter(p[:, 0], p[:, 1], s=1.5, c="k", alpha=0.4, lw=0)
        ax.set(xlim=(-2, 2), ylim=(-2, 2), xlabel="x [pc]",
               title=f"t = {times[i]:.2f} Myr ({times[i]/t_cross:.1f} t$_{{cross}}$)")
        ax.set_aspect("equal")
    axes[0].set_ylabel("y [pc]")
    fig.suptitle("Cold collapse of the gravoturbulent cluster (time-symmetric "
                 f"Hermite P(EC)³, dt={DT:.1e} Myr, ε = Δx/4 = {EPS:.3f} pc)", y=1.02)
    for ext_ in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig2_evolution.{ext_}"))
    plt.close(fig)

    # ── Fig 3: diagnostics + the substructure-memory decay ──
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    tt = rows[:, 0] / t_cross
    axes[0].plot(tt, rows[:, 1], "k-")
    axes[0].set(xlabel=r"$t/t_{\rm cross}$", ylabel=r"$r_h$ [pc]",
                title="Half-mass radius")
    axes[1].plot(tt, rows[:, 2], "k-")
    axes[1].axhline(0.79, color="gray", ls=":", lw=1,
                    label="CW04 uniform (Q=0.79)")
    axes[1].legend()
    axes[1].set(xlabel=r"$t/t_{\rm cross}$", ylabel="CW04 $Q$",
                title="Substructure memory decay")
    if dE is not None:
        axes[2].semilogy(tt, np.maximum(dE, 1e-16), "k-")
        axes[2].set(xlabel=r"$t/t_{\rm cross}$", ylabel=r"$|\Delta E/E|$",
                    title="Energy conservation")
    for ext_ in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig3_diagnostics.{ext_}"))
    plt.close(fig)


def main():
    print("=" * 78)
    print("GRAVOTURB CAREER PRODUCTION RUN (single seed, Anna-directed)")
    print("=" * 78)
    ic = build_ic()
    print(f"[ic] M_cl={float(ic.ledger.M_cl):.0f} M⊙, M_gas={float(ic.ledger.M_gas):.0f} M⊙, "
          f"Q0={float(ic.ledger.Q_virial):.4f}, α_vir={float(ic.ledger.alpha_vir):.3f}, "
          f"closure={float(ic.ledger.mass_closure_residual):.2e}")
    system0, systems, times, t_cross = run(ic)
    np.savez(os.path.join(OUT, "snapshots_raw.npz"),
             times=times, positions=np.asarray(systems.positions),
             velocities=np.asarray(systems.velocities),
             masses=np.asarray(systems.masses), t_cross=t_cross,
             eps=EPS, seed=SEED)  # saved FIRST: the expensive part is never lost
    rows, dE, e0 = diagnostics(system0, systems, times)
    np.savez(os.path.join(OUT, "snapshots.npz"),
             times=times, positions=np.asarray(systems.positions),
             velocities=np.asarray(systems.velocities),
             masses=np.asarray(systems.masses), rows=rows, t_cross=t_cross,
             e0=e0, eps=EPS, seed=SEED)
    figures(ic, system0, systems, times, t_cross, rows, dE)
    print(f"[done] diagnostics: r_h {rows[0,1]:.2f}→{rows[-1,1]:.2f} pc; "
          f"CW04 Q {rows[0,2]:.2f}→{rows[-1,2]:.2f}; "
          f"max|dE/E| = {np.nanmax(dE) if dE is not None else float('nan'):.2e}")
    print(f"[done] figures in {OUT}")


if __name__ == "__main__":
    main()

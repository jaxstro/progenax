"""CAREER feasibility figure (Anna-directed 2026-07-17): demonstrate the instrument.

Goal: show what the gravoturb generator PRODUCES and that it is validated — a
work-in-progress capability, not a blank slate. NO dynamical evolution and NO
inference are needed for feasibility; the compelling evidence is the natal
stars+gas state (a ~6 s IC build) plus the committed acceptance-gate results.

Output: validation/plots/feasibility/gravoturb_feasibility.{png,pdf} — a single
publication-quality multi-panel figure.
"""

import os

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jaxstro.units import STELLAR

from gravoturb.cluster import build_cluster_ic
from gravoturb.realization.helmholtz import helmholtz_velocity_field
from gravoturb.specs import (
    CloudSpec,
    CompositionSpec,
    GasSpec,
    GeometrySpec,
    VelocitySpec,
)
from progenax import Maschberger, PlummerProfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "plots", "feasibility")
os.makedirs(OUT, exist_ok=True)

N, BOX, NGRID = 3000, 4.0, 96
DX = BOX / NGRID
G = STELLAR.G

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 11,
    "axes.titlesize": 11, "legend.fontsize": 8.5, "figure.dpi": 120,
    "savefig.dpi": 300, "savefig.bbox": "tight", "mathtext.fontset": "cm",
})


def build():
    masses = Maschberger().sample(jax.random.PRNGKey(9), N)  # real IMF for panel (c)
    return build_cluster_ic(
        masses,
        cloud=CloudSpec(mach=8.0, b=0.5, alpha=1.8, beta=None, coupling="helmholtz"),
        geometry=GeometrySpec(profile=PlummerProfile(r_h=0.5), box_size=BOX,
                              shape=(NGRID,) * 3),
        velocity=VelocitySpec(beta_v=4.0, mode="physical", c_s=0.2),
        composition=CompositionSpec(lambda_corr=0.6),
        G=G, units=STELLAR, key=jax.random.PRNGKey(1), gas=GasSpec(sfe=0.2),
    )


def _column(rho3d, cell_volume):
    return rho3d.sum(axis=2) * cell_volume / DX**2  # M_sun/pc^2


def main():
    ic = build()
    led, geo = ic.ledger, ic.geometry
    origin = np.asarray(led.frame.origin)
    ext = [-origin[0], BOX - origin[0], -origin[1], BOX - origin[1]]
    pos = np.asarray(ic.stars.positions)
    masses = np.asarray(ic.stars.masses)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)

    # (a) parent cloud column density + stars
    ax = fig.add_subplot(gs[0, 0])
    col = _column(np.asarray(ic.gas.rho_cloud), float(ic.gas.cell_volume))
    im = ax.imshow(np.log10(col).T, origin="lower", extent=ext, cmap="magma")
    plt.colorbar(im, ax=ax, fraction=0.046,
                 label=r"$\log_{10}\,\Sigma_{\rm cl}$ [$M_\odot\,{\rm pc}^{-2}$]")
    ax.scatter(pos[:, 0], pos[:, 1], s=1.2, c="cyan", alpha=0.5, lw=0)
    ax.set(xlabel="x [pc]", ylabel="y [pc]", title="(a) Turbulent parent cloud + stars")
    ax.set_aspect("equal")

    # (b) residual gas (the Aim-2 handoff product)
    ax = fig.add_subplot(gs[0, 1])
    colg = _column(np.asarray(ic.gas.rho_residual), float(ic.gas.cell_volume))
    floor = colg[colg > 0].min()
    im = ax.imshow(np.log10(np.maximum(colg, floor)).T, origin="lower", extent=ext,
                   cmap="viridis")
    plt.colorbar(im, ax=ax, fraction=0.046,
                 label=r"$\log_{10}\,\Sigma_{g,0}$ [$M_\odot\,{\rm pc}^{-2}$]")
    ax.set(xlabel="x [pc]", title="(b) Residual gas ($\\epsilon_\\star$ partition)")
    ax.set_aspect("equal")

    # (c) primordial mass segregation (lambda_corr) — mass-coloured stars
    ax = fig.add_subplot(gs[0, 2])
    order = np.argsort(masses)
    sc = ax.scatter(pos[order, 0], pos[order, 1], s=3 + 6 * (masses[order] /
                    masses.max()), c=np.log10(masses[order]), cmap="plasma", lw=0)
    plt.colorbar(sc, ax=ax, fraction=0.046, label=r"$\log_{10}(m/M_\odot)$")
    ax.set(xlabel="x [pc]", title="(c) Primordial segregation ($\\lambda_{\\rm corr}{=}0.6$, IMF)")
    ax.set_aspect("equal")

    # (d) density PDF: realized vs BM19 target (the validated core)
    ax = fig.add_subplot(gs[1, 0])
    s = np.asarray(ic.fields.s_turb.s).ravel()
    ax.hist(s, bins=80, density=True, histtype="stepfilled", alpha=0.4,
            color="steelblue", label="realized field")
    ax.axvline(float(ic.fields.s_turb.s_t), color="crimson", ls="--", lw=1.2,
               label=r"BM19 transition $s_t$")
    ax.set(xlabel=r"$s=\ln(\rho/\rho_0)$", ylabel="PDF",
           title="(d) BM19 lognormal + power-law tail", yscale="log")
    ax.legend()

    # (e) Helmholtz coupling: velocity power split (compressive fraction chi)
    ax = fig.add_subplot(gs[1, 1])
    from gravoturb.theory.driving import chi_f10
    bs = np.linspace(0.34, 1.0, 30)
    ax.plot(bs, [float(chi_f10(b)) for b in bs], "k-", lw=1.5)
    for b_, lbl in [(1 / 3, "solenoidal"), (0.5, "natal mix"), (1.0, "compressive")]:
        ax.plot(b_, float(chi_f10(b_)), "o", ms=6)
        ax.annotate(lbl, (b_, float(chi_f10(b_))), textcoords="offset points",
                    xytext=(6, -2), fontsize=8)
    ax.set(xlabel=r"driving parameter $b$",
           ylabel=r"compressive fraction $\chi=E_\parallel/E_{\rm tot}$",
           title=r"(e) Helmholtz coupling: $\chi_{\rm F10}=b/\sqrt{3}$")

    # (f) validation scorecard (the "how good is it" panel)
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    rows = [
        ("BM19 dense-fraction fidelity", r"$1.8\times10^{-8}$", "AC1"),
        ("mass-conserving copula bias", r"$4\times10^{-5}$", "AC6"),
        (r"$\beta$ recovery (input vs realized)", r"$|\Delta|\leq 0.012$", "AC-IC6"),
        (r"derived slope $\beta=\beta_v-2$", r"$|{\rm err}|<0.05$", "AC-IC9"),
        ("convergence signature (coupled)", r"$+1.74\sigma$", "AC-IC9e"),
        ("stars+gas mass closure", r"$3\times10^{-15}$", "AC-G1"),
        (r"gas grid $\sigma_g=\mathcal{M}c_s$", r"$<10^{-10}$", "AC-G7"),
        (r"$\tau_\star$ root AD/FD agreement", r"$6\times10^{-11}$", "AC-G5"),
        (r"segregation $\Lambda_{\rm MSR}$ (off$\to$on)", r"$1.2\to10.4$", "AC-IC10"),
    ]
    ax.text(0.0, 1.02, "(f) Validation scorecard  —  all gates PASS",
            fontsize=11, weight="bold", transform=ax.transAxes)
    y = 0.90
    for name, val, tag in rows:
        ax.text(0.0, y, name, fontsize=8.5, transform=ax.transAxes)
        ax.text(0.80, y, val, fontsize=8.5, transform=ax.transAxes, ha="right",
                family="monospace")
        ax.text(1.0, y, tag, fontsize=7, transform=ax.transAxes, ha="right",
                color="gray")
        y -= 0.093
    ax.text(0.0, y - 0.02,
            "Status: validated forward IC generator + differentiable summary\n"
            "statistics (WIP). Dynamical inference is the proposed CAREER work.",
            fontsize=8, style="italic", transform=ax.transAxes, color="#444")

    fig.suptitle(
        "Differentiable turbulence-native cluster initial conditions "
        f"(gravoturb)  —  $N={N}$, $\\mathcal{{M}}=8$, "
        f"$M_{{\\rm cl}}={float(led.M_cl):.0f}\\,M_\\odot$, SFE $=0.2$, "
        f"$Q_0={float(led.Q_virial):.3f}$, $\\alpha_{{\\rm vir}}={float(led.alpha_vir):.2f}$",
        fontsize=13, y=0.98)

    for e in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"gravoturb_feasibility.{e}"))
    plt.close(fig)
    print(f"[feasibility] M_cl={float(led.M_cl):.0f} M_sun, M_gas={float(led.M_gas):.0f}, "
          f"Q0={float(led.Q_virial):.3f}, closure={float(led.mass_closure_residual):.1e}")
    print(f"[feasibility] wrote {OUT}/gravoturb_feasibility.png (+ .pdf)")


if __name__ == "__main__":
    main()

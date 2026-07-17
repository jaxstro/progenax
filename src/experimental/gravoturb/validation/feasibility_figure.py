"""CAREER figures (Anna-directed 2026-07-17): demonstrate the instrument.

Two products, one IC build (no dynamical evolution, no inference needed for
feasibility — the compelling evidence is the natal stars+gas state plus the
committed acceptance-gate results):

  1. gravoturb_career.{png,pdf}      — the NSF "money" figure: the three spatial
                                        panels only (cloud+stars, residual gas,
                                        primordial mass segregation), standalone.
  2. gravoturb_feasibility.{png,pdf} — the full dev figure: the same spatial row
                                        on top + BM19 PDF / Helmholtz coupling /
                                        validation scorecard beneath (for records).

Both share one `_spatial_panels()` helper so the money row can never drift from
the dev figure. Column densities are plotted in M_sun/pc^2 (cluster/galactic
convention); the g/cm^2 equivalents are printed to stdout for the SF/ISM reader.
"""

import os

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from jaxstro.units import STELLAR

from gravoturb.cluster import build_cluster_ic
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

N, BOX, NGRID = 5000, 6.0, 96
DX = BOX / NGRID
G = STELLAR.G

# Column-density unit bridge: cluster people read M_sun/pc^2, SF/ISM people read
# g/cm^2 (it sets optical depth / the KS threshold). 1 M_sun/pc^2 = 2.089e-4 g/cm^2
# (M_sun = 1.98892e33 g over pc^2 = 9.5214e36 cm^2).
MSUN_PC2_TO_G_CM2 = 1.98892e33 / (3.0857e18) ** 2

IMF = Maschberger()  # m_min=0.01, m_max=300, mu=0.2 M_sun

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 11,
    "axes.titlesize": 11, "legend.fontsize": 8.5, "figure.dpi": 120,
    "savefig.dpi": 300, "savefig.bbox": "tight", "mathtext.fontset": "cm",
})


def build():
    masses = IMF.sample(jax.random.PRNGKey(9), N)  # real IMF for the segregation panel
    return build_cluster_ic(
        masses,
        cloud=CloudSpec(mach=8.0, b=0.5, alpha=1.8, beta=None, coupling="helmholtz"),
        geometry=GeometrySpec(profile=PlummerProfile(r_h=1.5), box_size=BOX,
                              shape=(NGRID,) * 3),
        velocity=VelocitySpec(beta_v=4.0, mode="physical", c_s=0.2),
        composition=CompositionSpec(lambda_corr=0.6),
        G=G, units=STELLAR, key=jax.random.PRNGKey(1), gas=GasSpec(sfe=0.2),
    )


def _column(rho3d, cell_volume):
    return rho3d.sum(axis=2) * cell_volume / DX**2  # M_sun/pc^2


def _spatial_panels(fig, axes, ic, star_size=1.2):
    """Draw the three spatial panels into the given (ax_a, ax_b, ax_c).

    Returns (col_cloud_max, col_gas_max) in M_sun/pc^2 for unit reporting.
    """
    ax_a, ax_b, ax_c = axes
    origin = np.asarray(ic.ledger.frame.origin)
    ext = [-origin[0], BOX - origin[0], -origin[1], BOX - origin[1]]
    pos = np.asarray(ic.stars.positions)
    masses = np.asarray(ic.stars.masses)

    # (a) parent cloud column density + stars
    col = _column(np.asarray(ic.gas.rho_cloud), float(ic.gas.cell_volume))
    im = ax_a.imshow(np.log10(col).T, origin="lower", extent=ext, cmap="magma")
    plt.colorbar(im, ax=ax_a, fraction=0.046,
                 label=r"$\log_{10}\,\Sigma_{\rm cl}$ [$M_\odot\,{\rm pc}^{-2}$]")
    ax_a.scatter(pos[:, 0], pos[:, 1], s=star_size, c="cyan", alpha=0.5, lw=0)
    ax_a.set(xlabel="x [pc]", ylabel="y [pc]",
             title="(a) Turbulent parent cloud + stars")
    ax_a.set_aspect("equal")

    # (b) residual gas (the Aim-2 handoff product)
    colg = _column(np.asarray(ic.gas.rho_residual), float(ic.gas.cell_volume))
    floor = colg[colg > 0].min()
    im = ax_b.imshow(np.log10(np.maximum(colg, floor)).T, origin="lower", extent=ext,
                     cmap="viridis")
    plt.colorbar(im, ax=ax_b, fraction=0.046,
                 label=r"$\log_{10}\,\Sigma_{g,0}$ [$M_\odot\,{\rm pc}^{-2}$]")
    ax_b.set(xlabel="x [pc]", ylabel="y [pc]",
             title=r"(b) Residual gas ($\epsilon_\star$ partition)")
    ax_b.set_aspect("equal")

    # (c) primordial mass segregation (lambda_corr) — mass-coloured stars.
    # LogNorm clipped to the IMF's [m_min, m_max]; ticks read as physical masses.
    order = np.argsort(masses)
    sc = ax_c.scatter(pos[order, 0], pos[order, 1],
                      s=3 + 9 * (masses[order] / masses.max()),
                      c=masses[order], cmap="plasma", lw=0,
                      norm=LogNorm(vmin=IMF.m_min, vmax=IMF.m_max))
    plt.colorbar(sc, ax=ax_c, fraction=0.046, label=r"stellar mass $m$ [$M_\odot$]")
    ax_c.set(xlabel="x [pc]", ylabel="y [pc]",
             title=r"(c) Primordial segregation ($\lambda_{\rm corr}{=}0.6$, Maschberger IMF)")
    ax_c.set_aspect("equal")

    return float(col.max()), float(colg.max())


def _headline(led):
    return (f"$N={N}$, $\\mathcal{{M}}=8$, "
            f"$M_{{\\rm cl}}={float(led.M_cl):.0f}\\,M_\\odot$, SFE $=0.2$, "
            f"$Q_0={float(led.Q_virial):.3f}$, "
            f"$\\alpha_{{\\rm vir}}={float(led.alpha_vir):.2f}$")


def career_figure(ic):
    """The NSF 'money' figure: the three spatial panels, standalone."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    scales = _spatial_panels(fig, axes, ic, star_size=1.6)
    fig.suptitle(
        "Turbulence-native cluster initial conditions (gravoturb)  —  "
        + _headline(ic.ledger), fontsize=14, y=1.02)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"gravoturb_career.{e}"))
    plt.close(fig)
    return scales


def dev_figure(ic):
    """The full dev figure: spatial row on top + PDF / coupling / scorecard."""
    led = ic.ledger
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)
    top = [fig.add_subplot(gs[0, i]) for i in range(3)]
    _spatial_panels(fig, top, ic)

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
        "Differentiable turbulence-native cluster initial conditions (gravoturb)  —  "
        + _headline(led), fontsize=13, y=0.98)
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"gravoturb_feasibility.{e}"))
    plt.close(fig)


def main():
    ic = build()
    led = ic.ledger
    col_cloud_max, col_gas_max = career_figure(ic)
    dev_figure(ic)

    k = MSUN_PC2_TO_G_CM2
    print(f"[feasibility] M_cl={float(led.M_cl):.0f} M_sun, M_gas={float(led.M_gas):.0f}, "
          f"Q0={float(led.Q_virial):.3f}, alpha_vir={float(led.alpha_vir):.2f}, "
          f"closure={float(led.mass_closure_residual):.1e}")
    print(f"[feasibility] wrote {OUT}/gravoturb_career.png (+ .pdf)  [NSF money figure]")
    print(f"[feasibility] wrote {OUT}/gravoturb_feasibility.png (+ .pdf)  [dev figure]")
    print("[feasibility] column-density scales (peak, projected along z):")
    print(f"    parent cloud  Sigma_cl,max  = {col_cloud_max:.3e} M_sun/pc^2 "
          f"= {col_cloud_max * k:.3e} g/cm^2")
    print(f"    residual gas  Sigma_g0,max   = {col_gas_max:.3e} M_sun/pc^2 "
          f"= {col_gas_max * k:.3e} g/cm^2")
    print(f"    (conversion: 1 M_sun/pc^2 = {k:.4e} g/cm^2)")


if __name__ == "__main__":
    main()

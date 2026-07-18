"""CAREER figures (Anna-directed 2026-07-17): demonstrate the instrument.

No dynamical evolution, no inference — feasibility rests on the natal stars+gas
state plus the committed acceptance-gate results. Products (one IC build, or two
for the controlled option):

  1. gravoturb_career.{png,pdf}        — Option A, the 3-panel "money" figure:
                                          cloud+stars / residual gas / segregated
                                          luminosity-weighted spectral star field.
  2. gravoturb_career_4panel.{png,pdf} — Option B, the same two spatial panels + a
                                          MATCHED CONTROL: panel (c) lambda_corr=0
                                          vs (d) lambda_corr=0.6 at IDENTICAL cloud,
                                          positions, masses and rendering (only the
                                          mass<->position pairing differs), each
                                          annotated with the Allison+2009 Lambda_MSR.
  3. gravoturb_feasibility.{png,pdf}   — dev figure: Option-A row + BM19 PDF /
                                          Helmholtz coupling / validation scorecard.

Column densities are plotted in M_sun/pc^2 (cluster/galactic convention); g/cm^2
equivalents are printed to stdout for the SF/ISM reader.
"""

import os

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap
from jaxstro.units import STELLAR

from gravoturb.cluster import build_cluster_ic
from gravoturb.specs import (
    CloudSpec,
    CompositionSpec,
    GasSpec,
    GeometrySpec,
    VelocitySpec,
)
from scipy.stats import spearmanr

from progenax import Maschberger, PlummerProfile
from progenax.stellar import zams_effective_temperature, zams_radius

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

# ZAMS spectral sequence (Tout+1996 T_eff -> conventional stellar colours). A
# diverging temperature map: cool dark-red (M) -> white (F/A) -> hot blue-violet
# (O), binned into the canonical spectral classes. Validated (dataviz skill):
# adjacent-class CVD dE >= 15, every class clears 3:1 contrast on the dark surface.
SPECTRAL_EDGES = np.array([2400., 3700., 5200., 6000., 7500., 10000., 30000., 55000.])
SPECTRAL_LETTERS = ["M", "K", "G", "F", "A", "B", "O"]
SPECTRAL_COLORS = ["#c24a28", "#e8791f", "#f3c95a", "#f7f3e2",
                   "#cdd9ff", "#9ab8ff", "#8172ff"]
STARFIELD_BG = "#0d0d12"  # near-black so hot/pale stars pop; unifies with (a)/(b)

sns.set_theme(style="ticks", context="paper", font="serif")
plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 11,
    "axes.titlesize": 11, "legend.fontsize": 8.5, "figure.dpi": 120,
    "savefig.dpi": 300, "savefig.bbox": "tight", "mathtext.fontset": "cm",
})


def build(lambda_corr=0.6):
    """Build one IC. Star positions are independent of lambda_corr (they come from
    the density field); lambda_corr only sets the mass<->position pairing — so two
    builds with the same key differ ONLY in mass assignment (the clean control)."""
    masses = IMF.sample(jax.random.PRNGKey(9), N)
    return build_cluster_ic(
        masses,
        cloud=CloudSpec(mach=8.0, b=0.5, alpha=1.8, beta=None, coupling="helmholtz"),
        geometry=GeometrySpec(profile=PlummerProfile(r_h=1.5), box_size=BOX,
                              shape=(NGRID,) * 3),
        velocity=VelocitySpec(beta_v=4.0, mode="physical", c_s=0.2),
        composition=CompositionSpec(lambda_corr=lambda_corr),
        G=G, units=STELLAR, key=jax.random.PRNGKey(1), gas=GasSpec(sfe=0.2),
    )


def _column(rho3d, cell_volume):
    return rho3d.sum(axis=2) * cell_volume / DX**2  # M_sun/pc^2


def _extent(ic):
    origin = np.asarray(ic.ledger.frame.origin)
    return [-origin[0], BOX - origin[0], -origin[1], BOX - origin[1]]


def _mass_density_spearman(ic):
    """rho_S between stellar mass and local (natal) gas density — the quantity
    lambda_corr controls by construction (McLuster A1 shuffle on density rank)."""
    pos = np.asarray(ic.stars.positions)
    origin = np.asarray(ic.ledger.frame.origin)
    cell = np.clip(np.floor((pos + origin) / BOX * NGRID).astype(int), 0, NGRID - 1)
    rho = np.asarray(ic.gas.rho_cloud)
    local = rho[cell[:, 0], cell[:, 1], cell[:, 2]]
    return float(spearmanr(np.asarray(ic.stars.masses), local).statistic)


def _cloud_panel(ax, ic, ext, title="(a) Turbulent parent cloud + stars"):
    col = _column(np.asarray(ic.gas.rho_cloud), float(ic.gas.cell_volume))
    im = ax.imshow(np.log10(col).T, origin="lower", extent=ext, cmap="magma")
    plt.colorbar(im, ax=ax, fraction=0.046,
                 label=r"$\log_{10}\,\Sigma_{\rm cl}$ [$M_\odot\,{\rm pc}^{-2}$]")
    # stars as small uniform white points with a thin dark edge, sized by ZAMS
    # radius — reads on both the bright core and the dark envelope of the magma map
    # (the old cyan clashed with the warm colormap and blocked up into squares).
    pos = np.asarray(ic.stars.positions)
    rad = np.asarray(zams_radius(jnp.clip(jnp.asarray(ic.stars.masses), 0.08, 150.0)))
    ax.scatter(pos[:, 0], pos[:, 1], s=3.0 + 7.0 * np.sqrt(rad), c="white",
               alpha=0.6, edgecolors="#101014", linewidths=0.25)
    ax.set(xlabel="x [pc]", ylabel="y [pc]", title=title)
    ax.set_aspect("equal")
    return float(col.max()), col


def _gas_panel(ax, ic, ext, title=r"(b) Residual gas ($\epsilon_\star$ partition)"):
    colg = _column(np.asarray(ic.gas.rho_residual), float(ic.gas.cell_volume))
    floor = colg[colg > 0].min()
    im = ax.imshow(np.log10(np.maximum(colg, floor)).T, origin="lower", extent=ext,
                   cmap="viridis")
    plt.colorbar(im, ax=ax, fraction=0.046,
                 label=r"$\log_{10}\,\Sigma_{g,0}$ [$M_\odot\,{\rm pc}^{-2}$]")
    ax.set(xlabel="x [pc]", ylabel="y [pc]", title=title)
    ax.set_aspect("equal")
    return float(colg.max())


def _starfield(ax, pos, masses, ext, *, opaque=False, cloud_col=None,
               rho_s=None, title="", cbar=True):
    """Luminosity-weighted spectral star field (the segregation panel).

    All marks are physical ZAMS observables (Tout+1996): colour = spectral type
    (T_eff), size = radius, per-star alpha = luminosity; two-tier composite (M-dwarf
    haze behind, K..O depth-ordered on top); massive stars either white-outlined
    (opaque=False) or rendered in full-opacity spectral colour (opaque=True).
    """
    m_clip = jnp.clip(jnp.asarray(masses), 0.08, 150.0)
    teff = np.clip(np.asarray(zams_effective_temperature(m_clip)),
                   SPECTRAL_EDGES[0] + 1.0, SPECTRAL_EDGES[-1] - 1.0)
    radii = np.asarray(zams_radius(m_clip))
    cmap_sp = ListedColormap(SPECTRAL_COLORS)
    norm_sp = BoundaryNorm(SPECTRAL_EDGES, cmap_sp.N)
    z = np.asarray(pos[:, 2])
    zmin, zptp = z.min(), np.ptp(z) + 1e-9
    sizes_all = 6.0 + 14.0 * np.sqrt(radii)
    depth_all = (z - zmin) / zptp
    ax.set_facecolor(STARFIELD_BG)

    # faint gas-column contours behind the stars: shows the massive stars sinking
    # into the densest gas — the mechanism lambda_corr controls.
    if cloud_col is not None:
        lc = np.log10(np.maximum(cloud_col, cloud_col[cloud_col > 0].min()))
        xs = np.linspace(ext[0], ext[1], cloud_col.shape[0])
        ys = np.linspace(ext[2], ext[3], cloud_col.shape[1])
        ax.contour(xs, ys, lc.T, levels=np.percentile(lc, [80, 90, 96, 99]),
                   colors="#8ea3bf", alpha=0.25, linewidths=0.5)

    # Tier split at 1 Msun. The sub-solar field is drawn first as a faint backdrop;
    # the >1 Msun (resolved / alpha-slope) stars are then drawn most-massive-first
    # -> lowest, with per-star alpha encoding line-of-sight depth (near opaque, far
    # faint) — so depth reads as transparency and mass sets the draw order.
    sub = np.where(masses < 1.0)[0]
    sub = sub[np.argsort(z[sub])]  # far -> near within the field
    faceS = cmap_sp(norm_sp(teff[sub]))
    faceS[:, 3] = 0.12 + 0.22 * depth_all[sub]  # faint, depth-scaled
    ax.scatter(pos[sub, 0], pos[sub, 1], s=3.5 + 5.0 * np.sqrt(radii[sub]),
               facecolors=faceS, lw=0)

    res = np.where(masses >= 1.0)[0]
    res = res[np.argsort(masses[res])[::-1]]  # most massive first -> lowest last
    m_r = masses[res]
    faceR = cmap_sp(norm_sp(teff[res]))
    faceR[:, 3] = np.clip(0.30 + 0.70 * depth_all[res], 0.0, 1.0)  # alpha = depth
    if opaque:  # full spectral colour + thin bg-coloured separator (de-overlap)
        edge, lw = STARFIELD_BG, 0.7
    else:  # white outline (all resolved are the m>1 alpha-slope population)
        edge, lw = "#f7f7ff", 0.55
    rb = res[m_r > 3.0]  # soft PSF bloom behind the luminous (O/B/A) stars
    ax.scatter(pos[rb, 0], pos[rb, 1], s=3.5 * sizes_all[rb],
               c=cmap_sp(norm_sp(teff[rb])), alpha=0.10, lw=0)
    ax.scatter(pos[res, 0], pos[res, 1], s=sizes_all[res], facecolors=faceR,
               edgecolors=edge, linewidths=lw)

    if cbar:
        centers = 0.5 * (SPECTRAL_EDGES[:-1] + SPECTRAL_EDGES[1:])
        sm = plt.cm.ScalarMappable(cmap=cmap_sp, norm=norm_sp)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, fraction=0.046, ticks=centers, spacing="uniform")
        cb.ax.set_yticklabels(SPECTRAL_LETTERS)
        cb.set_label(r"ZAMS spectral type ($T_{\rm eff}$: M cool $\to$ O hot)")
    if rho_s is not None:
        ax.text(0.045, 0.955, rf"$\rho_S(m,\rho_{{\rm gas}})={rho_s:+.2f}$",
                transform=ax.transAxes, va="top", ha="left", color="white",
                fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="#0d0d12",
                                       ec="#556", alpha=0.8))
    ax.set(xlabel="x [pc]", ylabel="y [pc]", title=title)
    ax.set_aspect("equal")


def _spatial_panels(fig, axes, ic):
    """Option-A three-panel row (cloud+stars / residual gas / segregation)."""
    ax_a, ax_b, ax_c = axes
    ext = _extent(ic)
    cmax, _ = _cloud_panel(ax_a, ic, ext)
    gmax = _gas_panel(ax_b, ic, ext)
    _starfield(ax_c, np.asarray(ic.stars.positions), np.asarray(ic.stars.masses), ext,
               opaque=False, cbar=True,
               title=r"(c) Primordial segregation ($\lambda_{\rm corr}{=}0.6$, Maschberger IMF)")
    return cmax, gmax


def _headline(led):
    return (f"$N={N}$, $\\mathcal{{M}}=8$, $\\alpha_\\rho=1.8$, "
            f"$\\alpha_{{\\rm IMF}}={float(IMF.alpha):.1f}$, "
            f"$M_{{\\rm cl}}={float(led.M_cl):.0f}\\,M_\\odot$, SFE $=0.2$, "
            f"$Q_0={float(led.Q_virial):.3f}$, "
            f"$\\alpha_{{\\rm vir}}={float(led.alpha_vir):.2f}$")


def career_figure(ic0, ic6):
    """The NSF money figure: (a) cloud+stars, then the matched lambda_corr control
    (b: 0, c: 0.6) at identical cloud/positions/IMF. Residual gas -> dev figure."""
    led = ic6.ledger
    ext = _extent(ic6)
    pos0, m0 = np.asarray(ic0.stars.positions), np.asarray(ic0.stars.masses)
    pos6, m6 = np.asarray(ic6.stars.positions), np.asarray(ic6.stars.masses)
    rs0, rs6 = _mass_density_spearman(ic0), _mass_density_spearman(ic6)

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.4), layout="constrained")
    _, cloud_col = _cloud_panel(axes[0], ic6, ext, title="(a) Parent cloud + stars")
    _starfield(axes[1], pos0, m0, ext, opaque=True, cloud_col=cloud_col,
               cbar=False, title=r"(b) No coupling ($\lambda_{\rm corr}=0$)")
    _starfield(axes[2], pos6, m6, ext, opaque=True, cloud_col=cloud_col,
               cbar=True, title=r"(c) Mass$-$gas coupling ($\lambda_{\rm corr}=0.6$)")
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.05, hspace=0.0)
    fig.suptitle(
        "Gravoturbulent cluster initial conditions — controlled primordial "
        "mass segregation", fontsize=20)
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"gravoturb_career.{e}"))
    plt.close(fig)
    return rs0, rs6


def dev_figure(ic):
    """The full dev figure: Option-A row on top + PDF / coupling / scorecard."""
    led = ic.ledger
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.42)
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
    ic6 = build(0.6)  # realistic moderate primordial mass-density coupling (rho_S~0.6)
    ic0 = build(0.0)  # same key -> identical structure; only mass-assignment differs
    led = ic6.ledger
    dpos = float(np.max(np.abs(np.asarray(ic6.stars.positions)
                                - np.asarray(ic0.stars.positions))))

    rs0, rs6 = career_figure(ic0, ic6)   # the money figure (controlled segregation)
    dev_figure(ic6)

    col_cloud = _column(np.asarray(ic6.gas.rho_cloud), float(ic6.gas.cell_volume))
    col_gas = _column(np.asarray(ic6.gas.rho_residual), float(ic6.gas.cell_volume))
    k = MSUN_PC2_TO_G_CM2
    print(f"[feasibility] M_cl={float(led.M_cl):.0f} M_sun, M_gas={float(led.M_gas):.0f}, "
          f"Q0={float(led.Q_virial):.3f}, alpha_vir={float(led.alpha_vir):.2f}")
    print(f"[feasibility] control check: max|pos(lam=0)-pos(lam=0.6)| = {dpos:.2e} pc "
          "(rigid COM shift; structure identical)")
    print(f"[feasibility] rho_S(mass, local gas density): lam_corr=0 -> {rs0:+.3f} ; "
          f"lam_corr=0.6 -> {rs6:+.3f}")
    print(f"[feasibility] wrote {OUT}/gravoturb_career.png      [money figure: controlled]")
    print(f"[feasibility] wrote {OUT}/gravoturb_feasibility.png [dev figure]")
    print(f"    parent cloud  Sigma_cl,max = {float(col_cloud.max()):.3e} M_sun/pc^2 "
          f"= {float(col_cloud.max()) * k:.3e} g/cm^2")
    print(f"    residual gas  Sigma_g0,max  = {float(col_gas.max()):.3e} M_sun/pc^2 "
          f"= {float(col_gas.max()) * k:.3e} g/cm^2")


if __name__ == "__main__":
    main()

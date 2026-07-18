"""CAREER multiplicity money figure (extension C) — the λ_mult analog of the λ_corr money figure.

Mirrors ``feasibility_figure.career_figure`` (same cloud, positions, IMF, rendering style), but the
matched control varies **λ_mult** (environment-coupled multiplicity) instead of λ_corr: (b) λ_mult=0
(binaries placed independently of the gas) vs (c) λ_mult=0.6 (binaries concentrate in the dense natal
gas). Reuses the feasibility helpers so the two money figures are visually a pair. Panels:

  (a) parent cloud + stars (identical treatment to the mass figure)
  (b) λ_mult=0   — spectral star field, BINARY systems ringed; ρ_S(binary, ρ_gas) ≈ 0
  (c) λ_mult=0.6 — the SAME field/positions, binaries now sunk into the dense gas contours

Run:
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
      PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync \
      python -m gravoturb.validation.feasibility_multiplicity_figure

Writes plots/feasibility/gravoturb_career_multiplicity.{png,pdf}. Demo/figure script (LOC-exempt).
"""
import os

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jaxstro.units import STELLAR
from gravoturb.cluster import build_cluster_ic
from gravoturb.specs import CloudSpec, CompositionSpec, GasSpec, GeometrySpec, VelocitySpec
from progenax import Maschberger, PlummerProfile
from progenax.binaries.companions import MoeCompanions

# reuse the exact style + helpers of the mass money figure so the pair matches
from gravoturb.validation import feasibility_figure as ff
from gravoturb.validation.feasibility_figure import (
    BOX, G, N, NGRID, TITLE_FS, _cloud_panel, _extent, _finish, _starfield,
)

OUT = ff.OUT
IMF = Maschberger()


def build(lambda_mult):
    """One IC with a Moe binary population; λ_mult sets the multiplicity↔density placement.

    Positions come from the density field and the population is the same draw — only which
    positions host binaries changes with λ_mult (the clean matched control, like the mass figure)."""
    masses = IMF.sample(jax.random.PRNGKey(9), N)
    return build_cluster_ic(
        masses,
        cloud=CloudSpec(mach=8.0, b=0.5, alpha=1.8, beta=None, coupling="helmholtz"),
        geometry=GeometrySpec(profile=PlummerProfile(r_h=1.5), box_size=BOX, shape=(NGRID,) * 3),
        velocity=VelocitySpec(beta_v=4.0, mode="physical", c_s=0.2),
        composition=CompositionSpec(lambda_corr=0.6, lambda_mult=lambda_mult,
                                    companions=MoeCompanions()),
        G=G, units=STELLAR, key=jax.random.PRNGKey(1), gas=GasSpec(sfe=0.2),
    )


def _systems(ic):
    """Group resolved stars into systems: barycenter positions, is_binary, and the OBSERVED
    (blended-light) mass — m_obs = L⁻¹(L(m1)+L(m2)) for binaries, m1 for singles. The observed
    spectral type is set by m_obs, so an unresolved binary looks like a slightly hotter/bluer,
    more-massive single star (the 'binaries mimic mass' effect made visual)."""
    from progenax.stellar import inverse_zams_luminosity, zams_luminosity

    sid = np.asarray(ic.stars.system_id)
    pos = np.asarray(ic.stars.positions)
    mass = np.asarray(ic.stars.masses)
    order = np.argsort(sid, kind="stable")
    sid, pos, mass = sid[order], pos[order], mass[order]
    _, idx, counts = np.unique(sid, return_index=True, return_counts=True)
    is_binary = counts == 2
    bary = np.array([pos[i:i + c].mean(axis=0) for i, c in zip(idx, counts)])
    m1 = np.array([mass[i:i + c].max() for i, c in zip(idx, counts)])          # primary
    m2 = np.array([min(mass[i], mass[i + 1]) if c == 2 else 0.0 for i, c in zip(idx, counts)])
    l_tot = np.asarray(zams_luminosity(jnp.asarray(m1))) + np.where(
        is_binary, np.asarray(zams_luminosity(jnp.asarray(np.maximum(m2, 1e-3)))), 0.0)
    m_obs = np.asarray(jax.vmap(inverse_zams_luminosity)(jnp.asarray(l_tot)))
    return bary, m1, m_obs, is_binary


def _binary_density_spearman(ic, bary, is_binary):
    origin = np.asarray(ic.ledger.frame.origin)
    cell = np.clip(np.floor((bary + origin) / BOX * NGRID).astype(int), 0, NGRID - 1)
    rho = np.asarray(ic.gas.rho_cloud)
    local = rho[cell[:, 0], cell[:, 1], cell[:, 2]]
    return float(spearmanr(is_binary.astype(float), local).statistic)


_BG = "#0d0d12"
_SINGLE_C = "#6f7a90"   # cool muted slate — recessive backdrop population
_BINARY_C = "#38e0c6"   # vivid aqua — binaries glow and pop against the slate + black


def _multiplicity_field(ax, ic, cloud_col, title, cbar):
    """Elegant starfield (the mass figure's depth-alpha + PSF-bloom + two-tier recipe) keyed to
    MULTIPLICITY: colour = binary (aqua) vs single (slate), size = ZAMS radius, per-star alpha = LOS
    depth. Singles recede as a cool backdrop; binaries glow aqua and bloom, so as λ_mult turns on the
    aqua stars visibly gather into the dense natal gas. No spectral rainbow, no orange."""
    import matplotlib
    from progenax.stellar import zams_radius

    bary, m1, _, is_binary = _systems(ic)
    ext = _extent(ic)
    ax.set_facecolor(_BG)

    # faint gas-column contours behind (cool neutral)
    lc = np.log10(np.maximum(cloud_col, cloud_col[cloud_col > 0].min()))
    xs = np.linspace(ext[0], ext[1], cloud_col.shape[0])
    ys = np.linspace(ext[2], ext[3], cloud_col.shape[1])
    ax.contour(xs, ys, lc.T, levels=np.percentile(lc, [80, 90, 96, 99]),
               colors="#9aa3b2", alpha=0.65, linewidths=0.6)

    rad = np.asarray(zams_radius(jnp.clip(jnp.asarray(m1), 0.08, 150.0)))
    sizes = 6.0 + 14.0 * np.sqrt(rad)
    z = bary[:, 2]
    depth = (z - z.min()) / (np.ptp(z) + 1e-9)
    base = np.where(is_binary, _BINARY_C, _SINGLE_C)
    rgba = matplotlib.colors.to_rgba_array(base)

    # tier 1 — sub-solar backdrop (faint, depth-scaled)
    sub = np.where(m1 < 1.0)[0]
    sub = sub[np.argsort(z[sub])]
    fsub = rgba[sub].copy()
    fsub[:, 3] = np.where(is_binary[sub], 0.65, 0.30) + 0.30 * depth[sub]
    fsub[:, 3] = np.clip(fsub[:, 3], 0.0, 1.0)
    ax.scatter(bary[sub, 0], bary[sub, 1], s=4.0 + 6.0 * np.sqrt(rad[sub]),
               facecolors=fsub, lw=0, rasterized=True)

    # tier 2 — resolved (>1 M⊙), most-massive-first, depth-alpha, with a soft bloom for the luminous
    res = np.where(m1 >= 1.0)[0]
    res = res[np.argsort(m1[res])[::-1]]
    fres = rgba[res].copy()
    fres[:, 3] = np.clip(np.where(is_binary[res], 0.80, 0.55) + 0.45 * depth[res], 0.0, 1.0)
    bloom = res[m1[res] > 3.0]
    ax.scatter(bary[bloom, 0], bary[bloom, 1], s=3.0 * sizes[bloom], c=base[bloom],
               alpha=0.14, lw=0, rasterized=True)
    ax.scatter(bary[res, 0], bary[res, 1], s=sizes[res], facecolors=fres,
               edgecolors=_BG, linewidths=0.6, rasterized=True)
    # a subtle aqua bloom on the massive binaries so the coupled population reads at a glance
    bb = res[is_binary[res] & (m1[res] > 2.0)]
    ax.scatter(bary[bb, 0], bary[bb, 1], s=2.6 * sizes[bb], c=_BINARY_C, alpha=0.14, lw=0,
               rasterized=True)

    rs = _binary_density_spearman(ic, bary, is_binary)
    f_lo, f_hi = _fbin_split(ic, bary, is_binary)
    ax.text(0.045, 0.955,
            rf"$\rho_S(\mathrm{{bin}},\rho_{{\rm gas}})={rs:+.2f}$" + "\n"
            rf"$f_{{\rm bin}}$: {f_lo:.2f}$\to${f_hi:.2f} (low$\to$high $\rho$)",
            transform=ax.transAxes, va="top", ha="left", color="white", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc=_BG, ec="#556", alpha=0.85))
    if cbar:
        ax.scatter([], [], s=55, c=_BINARY_C, label="binary")
        ax.scatter([], [], s=40, c=_SINGLE_C, label="single")
        ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9, facecolor=_BG, labelcolor="white")
    _finish(ax, title, ext)


def _fbin_split(ic, bary, is_binary):
    origin = np.asarray(ic.ledger.frame.origin)
    cell = np.clip(np.floor((bary + origin) / BOX * NGRID).astype(int), 0, NGRID - 1)
    rho = np.asarray(ic.gas.rho_cloud)
    local = rho[cell[:, 0], cell[:, 1], cell[:, 2]]
    med = np.median(local)
    return is_binary[local <= med].mean(), is_binary[local > med].mean()


def _fbin_tertiles(ic):
    bary, _, _, is_binary = _systems(ic)
    origin = np.asarray(ic.ledger.frame.origin)
    cell = np.clip(np.floor((bary + origin) / BOX * NGRID).astype(int), 0, NGRID - 1)
    rho = np.asarray(ic.gas.rho_cloud)[tuple(cell.T)]
    ln = np.log(rho)
    lo, hi = np.percentile(ln, [33.33, 66.67])
    return [is_binary[ln <= lo].mean(), is_binary[(ln > lo) & (ln <= hi)].mean(),
            is_binary[ln > hi].mean()]


def _fbin_radial(ic, nb=8):
    bary, _, _, is_binary = _systems(ic)
    r = np.linalg.norm(bary, axis=1)
    edges = np.linspace(0, np.percentile(r, 95), nb)
    idx = np.clip(np.digitize(r, edges) - 1, 0, nb - 2)
    cen = 0.5 * (edges[:-1] + edges[1:])
    fb = [is_binary[idx == k].mean() if np.any(idx == k) else np.nan for k in range(nb - 1)]
    return cen, fb


def dev_figure(ic0, ic6):
    """Full/dev binaries figure: money row on top + λ_mult diagnostics + a descriptive text box."""
    fig = plt.figure(figsize=(17.5, 10.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.28, wspace=0.26)
    ext = _extent(ic6)
    axc = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1]); axcc = fig.add_subplot(gs[0, 2])
    _, cloud_col = _cloud_panel(axc, ic6, ext, title="(a) Parent cloud + stars", vmax_pct=99.5)
    _multiplicity_field(axb, ic0, cloud_col, r"(b) No coupling ($\lambda_{\rm mult}=0$)", cbar=False)
    _multiplicity_field(axcc, ic6, cloud_col,
                        r"(c) Multiplicity$-$gas coupling ($\lambda_{\rm mult}=0.6$)", cbar=True)

    # (d) f_bin vs density tertile
    ax = fig.add_subplot(gs[1, 0])
    for ic, lm, c in [(ic0, 0.0, "C3"), (ic6, 0.6, "C0")]:
        ax.plot([0, 1, 2], _fbin_tertiles(ic), "-o", color=c, label=rf"$\lambda_{{\rm mult}}={lm}$")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["low", "mid", "high"])
    ax.set(xlabel="local gas density tertile", ylabel=r"$f_{\rm bin}$",
           title="(d) binary fraction vs density")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (e) binary radial profile
    ax = fig.add_subplot(gs[1, 1])
    for ic, lm, c in [(ic0, 0.0, "C3"), (ic6, 0.6, "C0")]:
        cen, fb = _fbin_radial(ic)
        ax.plot(cen, fb, "-o", color=c, label=rf"$\lambda_{{\rm mult}}={lm}$")
    ax.set(xlabel="cluster-centric radius [pc]", ylabel=r"$f_{\rm bin}$",
           title="(e) binary radial profile")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (f) descriptive text box
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    bary0, _, _, isb0 = _systems(ic0)
    bary6, _, _, isb6 = _systems(ic6)
    rs0 = _binary_density_spearman(ic0, bary0, isb0)
    rs6 = _binary_density_spearman(ic6, bary6, isb6)
    t0 = _fbin_tertiles(ic0); t6 = _fbin_tertiles(ic6)
    ax.text(0.0, 1.0, r"(f) Environment-coupled multiplicity ($\lambda_{\rm mult}$)",
            fontsize=12, weight="bold", transform=ax.transAxes, va="top")
    body = (
        "Moe & Di Stefano (2017) binaries are placed onto natal positions by a\n"
        "Gaussian copula that couples system multiplicity (binary fraction +\n"
        "orbital compactness) to local gas density, independently of the mass\n"
        r"coupling $\lambda_{\rm corr}$. Whole self-consistent systems are reassigned, so" "\n"
        r"every marginal and the Moe $(P,q,e)$ joint are preserved exactly." "\n\n"
        r"$\bullet$  $\lambda_{\rm mult}=0$: density-independent — byte-identical to the" "\n"
        r"    Moe baseline. The residual $f_{\rm bin}$–$\rho$ trend ($\rho_S=" f"{rs0:+.2f}$)" "\n"
        "    is the emergent mass channel alone.\n"
        r"$\bullet$  $\lambda_{\rm mult}=0.6$: binaries sink into the dense gas" "\n"
        rf"    ($\rho_S={rs6:+.2f}$; high/low-$\rho$ $f_{{\rm bin}}$ ratio "
        rf"{t0[2]/max(t0[0],1e-3):.1f}$\times\!\to\!${t6[2]/max(t6[0],1e-3):.1f}$\times$)." "\n"
        rf"$\bullet$  Marginal preserved: $N_{{\rm bin}}={int(ic0.ledger.n_binaries)}$ at both."
        "\n\n"
        "A controlled birth variable (Aims 1–2) and an Aim-3 identifiability\n"
        "channel: multiplicity segregation aliases mass segregation — an\n"
        "unresolved binary is observed as a bluer, more-massive star."
    )
    ax.text(0.0, 0.90, body, fontsize=9.0, transform=ax.transAxes, va="top", family="serif",
            linespacing=1.35)

    fig.suptitle(r"Environment-coupled multiplicity in gravoturbulent cluster ICs  —  "
                 rf"$N={N}$, $\mathcal{{M}}=8$, Maschberger IMF + Moe binaries, "
                 rf"$\lambda_{{\rm corr}}=0.6$", fontsize=13, y=0.98)
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"gravoturb_multiplicity_feasibility.{e}"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.join(OUT, 'gravoturb_multiplicity_feasibility.png')}")


def main():
    ic0 = build(0.0)
    ic6 = build(0.6)
    dev_figure(ic0, ic6)
    ext = _extent(ic6)
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.4), layout="constrained")
    _, cloud_col = _cloud_panel(axes[0], ic6, ext, title="(a) Parent cloud + stars", vmax_pct=99.5)
    _multiplicity_field(axes[1], ic0, cloud_col,
                        r"(b) No coupling ($\lambda_{\rm mult}=0$)", cbar=False)
    _multiplicity_field(axes[2], ic6, cloud_col,
                        r"(c) Multiplicity$-$gas coupling ($\lambda_{\rm mult}=0.6$)", cbar=True)
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.05, hspace=0.0)
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"gravoturb_career_multiplicity.{e}"))
    plt.close(fig)
    print(f"wrote {os.path.join(OUT, 'gravoturb_career_multiplicity.png')}")
    print(f"n_binaries: lambda_mult=0 -> {int(ic0.ledger.n_binaries)}, "
          f"0.6 -> {int(ic6.ledger.n_binaries)} (marginal preserved)")


if __name__ == "__main__":
    main()

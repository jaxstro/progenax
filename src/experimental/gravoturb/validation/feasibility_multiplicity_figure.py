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
    """Group resolved stars into systems: barycenter positions, masses, is_binary."""
    sid = np.asarray(ic.stars.system_id)
    pos = np.asarray(ic.stars.positions)
    mass = np.asarray(ic.stars.masses)
    order = np.argsort(sid, kind="stable")
    sid, pos, mass = sid[order], pos[order], mass[order]
    _, idx, counts = np.unique(sid, return_index=True, return_counts=True)
    is_binary = counts == 2
    bary = np.array([pos[i:i + c].mean(axis=0) for i, c in zip(idx, counts)])
    m_sys = np.array([mass[i:i + c].sum() for i, c in zip(idx, counts)])
    return bary, m_sys, is_binary


def _binary_density_spearman(ic, bary, is_binary):
    origin = np.asarray(ic.ledger.frame.origin)
    cell = np.clip(np.floor((bary + origin) / BOX * NGRID).astype(int), 0, NGRID - 1)
    rho = np.asarray(ic.gas.rho_cloud)
    local = rho[cell[:, 0], cell[:, 1], cell[:, 2]]
    return float(spearmanr(is_binary.astype(float), local).statistic)


def _multiplicity_field(ax, ic, cloud_col, title, cbar):
    """Two-class field: singles as a faint grey haze, BINARY systems as bright cyan points.

    The multiplicity story is binary-vs-single, so a clean two-class encoding (not the mass
    figure's spectral map) makes the *concentration* of binaries into the dense gas the money shot.
    Point size still tracks system mass (√radius) for physical texture; gas contours behind."""
    from progenax.stellar import zams_radius

    bary, m_sys, is_binary = _systems(ic)
    ext = _extent(ic)
    ax.set_facecolor("#0d0d12")

    # faint gas-column contours behind (same style as the mass figure)
    lc = np.log10(np.maximum(cloud_col, cloud_col[cloud_col > 0].min()))
    xs = np.linspace(ext[0], ext[1], cloud_col.shape[0])
    ys = np.linspace(ext[2], ext[3], cloud_col.shape[1])
    ax.contour(xs, ys, lc.T, levels=np.percentile(lc, [80, 90, 96, 99]),
               colors="#8ea3bf", alpha=0.28, linewidths=0.5)

    rad = np.asarray(zams_radius(jnp.clip(jnp.asarray(m_sys), 0.08, 150.0)))
    size = 5.0 + 11.0 * np.sqrt(rad)
    sing, binr = ~is_binary, is_binary
    # singles: faint grey haze
    ax.scatter(bary[sing, 0], bary[sing, 1], s=0.5 * size[sing], c="#9aa0ad",
               alpha=0.20, linewidths=0, rasterized=True)
    # binaries: bright cyan, thin white edge — the eye tracks these clustering into the gas
    ax.scatter(bary[binr, 0], bary[binr, 1], s=size[binr], c="#31e7d6", alpha=0.85,
               edgecolors="#eafffb", linewidths=0.3, rasterized=True)

    rs = _binary_density_spearman(ic, bary, is_binary)
    f_lo, f_hi = _fbin_split(ic, bary, is_binary)
    ax.text(0.045, 0.955,
            rf"$\rho_S(\mathrm{{bin}},\rho_{{\rm gas}})={rs:+.2f}$" + "\n"
            rf"$f_{{\rm bin}}$: {f_lo:.2f}$\to${f_hi:.2f} (low$\to$high $\rho$)",
            transform=ax.transAxes, va="top", ha="left", color="white", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="#0d0d12", ec="#556", alpha=0.85))
    if cbar:  # a proxy handle so the shared legend/label reads
        ax.scatter([], [], s=40, c="#31e7d6", edgecolors="#eafffb", linewidths=0.3,
                   label="primordial binary")
        ax.scatter([], [], s=20, c="#9aa0ad", alpha=0.5, label="single star")
        ax.legend(loc="lower right", fontsize=9.5, framealpha=0.85, facecolor="#0d0d12",
                  labelcolor="white")
    _finish(ax, title, ext)


def _fbin_split(ic, bary, is_binary):
    origin = np.asarray(ic.ledger.frame.origin)
    cell = np.clip(np.floor((bary + origin) / BOX * NGRID).astype(int), 0, NGRID - 1)
    rho = np.asarray(ic.gas.rho_cloud)
    local = rho[cell[:, 0], cell[:, 1], cell[:, 2]]
    med = np.median(local)
    return is_binary[local <= med].mean(), is_binary[local > med].mean()


def main():
    ic0 = build(0.0)
    ic6 = build(0.6)
    ext = _extent(ic6)
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.4), layout="constrained")
    _, cloud_col = _cloud_panel(axes[0], ic6, ext, title="(a) Parent cloud + stars")
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

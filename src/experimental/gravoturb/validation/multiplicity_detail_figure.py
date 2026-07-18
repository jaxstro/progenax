"""Detailed λ_mult diagnostics (extension C) — the 'more detail for me' companion to the money figure.

Four panels quantifying the environment-coupled-multiplicity knob on the gravoturb cluster:
  (a) the controlled knob: ρ_S(is_binary, ρ_gas) vs λ_mult, with the EMERGENT (λ_mult=0) baseline
      from the mass channel marked — λ_mult is a controlled departure from a measured baseline;
  (b) f_bin vs local-density tertile for a few λ_mult (the coupling strengthening);
  (c) orbital-compactness coupling: median binary component separation vs local density
      (λ_mult=0 vs 0.8) — denser → tighter (the P/q signature, via resolved separations);
  (d) binary radial profile: f_bin vs cluster-centric radius (λ_mult=0 vs 0.6) — binaries sink in.

Run:
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
      PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync \
      python -m gravoturb.validation.multiplicity_detail_figure

Writes plots/feasibility/gravoturb_multiplicity_detail.png. Diagnostic/figure script (LOC-exempt).
"""
import os

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from jaxstro.units import STELLAR
from gravoturb.cluster import build_cluster_ic
from gravoturb.specs import CloudSpec, CompositionSpec, GasSpec, GeometrySpec, VelocitySpec
from progenax import Maschberger, PlummerProfile
from progenax.binaries.companions import MoeCompanions

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "plots", "feasibility")
os.makedirs(OUT, exist_ok=True)
N, BOX, NGRID = 4000, 6.0, 48
IMF = Maschberger()
sns.set_theme(style="ticks", context="paper", font="serif")


def build(lambda_mult):
    masses = IMF.sample(jax.random.PRNGKey(9), N)
    return build_cluster_ic(
        masses,
        cloud=CloudSpec(mach=8.0, b=0.5, alpha=1.8, beta=3.0),
        geometry=GeometrySpec(profile=PlummerProfile(r_h=1.0), box_size=BOX, shape=(NGRID,) * 3),
        velocity=VelocitySpec(beta_v=4.0, mode="physical", c_s=0.2),
        composition=CompositionSpec(lambda_corr=0.5, lambda_mult=lambda_mult,
                                    companions=MoeCompanions()),
        G=STELLAR.G, units=STELLAR, key=jax.random.PRNGKey(1), gas=GasSpec(sfe=0.2),
    )


def _binary_systems(ic):
    """Per binary system: barycenter, local gas density, component separation [pc]. Plus per-system
    (is_binary, barycenter, local density) for f_bin stats."""
    sid = np.asarray(ic.stars.system_id)
    pos = np.asarray(ic.stars.positions)
    order = np.argsort(sid, kind="stable")
    sid, pos = sid[order], pos[order]
    _, idx, counts = np.unique(sid, return_index=True, return_counts=True)
    is_binary = counts == 2
    bary = np.array([pos[i:i + c].mean(axis=0) for i, c in zip(idx, counts)])
    sep = np.array([np.linalg.norm(pos[i] - pos[i + 1]) if c == 2 else np.nan
                    for i, c in zip(idx, counts)])
    origin = np.asarray(ic.ledger.frame.origin)
    cell = np.clip(np.floor((bary + origin) / BOX * NGRID).astype(int), 0, NGRID - 1)
    rho = np.asarray(ic.gas.rho_cloud)
    local = rho[cell[:, 0], cell[:, 1], cell[:, 2]]
    radius = np.linalg.norm(bary, axis=1)
    return is_binary, local, sep, radius


def _tertile_fbin(is_binary, local):
    lo, hi = np.percentile(np.log(local), [33.33, 66.67])
    ln = np.log(local)
    return [is_binary[ln <= lo].mean(),
            is_binary[(ln > lo) & (ln <= hi)].mean(),
            is_binary[ln > hi].mean()]


def main():
    lams = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    data = {lm: _binary_systems(build(lm)) for lm in lams}

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))
    fig.suptitle(r"Environment-coupled multiplicity $\lambda_{\rm mult}$ — diagnostics "
                 r"($\lambda_{\rm corr}=0.5$ fixed, Maschberger IMF + Moe binaries)", fontsize=13)

    # (a) controlled knob: rho_S(bin, rho) vs lambda_mult
    ax = axes[0, 0]
    rs = [spearmanr(data[lm][0].astype(float), data[lm][1]).statistic for lm in lams]
    ax.plot(lams, rs, "-o", color="C0")
    ax.axhline(rs[0], color="0.5", ls="--", lw=1,
               label=rf"emergent baseline ($\lambda_{{\rm mult}}=0$): {rs[0]:+.2f}")
    ax.set(xlabel=r"$\lambda_{\rm mult}$", ylabel=r"$\rho_S(\mathrm{binary},\,\rho_{\rm gas})$",
           title="(a) controlled knob: coupling vs strength")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)

    # (b) f_bin vs density tertile
    ax = axes[0, 1]
    for lm, c in zip([0.0, 0.5 if 0.5 in data else 0.4, 1.0], ["C3", "C1", "C0"]):
        if lm not in data:
            continue
        ax.plot([0, 1, 2], _tertile_fbin(data[lm][0], data[lm][1]), "-o",
                label=rf"$\lambda_{{\rm mult}}={lm}$", color=c)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["low", "mid", "high"])
    ax.set(xlabel=r"local gas density tertile", ylabel=r"$f_{\rm bin}$",
           title="(b) binary fraction vs density")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)

    # (c) orbital-compactness coupling: median separation vs density (equal-count bins, AU)
    ax = axes[1, 0]
    _PC_TO_AU = 206264.806
    for lm, c in zip([0.0, 0.8], ["C3", "C0"]):
        isb, local, sep, _ = data[lm]
        m = isb & np.isfinite(sep)
        ln = np.log10(local[m])
        sep_au = sep[m] * _PC_TO_AU
        o = np.argsort(ln)
        ln, sep_au = ln[o], sep_au[o]
        nb = 6
        edges = np.linspace(0, len(ln), nb + 1).astype(int)
        cen = [np.median(ln[a:b]) for a, b in zip(edges[:-1], edges[1:]) if b > a]
        med = [np.median(sep_au[a:b]) for a, b in zip(edges[:-1], edges[1:]) if b > a]
        ax.plot(cen, med, "-o", label=rf"$\lambda_{{\rm mult}}={lm}$", color=c)
    ax.set(xlabel=r"$\log_{10}\,\rho_{\rm gas,local}$",
           ylabel="median component sep. [AU]",
           title="(c) orbital compactness: denser $\\to$ tighter")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)

    # (d) binary radial profile
    ax = axes[1, 1]
    for lm, c in zip([0.0, 0.6], ["C3", "C0"]):
        isb, _, _, radius = data[lm]
        bins = np.linspace(0, np.percentile(radius, 95), 8)
        idx = np.clip(np.digitize(radius, bins) - 1, 0, len(bins) - 2)
        cen = 0.5 * (bins[:-1] + bins[1:])
        fb = [isb[idx == k].mean() if np.any(idx == k) else np.nan for k in range(len(cen))]
        ax.plot(cen, fb, "-o", label=rf"$\lambda_{{\rm mult}}={lm}$", color=c)
    ax.set(xlabel="cluster-centric radius [pc]", ylabel=r"$f_{\rm bin}$",
           title="(d) binary radial profile (binaries sink in)")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT, "gravoturb_multiplicity_detail.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print("rho_S(bin,rho) vs lambda_mult:", {lm: round(r, 3) for lm, r in zip(lams, rs)})


if __name__ == "__main__":
    main()

"""CAREER Aim-3 demo figure: 'the cluster you see depends on the gas you don't'.

gravoturb emits a natal residual-gas grid; ``GravoturbDustModel`` turns it into a physical,
spatially-correlated differential-extinction screen (star-embedded LOS depth), and fluxax renders the
same cluster through a series of passbands. Optical light from the deeply embedded stars is
extinguished by their own natal gas; the near-IR reveals them. This is the Aim-3 question made
visual: *which observation recovers the embedded population through the reddening screen?*

The figure is driven by the *differential extinction per band* — the physical quantity extension A
produces. fluxax's ``a_band`` (Fitzpatrick-99) turns each star's A_V column into a per-band A_λ; a
star is "recovered" in band b if its extinction there stays under a fixed survey margin (holding
intrinsic luminosity fixed — the natal gas is the only thing hiding it). A common 5000 K SED is used
for the extinction (constant-treatment; A_λ/A_V is only weakly Teff-dependent in this regime).

Panels
------
(top)  the cluster on the sky in LSST g, y and 2MASS Ks — point size ∝ recoverability; stars whose
       band extinction exceeds the survey margin are greyed 'lost'. Embedded stars vanish in g,
       emerge in Ks.
(bottom-left)  per-band recovery curve: recovered fraction vs effective wavelength (g→Ks).
(bottom-mid)   the differential-extinction screen: A_V vs LOS depth (front stars light, back heavy).
(bottom-right) the metallicity memory imprint: A_V distribution at solar vs a low-Z birth (same gas).

Run:
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
      PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_extinction_bands.py

Writes src/experimental/gravoturb/validation/plots/extinction_bands.png. Demo script (LOC-exempt).
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

from jaxstro.units import STELLAR
from gravoturb.cluster import build_cluster_ic
from gravoturb.extinction import GravoturbDustModel
from gravoturb.specs import CloudSpec, CompositionSpec, GasSpec, GeometrySpec, VelocitySpec
from progenax import PlummerProfile
from progenax.imf import BirthEnvironment
from fluxax.photometry.extinction import F99_TEFF

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "..", "src", "experimental", "gravoturb", "validation", "plots")
os.makedirs(PLOTS, exist_ok=True)

BOX, SHAPE = 6.0, (32, 32, 32)
MACH, B, ALPHA, BETA_V = 8.0, 0.5, 1.8, 4.0
N, R_H, SFE = 1000, 0.7, 0.25   # a realistically deeply-embedded cluster: median A_V ~ 20 mag
T_EFF = 5000.0        # representative SED for the (weakly Teff-dependent) band extinction
SURVEY_MARGIN = 5.0   # mag of band extinction a survey tolerates before a star is 'lost'
# (band, system, effective wavelength [um]): LSST grizy + 2MASS JHKs
_BANDS = [("g", "lsst", 0.48), ("r", "lsst", 0.62), ("i", "lsst", 0.75), ("z", "lsst", 0.87),
          ("y", "lsst", 1.02), ("J", "johnson", 1.24), ("H", "johnson", 1.66), ("K", "johnson", 2.20)]


def _build_ic(seed: int = 0):
    return build_cluster_ic(
        jnp.ones(N),
        cloud=CloudSpec(mach=MACH, b=B, alpha=ALPHA, beta=3.0),
        geometry=GeometrySpec(profile=PlummerProfile(r_h=R_H), box_size=BOX, shape=SHAPE),
        velocity=VelocitySpec(beta_v=BETA_V, mode="physical", c_s=0.2),
        composition=CompositionSpec(placement="two_population", f_sub=0.3),
        G=STELLAR.G, units=STELLAR, key=jax.random.PRNGKey(seed), gas=GasSpec(sfe=SFE),
    )


def _a_band(av, system, band):
    """Per-star band extinction A_λ [mag] from the A_V column (vectorized over stars, fixed SED)."""
    fn = lambda a0: F99_TEFF.a_band(T_EFF, a0, system, band)
    return np.asarray(jax.vmap(fn)(jnp.asarray(av)))


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ic = _build_ic()
    pos = ic.stars.positions
    dust = GravoturbDustModel.from_ic(ic, env=BirthEnvironment.solar())
    av = np.asarray(dust.column(pos))

    # per-band extinction A_λ for every star, and recovery (extinction under the survey margin)
    a_lambda = {b: _a_band(av, system, b) for (b, system, _) in _BANDS}
    detected = {b: a_lambda[b] < SURVEY_MARGIN for b in a_lambda}
    frac = {b: detected[b].mean() for b in a_lambda}

    # small-angle sky positions from the transverse coordinates (LOS = z is not on-sky)
    ra = np.asarray(pos[:, 0])
    dec = np.asarray(pos[:, 1])

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.32, wspace=0.28)

    # --- top row: sky in g, y, Ks ---
    for col, band in enumerate(["g", "y", "K"]):
        ax = fig.add_subplot(gs[0, col])
        det = detected[band]
        ax.scatter(ra[~det], dec[~det], s=4, c="0.85", marker=".", linewidths=0, zorder=1)
        sc = ax.scatter(ra[det], dec[det], s=14, c=a_lambda[band][det], cmap="inferno_r",
                        vmin=0, vmax=SURVEY_MARGIN, linewidths=0, zorder=2)
        med = np.median(a_lambda[band])
        ax.set_title(f"{band}-band  ({100*frac[band]:.0f}% recovered, med $A_{{{band}}}$={med:.1f})",
                     fontsize=11)
        ax.set_xlabel("x [pc]"); ax.set_aspect("equal")
        if col == 0:
            ax.set_ylabel("y [pc]")
    cbar = fig.colorbar(sc, ax=fig.axes[:3], fraction=0.025, pad=0.01)
    cbar.set_label(r"band extinction $A_\lambda$ [mag]")

    # --- bottom-left: recovery curve vs wavelength ---
    ax = fig.add_subplot(gs[1, 0])
    xs = [lam for (_, _, lam) in _BANDS]
    ys = [100 * frac[b] for (b, _, _) in _BANDS]
    ax.plot(xs, ys, "-o", color="C3")
    for (b, _, lam), yy in zip(_BANDS, ys):
        ax.annotate(b, (lam, yy), textcoords="offset points", xytext=(0, 6), fontsize=8, ha="center")
    ax.set_xscale("log")
    ax.set_xlabel(r"effective wavelength [$\mu$m]"); ax.set_ylabel("recovered fraction [%]")
    ax.set_title("recovery vs band", fontsize=11); ax.grid(alpha=0.3)

    # --- bottom-mid: differential extinction vs LOS depth ---
    ax = fig.add_subplot(gs[1, 1])
    zlos = np.asarray(pos[:, 2])
    ax.scatter(zlos, av, s=6, c=av, cmap="inferno_r", vmax=np.percentile(av, 98), linewidths=0)
    ax.set_xlabel("LOS position z [pc]  (observer at −z)"); ax.set_ylabel(r"$A_V$ [mag]")
    ax.set_title("star-embedded reddening", fontsize=11)

    # --- bottom-right: metallicity memory imprint ---
    ax = fig.add_subplot(gs[1, 2])
    av_poor = np.asarray(
        GravoturbDustModel.from_ic(ic, env=BirthEnvironment.from_cluster_mass(1e4, FeH=-1.5)).column(pos)
    )
    bins = np.linspace(0, np.percentile(av, 99), 40)
    ax.hist(av, bins=bins, color="C0", alpha=0.7, label="solar birth")
    ax.hist(av_poor, bins=bins, color="C1", alpha=0.7, label="[Fe/H]=−1.5 birth")
    ax.set_xlabel(r"$A_V$ [mag]"); ax.set_ylabel("stars")
    ax.set_title("metallicity memory (same gas)", fontsize=11); ax.legend(fontsize=8)

    fig.suptitle("The cluster you see depends on the gas you don't — and which colours recover it",
                 fontsize=13, y=0.98)
    out = os.path.abspath(os.path.join(PLOTS, "extinction_bands.png"))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    print("recovered fraction by band:", {b: round(frac[b], 3) for (b, _, _) in _BANDS})
    print("median A_lambda by band:", {b: round(float(np.median(a_lambda[b])), 2) for (b, _, _) in _BANDS})
    print(f"A_V solar: med={np.median(av):.2f} max={av.max():.1f} mag; "
          f"[Fe/H]=-1.5: med={np.median(av_poor):.3f} mag")


if __name__ == "__main__":
    main()

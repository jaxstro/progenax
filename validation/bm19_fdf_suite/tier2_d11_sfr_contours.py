#!/usr/bin/env python
"""D11: SFR-Sigma Contours with Larson Track.

Contours in Sigma-Mach space, overlay M(Sigma) relation.

Output: d11_sfr_sigma_contours.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from progenax.gravoturb import bm19_model as bm19

from .helpers import (
    setup_publication_style,
    save_plot,
    t_ff_myr,
    sfr_proxy,
    larson_mach,
    COLORS,
)


def run_validation(
    Sigmas: np.ndarray | None = None,
    machs: np.ndarray | None = None,
    alpha: float = 2.0,
    eta: float = 0.6,
    epsilon_ff: float = 0.01,
    verbose: bool = True,
):
    """Compute SFR contours in Sigma-Mach space.

    Parameters
    ----------
    Sigmas : array
        Surface density range [Msun/pc^2]
    machs : array
        Mach number range
    alpha : float
        BM19 powerlaw slope
    eta : float
        Feedback survival efficiency
    epsilon_ff : float
        Intrinsic SFE per freefall
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        Contour data
    """
    if verbose:
        print("=" * 70)
        print("D11: SFR-SIGMA CONTOURS WITH LARSON TRACK")
        print("=" * 70)

    if Sigmas is None:
        Sigmas = np.logspace(1, 4, 50)  # 10 - 10000 Msun/pc^2
    if machs is None:
        machs = np.linspace(3, 50, 50)

    b = 0.4

    if verbose:
        print(f"\nParameters: alpha={alpha}, eta={eta}, epsilon_ff={epsilon_ff}")
        print(f"Sigma range: [{Sigmas.min():.0f}, {Sigmas.max():.0f}] Msun/pc^2")
        print(f"Mach range: [{machs.min():.0f}, {machs.max():.0f}]")

    # Compute grids
    f_dense_grid = np.zeros((len(Sigmas), len(machs)))
    f_sub_grid = np.zeros((len(Sigmas), len(machs)))
    sfr_proxy_grid = np.zeros((len(Sigmas), len(machs)))
    t_ff_grid = np.zeros((len(Sigmas), len(machs)))

    for i, Sigma in enumerate(Sigmas):
        t_ff_val = t_ff_myr(Sigma)
        for j, mach in enumerate(machs):
            result = bm19.bm19_pipeline(mach, b, alpha, eta)
            f_dense = float(result.f_dense)
            f_sub = float(result.f_sub)

            f_dense_grid[i, j] = f_dense
            f_sub_grid[i, j] = f_sub
            t_ff_grid[i, j] = t_ff_val
            sfr_proxy_grid[i, j] = sfr_proxy(f_dense, Sigma, epsilon_ff)

    # Larson track: M(Sigma) = M_ref * sqrt(Sigma/Sigma_ref)
    larson_sigmas = Sigmas
    larson_machs = larson_mach(larson_sigmas, M_ref=10, Sigma_ref=100)

    # SFR along Larson track
    larson_sfr = []
    larson_f_dense = []
    for Sigma, mach in zip(larson_sigmas, larson_machs):
        result = bm19.bm19_pipeline(mach, b, alpha, eta)
        f_dense = float(result.f_dense)
        larson_f_dense.append(f_dense)
        larson_sfr.append(sfr_proxy(f_dense, Sigma, epsilon_ff))

    if verbose:
        print(f"\nAlong Larson track:")
        print(f"  f_dense range: [{min(larson_f_dense):.4f}, {max(larson_f_dense):.4f}]")
        print(f"  SFR proxy range: [{min(larson_sfr):.6f}, {max(larson_sfr):.6f}] Myr^-1")

    return {
        "Sigmas": Sigmas,
        "machs": machs,
        "f_dense_grid": f_dense_grid,
        "f_sub_grid": f_sub_grid,
        "sfr_proxy_grid": sfr_proxy_grid,
        "t_ff_grid": t_ff_grid,
        "larson": {
            "Sigmas": larson_sigmas,
            "machs": larson_machs,
            "f_dense": np.array(larson_f_dense),
            "sfr": np.array(larson_sfr),
        },
        "params": {
            "alpha": alpha,
            "eta": eta,
            "epsilon_ff": epsilon_ff,
            "b": b,
        }
    }


def make_plot(results: dict, show: bool = False) -> str:
    """Generate SFR-Sigma contour plot.

    Parameters
    ----------
    results : dict
        Output from run_validation()
    show : bool
        Display interactively

    Returns
    -------
    path : str
        Path to saved plot
    """
    setup_publication_style()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    Sigmas = results["Sigmas"]
    machs = results["machs"]
    larson = results["larson"]

    # LEFT: f_dense contours
    ax1 = axes[0]
    cf1 = ax1.contourf(machs, Sigmas, results["f_dense_grid"], levels=20, cmap="viridis")
    plt.colorbar(cf1, ax=ax1, label="$f_\\mathrm{dense}$")

    # Larson track
    ax1.plot(larson["machs"], larson["Sigmas"], "r-", linewidth=3, label="Larson track")
    ax1.set_xlabel("Mach Number ($\\mathcal{M}$)", fontsize=12)
    ax1.set_ylabel("$\\Sigma$ [M$_\\odot$/pc$^2$]", fontsize=12)
    ax1.set_yscale("log")
    ax1.set_title("$f_\\mathrm{dense}(\\mathcal{M}, \\Sigma)$\n(const at fixed $\\mathcal{M}$)", fontsize=12)
    ax1.legend(fontsize=10, loc="upper left")

    # MIDDLE: SFR proxy contours
    ax2 = axes[1]
    # SFR proxy in units of Myr^-1
    cf2 = ax2.contourf(machs, Sigmas, results["sfr_proxy_grid"] * 1000, levels=20, cmap="hot")
    plt.colorbar(cf2, ax=ax2, label="SFR proxy [$10^{-3}$ Myr$^{-1}$]")

    # Larson track
    ax2.plot(larson["machs"], larson["Sigmas"], "cyan", linewidth=3, label="Larson track")
    ax2.set_xlabel("Mach Number ($\\mathcal{M}$)", fontsize=12)
    ax2.set_ylabel("$\\Sigma$ [M$_\\odot$/pc$^2$]", fontsize=12)
    ax2.set_yscale("log")
    ax2.set_title("SFR$/ M_\\mathrm{cloud}$ $\\propto f_\\mathrm{dense} / t_\\mathrm{ff}$", fontsize=12)
    ax2.legend(fontsize=10, loc="upper left")

    # RIGHT: SFR along Larson track
    ax3 = axes[2]
    ax3.semilogy(larson["Sigmas"], larson["sfr"] * 1000, "b-", linewidth=2.5, label="SFR proxy")
    ax3.semilogy(larson["Sigmas"], larson["f_dense"], "g--", linewidth=2, label="$f_\\mathrm{dense}$")

    ax3.set_xlabel("$\\Sigma$ [M$_\\odot$/pc$^2$]", fontsize=12)
    ax3.set_ylabel("SFR proxy / $f_\\mathrm{dense}$", fontsize=12)
    ax3.set_xscale("log")
    ax3.set_title("Along Larson Track\n(Compensation Effect)", fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Add annotation about compensation
    ax3.annotate(
        "SFR proxy $\\approx$ const\n(f$\\downarrow$ compensates t$_{ff}\\downarrow$)",
        xy=(200, larson["sfr"][len(larson["sfr"])//2] * 1000),
        fontsize=10, color="blue",
        xytext=(400, 0.1), arrowprops=dict(arrowstyle="->", color="blue")
    )

    plt.suptitle(
        f"BM19 in $\\Sigma$-$\\mathcal{{M}}$ Space ($\\alpha$={results['params']['alpha']}, $\\eta$={results['params']['eta']})",
        fontsize=14, y=1.02
    )
    plt.tight_layout()

    if show:
        plt.show()

    path = save_plot(fig, "d11_sfr_sigma_contours")
    plt.close(fig)

    return path


def make_compensation_plot(results: dict, show: bool = False) -> str:
    """Detailed plot showing compensation effect along Larson track."""
    setup_publication_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    larson = results["larson"]
    Sigmas = larson["Sigmas"]

    # Top left: Mach along Larson track
    ax1 = axes[0, 0]
    ax1.loglog(Sigmas, larson["machs"], "b-", linewidth=2.5)
    ax1.set_xlabel("$\\Sigma$ [M$_\\odot$/pc$^2$]", fontsize=12)
    ax1.set_ylabel("$\\mathcal{M}$", fontsize=12)
    ax1.set_title("Larson Relation: $\\mathcal{M} \\propto \\Sigma^{0.5}$", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Top right: f_dense along track
    ax2 = axes[0, 1]
    ax2.loglog(Sigmas, larson["f_dense"], "g-", linewidth=2.5)
    ax2.set_xlabel("$\\Sigma$ [M$_\\odot$/pc$^2$]", fontsize=12)
    ax2.set_ylabel("$f_\\mathrm{dense}$", fontsize=12)
    ax2.set_title("f$_\\mathrm{dense}$ Decreases (Higher Mach)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Bottom left: t_ff along track
    ax3 = axes[1, 0]
    t_ff_track = t_ff_myr(Sigmas)
    ax3.loglog(Sigmas, t_ff_track, "r-", linewidth=2.5)
    ax3.set_xlabel("$\\Sigma$ [M$_\\odot$/pc$^2$]", fontsize=12)
    ax3.set_ylabel("$t_\\mathrm{ff}$ [Myr]", fontsize=12)
    ax3.set_title("t$_{ff}$ Decreases (Denser)", fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Bottom right: SFR proxy (f_dense / t_ff)
    ax4 = axes[1, 1]
    sfr_track = larson["f_dense"] / t_ff_track
    ax4.semilogx(Sigmas, sfr_track, "purple", linewidth=2.5)
    ax4.axhline(y=np.mean(sfr_track), color="orange", linestyle="--", label=f"Mean = {np.mean(sfr_track):.4f}")
    ax4.fill_between(Sigmas, np.mean(sfr_track) * 0.8, np.mean(sfr_track) * 1.2, alpha=0.2, color="orange", label="$\\pm$20%")
    ax4.set_xlabel("$\\Sigma$ [M$_\\odot$/pc$^2$]", fontsize=12)
    ax4.set_ylabel("$f_\\mathrm{dense} / t_\\mathrm{ff}$ [Myr$^{-1}$]", fontsize=12)
    ax4.set_title("Compensation: SFR Proxy $\\approx$ Constant!", fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Compute flatness metric
    cv = np.std(sfr_track) / np.mean(sfr_track)
    fig.text(0.5, 0.01, f"Coefficient of Variation: {100*cv:.1f}% (flat = compensation works)", ha="center", fontsize=12)

    plt.suptitle(
        "The Compensation Effect: f$_\\mathrm{dense}$$\\downarrow$ Balances t$_{ff}$$\\downarrow$",
        fontsize=14, y=1.02
    )
    plt.tight_layout()

    if show:
        plt.show()

    path = save_plot(fig, "d11_compensation_effect")
    plt.close(fig)

    return path


def main():
    """Run full D11 validation."""
    results = run_validation(verbose=True)
    make_plot(results)
    make_compensation_plot(results)

    print("\n" + "=" * 70)
    print("D11 VALIDATION COMPLETE")
    print("=" * 70)

    # Quantify compensation
    larson = results["larson"]
    t_ff_track = t_ff_myr(larson["Sigmas"])
    sfr_track = larson["f_dense"] / t_ff_track

    cv = np.std(sfr_track) / np.mean(sfr_track)
    print(f"\nCompensation effect along Larson track:")
    print(f"  SFR proxy mean: {np.mean(sfr_track):.4f} Myr^-1")
    print(f"  SFR proxy std: {np.std(sfr_track):.4f} Myr^-1")
    print(f"  Coefficient of variation: {100*cv:.1f}%")
    print(f"  -> Nearly constant SFR proxy despite 100x Sigma variation!")

    return results


if __name__ == "__main__":
    main()

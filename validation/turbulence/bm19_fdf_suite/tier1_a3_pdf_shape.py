#!/usr/bin/env python
"""A3: PDF Shape with s_t — INTUITIVE VISUALIZATION.

3 panels (M=5,15,30), histogram of s=ln(rho/rho_mean), overlay BM19 PDF, mark s_t.

Makes BM19 tangible, visual check of tail slope.

Output: a3_pdf_shape.png
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

from progenax.gravoturb import bm19_model as bm19
from progenax.gravoturb import gaussian_to_bm19, build_bm19_cdf_table

from .helpers import (
    setup_publication_style,
    save_plot,
    COLORS,
)


def lognormal_pdf(s, sigma_s_sq):
    """Theoretical lognormal PDF for s = ln(rho/rho_mean).

    p(s) = (1 / sqrt(2*pi*sigma_s^2)) * exp(-(s + sigma_s^2/2)^2 / (2*sigma_s^2))

    Note: s_0 = -sigma_s^2/2 for mass conservation.
    """
    sigma_s = np.sqrt(sigma_s_sq)
    s_0 = -sigma_s_sq / 2
    return scipy_stats.norm.pdf(s, loc=s_0, scale=sigma_s)


def bm19_piecewise_pdf(s, sigma_s_sq, s_t, alpha):
    """BM19 piecewise lognormal + powerlaw PDF.

    For s < s_t: lognormal
    For s >= s_t: powerlaw with continuity at s_t

    This is for visualization only (not normalized for mass-weighting).
    """
    sigma_s = np.sqrt(sigma_s_sq)
    s_0 = -sigma_s_sq / 2

    # Lognormal part
    p_ln = scipy_stats.norm.pdf(s, loc=s_0, scale=sigma_s)

    # Powerlaw normalization: p_PL(s_t) = p_LN(s_t)
    p_ln_at_st = scipy_stats.norm.pdf(s_t, loc=s_0, scale=sigma_s)
    A = p_ln_at_st * np.exp(alpha * s_t)

    # Powerlaw part: p_PL(s) = A * exp(-alpha * s)
    p_pl = A * np.exp(-alpha * s)

    # Piecewise combination
    return np.where(s < s_t, p_ln, p_pl)


def run_validation(
    machs: list[float] = [5.0, 15.0, 30.0],
    alphas: list[float] = [1.5, 2.0, 2.5],
    b: float = 0.4,
    grid_size: int = 128,
    verbose: bool = True,
):
    """Generate PDF visualizations for multiple Mach and alpha values.

    Parameters
    ----------
    machs : list
        Mach numbers to visualize (columns)
    alphas : list
        BM19 powerlaw slopes to visualize (rows)
    b : float
        Driving parameter
    grid_size : int
        Grid resolution for field generation
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        Histograms and theory curves indexed by (alpha, mach)
    """
    if verbose:
        print("=" * 70)
        print("A3: PDF SHAPE VISUALIZATION")
        print("=" * 70)
        print(f"\nParameters: alphas={alphas}, machs={machs}, b={b}, grid={grid_size}^3")

    results_by_alpha_mach = {}

    for alpha in alphas:
        for mach in machs:
            if verbose:
                print(f"\nalpha = {alpha}, Mach = {mach}")

            # BM19 theory
            bm19_result = bm19.bm19_pipeline(mach, b, alpha, eta_survive=0.6)
            sigma_s = float(bm19_result.sigma_s)
            sigma_s_sq = float(bm19_result.sigma_s_sq)
            s_t = float(bm19_result.s_t)
            f_dense = float(bm19_result.f_dense)

            if verbose:
                print(f"  sigma_s = {sigma_s:.3f}, s_t = {s_t:.3f}, f_dense = {f_dense:.4f}")

            # Generate field with exact BM19 PDF via CDF remap
            key = random.PRNGKey(int(alpha * 1000 + mach * 100))
            g = random.normal(key, (grid_size, grid_size, grid_size))
            s_grid, F_grid = build_bm19_cdf_table(sigma_s_sq, s_t, alpha)
            s_field = gaussian_to_bm19(g, sigma_s_sq, s_t, alpha, s_grid, F_grid)
            s_flat = np.array(s_field.flatten())

            # Histogram - expand range to show full tail
            s_min = min(s_flat.min(), -6 * sigma_s) - 0.5
            s_max = max(s_flat.max(), s_t + 5) + 0.5
            n_bins = 100
            s_range = (s_min, s_max)
            hist, bin_edges = np.histogram(s_flat, bins=n_bins, range=s_range, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Theory curves - extend to show full tail
            s_theory = np.linspace(s_range[0], s_range[1], 500)
            p_lognormal = lognormal_pdf(s_theory, sigma_s_sq)
            p_bm19 = bm19_piecewise_pdf(s_theory, sigma_s_sq, s_t, alpha)

            results_by_alpha_mach[(alpha, mach)] = {
                "s_flat": s_flat,
                "hist": hist,
                "bin_centers": bin_centers,
                "s_theory": s_theory,
                "p_lognormal": p_lognormal,
                "p_bm19": p_bm19,
                "sigma_s": sigma_s,
                "sigma_s_sq": sigma_s_sq,
                "s_t": s_t,
                "f_dense": f_dense,
                "alpha": alpha,
                "mach": mach,
            }

    return {
        "by_alpha_mach": results_by_alpha_mach,
        "params": {
            "machs": machs,
            "alphas": alphas,
            "b": b,
            "grid_size": grid_size,
        }
    }


def make_plot(results: dict, show: bool = False) -> str:
    """Generate PDF shape visualization: α rows × Mach columns, log scale.

    3×3 grid layout:
    - Rows: Different α values (1.5, 2.0, 2.5)
    - Columns: Different Mach values (5, 15, 30)
    - All panels: Log scale to show powerlaw tail

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

    machs = results["params"]["machs"]
    alphas = results["params"]["alphas"]
    b = results["params"]["b"]
    n_rows = len(alphas)
    n_cols = len(machs)

    # Create grid: rows = α, cols = Mach
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    for row, alpha in enumerate(alphas):
        for col, mach in enumerate(machs):
            ax = axes[row, col]
            data = results["by_alpha_mach"][(alpha, mach)]
            s_t = data["s_t"]

            # Histogram - scatter for log scale clarity
            nonzero = data["hist"] > 0
            ax.scatter(
                data["bin_centers"][nonzero], data["hist"][nonzero],
                s=12, alpha=0.5, color="gray", label="3D FDF", zorder=1
            )

            # Lognormal theory
            ax.plot(
                data["s_theory"], data["p_lognormal"],
                color=COLORS["pn11"], linewidth=2, linestyle="--",
                label="Pure LN", zorder=2
            )

            # BM19 piecewise
            ax.plot(
                data["s_theory"], data["p_bm19"],
                color=COLORS["bm19"], linewidth=2.5,
                label="BM19 (LN+PL)", zorder=3
            )

            # Mark s_t
            ax.axvline(x=s_t, color="red", linestyle=":", linewidth=1.5,
                       label=f"$s_t$={s_t:.1f}", zorder=4)

            # Log scale settings
            ax.set_yscale("log")
            ax.set_ylim(1e-8, 1)
            ax.set_xlim(data["s_theory"].min(), data["s_theory"].max())
            ax.grid(True, alpha=0.3, which="both")

            # Labels
            if row == n_rows - 1:
                ax.set_xlabel("$s = \\ln(\\rho / \\bar{\\rho})$", fontsize=10)
            if col == 0:
                ax.set_ylabel("$p(s)$", fontsize=10)

            # Title: show key params
            ax.set_title(
                f"$\\mathcal{{M}}$={int(mach)}, $\\sigma_s$={data['sigma_s']:.2f}, "
                f"$f_{{dense}}$={data['f_dense']:.3f}",
                fontsize=9
            )

            # Legend only on first panel
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right")

        # Row label on left side
        axes[row, 0].annotate(
            f"$\\alpha$ = {alpha}",
            xy=(-0.25, 0.5), xycoords="axes fraction",
            fontsize=12, fontweight="bold",
            ha="center", va="center", rotation=90
        )

    # Column labels at top
    for col, mach in enumerate(machs):
        axes[0, col].annotate(
            f"$\\mathcal{{M}}$ = {int(mach)}",
            xy=(0.5, 1.15), xycoords="axes fraction",
            fontsize=12, fontweight="bold",
            ha="center", va="bottom"
        )

    plt.suptitle(
        f"BM19 Density PDF: Lognormal + Powerlaw Tail ($b$ = {b})",
        fontsize=14, y=0.98
    )
    plt.tight_layout(rect=[0.03, 0, 1, 0.95])

    if show:
        plt.show()

    path = save_plot(fig, "a3_pdf_shape")
    plt.close(fig)

    return path


def main():
    """Run full A3 validation."""
    results = run_validation(verbose=True)
    make_plot(results)

    print("\n" + "=" * 70)
    print("A3 VALIDATION COMPLETE")
    print("=" * 70)
    print("\nKey observations (s_t and f_dense by α and Mach):")
    print(f"{'α':<6} {'M':<6} {'s_t':<8} {'f_dense':<10}")
    print("-" * 32)
    for (alpha, mach), data in sorted(results["by_alpha_mach"].items()):
        print(f"{alpha:<6.1f} {int(mach):<6} {data['s_t']:<8.2f} {data['f_dense']:<10.4f}")

    return results


if __name__ == "__main__":
    main()

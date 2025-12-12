#!/usr/bin/env python
"""E3: Morphology Grid — β, χ → Q Parameter Mapping.

Maps FDF geometry knobs (β=power spectrum slope, χ=turbulence shaping)
to cluster Q-parameter (Cartwright & Whitworth morphology statistic).

Goal: Calibrate β(Mach, χ) to reproduce observed cluster Q ∈ [0.4, 0.8].

Output: e3_morphology.png

FULLY JAX-NATIVE: Single JIT compilation, vectorized over all configs.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt

from progenax.gravoturb import bm19_model as bm19
from progenax.cluster.fdf_density import (
    init_bm19_density_field,
    sample_positions_tail,
)
from progenax.diagnostics.q_approx import q_approx

from .helpers import (
    setup_publication_style,
    save_plot,
    COLORS,
)


class MorphologyResult(NamedTuple):
    """Result for a single (β, f_sub) configuration."""
    beta: float
    f_sub: float
    q_values: list[float]
    q_mean: float
    q_std: float


# =============================================================================
# Fully JAX-Native Q Computation
# =============================================================================


def _single_realization_q(
    key: jax.Array,
    beta: float,
    f_sub: float,
    sigma_s_sq: float,
    s_t: float,
    alpha: float,
    grid_size: int,
    n_stars: int,
    kappa: float,
) -> jax.Array:
    """Compute Q for a single realization (pure JAX).

    All parameters are concrete (not traced) for maximum JIT efficiency.
    """
    key_field, key_sample = random.split(key)

    field = init_bm19_density_field(
        key=key_field,
        sigma_s_sq=sigma_s_sq,
        s_t=s_t,
        alpha=alpha,
        grid_size=grid_size,
        box_half_size=1.0,
        beta=beta,
    )

    positions = sample_positions_tail(
        key=key_sample,
        field=field,
        N_stars=n_stars,
        f_sub=f_sub,
        mode="bm19",
        s_t=s_t,
        kappa=kappa,
    )

    return q_approx(positions, project_to_2d=True)


def _compute_q_for_config(
    keys: jax.Array,
    beta: float,
    f_sub: float,
    sigma_s_sq: float,
    s_t: float,
    alpha: float,
    grid_size: int,
    n_stars: int,
    kappa: float,
) -> jax.Array:
    """Compute Q values for all realizations of a single config.

    Uses jax.lax.map for memory-efficient sequential execution over realizations.

    Returns: Q_values[n_realizations]
    """
    return jax.lax.map(
        lambda k: _single_realization_q(
            k, beta, f_sub, sigma_s_sq, s_t, alpha, grid_size, n_stars, kappa
        ),
        keys
    )


def _make_compute_all_configs_fn(
    sigma_s_sq: float,
    s_t: float,
    alpha: float,
    grid_size: int,
    n_stars: int,
    kappa: float,
    betas: tuple[float, ...],
    f_subs: tuple[float, ...],
):
    """Create a JIT-compiled function that computes Q for ALL configs at once.

    This achieves single JIT compilation by:
    1. Embedding the config grid as static constants
    2. Using jax.lax.map over the flattened config grid

    Returns: fn(all_keys) -> Q_grid[n_configs, n_realizations]
    """
    n_betas = len(betas)
    n_f_subs = len(f_subs)
    n_configs = n_betas * n_f_subs

    # Convert to JAX arrays for indexing
    betas_arr = jnp.array(betas)
    f_subs_arr = jnp.array(f_subs)

    def compute_one_config(args):
        """Compute Q for config at flat index."""
        config_idx, keys = args
        # Decode flat index -> (beta_idx, f_sub_idx)
        beta_idx = config_idx // n_f_subs
        f_sub_idx = config_idx % n_f_subs
        beta = betas_arr[beta_idx]
        f_sub = f_subs_arr[f_sub_idx]

        # Compute Q for all realizations
        return jax.lax.map(
            lambda k: _single_realization_q(
                k, beta, f_sub, sigma_s_sq, s_t, alpha, grid_size, n_stars, kappa
            ),
            keys
        )

    @jax.jit
    def compute_all_configs(all_keys: jax.Array) -> jax.Array:
        """Compute Q for all configs.

        Args:
            all_keys: [n_configs, n_realizations, 2] PRNGKeys

        Returns:
            Q_grid: [n_configs, n_realizations]
        """
        config_indices = jnp.arange(n_configs)
        return jax.lax.map(
            compute_one_config,
            (config_indices, all_keys)
        )

    return compute_all_configs


def run_validation(
    betas: list[float] | None = None,
    f_subs: list[float] | None = None,
    mach: float = 10.0,
    alpha: float = 2.0,
    b: float = 0.4,
    n_stars: int = 1000,
    grid_size: int = 128,
    n_realizations: int = 10,
    kappa: float = 10.0,
    verbose: bool = True,
):
    """Sweep β and f_sub to measure Q-parameter.

    Parameters
    ----------
    betas : list[float], optional
        Power spectrum slopes to test (default: 3.4-4.0).
    f_subs : list[float], optional
        Dense tail fractions to test (default: 0.2-0.5).
    mach : float
        Fixed Mach number for BM19 PDF.
    alpha : float
        Fixed BM19 powerlaw slope.
    b : float
        Driving parameter.
    n_stars : int
        Number of stars to sample per realization.
    grid_size : int
        Density field grid resolution.
    n_realizations : int
        Realizations per configuration.
    kappa : float
        Sigmoid sharpness for tail selection.
    verbose : bool
        Print progress.

    Returns
    -------
    results : dict
        Results organized by (beta, f_sub).
    """
    # Defaults
    if betas is None:
        betas = [3.4, 3.5, 3.67, 3.8, 4.0]
    if f_subs is None:
        f_subs = [0.2, 0.3, 0.4, 0.5]

    # Convert to tuples for static embedding in JIT
    betas_tuple = tuple(betas)
    f_subs_tuple = tuple(f_subs)
    n_configs = len(betas) * len(f_subs)

    if verbose:
        print("=" * 70)
        print("E3: MORPHOLOGY GRID (β, f_sub → Q)")
        print("=" * 70)
        print(f"\nFixed BM19: Mach={mach}, α={alpha}, b={b}")
        print(f"Grid: {grid_size}³, N*={n_stars}, κ={kappa}")
        print(f"β ∈ {betas}")
        print(f"f_sub ∈ {f_subs}")
        print(f"Total configs: {n_configs}")
        print(f"Realizations per point: {n_realizations}")

    # Get BM19 parameters
    bm19_result = bm19.bm19_pipeline(mach, b, alpha, eta_survive=0.5)
    sigma_s_sq = float(bm19_result.sigma_s_sq)
    s_t = float(bm19_result.s_t)

    if verbose:
        print(f"\nBM19: σ_s²={sigma_s_sq:.4f}, s_t={s_t:.4f}")
        print("\nCompiling JAX function (single compilation for all configs)...")

    # Pre-generate all keys: [n_configs, n_realizations]
    master_key = random.PRNGKey(42)
    all_keys = random.split(master_key, n_configs * n_realizations)
    all_keys = all_keys.reshape(n_configs, n_realizations, 2)

    # Create the all-in-one JIT function
    compute_all_fn = _make_compute_all_configs_fn(
        sigma_s_sq=sigma_s_sq,
        s_t=s_t,
        alpha=alpha,
        grid_size=grid_size,
        n_stars=n_stars,
        kappa=kappa,
        betas=betas_tuple,
        f_subs=f_subs_tuple,
    )

    # Single JIT call for ALL configs
    t_start = time.time()
    Q_grid = compute_all_fn(all_keys)
    Q_grid = jax.block_until_ready(Q_grid)
    total_time = time.time() - t_start

    if verbose:
        print(f"\nTotal computation time: {total_time:.1f}s")
        print(f"Time per config: {total_time / n_configs:.2f}s")
        print(f"Time per realization: {total_time / (n_configs * n_realizations):.3f}s")

    # Unpack results into dictionary
    results_by_config = {}

    if verbose:
        print(f"\n{'β':>6} | {'f_sub':>6} | {'Q (mean±std)':>20}")
        print("-" * 45)

    config_idx = 0
    for beta_idx, beta in enumerate(betas):
        for f_sub_idx, f_sub in enumerate(f_subs):
            q_arr = Q_grid[config_idx]
            q_values = [float(q) for q in q_arr]
            q_mean = float(jnp.mean(q_arr))
            q_std = float(jnp.std(q_arr))

            result = MorphologyResult(
                beta=beta,
                f_sub=f_sub,
                q_values=q_values,
                q_mean=q_mean,
                q_std=q_std,
            )
            results_by_config[(beta, f_sub)] = result

            if verbose:
                print(f"{beta:>6.2f} | {f_sub:>6.2f} | {q_mean:>8.3f} ± {q_std:<8.3f}")

            config_idx += 1

    if verbose:
        print("-" * 45)

    return {
        "by_config": results_by_config,
        "params": {
            "betas": betas,
            "f_subs": f_subs,
            "mach": mach,
            "alpha": alpha,
            "b": b,
            "n_stars": n_stars,
            "grid_size": grid_size,
            "n_realizations": n_realizations,
            "kappa": kappa,
            "sigma_s_sq": sigma_s_sq,
            "s_t": s_t,
        }
    }


def make_plot(results: dict, show: bool = False) -> str:
    """Generate morphology grid plot.

    Parameters
    ----------
    results : dict
        Output from run_validation().
    show : bool
        Display interactively.

    Returns
    -------
    path : str
        Path to saved plot.
    """
    setup_publication_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    betas = results["params"]["betas"]
    f_subs = results["params"]["f_subs"]
    data = results["by_config"]

    # Color palette for f_sub values (use jnp for linspace)
    f_sub_colors = jnp.linspace(0.2, 0.9, len(f_subs))
    beta_colors = jnp.linspace(0.2, 0.9, len(betas))

    # LEFT: Q vs β (curves for different f_sub)
    ax1 = axes[0]

    for i, f_sub in enumerate(f_subs):
        q_means = [data[(b, f_sub)].q_mean for b in betas]
        q_stds = [data[(b, f_sub)].q_std for b in betas]

        ax1.errorbar(
            betas, q_means, yerr=q_stds,
            fmt="o-", color=plt.cm.viridis(float(f_sub_colors[i])),
            markersize=8, capsize=4,
            linewidth=2, label=f"$f_{{sub}}$={f_sub:.2f}"
        )

    # Observed Q range (Cartwright & Whitworth 2004)
    ax1.axhspan(0.4, 0.8, alpha=0.2, color="green", label="Observed range")
    ax1.axhline(y=0.79, color="gray", linestyle=":", linewidth=1.5, label="Uniform sphere")

    ax1.set_xlabel("Power Spectrum Slope $\\beta$", fontsize=12)
    ax1.set_ylabel("Q Parameter", fontsize=12)
    ax1.set_title("Cluster Morphology vs Power Spectrum", fontsize=14)
    ax1.legend(fontsize=10, loc="upper left")
    ax1.set_xlim(3.3, 4.1)
    ax1.set_ylim(0.2, 1.0)
    ax1.grid(True, alpha=0.3)

    # Add reference points for β
    ax1_twin = ax1.twiny()
    ax1_twin.set_xlim(ax1.get_xlim())
    ax1_twin.set_xticks([11/3, 4.0])
    ax1_twin.set_xticklabels(["Kolmogorov\n(11/3)", "Burgers\n(4)"], fontsize=9)

    # RIGHT: Q vs f_sub (curves for different β)
    ax2 = axes[1]

    for i, beta in enumerate(betas):
        q_means = [data[(beta, f)].q_mean for f in f_subs]
        q_stds = [data[(beta, f)].q_std for f in f_subs]

        ax2.errorbar(
            f_subs, q_means, yerr=q_stds,
            fmt="s-", color=plt.cm.plasma(float(beta_colors[i])),
            markersize=8, capsize=4,
            linewidth=2, label=f"$\\beta$={beta:.2f}"
        )

    # Observed Q range
    ax2.axhspan(0.4, 0.8, alpha=0.2, color="green", label="Observed range")
    ax2.axhline(y=0.79, color="gray", linestyle=":", linewidth=1.5, label="Uniform sphere")

    ax2.set_xlabel("Dense Tail Fraction $f_{sub}$", fontsize=12)
    ax2.set_ylabel("Q Parameter", fontsize=12)
    ax2.set_title("Cluster Morphology vs Substructure", fontsize=14)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.set_xlim(0.15, 0.55)
    ax2.set_ylim(0.2, 1.0)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"FDF Morphology Calibration: Q(β, $f_{{sub}}$)\n"
        f"(M={results['params']['mach']}, α={results['params']['alpha']}, "
        f"N*={results['params']['n_stars']}, {results['params']['n_realizations']} realizations/point)",
        fontsize=14, y=1.02
    )
    plt.tight_layout()

    if show:
        plt.show()

    path = save_plot(fig, "e3_morphology")
    plt.close(fig)

    return path


def make_heatmap(results: dict, show: bool = False) -> str:
    """Generate Q-parameter heatmap over (β, f_sub) grid.

    Parameters
    ----------
    results : dict
        Output from run_validation().
    show : bool
        Display interactively.

    Returns
    -------
    path : str
        Path to saved plot.
    """
    setup_publication_style()

    betas = results["params"]["betas"]
    f_subs = results["params"]["f_subs"]
    data = results["by_config"]

    # Build Q grid using JAX
    Q_grid = jnp.zeros((len(f_subs), len(betas)))
    for i, f_sub in enumerate(f_subs):
        for j, beta in enumerate(betas):
            Q_grid = Q_grid.at[i, j].set(data[(beta, f_sub)].q_mean)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Convert to numpy only for matplotlib imshow
    import numpy as np
    im = ax.imshow(
        np.asarray(Q_grid), origin="lower", aspect="auto",
        extent=[min(betas) - 0.05, max(betas) + 0.05,
                min(f_subs) - 0.025, max(f_subs) + 0.025],
        cmap="RdYlGn", vmin=0.3, vmax=0.9
    )

    # JAX-native bilinear interpolation for contours
    betas_arr = jnp.array(betas)
    f_subs_arr = jnp.array(f_subs)

    beta_fine = jnp.linspace(min(betas), max(betas), 50)
    f_sub_fine = jnp.linspace(min(f_subs), max(f_subs), 50)
    B, F = jnp.meshgrid(beta_fine, f_sub_fine)

    def bilinear_interp(beta_query, f_sub_query):
        """JAX-native bilinear interpolation."""
        beta_idx = (beta_query - betas_arr[0]) / (betas_arr[-1] - betas_arr[0]) * (len(betas) - 1)
        f_idx = (f_sub_query - f_subs_arr[0]) / (f_subs_arr[-1] - f_subs_arr[0]) * (len(f_subs) - 1)

        beta_idx = jnp.clip(beta_idx, 0, len(betas) - 1.001)
        f_idx = jnp.clip(f_idx, 0, len(f_subs) - 1.001)

        i0 = jnp.floor(f_idx).astype(jnp.int32)
        j0 = jnp.floor(beta_idx).astype(jnp.int32)
        i1 = jnp.minimum(i0 + 1, len(f_subs) - 1)
        j1 = jnp.minimum(j0 + 1, len(betas) - 1)

        fi = f_idx - i0
        fj = beta_idx - j0

        return (
            Q_grid[i0, j0] * (1 - fi) * (1 - fj) +
            Q_grid[i0, j1] * (1 - fi) * fj +
            Q_grid[i1, j0] * fi * (1 - fj) +
            Q_grid[i1, j1] * fi * fj
        )

    # Vectorize over grid
    Q_fine = jax.vmap(jax.vmap(bilinear_interp))(B, F)

    # Convert to numpy for matplotlib contour
    ax.contour(np.asarray(B), np.asarray(F), np.asarray(Q_fine),
               levels=[0.4, 0.8], colors=["black"], linewidths=2)

    plt.colorbar(im, ax=ax, label="Q Parameter")

    ax.set_xlabel("Power Spectrum Slope $\\beta$", fontsize=12)
    ax.set_ylabel("Dense Tail Fraction $f_{sub}$", fontsize=12)
    ax.set_title(
        f"Q-Parameter Heatmap\n"
        f"(M={results['params']['mach']}, α={results['params']['alpha']})",
        fontsize=14
    )

    ax.text(0.05, 0.95, "Black contours: Q ∈ [0.4, 0.8]",
            transform=ax.transAxes, fontsize=10, va="top")

    if show:
        plt.show()

    path = save_plot(fig, "e3_morphology_heatmap")
    plt.close(fig)

    return path


def main():
    """Run full E3 validation."""
    results = run_validation(verbose=True)
    make_plot(results)
    make_heatmap(results)

    print("\n" + "=" * 70)
    print("E3 VALIDATION COMPLETE")
    print("=" * 70)

    # Summary analysis
    data = results["by_config"]
    betas = results["params"]["betas"]
    f_subs = results["params"]["f_subs"]

    print("\nKey findings:")

    # Find configurations in observed range
    in_range = []
    for (beta, f_sub), r in data.items():
        if 0.4 <= r.q_mean <= 0.8:
            in_range.append((beta, f_sub, r.q_mean))

    if in_range:
        print(f"  1. {len(in_range)} configs produce Q ∈ [0.4, 0.8]:")
        for beta, f_sub, q in sorted(in_range):
            print(f"     β={beta:.2f}, f_sub={f_sub:.2f} → Q={q:.3f}")

    # Effect of β at fixed f_sub=0.3
    if 0.3 in f_subs:
        q_at_beta = {b: data[(b, 0.3)].q_mean for b in betas}
        print(f"\n  2. Effect of β at f_sub=0.3:")
        for b in betas:
            print(f"     β={b:.2f} → Q={q_at_beta[b]:.3f}")

    # Effect of f_sub at fixed β=3.67
    beta_ref = 3.67 if 3.67 in betas else betas[len(betas)//2]
    if beta_ref in betas:
        q_at_fsub = {f: data[(beta_ref, f)].q_mean for f in f_subs}
        print(f"\n  3. Effect of f_sub at β={beta_ref:.2f}:")
        for f in f_subs:
            print(f"     f_sub={f:.2f} → Q={q_at_fsub[f]:.3f}")

    print("\n  4. General trends:")
    print("     - Lower β → more power at small scales → more clumpy → lower Q")
    print("     - Higher f_sub → more stars in dense clumps → lower Q")
    print("     - f_sub has stronger effect on Q than β")

    return results


if __name__ == "__main__":
    main()

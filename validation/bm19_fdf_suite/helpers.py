#!/usr/bin/env python
"""Shared helper functions for BM19+FDF validation suite.

Contains:
- Physical conversions (t_ff, column density, etc.)
- Environment presets (GMC, CMZ, YMC-forming)
- Plot styling utilities
- Observational anchors
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from typing import NamedTuple

# =============================================================================
# Observational Anchors (for reference in validation)
# =============================================================================

OBSERVATIONAL_ANCHORS = {
    "lada_threshold_cm2": 7e21,       # Lada+2010: A_K > 0.8 mag
    "gmc_sfe_low": 0.01,              # Evans+2014: 1-5%
    "gmc_sfe_high": 0.05,
    "ymc_sfe_low": 0.10,              # Kruijssen+2012: 10-30%
    "ymc_sfe_high": 0.30,
    "epsilon_ff_int": 0.01,           # Krumholz & McKee 2005: ~1%
    "epsilon_ff_int_high": 0.02,
}


class EnvironmentPreset(NamedTuple):
    """Preset cloud environment for validation."""
    name: str
    Sigma: float    # Surface density [Msun/pc^2]
    Mach: float     # Turbulent Mach number
    alpha: float    # BM19 powerlaw slope
    b: float        # Driving parameter
    description: str


# Standard environment presets
ENVIRONMENT_PRESETS = {
    "gmc_solar": EnvironmentPreset(
        name="GMC (Solar)",
        Sigma=100.0,
        Mach=10.0,
        alpha=2.0,
        b=0.4,
        description="Typical solar neighborhood GMC"
    ),
    "gmc_low_mass": EnvironmentPreset(
        name="GMC (Low-mass)",
        Sigma=50.0,
        Mach=7.0,
        alpha=2.2,
        b=0.4,
        description="Lower surface density GMC"
    ),
    "cmz_like": EnvironmentPreset(
        name="CMZ-like",
        Sigma=1000.0,
        Mach=30.0,
        alpha=1.8,
        b=0.4,
        description="Central Molecular Zone conditions"
    ),
    "ymc_forming": EnvironmentPreset(
        name="YMC-forming",
        Sigma=500.0,
        Mach=20.0,
        alpha=2.0,
        b=0.4,
        description="Young massive cluster forming cloud"
    ),
    "low_mach": EnvironmentPreset(
        name="Low Mach",
        Sigma=100.0,
        Mach=5.0,
        alpha=2.0,
        b=0.4,
        description="Low turbulence reference"
    ),
}


# =============================================================================
# Physical Conversions
# =============================================================================

def t_ff_myr(Sigma: float | np.ndarray) -> float | np.ndarray:
    """Freefall time [Myr] calibrated to 1 Myr at Sigma=100.

    t_ff ~ 1/sqrt(rho) ~ sqrt(1/Sigma) for constant depth clouds.

    Parameters
    ----------
    Sigma : float or array
        Surface density [Msun/pc^2]

    Returns
    -------
    t_ff : float or array
        Freefall time [Myr]

    Notes
    -----
    Calibration: t_ff(100 Msun/pc^2) = 1 Myr (typical GMC value)
    """
    return 1.0 * np.sqrt(100.0 / Sigma)


def s_to_column_density(s: float | np.ndarray, Sigma: float, depth_pc: float = 1.0) -> float | np.ndarray:
    """Convert log-density s to column density N_H [cm^-2].

    Parameters
    ----------
    s : float or array
        Log-density contrast s = ln(rho/rho_mean)
    Sigma : float
        Surface density [Msun/pc^2]
    depth_pc : float
        Cloud depth [pc] (default 1.0)

    Returns
    -------
    N_H : float or array
        Column density [cm^-2]

    Notes
    -----
    Conversion: N_H ~ 2.1e21 cm^-2 per (Sigma/100) per pc depth
    This assumes molecular gas with mean molecular weight ~2.8.
    """
    N_H_mean = 2.1e21 * (Sigma / 100.0) * depth_pc
    return N_H_mean * np.exp(s)


def column_density_to_s(N_H: float | np.ndarray, Sigma: float, depth_pc: float = 1.0) -> float | np.ndarray:
    """Convert column density N_H [cm^-2] to log-density s.

    Inverse of s_to_column_density().

    Parameters
    ----------
    N_H : float or array
        Column density [cm^-2]
    Sigma : float
        Surface density [Msun/pc^2]
    depth_pc : float
        Cloud depth [pc] (default 1.0)

    Returns
    -------
    s : float or array
        Log-density contrast
    """
    N_H_mean = 2.1e21 * (Sigma / 100.0) * depth_pc
    return np.log(N_H / N_H_mean)


def larson_mach(Sigma: float | np.ndarray, M_ref: float = 10.0, Sigma_ref: float = 100.0) -> float | np.ndarray:
    """Larson-type Mach-Sigma relation.

    M(Sigma) = M_ref * (Sigma/Sigma_ref)^0.5

    Parameters
    ----------
    Sigma : float or array
        Surface density [Msun/pc^2]
    M_ref : float
        Reference Mach at Sigma_ref (default 10)
    Sigma_ref : float
        Reference surface density (default 100)

    Returns
    -------
    Mach : float or array
        Turbulent Mach number

    Notes
    -----
    This follows from Larson's velocity-size relation combined with
    surface-density scaling. The exponent 0.5 is typical but uncertain.
    """
    return M_ref * np.sqrt(Sigma / Sigma_ref)


def sfr_proxy(f_dense: float | np.ndarray, Sigma: float | np.ndarray, epsilon_ff: float = 0.01) -> float | np.ndarray:
    """SFR proxy from BM19 f_dense and cloud properties.

    SFR/M_cloud ~ epsilon_ff * f_dense / t_ff

    Parameters
    ----------
    f_dense : float or array
        BM19 self-gravitating fraction
    Sigma : float or array
        Surface density [Msun/pc^2]
    epsilon_ff : float
        Intrinsic SFE per freefall (default 0.01)

    Returns
    -------
    sfr_proxy : float or array
        SFR per unit cloud mass [Myr^-1]
    """
    t_ff = t_ff_myr(Sigma)
    return epsilon_ff * f_dense / t_ff


# =============================================================================
# PP20 Related
# =============================================================================

def pp20_sfr_per_mdg(zeta: float | np.ndarray, epsilon_ff: float = 0.01, t_ff_dg_myr: float = 0.5) -> float | np.ndarray:
    """PP20 SFR per unit dense gas mass.

    SFR/M_dg = zeta * epsilon_ff / t_ff,dg

    Parameters
    ----------
    zeta : float or array
        Magnification factor
    epsilon_ff : float
        Intrinsic SFE per freefall (default 0.01)
    t_ff_dg_myr : float
        Mean freefall time of dense gas [Myr] (default 0.5)

    Returns
    -------
    sfr_per_mdg : float or array
        SFR per M_dg [Myr^-1]
    """
    return zeta * epsilon_ff / t_ff_dg_myr


def p_from_alpha(alpha: float | np.ndarray) -> float | np.ndarray:
    """PP20 profile slope from BM19 powerlaw slope.

    p = 3/alpha

    Parameters
    ----------
    alpha : float or array
        BM19 powerlaw slope (>1)

    Returns
    -------
    p : float or array
        Radial density profile slope (rho ~ r^-p)
    """
    return 3.0 / alpha


def alpha_from_p(p: float | np.ndarray) -> float | np.ndarray:
    """BM19 powerlaw slope from PP20 profile slope.

    alpha = 3/p

    Parameters
    ----------
    p : float or array
        Radial density profile slope

    Returns
    -------
    alpha : float or array
        BM19 powerlaw slope
    """
    return 3.0 / p


# =============================================================================
# Plot Styling
# =============================================================================

# Color palette (colorblind-friendly)
COLORS = {
    "bm19": "#1f77b4",       # Blue
    "pn11": "#ff7f0e",       # Orange
    "fdense": "#2ca02c",     # Green
    "ftail": "#d62728",      # Red
    "zeta_analytic": "#9467bd",  # Purple
    "zeta_fdf": "#8c564b",   # Brown
    "gmc": "#17becf",        # Cyan
    "cmz": "#bcbd22",        # Olive
    "ymc": "#e377c2",        # Pink
}


def setup_publication_style():
    """Configure matplotlib for publication-quality plots."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.figsize": (8, 6),
        "figure.dpi": 150,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
    })


def add_1to1_line(ax, color="k", linestyle="--", label="1:1"):
    """Add 1:1 reference line to axes."""
    lims = [
        max(ax.get_xlim()[0], ax.get_ylim()[0]),
        min(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, color=color, linestyle=linestyle, linewidth=1.5, label=label)


def add_percent_bands(ax, center_line, percents=[20], colors=["gray"], alpha=0.2):
    """Add percentage error bands around a center line."""
    x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
    for pct, color in zip(percents, colors):
        factor = pct / 100
        ax.fill_between(x, x * (1 - factor), x * (1 + factor), alpha=alpha, color=color, label=f"$\\pm${pct}%")


# =============================================================================
# Validation Utilities
# =============================================================================

def compute_statistics(values: np.ndarray) -> dict:
    """Compute summary statistics for validation results.

    Parameters
    ----------
    values : array
        Sample values

    Returns
    -------
    stats : dict
        Dictionary with mean, std, median, min, max, percentiles
    """
    return {
        "mean": np.mean(values),
        "std": np.std(values),
        "median": np.median(values),
        "min": np.min(values),
        "max": np.max(values),
        "p5": np.percentile(values, 5),
        "p25": np.percentile(values, 25),
        "p75": np.percentile(values, 75),
        "p95": np.percentile(values, 95),
        "n": len(values),
    }


def relative_error_percent(measured: float | np.ndarray, expected: float | np.ndarray) -> float | np.ndarray:
    """Compute relative error in percent.

    Parameters
    ----------
    measured : float or array
        Measured/actual value
    expected : float or array
        Expected/theoretical value

    Returns
    -------
    error_pct : float or array
        Relative error in percent
    """
    return 100 * (measured - expected) / expected


def save_plot(fig, name: str, base_dir: str = "/Users/anna/projects/jaxstro-dev/progenax/validation/plots/bm19_fdf_suite"):
    """Save plot with standard settings.

    Parameters
    ----------
    fig : matplotlib Figure
        Figure to save
    name : str
        Filename (without extension)
    base_dir : str
        Output directory
    """
    import os
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {path}")
    return path

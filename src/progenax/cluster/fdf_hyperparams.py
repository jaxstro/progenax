# progenax/src/progenax/cluster/fdf_hyperparams.py
"""Hyperparameter dataclasses for FDF calibration.

These dataclasses encapsulate tunable parameters that are NOT derived
from physics but are heuristics awaiting calibration.

Status
------
All hyperparameters in this module are UNCALIBRATED. The values are
reasonable starting points but should not be trusted for quantitative
predictions without calibration against simulations or observations.

Calibration Targets
-------------------
- Cartwright & Whitworth (2004) Q(D) measurements
- MHD turbulence simulations (e.g., Federrath+2010)
- Observed cluster structure (Q, m, bar)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FDFDensityHyperparams:
    """Hyperparameters for density-field FDF (fdf_density.py).

    Controls the spectral slope and amplitude of the lognormal
    density field. Currently UNCALIBRATED heuristics.

    Attributes
    ----------
    beta_0 : float
        Baseline spectral slope for δ field. Default 2.0.
        Physical reference: Kolmogorov ~3.67, Burgers ~4.
    beta_1 : float
        Sensitivity of β to chi parameter. Default 1.5.
        Maps: β = beta_0 + beta_1 × (χ - 1.5)
    sigma_ln_rho_default : float
        Default amplitude of log-density fluctuations.
        Physical range: σ_ln_ρ ~ 1.0-2.5 for typical clusters.
    version : str
        Version identifier for tracking calibration state.

    Notes
    -----
    Version history:
        - v0_uncalibrated (2024-12): Initial heuristics, NOT calibrated
    """

    beta_0: float = 2.0
    beta_1: float = 1.5
    sigma_ln_rho_default: float = 2.0
    version: str = "v0_uncalibrated"


@dataclass(frozen=True)
class FDFDisplacementHyperparams:
    """Hyperparameters for displacement-field FDF (fdf.py).

    Controls the spectral shape and amplitude of the Fourier-mode
    displacement field. Currently UNCALIBRATED heuristics.

    Attributes
    ----------
    beta_base : float
        Mild baseline power-law slope. Default 1.5.
    sigma_logk : float
        Width of lognormal envelope in log-k space. Default 0.8.
    sigma_u_default : float
        Default displacement amplitude scale (dimensionless).
    version : str
        Version identifier for tracking calibration state.

    Notes
    -----
    Version history:
        - v0_uncalibrated (2024-12): Initial heuristics, NOT calibrated
    """

    beta_base: float = 1.5
    sigma_logk: float = 0.8
    sigma_u_default: float = 0.3
    version: str = "v0_uncalibrated"


@dataclass(frozen=True)
class FDFUncalibratedHeuristics:
    """HEURISTIC CONSTANTS - NOT PHYSICS-DERIVED.

    WARNING: These values are placeholders awaiting calibration.

    DO NOT use in env_to_fdf_layer(). They exist only for:
    - Legacy API compatibility (density_layer_from_D)
    - Manual experimentation
    - Calibration sweeps

    Attributes
    ----------
    beta_0 : float
        Baseline spectral slope for legacy χ→β mapping.
    beta_1 : float
        Sensitivity of β to chi.
    sigma_ln_rho_manual : float
        Default σ_ln_ρ for manual construction.
    beta_base_displacement : float
        Spectral slope for displacement FDF.
    sigma_logk_displacement : float
        Envelope width for displacement FDF.
    sigma_u_default : float
        Displacement amplitude scale.

    Notes
    -----
    CW04 calibration targets (NOT currently achieved):
        D=1.5 → Q ≈ 0.47
        D=2.0 → Q ≈ 0.58
        D=2.5 → Q ≈ 0.70
        D=3.0 → Q ≈ 0.79-0.82
    """

    beta_0: float = 2.0
    beta_1: float = 1.5
    sigma_ln_rho_manual: float = 2.0
    beta_base_displacement: float = 1.5
    sigma_logk_displacement: float = 0.8
    sigma_u_default: float = 0.3
    version: str = "v0_uncalibrated_2024-12"


# Default instances
FDF_DENSITY_DEFAULTS = FDFDensityHyperparams()
FDF_DISPLACEMENT_DEFAULTS = FDFDisplacementHyperparams()
FDF_HEURISTICS = FDFUncalibratedHeuristics()


__all__ = [
    "FDFDensityHyperparams",
    "FDFDisplacementHyperparams",
    "FDFUncalibratedHeuristics",
    "FDF_DENSITY_DEFAULTS",
    "FDF_DISPLACEMENT_DEFAULTS",
    "FDF_HEURISTICS",
]

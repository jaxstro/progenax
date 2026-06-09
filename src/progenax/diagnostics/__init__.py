# progenax/src/progenax/diagnostics/__init__.py
"""
Diagnostic functions for star cluster initial conditions.

This module provides tools for measuring and quantifying:
- Mass segregation (Λ_MSR ratio from Allison+2009)
- Spatial substructure (Cartwright-Whitworth Q parameter)
- Azimuthal density variation (correlated with fractal dimension)

All functions in this module use NumPy and SciPy for simplicity and are
NOT part of the jitted, differentiable IC generator. They are intended
for validation, calibration, and visualization.

Functions:
    compute_lambda_msr: Mass segregation ratio via MST (Allison+2009)
    compute_q_parameter: Cartwright-Whitworth Q for substructure
    compute_azimuthal_variation: σ_Σ/<Σ> density variation

Example:
    >>> from progenax.diagnostics import compute_lambda_msr, compute_q_parameter
    >>> import numpy as np
    >>>
    >>> # Convert JAX arrays to NumPy for diagnostics
    >>> positions_np = np.array(cluster.positions)
    >>> masses_np = np.array(cluster.masses)
    >>>
    >>> # Compute mass segregation
    >>> lam, err = compute_lambda_msr(positions_np, masses_np, N_massive=20)
    >>> print(f"Λ_MSR = {lam:.2f} ± {err:.2f}")
    >>>
    >>> # Compute substructure
    >>> Q = compute_q_parameter(positions_np)
    >>> print(f"Q = {Q:.3f}")

References:
    Allison et al. (2009), ApJ 700, L99 - Λ_MSR metric
    Cartwright & Whitworth (2004), MNRAS 348, 589 - Q parameter
    Küpper et al. (2011), MNRAS 417, 2300 - Azimuthal variation
"""

from progenax.diagnostics.mass_segregation import compute_lambda_msr
from progenax.diagnostics.substructure import (
    compute_q_parameter,
    compute_azimuthal_variation,
)
from progenax.diagnostics.q_approx import (
    q_approx,
    q_approx_naive,
    q_approx_fast,
    calibrate_q_approx,
    DEFAULT_CALIBRATION,
)
from progenax.diagnostics.segregation_approx import (
    soft_mass_weights,
    radial_concentration_approx,
    lambda_msr_approx,
    sigma_m_approx,
    calibrate_segregation_approx,
)

__all__ = [
    "compute_lambda_msr",
    "compute_q_parameter",
    "compute_azimuthal_variation",
    "q_approx",
    "q_approx_naive",
    "q_approx_fast",
    "calibrate_q_approx",
    "DEFAULT_CALIBRATION",
    # Differentiable segregation observables
    "soft_mass_weights",
    "radial_concentration_approx",
    "lambda_msr_approx",
    "sigma_m_approx",
    "calibrate_segregation_approx",
]

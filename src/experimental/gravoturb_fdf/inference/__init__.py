"""Differentiable physics-direct inference for gravoturb_fdf (Phase 5+).

Milestone 1: analytic covariance + Gaussian likelihood on a data vector of log-density
power-spectrum band-powers P_s(k_i) + CIC variance sigma^2_N(R), and a Fisher forecast.
Milestone 2 (Phase 6): compound-Poisson count likelihood + blackjax HMC recovery.

All differentiable in theta=(mach, b, alpha, beta); numpy/scipy only in the mock/validation paths.
"""

from gravoturb_fdf.inference.diagnostics import compute_hmc_diagnostics
from gravoturb_fdf.inference.priors import BM19Prior
from gravoturb_fdf.inference.sbc import build_logdensity, sbc_ranks

__all__ = [
    "BM19Prior",
    "compute_hmc_diagnostics",
    "build_logdensity",
    "sbc_ranks",
]

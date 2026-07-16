"""Differentiable physics-direct inference for gravoturb (Phase 5+).

Milestone 1: analytic covariance + Gaussian likelihood on a data vector of log-density
power-spectrum band-powers P_s(k_i) + CIC variance sigma^2_N(R), and a Fisher forecast.
Milestone 2 (Phase 6): compound-Poisson count likelihood + blackjax HMC recovery.

All differentiable in theta=(mach, b, alpha, beta); numpy/scipy only in the mock/validation paths.
"""

from gravoturb.inference.diagnostics import compute_hmc_diagnostics
from gravoturb.inference.flow_npe import (
    beta_to_z,
    build_npe_flow,
    npe_posterior_z,
    train_npe,
    whiten,
    whiten_stats,
    z_to_beta,
)
from gravoturb.inference.model import build_logdensity
from gravoturb.inference.priors import BM19Prior

# The 2-D projected-beta forward model -- the ACTIVE HEADLINE path for star-catalog beta
# (Anna's 2026-07-16 adjudication); the validation/_banked_2d_beta scratch drivers are
# archived evidence only. Analytic log+ band-power chain (projected_logp) + the flow-NPE
# low-N fallback (flow_npe).
from gravoturb.inference.projected_logp import (
    analytic_logdensity_bandpowers,
    calibrate_transfer,
    logp_loglike,
    logp_shot_components,
    predict_logp_bandpowers,
    predict_logp_bandpowers_shot,
)
from gravoturb.inference.sbc import sbc_ranks

__all__ = [
    "BM19Prior",
    "compute_hmc_diagnostics",
    "build_logdensity",
    "sbc_ranks",
    # 2-D projected-beta headline: analytic log+ band-power forward model
    "analytic_logdensity_bandpowers",
    "calibrate_transfer",
    "logp_loglike",
    "logp_shot_components",
    "predict_logp_bandpowers",
    "predict_logp_bandpowers_shot",
    # 2-D projected-beta headline: flow-NPE low-N path
    "beta_to_z",
    "build_npe_flow",
    "npe_posterior_z",
    "train_npe",
    "whiten",
    "whiten_stats",
    "z_to_beta",
]

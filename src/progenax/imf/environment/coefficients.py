"""Marks+2012 / Jerabkova+2018 calibration coefficients (split from environment.py)."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


# =============================================================================
# Paper Coefficients - EXACT values from literature
# =============================================================================

JERABKOVA_COEFFICIENTS = {
    # Mass-based formula (derived from Eq. 7 + Marks r_h-M_ecl relation + 8π density)
    # x = -0.14×[Fe/H] + 0.6039×log₁₀(M_ecl/10⁶) + 0.2161 - 0.99×log₁₀(ε/0.33)
    #
    # Derivation (8π half-mass density convention, Marks+2012):
    #   1. r_h = 0.1 × M_ecl^0.13
    #   2. ρ_ecl = 3M_ecl/(8π×r_h³)  → log₁₀(ρ_ecl) = 0.61×log₁₀(M_ecl) + 2.0768
    #   3. ρ_cl = ρ_ecl/ε            → log₁₀(ρ_cl) = 0.61×log₁₀(M_ecl) + 2.5583 (ε=0.33)
    #   4. Eq. 7: x = -0.14×[Fe/H] + 0.99×log₁₀(ρ_cl/10⁶)
    #   5. Substituting: x = -0.14×[Fe/H] + 0.6039×log₁₀(M_ecl/10⁶) + 0.2161
    #
    # Constants derived from: log₁₀(3)=0.4771, log₁₀(8π)=1.4003, log₁₀(0.33)=-0.4815
    "FeH_coeff": -0.14,
    "logMecl_coeff": 0.6039,  # 0.99 × 0.61 = 0.6039 (exact)
    "constant": 0.2161,       # 0.99 × (2.5583 - 6) + 3.6234 = 0.2161 (8π-derived)
    # α₃(x) relation — Jerabkova+2018 Eq. 6 (= Marks+2012 + 2014 erratum):
    #   α₃ = 2.3 (x < -0.87);  α₃ = -0.41·x + 1.94 (x ≥ -0.87)
    # NOTE: distinct from the Marks+2012 MNRAS Fundamental-Plane fit (-0.4072 / 1.9383),
    # which is kept separately in MARKS_COEFFICIENTS.
    "x_threshold": -0.87,     # NEGATIVE threshold (top-heavy when x ≥ -0.87); Jerabkova Eq. 6
    "alpha3_slope": -0.41,    # Jerabkova+2018 Eq. 6
    "alpha3_intercept": 1.94, # Jerabkova+2018 Eq. 6
    "alpha3_canonical": 2.3,  # Kroupa (2001) canonical high-mass slope
    # Jerabkova+2018 Eq. 7 (density-based, NO constant term)
    "rho_logRho_coeff": 0.99,
}

MARKS_COEFFICIENTS = {
    # Marks+2012 Eq. 14-15 (Fundamental Plane)
    "cos_theta": -0.139,  # cos(98 deg)
    "sin_theta": 0.990,   # sin(98 deg)
    "x_hat_threshold": 0.87,  # POSITIVE threshold
    "alpha3_slope": -0.4072,
    "alpha3_intercept": 1.9383,
    "alpha3_canonical": 2.3,
    # Marks+2012 Eq. 12 (low-mass slopes)
    "lowmass_slope": 0.5,  # delta_alpha per [Fe/H]
}

# Marks+2012 Table 3: 1D relations for α₃
# α₃(λ) = p × λ + q, if λ ≷ λ_lim (else 2.3)
MARKS_TABLE3_COEFFICIENTS = {
    "mcl": {"p": -0.94, "q": 2.14, "lim": 0.68, "branch": ">"},
    "mecl": {"p": -0.77, "q": 1.59, "lim": 0.27, "branch": ">"},
    "rho": {"p": -0.43, "q": 1.86, "lim": 0.095, "branch": ">"},
    "feh": {"p": 0.66, "q": 2.63, "lim": -0.5, "branch": "<"},
}

# Default SFE from literature (Jerabkova+2018 assumes ε = 0.33)
DEFAULT_SFE = 0.33



"""Environment-dependent IMF: Marks+2012 and Jerabkova+2018 prescriptions.

This module provides fully-differentiable functions for computing IMF parameters
from stellar birth environment conditions. Designed as a wrapper/modifier layer
that transforms IMFParams based on physical conditions.

Core Concept:
    BirthEnvironment + baseline IMFParams
          ↓
      env_to_imf_params(...)
          ↓
    modified 4-segment IMFParams (α₀, α₁, α₂, α₃)
          ↓
    existing PowerLawIMF sampling / IC pipelines

4-Segment IMF Convention (Marks+2012 Eq. 2):
    Segment 0: [0.01, 0.08] M☉ - sub-stellar (α₀ = 0.3, FIXED)
    Segment 1: [0.08, 0.50] M☉ - low-mass (α₁, metallicity-dependent)
    Segment 2: [0.50, 1.00] M☉ - intermediate (α₂, metallicity-dependent)
    Segment 3: [1.00, m_max] M☉ - high-mass (α₃, environment-dependent)

Key Physics:
    - SFE (ε): Star Formation Efficiency = M_ecl / M_cl (default 0.33)
    - Radius-mass relation: r_h [pc] = 0.1 × (M_ecl/M☉)^0.13
    - Density: ρ_cl = ρ_ecl / ε

References:
    Marks et al. (2012), MNRAS 422, 2246 - Fundamental Plane + Table 3
    Jerabkova et al. (2018), A&A 620, A39 - Mass-based prescription (Eq. 9)
    Kroupa (2001), MNRAS 322, 231 - Canonical IMF slopes
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from .params import IMFParams


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
    "x_threshold": -0.87,     # NEGATIVE threshold (top-heavy when x > -0.87)
    "alpha3_slope": -0.41,
    "alpha3_intercept": 1.94,
    "alpha3_canonical": 2.3,
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


# =============================================================================
# Density Computation Functions
# =============================================================================

def compute_r_half(M_ecl: Float[Array, "..."]) -> Float[Array, "..."]:
    """Radius-mass relation from Marks & Kroupa (2012).

    r_h [pc] = 0.1 × (M_ecl / M☉)^0.13

    Args:
        M_ecl: Stellar mass of embedded cluster [M☉]

    Returns:
        Half-mass radius [pc]
    """
    return 0.1 * jnp.power(M_ecl, 0.13)


def compute_rho_ecl(M_ecl: Float[Array, "..."]) -> Float[Array, "..."]:
    """Half-mass density (Marks+2012 convention).

    ρ_ecl = 3 × M_ecl / (8π × r_h³)

    The 8π factor arises because half the cluster mass is within r_h:
        ρ = (M_ecl/2) / (4π/3 × r_h³) = 3M_ecl / (8π × r_h³)

    This convention matches Marks+2012 Table 1 densities exactly.

    Args:
        M_ecl: Stellar mass of embedded cluster [M☉]

    Returns:
        Stellar half-mass density [M☉ pc⁻³]
    """
    r_h = compute_r_half(M_ecl)
    return 3.0 * M_ecl / (8.0 * jnp.pi * r_h**3)


def compute_rho_cl(
    M_ecl: Float[Array, "..."],
    sfe: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Cloud-core density from stellar mass and SFE.

    ρ_cl = ρ_ecl / ε

    Args:
        M_ecl: Stellar mass of embedded cluster [M☉]
        sfe: Star formation efficiency ε = M_ecl / M_cl

    Returns:
        Cloud density [M☉ pc⁻³]
    """
    return compute_rho_ecl(M_ecl) / sfe


def compute_log_rho_cl_6(
    M_ecl: Float[Array, "..."],
    sfe: Float[Array, "..."],
) -> Float[Array, "..."]:
    """log₁₀(ρ_cl / 10⁶ M☉ pc⁻³) from cluster mass and SFE.

    Args:
        M_ecl: Stellar mass of embedded cluster [M☉]
        sfe: Star formation efficiency

    Returns:
        log₁₀(ρ_cl / 10⁶)
    """
    rho_cl = compute_rho_cl(M_ecl, sfe)
    return jnp.log10(rho_cl) - 6.0


# =============================================================================
# BirthEnvironment - Primary inference target
# =============================================================================

class BirthEnvironment(eqx.Module):
    """Physical birth environment with SFE parameter.

    Represents conditions during star formation that influence the IMF.
    All fields are JAX arrays for gradient-based inference.

    Attributes:
        metallicity: [Fe/H], default 0.0 (solar). Calibrated range [-2.5, +0.5]
        log_mecl: log₁₀(M_ecl / M☉). Calibrated range [3, 8]
        sfe: Star formation efficiency ε = M_ecl / M_cl, default 0.33
        log_rho_cl: Optional override for log₁₀(ρ_cl / 10⁶ M☉ pc⁻³)

    Example:
        >>> # Solar neighborhood cluster
        >>> env = BirthEnvironment.solar()
        >>> params = env_to_imf_params(env)
        >>> print(f"α₃ = {float(params.alpha3):.2f}")  # ~2.3

        >>> # Dense, metal-poor globular cluster with low SFE
        >>> env = BirthEnvironment.from_cluster_mass(
        ...     M_ecl=1e6, FeH=-1.5, sfe=0.1
        ... )
        >>> params = env_to_imf_params(env)
        >>> print(f"α₃ = {float(params.alpha3):.2f}")  # Top-heavy!
    """

    metallicity: Float[Array, ""]
    log_mecl: Float[Array, ""]
    sfe: Float[Array, ""]
    log_rho_cl: Float[Array, ""] | None = None

    def __init__(
        self,
        metallicity: Float[Array, ""] | None = None,
        log_mecl: Float[Array, ""] | None = None,
        sfe: Float[Array, ""] | None = None,
        log_rho_cl: Float[Array, ""] | None = None,
    ):
        """Initialize BirthEnvironment with defaults.

        Args:
            metallicity: [Fe/H], default 0.0 (solar)
            log_mecl: log₁₀(M_ecl/M☉), required
            sfe: Star formation efficiency, default 0.33
            log_rho_cl: Optional density override
        """
        if log_mecl is None:
            raise ValueError("log_mecl is required")

        self.metallicity = metallicity if metallicity is not None else jnp.array(0.0)
        self.log_mecl = log_mecl
        self.sfe = sfe if sfe is not None else jnp.array(DEFAULT_SFE)
        self.log_rho_cl = log_rho_cl

    @classmethod
    def from_cluster_mass(
        cls,
        M_ecl: float,
        FeH: float = 0.0,
        sfe: float = DEFAULT_SFE,
        log_rho_cl: float | None = None,
    ) -> BirthEnvironment:
        """Create from embedded cluster mass with solar metallicity default.

        Args:
            M_ecl: Embedded cluster mass [M☉]
            FeH: Metallicity [Fe/H], default 0.0 (solar)
            sfe: Star formation efficiency, default 0.33
            log_rho_cl: Optional log₁₀(ρ_cl / 10⁶ M☉ pc⁻³) override

        Returns:
            BirthEnvironment with computed log_mecl
        """
        return cls(
            metallicity=jnp.array(FeH),
            log_mecl=jnp.log10(jnp.array(M_ecl)),
            sfe=jnp.array(sfe),
            log_rho_cl=jnp.array(log_rho_cl) if log_rho_cl is not None else None,
        )

    @classmethod
    def solar(cls) -> BirthEnvironment:
        """Solar neighborhood: 1000 M☉ cluster, solar metallicity, default SFE."""
        return cls(
            metallicity=jnp.array(0.0),
            log_mecl=jnp.array(3.0),  # 1000 M☉
            sfe=jnp.array(DEFAULT_SFE),
        )

    @classmethod
    def massive_gc(cls, FeH: float = -1.5) -> BirthEnvironment:
        """Typical massive globular cluster birth conditions."""
        return cls(
            metallicity=jnp.array(FeH),
            log_mecl=jnp.array(6.0),  # 10^6 M☉
            sfe=jnp.array(DEFAULT_SFE),
        )

    @classmethod
    def ngc_7078(cls) -> BirthEnvironment:
        """NGC 7078 (M15) - most top-heavy in Marks+2012 Table 1.

        From Marks+2012:
            [Fe/H] = -2.16
            ρ_cl = 258.13 × 10⁶ M☉/pc³
            α₃ = 0.76 (extremely top-heavy!)
        """
        return cls(
            metallicity=jnp.array(-2.16),
            log_mecl=jnp.array(6.5),
            sfe=jnp.array(DEFAULT_SFE),
            log_rho_cl=jnp.log10(jnp.array(258.13)),  # log₁₀(ρ/10⁶)
        )

    @classmethod
    def ngc_104(cls) -> BirthEnvironment:
        """NGC 104 (47 Tuc) - well-studied globular cluster."""
        return cls(
            metallicity=jnp.array(-0.76),
            log_mecl=jnp.array(6.0),
            sfe=jnp.array(DEFAULT_SFE),
            log_rho_cl=jnp.log10(jnp.array(9.54)),
        )


# =============================================================================
# Generalized x Functions
# =============================================================================

def x_jerabkova_generalized(
    FeH: Float[Array, "..."],
    M_ecl: Float[Array, "..."],
    sfe: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Mass-based x with explicit ε dependence.

    x = -0.14 × [Fe/H] + 0.6039 × log₁₀(M_ecl/10⁶) + 0.2161 - 0.99 × log₁₀(ε/0.33)

    Derived from Jeřábková Eq. 7 using Marks & Kroupa r_h-M_ecl relation
    and 8π half-mass density convention. See JERABKOVA_COEFFICIENTS for derivation.

    Args:
        FeH: Metallicity [Fe/H]
        M_ecl: Cluster stellar mass [M☉]
        sfe: Star formation efficiency ε

    Returns:
        x parameter for α₃ calculation
    """
    c = JERABKOVA_COEFFICIENTS
    log_mecl_6 = jnp.log10(M_ecl) - 6.0
    epsilon_correction = -0.99 * jnp.log10(sfe / DEFAULT_SFE)
    return c["FeH_coeff"] * FeH + c["logMecl_coeff"] * log_mecl_6 + c["constant"] + epsilon_correction


def x_jerabkova_rho(
    FeH: Float[Array, "..."],
    log_rho_6: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Jerabkova+2018 Eq. 7: x from density.

    x = -0.14 × [Fe/H] + 0.99 × log₁₀(ρ_cl/10⁶)

    NOTE: This density-based formula has NO constant term (unlike the mass-based Eq. 9).

    Args:
        FeH: Metallicity [Fe/H]
        log_rho_6: log₁₀(ρ_cl / 10⁶ M☉ pc⁻³)

    Returns:
        x parameter for α₃ calculation
    """
    c = JERABKOVA_COEFFICIENTS
    # Eq. 7 has NO constant term - just FeH and log_rho terms
    return c["FeH_coeff"] * FeH + c["rho_logRho_coeff"] * log_rho_6


def x_hat_marks_plane(
    FeH: Float[Array, "..."],
    log_rho_6: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Marks+2012 Fundamental Plane coordinate (Eq. 14).

    x̂ = cos(θ) × [Fe/H] + sin(θ) × log₁₀(ρ_cl/10⁶)
       = -0.139 × [Fe/H] + 0.990 × log₁₀(ρ_cl/10⁶)

    Args:
        FeH: Metallicity [Fe/H]
        log_rho_6: log₁₀(ρ_cl / 10⁶ M☉ pc⁻³)

    Returns:
        x̂ coordinate on Fundamental Plane
    """
    c = MARKS_COEFFICIENTS
    return c["cos_theta"] * FeH + c["sin_theta"] * log_rho_6


# =============================================================================
# α₃ Functions (High-Mass Slope)
# =============================================================================

def _alpha3_from_x(
    x: Float[Array, "..."],
    threshold: float,
    slope: float,
    intercept: float,
    canonical: float = 2.3,
    smooth: bool = False,
    smooth_width: float = 0.2,
) -> Float[Array, "..."]:
    """Compute α₃ from x parameter with optional smoothing.

    α₃(x) = {
        slope × x + intercept,  if x ≥ threshold
        canonical,               otherwise
    }

    Args:
        x: x parameter (Jerabkova or Marks)
        threshold: Transition threshold
        slope: Linear slope
        intercept: Linear intercept
        canonical: Canonical α₃ value (2.3)
        smooth: Use tanh smoothing for gradients
        smooth_width: Width of tanh transition

    Returns:
        α₃, clipped to [0.5, 2.3]
    """
    if smooth:
        transition = 0.5 * (1.0 + jnp.tanh((x - threshold) / smooth_width))
        alpha3_varied = slope * x + intercept
        alpha3 = canonical * (1 - transition) + alpha3_varied * transition
    else:
        alpha3 = jnp.where(
            x >= threshold,
            slope * x + intercept,
            canonical,
        )
    return jnp.clip(alpha3, 0.5, 2.3)


def alpha3_jerabkova_generalized(
    FeH: Float[Array, "..."],
    M_ecl: Float[Array, "..."],
    sfe: Float[Array, "..."],
    smooth: bool = False,
    smooth_width: float = 0.2,
) -> Float[Array, "..."]:
    """α₃ from generalized Jerabkova with explicit ε (RECOMMENDED).

    Uses x_jerabkova_generalized() which reduces to Eq. 9 at ε = 0.33.

    Args:
        FeH: Metallicity [Fe/H]
        M_ecl: Cluster stellar mass [M☉]
        sfe: Star formation efficiency
        smooth: Use tanh smoothing
        smooth_width: Smoothing width

    Returns:
        α₃, clipped to [0.5, 2.3]
    """
    c = JERABKOVA_COEFFICIENTS
    x = x_jerabkova_generalized(FeH, M_ecl, sfe)
    return _alpha3_from_x(
        x, c["x_threshold"], c["alpha3_slope"], c["alpha3_intercept"],
        c["alpha3_canonical"], smooth, smooth_width
    )


def alpha3_jerabkova_mecl(
    log_mecl_6: Float[Array, "..."],
    FeH: Float[Array, "..."],
    smooth: bool = False,
    smooth_width: float = 0.2,
) -> Float[Array, "..."]:
    """Mass-based α₃ from cluster mass (assumes ε = 0.33).

    x = -0.14 × [Fe/H] + 0.6039 × log₁₀(M_ecl/10⁶) + 0.2161

    Args:
        log_mecl_6: log₁₀(M_ecl / 10⁶ M☉)
        FeH: Metallicity [Fe/H]
        smooth: Use tanh smoothing
        smooth_width: Smoothing width

    Returns:
        α₃, clipped to [0.5, 2.3]
    """
    c = JERABKOVA_COEFFICIENTS
    x = c["FeH_coeff"] * FeH + c["logMecl_coeff"] * log_mecl_6 + c["constant"]
    return _alpha3_from_x(
        x, c["x_threshold"], c["alpha3_slope"], c["alpha3_intercept"],
        c["alpha3_canonical"], smooth, smooth_width
    )


def alpha3_jerabkova_rho(
    log_rho_6: Float[Array, "..."],
    FeH: Float[Array, "..."],
    smooth: bool = False,
    smooth_width: float = 0.2,
) -> Float[Array, "..."]:
    """Jerabkova+2018 Eq. 7: α₃ from density.

    Args:
        log_rho_6: log₁₀(ρ_cl / 10⁶ M☉ pc⁻³)
        FeH: Metallicity [Fe/H]
        smooth: Use tanh smoothing
        smooth_width: Smoothing width

    Returns:
        α₃, clipped to [0.5, 2.3]
    """
    c = JERABKOVA_COEFFICIENTS
    x = x_jerabkova_rho(FeH, log_rho_6)
    return _alpha3_from_x(
        x, c["x_threshold"], c["alpha3_slope"], c["alpha3_intercept"],
        c["alpha3_canonical"], smooth, smooth_width
    )


def alpha3_marks_plane(
    log_rho_6: Float[Array, "..."],
    FeH: Float[Array, "..."],
    smooth: bool = False,
    smooth_width: float = 0.2,
) -> Float[Array, "..."]:
    """Marks+2012 Fundamental Plane (Eq. 14-15).

    x̂ = -0.139 × [Fe/H] + 0.990 × log₁₀(ρ_cl/10⁶)

    CRITICAL: Threshold is POSITIVE (+0.87), unlike Jerabkova's -0.87!

    Args:
        log_rho_6: log₁₀(ρ_cl / 10⁶ M☉ pc⁻³)
        FeH: Metallicity [Fe/H]
        smooth: Use tanh smoothing
        smooth_width: Smoothing width

    Returns:
        α₃, clipped to [0.5, 2.3]
    """
    c = MARKS_COEFFICIENTS
    x_hat = x_hat_marks_plane(FeH, log_rho_6)
    return _alpha3_from_x(
        x_hat, c["x_hat_threshold"], c["alpha3_slope"], c["alpha3_intercept"],
        c["alpha3_canonical"], smooth, smooth_width
    )


def alpha3_marks_table3(
    lambda_param: Float[Array, "..."],
    relation: str,
    smooth: bool = False,
    smooth_width: float = 0.2,
) -> Float[Array, "..."]:
    """Marks+2012 Table 3: 1D relations for α₃.

    α₃(λ) = p × λ + q, if λ ≷ λ_lim

    Args:
        lambda_param: Parameter value (log₁₀(M/10⁶), log₁₀(ρ/10⁶), or [Fe/H])
        relation: One of "mcl", "mecl", "rho", "feh"
        smooth: Use tanh smoothing
        smooth_width: Smoothing width

    Returns:
        α₃, clipped to [0.5, 2.3]
    """
    if relation not in MARKS_TABLE3_COEFFICIENTS:
        raise ValueError(f"Unknown relation: {relation}. Use 'mcl', 'mecl', 'rho', or 'feh'")

    coef = MARKS_TABLE3_COEFFICIENTS[relation]
    p, q, lim, branch = coef["p"], coef["q"], coef["lim"], coef["branch"]

    alpha3_varied = p * lambda_param + q

    if smooth:
        if branch == ">":
            transition = 0.5 * (1.0 + jnp.tanh((lambda_param - lim) / smooth_width))
        else:  # branch == "<"
            transition = 0.5 * (1.0 - jnp.tanh((lambda_param - lim) / smooth_width))
        alpha3 = 2.3 * (1 - transition) + alpha3_varied * transition
    else:
        if branch == ">":
            alpha3 = jnp.where(lambda_param > lim, alpha3_varied, 2.3)
        else:  # branch == "<"
            alpha3 = jnp.where(lambda_param < lim, alpha3_varied, 2.3)

    return jnp.clip(alpha3, 0.5, 2.3)


# =============================================================================
# Low-Mass Slope Metallicity Dependence
# =============================================================================

def lowmass_slopes_metallicity(
    FeH: Float[Array, "..."],
    clamp_FeH: bool = True,
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Marks+2012 Eq. 12: Low-mass slopes from metallicity.

    For 4-segment IMF convention:
        α₁([Fe/H]) = 1.3 + 0.5 × [Fe/H]  (0.08-0.50 M☉)
        α₂([Fe/H]) = 2.3 + 0.5 × [Fe/H]  (0.50-1.00 M☉)

    TENTATIVE: Extrapolation for [Fe/H] < -0.5 is uncertain.

    Args:
        FeH: Metallicity [Fe/H]
        clamp_FeH: If True, clamp to calibrated range [-2.5, +0.5]

    Returns:
        (α₁, α₂) - slopes for segments 1 and 2
    """
    FeH_use = jnp.clip(FeH, -2.5, 0.5) if clamp_FeH else FeH

    c = MARKS_COEFFICIENTS
    alpha1 = 1.3 + c["lowmass_slope"] * FeH_use
    alpha2 = 2.3 + c["lowmass_slope"] * FeH_use

    # Ensure physically reasonable bounds
    alpha1 = jnp.maximum(alpha1, 0.0)
    alpha2 = jnp.maximum(alpha2, 0.5)

    return alpha1, alpha2


# =============================================================================
# Unified API: env_to_imf_params()
# =============================================================================

def env_to_imf_params(
    env: BirthEnvironment,
    model: str = "jerabkova_generalized",
    include_lowmass_variation: bool = False,
    smooth_alpha3: bool = False,
    smooth_width: float = 0.2,
    clamp_domain: bool = True,
) -> IMFParams:
    """Convert birth environment to 4-segment IMF parameters.

    SINGLE API for all environment-dependent IMF models.
    Fully differentiable - gradients flow from IMFParams back to BirthEnvironment.

    Models:
        - "kroupa": Standard Kroupa (2001), ignores environment
        - "jerabkova_generalized": Eq. 9 with explicit ε (RECOMMENDED)
        - "jerabkova_mecl": Eq. 9 (mass-based, assumes ε = 0.33)
        - "jerabkova_rho": Eq. 7 (density-based, requires log_rho_cl)
        - "marks_plane": Fundamental Plane Eq. 14-15 (requires log_rho_cl)
        - "marks_mcl": Table 3 (cloud mass)
        - "marks_mecl": Table 3 (stellar mass)
        - "marks_rho": Table 3 (density, requires log_rho_cl)
        - "marks_feh": Table 3 (metallicity only)

    Args:
        env: BirthEnvironment with metallicity, log_mecl, sfe
        model: Model name (see above)
        include_lowmass_variation: Apply Marks Eq. 12 to α₁, α₂ (default False)
        smooth_alpha3: Use tanh smoothing at threshold (for gradients)
        smooth_width: Width of tanh transition
        clamp_domain: Clamp inputs to calibrated ranges

    Returns:
        4-segment IMFParams (α₀, α₁, α₂, α₃)

    Example:
        >>> env = BirthEnvironment.from_cluster_mass(M_ecl=1e6, FeH=-1.5)
        >>> params = env_to_imf_params(env)
        >>> print(f"α₃ = {float(params.alpha3):.2f}")

        >>> # With explicit SFE
        >>> env = BirthEnvironment.from_cluster_mass(M_ecl=1e6, FeH=-1.5, sfe=0.1)
        >>> params = env_to_imf_params(env, model="jerabkova_generalized")
    """
    # Clamp to calibrated domain if requested
    if clamp_domain:
        FeH = jnp.clip(env.metallicity, -2.5, 0.5)
        log_mecl = jnp.clip(env.log_mecl, 3.0, 8.0)
    else:
        FeH = env.metallicity
        log_mecl = env.log_mecl

    # Convert to paper units
    log_mecl_6 = log_mecl - 6.0
    M_ecl = 10.0 ** log_mecl

    # Compute α₃ based on model
    if model == "kroupa":
        alpha3 = jnp.array(2.3)

    elif model == "jerabkova_generalized":
        alpha3 = alpha3_jerabkova_generalized(
            FeH, M_ecl, env.sfe, smooth=smooth_alpha3, smooth_width=smooth_width
        )

    elif model == "jerabkova_mecl":
        alpha3 = alpha3_jerabkova_mecl(
            log_mecl_6, FeH, smooth=smooth_alpha3, smooth_width=smooth_width
        )

    elif model == "jerabkova_rho":
        if env.log_rho_cl is None:
            log_rho_6 = compute_log_rho_cl_6(M_ecl, env.sfe)
        else:
            log_rho_6 = env.log_rho_cl
        alpha3 = alpha3_jerabkova_rho(
            log_rho_6, FeH, smooth=smooth_alpha3, smooth_width=smooth_width
        )

    elif model == "marks_plane":
        if env.log_rho_cl is None:
            log_rho_6 = compute_log_rho_cl_6(M_ecl, env.sfe)
        else:
            log_rho_6 = env.log_rho_cl
        alpha3 = alpha3_marks_plane(
            log_rho_6, FeH, smooth=smooth_alpha3, smooth_width=smooth_width
        )

    elif model == "marks_mcl":
        # M_cl = M_ecl / ε
        log_mcl_6 = jnp.log10(M_ecl / env.sfe) - 6.0
        alpha3 = alpha3_marks_table3(
            log_mcl_6, "mcl", smooth=smooth_alpha3, smooth_width=smooth_width
        )

    elif model == "marks_mecl":
        alpha3 = alpha3_marks_table3(
            log_mecl_6, "mecl", smooth=smooth_alpha3, smooth_width=smooth_width
        )

    elif model == "marks_rho":
        if env.log_rho_cl is None:
            log_rho_6 = compute_log_rho_cl_6(M_ecl, env.sfe)
        else:
            log_rho_6 = env.log_rho_cl
        alpha3 = alpha3_marks_table3(
            log_rho_6, "rho", smooth=smooth_alpha3, smooth_width=smooth_width
        )

    elif model == "marks_feh":
        alpha3 = alpha3_marks_table3(
            FeH, "feh", smooth=smooth_alpha3, smooth_width=smooth_width
        )

    else:
        valid_models = [
            "kroupa", "jerabkova_generalized", "jerabkova_mecl", "jerabkova_rho",
            "marks_plane", "marks_mcl", "marks_mecl", "marks_rho", "marks_feh"
        ]
        raise ValueError(f"Unknown model: '{model}'. Valid: {valid_models}")

    # Compute low-mass slopes
    if include_lowmass_variation and model != "kroupa":
        alpha1, alpha2 = lowmass_slopes_metallicity(FeH, clamp_FeH=clamp_domain)
    else:
        alpha1 = jnp.array(1.3)
        alpha2 = jnp.array(2.3)

    # α₀ is always fixed at 0.3 (brown dwarf regime)
    alpha0 = jnp.array(0.3)

    return IMFParams(
        alpha0=alpha0,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
    )


__all__ = [
    # Data class
    "BirthEnvironment",
    # Main API
    "env_to_imf_params",
    # Density functions
    "compute_r_half",
    "compute_rho_ecl",
    "compute_rho_cl",
    "compute_log_rho_cl_6",
    # x functions
    "x_jerabkova_generalized",
    "x_jerabkova_rho",
    "x_hat_marks_plane",
    # α₃ functions
    "alpha3_jerabkova_generalized",
    "alpha3_jerabkova_mecl",
    "alpha3_jerabkova_rho",
    "alpha3_marks_plane",
    "alpha3_marks_table3",
    # Low-mass slopes
    "lowmass_slopes_metallicity",
    # Coefficients
    "JERABKOVA_COEFFICIENTS",
    "MARKS_COEFFICIENTS",
    "MARKS_TABLE3_COEFFICIENTS",
    "DEFAULT_SFE",
]

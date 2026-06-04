"""Cluster density / radius helpers (split from environment.py)."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


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
    """Half-mass density (Marks & Kroupa 2012 definition).

    ρ_ecl = 3 × M_ecl / (8π × r_h³)

    The 8π factor arises because half the cluster mass is within r_h:
        ρ = (M_ecl/2) / (4π/3 × r_h³) = 3M_ecl / (8π × r_h³)

    This is the authoritative definition in Marks & Kroupa (2012), A&A 543, A8 (p. 2,
    "ρ_ecl = 3 M_ecl/8π r_h³"), and reproduces Marks+2012 (MNRAS 422, 2246) Table 1
    densities exactly (e.g. NGC 104: 3·9.40e6/(8π·0.49³)=9.54e6 = the tabulated ρ_cl).
    NOTE: Jerabkova+2018 Eq. 8 writes 4π, but that is internally inconsistent with its
    own ρ_ecl=0.61·logM+2.08 relation (which is 8π); progenax follows the 8π convention
    that matches the actual α₃–ρ calibration data.

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



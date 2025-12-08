"""Differentiable IMF sampling and likelihood functions.

This module provides JAX-native functions for:
- Evaluating IMF probability density (log_prob_masses)
- Sampling masses via inverse CDF (sample_masses_from_params)
- Computing likelihood for observed masses (individual_mass_nll)

All functions are fully differentiable and JIT-compatible.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .params import IMFParams


def _compute_normalization(params: IMFParams) -> Float[Array, ""]:
    """Compute normalization constant for piecewise power-law IMF.

    Integrates ξ(m) = k * m^(-α) over each segment.

    For α ≠ 1: ∫ m^(-α) dm = m^(1-α) / (1-α)
    """
    m_min = params.m_min
    m_b1 = params.m_break1
    m_b2 = params.m_break2
    m_max = params.m_max

    a1 = params.alpha_low
    a2 = params.alpha_mid
    a3 = params.alpha_high

    # Integral of m^(-α) from a to b: [m^(1-α)/(1-α)]_a^b
    def power_integral(m_lo, m_hi, alpha):
        exp = 1.0 - alpha
        return (m_hi**exp - m_lo**exp) / exp

    # Segment 1: [m_min, m_b1)
    I1 = power_integral(m_min, m_b1, a1)

    # Continuity factor at m_b1: C1 such that C1 * m_b1^(-a2) = m_b1^(-a1)
    C1 = m_b1 ** (a2 - a1)

    # Segment 2: [m_b1, m_b2)
    I2 = C1 * power_integral(m_b1, m_b2, a2)

    # Continuity factor at m_b2
    C2 = C1 * m_b2 ** (a3 - a2)

    # Segment 3: [m_b2, m_max]
    I3 = C2 * power_integral(m_b2, m_max, a3)

    return I1 + I2 + I3


def log_prob_masses(
    masses: Float[Array, "N"],
    params: IMFParams,
) -> Float[Array, "N"]:
    """Compute log probability of each mass under the IMF.

    Evaluates the normalized piecewise power-law PDF:
        log p(m | params) = log(ξ(m)) - log(normalization)

    where ξ(m) is the unnormalized IMF.

    Args:
        masses: Stellar masses [M☉], shape (N,)
        params: IMF parameters

    Returns:
        Log probability for each mass, shape (N,)

    Example:
        >>> params = IMFParams.kroupa()
        >>> masses = jnp.array([0.5, 1.0, 10.0])
        >>> log_probs = log_prob_masses(masses, params)
    """
    m_b1 = params.m_break1
    m_b2 = params.m_break2

    a1 = params.alpha_low
    a2 = params.alpha_mid
    a3 = params.alpha_high

    # Continuity factors
    C1 = m_b1 ** (a2 - a1)
    C2 = C1 * m_b2 ** (a3 - a2)

    # Determine which segment each mass belongs to
    in_seg1 = masses < m_b1
    in_seg2 = (masses >= m_b1) & (masses < m_b2)
    in_seg3 = masses >= m_b2

    # Unnormalized log PDF (log of ξ(m) = C * m^(-α))
    log_xi = jnp.where(
        in_seg1,
        -a1 * jnp.log(masses),
        jnp.where(
            in_seg2,
            jnp.log(C1) - a2 * jnp.log(masses),
            jnp.log(C2) - a3 * jnp.log(masses),
        ),
    )

    # Normalize
    norm = _compute_normalization(params)
    log_prob = log_xi - jnp.log(norm)

    return log_prob


__all__ = ["log_prob_masses"]

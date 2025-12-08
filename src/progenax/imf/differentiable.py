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


def _compute_cdf_at_breaks(params: IMFParams) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Compute CDF values at mass break points.

    Returns (F(m_b1), F(m_b2)) where F is the cumulative distribution.
    """
    m_min = params.m_min
    m_b1 = params.m_break1
    m_b2 = params.m_break2

    a1 = params.alpha_low
    a2 = params.alpha_mid
    a3 = params.alpha_high

    norm = _compute_normalization(params)

    # Integral from m_min to m_b1
    def power_integral(m_lo, m_hi, alpha):
        exp = 1.0 - alpha
        return (m_hi**exp - m_lo**exp) / exp

    I1 = power_integral(m_min, m_b1, a1)
    F_b1 = I1 / norm

    C1 = m_b1 ** (a2 - a1)
    I2 = C1 * power_integral(m_b1, m_b2, a2)
    F_b2 = (I1 + I2) / norm

    return F_b1, F_b2


def sample_masses_from_params(
    params: IMFParams,
    u: Float[Array, "N"],
) -> Float[Array, "N"]:
    """Sample masses via inverse CDF - fully differentiable.

    Uses the reparameterization trick: masses = F⁻¹(u; params)
    where F is the cumulative distribution function.

    Gradients flow through params, not through the random samples u.

    Args:
        params: IMF parameters
        u: Uniform samples in [0, 1], shape (N,)

    Returns:
        Sampled masses [M☉], shape (N,)

    Example:
        >>> params = IMFParams.kroupa()
        >>> key = jax.random.PRNGKey(42)
        >>> u = jax.random.uniform(key, (1000,))
        >>> masses = sample_masses_from_params(params, u)
    """
    m_min = params.m_min
    m_b1 = params.m_break1
    m_b2 = params.m_break2
    m_max = params.m_max

    a1 = params.alpha_low
    a2 = params.alpha_mid
    a3 = params.alpha_high

    # Get CDF values at breaks
    F_b1, F_b2 = _compute_cdf_at_breaks(params)
    norm = _compute_normalization(params)

    # Continuity factors
    C1 = m_b1 ** (a2 - a1)
    C2 = C1 * m_b2 ** (a3 - a2)

    # Inverse CDF for each segment
    # For segment with ξ(m) = C * m^(-α):
    # F(m) - F(m_lo) = C * [m^(1-α) - m_lo^(1-α)] / [(1-α) * norm]
    # Solving for m: m = [m_lo^(1-α) + (u - F(m_lo)) * norm * (1-α) / C]^(1/(1-α))

    def inv_cdf_segment(u_val, m_lo, F_lo, C, alpha):
        """Inverse CDF within a single power-law segment."""
        exp = 1.0 - alpha
        inner = m_lo**exp + (u_val - F_lo) * norm * exp / C
        return inner ** (1.0 / exp)

    # Determine segment for each u
    in_seg1 = u < F_b1
    in_seg2 = (u >= F_b1) & (u < F_b2)
    in_seg3 = u >= F_b2

    # Compute mass for each segment
    m_seg1 = inv_cdf_segment(u, m_min, 0.0, 1.0, a1)
    m_seg2 = inv_cdf_segment(u, m_b1, F_b1, C1, a2)
    m_seg3 = inv_cdf_segment(u, m_b2, F_b2, C2, a3)

    # Select based on segment
    masses = jnp.where(
        in_seg1,
        m_seg1,
        jnp.where(in_seg2, m_seg2, m_seg3),
    )

    # Clip to valid range (numerical safety)
    masses = jnp.clip(masses, m_min, m_max)

    return masses


def individual_mass_nll(
    masses: Float[Array, "N"],
    params: IMFParams,
) -> Float[Array, ""]:
    """Compute negative log-likelihood for individual resolved masses.

    NLL = -Σᵢ log p(mᵢ | params)

    This is the simplest likelihood for gradient-based IMF inference.

    Args:
        masses: Observed stellar masses [M☉], shape (N,)
        params: IMF parameters to evaluate

    Returns:
        Negative log-likelihood (scalar)

    Example:
        >>> params = IMFParams.kroupa()
        >>> masses = jnp.array([0.5, 1.0, 10.0, 50.0])
        >>> nll = individual_mass_nll(masses, params)
        >>> # Minimize NLL to fit params to data
    """
    log_probs = log_prob_masses(masses, params)
    return -jnp.sum(log_probs)


__all__ = ["log_prob_masses", "sample_masses_from_params", "individual_mass_nll"]

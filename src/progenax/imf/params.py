"""Differentiable IMF parameters for inference.

This module provides IMFParams, a JAX-compatible dataclass representing
the parameters of a piecewise power-law IMF. Designed for gradient-based
inference of IMF slopes from stellar mass data.

References:
    Kroupa (2001), MNRAS 322, 231 - Canonical IMF slopes
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class IMFParams(eqx.Module):
    """Differentiable IMF parameters - primary inference target.

    Represents a 3-segment piecewise power-law IMF:
        ξ(m) ∝ m^(-α_low)   for m < m_break1
        ξ(m) ∝ m^(-α_mid)   for m_break1 ≤ m < m_break2
        ξ(m) ∝ m^(-α_high)  for m ≥ m_break2

    v1 Strategy:
    - Fixed breaks at 0.08, 0.5 M☉ (well-established from observations)
    - Primary inference target: alpha_high (main science driver)
    - Secondary: alpha_mid (optional refinement)
    - alpha_low rarely constrained by data

    Attributes:
        alpha_low: Power-law slope below 0.08 M☉ (typically ~0.3)
        alpha_mid: Power-law slope 0.08-0.5 M☉ (typically ~1.3)
        alpha_high: Power-law slope above 0.5 M☉ (typically ~2.3)
        m_break1: First mass break [M☉] (fixed at 0.08)
        m_break2: Second mass break [M☉] (fixed at 0.50)
        m_min: Minimum stellar mass [M☉] (fixed at 0.01)
        m_max: Maximum stellar mass [M☉] (fixed at 150.0)

    Example:
        >>> params = IMFParams.kroupa()
        >>> print(f"High-mass slope: {params.alpha_high}")
        High-mass slope: 2.3

        >>> # Custom top-heavy IMF
        >>> params = IMFParams(
        ...     alpha_low=jnp.array(0.3),
        ...     alpha_mid=jnp.array(1.3),
        ...     alpha_high=jnp.array(1.8),  # Top-heavy
        ... )
    """

    # Inference targets (JAX arrays for autodiff)
    alpha_low: Float[Array, ""]
    alpha_mid: Float[Array, ""]
    alpha_high: Float[Array, ""]

    # Fixed in v1 (static - not traced by JAX)
    m_break1: float = eqx.field(static=True, default=0.08)
    m_break2: float = eqx.field(static=True, default=0.50)
    m_min: float = eqx.field(static=True, default=0.01)
    m_max: float = eqx.field(static=True, default=150.0)

    @classmethod
    def kroupa(cls) -> "IMFParams":
        """Create canonical Kroupa (2001) IMF parameters.

        Returns:
            IMFParams with standard Kroupa slopes:
            - α_low = 0.3 (below 0.08 M☉)
            - α_mid = 1.3 (0.08-0.5 M☉)
            - α_high = 2.3 (above 0.5 M☉)

        References:
            Kroupa (2001), MNRAS 322, 231
        """
        return cls(
            alpha_low=jnp.array(0.3),
            alpha_mid=jnp.array(1.3),
            alpha_high=jnp.array(2.3),
        )


__all__ = ["IMFParams"]

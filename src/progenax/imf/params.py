"""Differentiable IMF parameters for inference.

This module provides IMFParams, a JAX-compatible dataclass representing
the parameters of a 4-segment piecewise power-law IMF. Designed for
gradient-based inference of IMF slopes from stellar mass data.

The 4-segment convention matches Marks+2012 exactly:
    - Segment 0: [0.01, 0.08] M☉ - sub-stellar/brown dwarf
    - Segment 1: [0.08, 0.50] M☉ - low-mass stars
    - Segment 2: [0.50, 1.00] M☉ - intermediate-mass
    - Segment 3: [1.00, m_max] M☉ - high-mass (main science target)

References:
    Kroupa (2001), MNRAS 322, 231 - Canonical IMF slopes
    Marks et al. (2012), MNRAS 422, 2246 - Environment-dependent IMF
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class IMFParams(eqx.Module):
    """4-segment piecewise power-law IMF parameters.

    Matches Marks+2012 Eq. 2 convention:
        ξ(m) ∝ m^(-α₀) for m ∈ [0.01, 0.08] M☉ (sub-stellar, typically FIXED)
        ξ(m) ∝ m^(-α₁) for m ∈ [0.08, 0.50] M☉ (low-mass)
        ξ(m) ∝ m^(-α₂) for m ∈ [0.50, 1.00] M☉ (intermediate)
        ξ(m) ∝ m^(-α₃) for m ∈ [1.00, m_max] M☉ (high-mass, main target)

    Primary inference targets:
        - α₃: Environment-dependent high-mass slope (main science driver)
        - α₁, α₂: Metallicity-dependent (optional refinement via Marks Eq. 12)
        - α₀: Fixed at 0.3 (brown dwarf regime, rarely constrained)

    Attributes:
        alpha0: Slope for [0.01, 0.08] M☉ (canonical 0.3, typically FIXED)
        alpha1: Slope for [0.08, 0.50] M☉ (canonical 1.3)
        alpha2: Slope for [0.50, 1.00] M☉ (canonical 2.3)
        alpha3: Slope for [1.00, m_max] M☉ (canonical 2.3, environment-dependent)
        m_break0: First break at 0.08 M☉ (static)
        m_break1: Second break at 0.50 M☉ (static)
        m_break2: Third break at 1.00 M☉ (static)
        m_min: Minimum mass 0.01 M☉ (static)
        m_max: Maximum mass 150.0 M☉ (static)

    Example:
        >>> params = IMFParams.kroupa()
        >>> print(f"High-mass slope: {params.alpha3}")
        High-mass slope: 2.3

        >>> # Top-heavy IMF (e.g., in dense starburst)
        >>> params = IMFParams(
        ...     alpha0=jnp.array(0.3),
        ...     alpha1=jnp.array(1.3),
        ...     alpha2=jnp.array(2.3),
        ...     alpha3=jnp.array(1.8),  # Top-heavy
        ... )
    """

    # Inference targets (JAX arrays for autodiff)
    alpha0: Float[Array, ""]  # Sub-stellar, typically fixed at 0.3
    alpha1: Float[Array, ""]  # Low-mass, canonical 1.3
    alpha2: Float[Array, ""]  # Intermediate, canonical 2.3
    alpha3: Float[Array, ""]  # High-mass, canonical 2.3 (environment-dependent)

    # Fixed breakpoints (static - not traced by JAX)
    m_break0: float = eqx.field(static=True, default=0.08)
    m_break1: float = eqx.field(static=True, default=0.50)
    m_break2: float = eqx.field(static=True, default=1.00)
    m_min: float = eqx.field(static=True, default=0.01)
    m_max: float = eqx.field(static=True, default=150.0)

    @classmethod
    def kroupa(cls) -> "IMFParams":
        """Create canonical Kroupa (2001) IMF parameters.

        Returns:
            IMFParams with standard Kroupa slopes:
            - α₀ = 0.3 (sub-stellar, [0.01, 0.08] M☉)
            - α₁ = 1.3 (low-mass, [0.08, 0.50] M☉)
            - α₂ = 2.3 (intermediate, [0.50, 1.00] M☉)
            - α₃ = 2.3 (high-mass, [1.00, m_max] M☉)

        References:
            Kroupa (2001), MNRAS 322, 231
            Marks et al. (2012), MNRAS 422, 2246 (4-segment convention)
        """
        return cls(
            alpha0=jnp.array(0.3),
            alpha1=jnp.array(1.3),
            alpha2=jnp.array(2.3),
            alpha3=jnp.array(2.3),
        )


__all__ = ["IMFParams"]

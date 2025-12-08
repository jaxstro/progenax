# progenax/src/progenax/imf/environment_v2.py
"""Differentiable birth environment for environment-dependent IMF.

This module provides BirthEnvironment, representing the physical conditions
at star formation that may influence the IMF. Designed for gradient-based
inference of birth environment from observed stellar masses.

References:
    Marks et al. (2012), MNRAS 422, 2246 - Density-dependent α₃
    Jeřábková et al. (2018), A&A 620, A39 - Density + metallicity dependence
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class BirthEnvironment(eqx.Module):
    """Physical birth environment - CAN be inference target.

    Represents conditions during star formation that may affect IMF:
    - Gas density (correlates with cluster mass, affects massive star formation)
    - Metallicity (affects cooling, fragmentation)
    - Star formation rate (optional, for IGIMF-style models)

    Use cases:
    1. Fixed context: provide environment as prior, infer IMF params
    2. Inference target: "What environment produced this observed IMF?"

    Attributes:
        log_density: log₁₀(ρ / M☉ pc⁻³), typical range [2, 6]
        metallicity: [Fe/H], typical range [-2, +0.5]
        sfr: Star formation rate [M☉/yr], default 1.0

    Example:
        >>> # Dense, metal-poor environment (early universe)
        >>> env = BirthEnvironment(
        ...     log_density=jnp.array(5.0),
        ...     metallicity=jnp.array(-1.5),
        ... )
    """

    log_density: Float[Array, ""]
    metallicity: Float[Array, ""]
    sfr: Float[Array, ""] = eqx.field(default_factory=lambda: jnp.array(1.0))


__all__ = ["BirthEnvironment"]

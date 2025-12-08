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

from .params import IMFParams


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


def env_to_imf_params(
    env: BirthEnvironment,
    model: str = "universal_kroupa",
) -> IMFParams:
    """Map birth environment to IMF parameters.

    Pure JAX function - fully differentiable. Gradients flow from
    IMFParams back to BirthEnvironment.

    Models:
        - "universal_kroupa": Standard Kroupa (2001), ignores environment
        - "marks2012_like": α₃ decreases with density (PROVISIONAL coefficients)
        - "jerabkova2018_like": α₃ depends on density + metallicity (PROVISIONAL)

    IMPORTANT: The "marks2012_like" and "jerabkova2018_like" models use
    placeholder coefficients that capture the qualitative trends but are
    NOT calibrated to the original papers. Use for prototyping only.

    Args:
        env: Birth environment conditions
        model: Model name (see above)

    Returns:
        IMFParams with slopes determined by model prescription

    Raises:
        ValueError: If model name is unknown

    Example:
        >>> env = BirthEnvironment(log_density=jnp.array(5.0), metallicity=jnp.array(-0.5))
        >>> params = env_to_imf_params(env, model="marks2012_like")
        >>> print(f"α_high = {params.alpha_high}")  # Will be < 2.3 due to high density
    """
    if model == "universal_kroupa":
        # Standard Kroupa - environment independent
        return IMFParams(
            alpha_low=jnp.array(0.3),
            alpha_mid=jnp.array(1.3),
            alpha_high=jnp.array(2.3),
        )

    elif model == "marks2012_like":
        # PROVISIONAL: α₃ decreases with density above threshold
        # Qualitative trend: dense environments → top-heavy IMF
        # Actual Marks+2012 uses more complex prescription
        density_excess = jnp.clip(env.log_density - 3.0, 0.0, 5.0)
        alpha_high = 2.3 - 0.15 * density_excess

        return IMFParams(
            alpha_low=jnp.array(0.3),
            alpha_mid=jnp.array(1.3),
            alpha_high=alpha_high,
        )

    elif model == "jerabkova2018_like":
        # PROVISIONAL: adds metallicity dependence to Marks-like model
        # Qualitative: metal-rich environments may have slightly steeper IMF
        density_excess = jnp.clip(env.log_density - 3.0, 0.0, 5.0)
        alpha_high = 2.3 - 0.15 * density_excess + 0.1 * env.metallicity

        return IMFParams(
            alpha_low=jnp.array(0.3),
            alpha_mid=jnp.array(1.3),
            alpha_high=alpha_high,
        )

    else:
        raise ValueError(
            f"Unknown model: {model}. "
            f"Available: 'universal_kroupa', 'marks2012_like', 'jerabkova2018_like'"
        )


__all__ = ["BirthEnvironment", "env_to_imf_params"]

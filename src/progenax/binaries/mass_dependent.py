"""Radial and mass-dependent binary prescriptions.

- :class:`RadialBinaryFraction` — a phenomenological radial binary-fraction knob.
- :class:`MassDependentBinaryConfig` / :func:`sample_mass_dependent_orbits` —
  route stars to low-/high-mass period+eccentricity distributions by primary mass.

References:
    Duquennoy & Mayor (1991) A&A 248, 485 — solar-type (low-mass) binaries.
    Sana et al. (2012) Science 337, 444; Moe & Di Stefano (2017) ApJS 230, 15 —
        massive (high-mass) binaries.
"""

from __future__ import annotations

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from .eccentricity import (
    LogisticThermalEccentricity,
    MoeEccentricity,
    ThermalEccentricity,
    UniformEccentricity,
)
from .period import LogNormalPeriod, LogUniformPeriod, SanaOBPeriod


class RadialBinaryFraction(eqx.Module):
    """Phenomenological radially-varying binary fraction.

    A simple parametric knob for a spatially varying binary fraction:

        f_b(r) = fb0 × (1 + A × (r/r_scale)^(-α)),  clipped to [0, 1]

    where:
        - A > 0: core-enhanced (more binaries in center)
        - A < 0: core-depleted (fewer binaries in center)
        - A = 0: constant binary fraction everywhere

    NOTE: this functional form is a PHENOMENOLOGICAL model, NOT taken from any
    specific paper. The references below establish that the binary fraction
    *varies* with primary mass / environment (motivating a spatial knob), but none
    provides this closed-form radial f_b(r) profile. The (r/r_scale)^(-α) term
    diverges as r -> 0 for α > 0; the clip to [0, 1] caps it (so the core value is
    set by the clip, not the power law) — intended for r > 0.

    References (motivation only — not the source of this functional form):
        Raghavan et al. (2010) ApJS 190, 1 - solar-neighborhood multiplicity census.
        Sana et al. (2012) Science 337, 444 - O-star binary fraction.
        Moe & Di Stefano (2017) ApJS 230, 15 - binary statistics review.

    Parameters:
        fb0: Baseline binary fraction (default: 0.5)
        A: Amplitude of radial variation (default: 0.5 for core-enhanced)
        alpha: Power-law index (default: 1.0)
        r_scale: Scale radius for radial variation (default: 1.0)

    Examples:
        >>> # Core-enhanced: more binaries in center
        >>> rbf = RadialBinaryFraction(fb0=0.5, A=0.5, alpha=1.0, r_scale=1.0)
        >>> radii = jnp.array([0.1, 1.0, 5.0])
        >>> fb_r = rbf.compute(radii)  # Higher at r=0.1, lower at r=5.0
        >>>
        >>> # Sample binary membership
        >>> key = jax.random.PRNGKey(42)
        >>> is_binary = rbf.sample_membership(radii, key)
    """

    fb0: float = 0.5
    A: float = 0.5
    alpha: float = 1.0
    r_scale: float = 1.0

    def compute(self, radii: Float[Array, "N"]) -> Float[Array, "N"]:
        """Compute binary fraction at given radii.

        Args:
            radii: Radial distances (shape N,)

        Returns:
            Binary fraction at each radius, clipped to [0, 1] (shape N,)
        """
        # f_b(r) = fb0 × (1 + A × (r/r_scale)^(-α))
        r_normalized = radii / self.r_scale
        fb_r = self.fb0 * (1.0 + self.A * jnp.power(r_normalized, -self.alpha))

        # Clip to valid range [0, 1]
        return jnp.clip(fb_r, 0.0, 1.0)

    def sample_membership(
        self, radii: Float[Array, "N"], key: PRNGKeyArray
    ) -> Bool[Array, "N"]:
        """Sample binary membership based on radial binary fraction.

        Each particle becomes a binary with probability f_b(r) where r is its radius.

        Args:
            radii: Radial distances for each particle (shape N,)
            key: JAX random key

        Returns:
            Boolean array indicating binary membership (shape N,)
            True = particle is a binary, False = single star
        """
        fb_r = self.compute(radii)
        u = jax.random.uniform(key, radii.shape)
        return u < fb_r


class MassDependentBinaryConfig(eqx.Module):
    """Configuration for mass-dependent binary orbital parameter sampling.

    Routes stars to different period/eccentricity distributions based on mass:
    - Low-mass stars (M < m_break): Solar-type binaries (Duquennoy & Mayor)
    - High-mass stars (M >= m_break): O/B-type binaries (Sana+2012, Moe+2017)

    References:
        Sana et al. (2012) Science 337, 444 - O-star binary fraction
        Moe & Di Stefano (2017) ApJS 230, 15 - Binary statistics review
        Duquennoy & Mayor (1991) A&A 248, 485 - Solar-type binary periods

    Parameters:
        m_break: Mass threshold [Msun] separating low/high-mass prescriptions (default: 8.0)
        low_mass_period: Period distribution for M < m_break
        high_mass_period: Period distribution for M >= m_break
        low_mass_eccentricity: Eccentricity distribution for M < m_break
        high_mass_eccentricity: Eccentricity distribution for M >= m_break

    Example:
        >>> config = MassDependentBinaryConfig(
        ...     m_break=8.0,
        ...     low_mass_period=LogNormalPeriod(mu_log_P=4.8, sigma_log_P=2.3),
        ...     high_mass_period=SanaOBPeriod(),
        ...     low_mass_eccentricity=ThermalEccentricity(),
        ...     high_mass_eccentricity=MoeEccentricity(),
        ... )
        >>> masses = jnp.array([1.0, 5.0, 10.0, 20.0])
        >>> key = jax.random.PRNGKey(42)
        >>> periods, ecc = sample_mass_dependent_orbits(masses, config, key)
    """

    m_break: float
    low_mass_period: LogNormalPeriod | LogUniformPeriod
    high_mass_period: SanaOBPeriod
    low_mass_eccentricity: ThermalEccentricity | UniformEccentricity
    high_mass_eccentricity: MoeEccentricity | LogisticThermalEccentricity


def sample_mass_dependent_orbits(
    masses: Float[Array, "N"],
    config: MassDependentBinaryConfig,
    key: PRNGKeyArray,
) -> Tuple[Float[Array, "N"], Float[Array, "N"]]:
    """Sample orbital parameters with mass-dependent prescriptions.

    Routes each star to appropriate period/eccentricity distribution based on mass:
    - M < m_break: Low-mass prescription (e.g., solar-type binaries)
    - M >= m_break: High-mass prescription (e.g., O/B-type binaries)

    Uses JAX-native branching (jnp.where) for JIT compatibility.

    Args:
        masses: Stellar masses [Msun] (shape N,)
        config: Mass-dependent binary configuration
        key: JAX random key

    Returns:
        Tuple of:
            - periods: Orbital periods [days] (shape N,)
            - eccentricities: Orbital eccentricities (shape N,)

    Note:
        Only the high-mass branch supports period-dependent eccentricity
        (MoeEccentricity, conditioned on the high-mass periods). The low-mass
        eccentricity is sampled unconditionally (Thermal/Uniform), enforced by
        the config type hints.

    Example:
        >>> config = MassDependentBinaryConfig(
        ...     m_break=8.0,
        ...     low_mass_period=LogNormalPeriod(),
        ...     high_mass_period=SanaOBPeriod(),
        ...     low_mass_eccentricity=ThermalEccentricity(),
        ...     high_mass_eccentricity=MoeEccentricity(),
        ... )
        >>> masses = jnp.array([1.0, 5.0, 10.0, 20.0])
        >>> key = jax.random.PRNGKey(42)
        >>> periods, ecc = sample_mass_dependent_orbits(masses, config, key)
    """
    N = masses.shape[0]
    # One split: the four sub-keys below are independent; we deliberately consume
    # one slot of the first split so period/eccentricity entropy is separable if
    # this is later extended. (The second slot is unused by design.)
    key_period, _ = jax.random.split(key)

    # Sample from both distributions (always sample both for JIT)
    key_p_low, key_p_high, key_e_low, key_e_high = jax.random.split(key_period, 4)

    # Sample periods from both distributions
    periods_low = config.low_mass_period.sample(key_p_low, N)
    periods_high = config.high_mass_period.sample(key_p_high, N)

    # Sample eccentricities from both distributions. The high-mass eccentricity
    # is period+mass conditional (faithful MoeEccentricity uses both; the
    # LogisticThermalEccentricity heuristic accepts masses but ignores them), so
    # it is always passed (periods_high, masses). The low-mass eccentricity is
    # unconditional (Thermal/Uniform).
    ecc_low = config.low_mass_eccentricity.sample(key_e_low, N)
    ecc_high = config.high_mass_eccentricity.sample(key_e_high, periods_high, masses)

    # Route based on mass threshold
    is_high_mass = masses >= config.m_break

    # Select appropriate distribution
    periods = jnp.where(is_high_mass, periods_high, periods_low)
    eccentricities = jnp.where(is_high_mass, ecc_high, ecc_low)

    return periods, eccentricities


__all__ = [
    "RadialBinaryFraction",
    "MassDependentBinaryConfig",
    "sample_mass_dependent_orbits",
]

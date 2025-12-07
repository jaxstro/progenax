"""Binary population parameter distributions.

Implements orbital parameter sampling for binary populations:
- Period distributions (Öpik, log-normal, Moe+17)
- Eccentricity distributions (thermal, Moe+17)
- Orientation sampling (isotropic)

References:
    Öpik (1924) - Log-uniform period distribution
    Duquennoy & Mayor (1991) A&A 248, 485 - Log-normal periods for solar-type
    Moe & Di Stefano (2017) ApJS 230, 15 - Comprehensive binary statistics
    Heggie (1975) MNRAS 173, 729 - Thermal eccentricity distribution
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray


class LogUniformPeriod(eqx.Module):
    """Log-uniform (Öpik) period distribution.

    Öpik's law: binary periods are uniformly distributed in log space.
    p(log P) = const  =>  p(P) ∝ 1/P

    Reference:
        Öpik (1924) Publications de l'Observatoire Astronomique de l'Université de Tartu

    Parameters:
        log_P_min: Minimum log10(P/days) (default: 0.0 = 1 day)
        log_P_max: Maximum log10(P/days) (default: 8.0 = ~27,000 years)
    """

    log_P_min: float = 0.0
    log_P_max: float = 8.0

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample n periods [days]."""
        u = jax.random.uniform(key, (n,))
        log_P = self.log_P_min + u * (self.log_P_max - self.log_P_min)
        return 10.0 ** log_P

    def pdf(self, P: Float[Array, "..."]) -> Float[Array, "..."]:
        """PDF: p(P) = 1 / (P * ln(10) * (log_P_max - log_P_min))."""
        log_P = jnp.log10(P)
        in_range = (log_P >= self.log_P_min) & (log_P <= self.log_P_max)
        norm = jnp.log(10.0) * (self.log_P_max - self.log_P_min)
        return jnp.where(in_range, 1.0 / (P * norm), 0.0)

    def cdf(self, P: Float[Array, "..."]) -> Float[Array, "..."]:
        """CDF: F(P) = (log P - log_P_min) / (log_P_max - log_P_min)."""
        log_P = jnp.log10(P)
        cdf_val = (log_P - self.log_P_min) / (self.log_P_max - self.log_P_min)
        return jnp.clip(cdf_val, 0.0, 1.0)

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Inverse CDF."""
        log_P = self.log_P_min + u * (self.log_P_max - self.log_P_min)
        return 10.0 ** log_P


class LogNormalPeriod(eqx.Module):
    """Log-normal period distribution.

    log10(P) ~ Normal(mu, sigma)

    Reference:
        Duquennoy & Mayor (1991) A&A 248, 485
        For solar-type stars: mu ≈ 4.8, sigma ≈ 2.3 (P in days)

    Parameters:
        mu_log_P: Mean of log10(P/days) (default: 4.8)
        sigma_log_P: Std dev of log10(P/days) (default: 2.3)
    """

    mu_log_P: float = 4.8
    sigma_log_P: float = 2.3

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample n periods [days]."""
        log_P = self.mu_log_P + self.sigma_log_P * jax.random.normal(key, (n,))
        return 10.0 ** log_P

    def pdf(self, P: Float[Array, "..."]) -> Float[Array, "..."]:
        """Log-normal PDF."""
        log_P = jnp.log10(P)
        z = (log_P - self.mu_log_P) / self.sigma_log_P
        p_log = jnp.exp(-0.5 * z**2) / (self.sigma_log_P * jnp.sqrt(2 * jnp.pi))
        return p_log / (P * jnp.log(10.0))

    def cdf(self, P: Float[Array, "..."]) -> Float[Array, "..."]:
        """Log-normal CDF."""
        log_P = jnp.log10(P)
        z = (log_P - self.mu_log_P) / self.sigma_log_P
        return 0.5 * (1.0 + jax.scipy.special.erf(z / jnp.sqrt(2.0)))

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Inverse CDF via inverse error function."""
        z = jnp.sqrt(2.0) * jax.scipy.special.erfinv(2.0 * u - 1.0)
        log_P = self.mu_log_P + self.sigma_log_P * z
        return 10.0 ** log_P


# =============================================================================
# Eccentricity Distributions
# =============================================================================


class ThermalEccentricity(eqx.Module):
    """Thermal eccentricity distribution f(e) = 2e.

    Arises from energy equipartition in dynamically relaxed systems.
    CDF: F(e) = e²  =>  PPF: e = √u

    Reference:
        Heggie (1975) MNRAS 173, 729
        Jeans (1919) "Problems of Cosmogony and Stellar Dynamics"

    Parameters:
        e_max: Maximum eccentricity (default: 0.99, avoids singularity)
    """

    e_max: float = 0.99

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample n eccentricities from f(e) = 2e."""
        u = jax.random.uniform(key, (n,))
        return self.e_max * jnp.sqrt(u)

    def pdf(self, e: Float[Array, "..."]) -> Float[Array, "..."]:
        """PDF: p(e) = 2e / e_max²."""
        in_range = (e >= 0.0) & (e <= self.e_max)
        return jnp.where(in_range, 2.0 * e / self.e_max**2, 0.0)

    def cdf(self, e: Float[Array, "..."]) -> Float[Array, "..."]:
        """CDF: F(e) = e² / e_max²."""
        cdf_val = (e / self.e_max) ** 2
        return jnp.clip(cdf_val, 0.0, 1.0)

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Inverse CDF: e = e_max × √u."""
        return self.e_max * jnp.sqrt(u)


class UniformEccentricity(eqx.Module):
    """Uniform eccentricity distribution.

    Simple uniform distribution, useful for circular-dominated populations.

    Parameters:
        e_min: Minimum eccentricity (default: 0.0)
        e_max: Maximum eccentricity (default: 0.9)
    """

    e_min: float = 0.0
    e_max: float = 0.9

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample n eccentricities uniformly."""
        u = jax.random.uniform(key, (n,))
        return self.e_min + u * (self.e_max - self.e_min)

    def pdf(self, e: Float[Array, "..."]) -> Float[Array, "..."]:
        """Uniform PDF."""
        in_range = (e >= self.e_min) & (e <= self.e_max)
        return jnp.where(in_range, 1.0 / (self.e_max - self.e_min), 0.0)

    def cdf(self, e: Float[Array, "..."]) -> Float[Array, "..."]:
        """Uniform CDF."""
        cdf_val = (e - self.e_min) / (self.e_max - self.e_min)
        return jnp.clip(cdf_val, 0.0, 1.0)

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Inverse CDF."""
        return self.e_min + u * (self.e_max - self.e_min)


# =============================================================================
# Orientation Sampling
# =============================================================================


def sample_isotropic_orientations(
    key: PRNGKeyArray,
    n: int,
) -> Tuple[Float[Array, "n"], Float[Array, "n"], Float[Array, "n"], Float[Array, "n"]]:
    """Sample isotropic orbital orientations.

    For randomly oriented orbits in 3D space:
    - cos(i) ~ U(-1, 1)  =>  i = arccos(u) where u ~ U(-1, 1)
    - Ω ~ U(0, 2π)  (longitude of ascending node)
    - ω ~ U(0, 2π)  (argument of periapsis)
    - M₀ ~ U(0, 2π)  (mean anomaly at epoch)

    Args:
        key: JAX random key
        n: Number of orientations to sample

    Returns:
        Tuple of (inclination, Omega, omega, M_anom) arrays, each shape (n,)
        - inclination: [0, π] radians
        - Omega: [0, 2π) radians
        - omega: [0, 2π) radians
        - M_anom: [0, 2π) radians

    Reference:
        Binney & Tremaine (2008) "Galactic Dynamics" Section 3.1
    """
    key1, key2, key3, key4 = jax.random.split(key, 4)

    # Inclination: cos(i) ~ U(-1, 1) for isotropic
    cos_i = jax.random.uniform(key1, (n,), minval=-1.0, maxval=1.0)
    inclination = jnp.arccos(cos_i)

    # Other angles: uniform on [0, 2π)
    Omega = jax.random.uniform(key2, (n,), minval=0.0, maxval=2.0 * jnp.pi)
    omega = jax.random.uniform(key3, (n,), minval=0.0, maxval=2.0 * jnp.pi)
    M_anom = jax.random.uniform(key4, (n,), minval=0.0, maxval=2.0 * jnp.pi)

    return inclination, Omega, omega, M_anom


# =============================================================================
# Radially Varying Binary Fraction
# =============================================================================


class SanaOBPeriod(eqx.Module):
    """Sana+2012 period distribution for O/B stars.

    Power-law distribution in log-space:
        p(log P) ∝ (log P)^(-0.55)

    for log P in [0.3, 3.5] (P in days).

    This corresponds to shorter periods than solar-type binaries,
    consistent with observations of massive star binaries.

    Reference:
        Sana et al. (2012) Science 337, 444 - O-star binary survey

    Parameters:
        log_P_min: Minimum log10(P/days) (default: 0.3 = ~2 days)
        log_P_max: Maximum log10(P/days) (default: 3.5 = ~3162 days)
        power: Power-law index (default: -0.55 from Sana+2012)
    """

    log_P_min: float = 0.3
    log_P_max: float = 3.5
    power: float = -0.55

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample n periods [days] from power-law distribution.

        Uses inverse transform sampling:
            p(log P) ∝ (log P)^α  =>  CDF: F(log P) = [(log P)^(α+1) - min^(α+1)] / [max^(α+1) - min^(α+1)]
        """
        u = jax.random.uniform(key, (n,))

        # Power-law CDF inversion
        # For p(x) ∝ x^α with x ∈ [a, b]:
        # F(x) = [x^(α+1) - a^(α+1)] / [b^(α+1) - a^(α+1)]
        # F^(-1)(u) = [u × (b^(α+1) - a^(α+1)) + a^(α+1)]^(1/(α+1))

        alpha = self.power
        a = self.log_P_min
        b = self.log_P_max

        # Special case: α = -1 (log-uniform)
        is_log_uniform = jnp.abs(alpha + 1.0) < 1e-10

        # General power-law case
        a_pow = a ** (alpha + 1.0)
        b_pow = b ** (alpha + 1.0)
        log_P_general = jnp.power(
            u * (b_pow - a_pow) + a_pow,
            1.0 / (alpha + 1.0)
        )

        # Log-uniform case (α = -1)
        log_P_log_uniform = a + u * (b - a)

        # Select appropriate formula
        log_P = jnp.where(is_log_uniform, log_P_log_uniform, log_P_general)

        return 10.0 ** log_P


class MoeEccentricity(eqx.Module):
    """Moe+2017 period-dependent eccentricity distribution.

    Implements period-dependent eccentricity following Moe & Di Stefano (2017):
    - Short periods (P < P_circ ~ 10d): Tidally circularized, e ≈ 0
    - Intermediate periods: Smooth transition
    - Long periods (P > P_thermal ~ 1000d): Thermal-like distribution f(e) = 2e

    The transition uses a smooth logistic function to ensure differentiability.

    Reference:
        Moe & Di Stefano (2017) ApJS 230, 15 - Binary statistics review

    Parameters:
        P_circ: Circularization period [days] (default: 10.0)
        P_thermal: Thermalization period [days] (default: 1000.0)
        e_max: Maximum eccentricity (default: 0.99)
        transition_width: Width of transition region in log10(P) (default: 0.5)
    """

    P_circ: float = 10.0
    P_thermal: float = 1000.0
    e_max: float = 0.99
    transition_width: float = 0.5

    def sample(
        self,
        periods: Float[Array, "N"],
        key: PRNGKeyArray
    ) -> Float[Array, "N"]:
        """Sample eccentricities given periods.

        Args:
            periods: Orbital periods [days] (shape N,)
            key: JAX random key

        Returns:
            Eccentricities (shape N,), period-dependent
        """
        # Sample thermal eccentricities (f(e) = 2e)
        u = jax.random.uniform(key, periods.shape)
        e_thermal = self.e_max * jnp.sqrt(u)

        # Compute blending factor based on period
        # For P < P_circ: blend ≈ 0 (circular)
        # For P > P_thermal: blend ≈ 1 (thermal)
        log_P = jnp.log10(periods)
        log_P_mid = jnp.log10(jnp.sqrt(self.P_circ * self.P_thermal))

        # Smooth transition via logistic function
        blend = 1.0 / (1.0 + jnp.exp(-(log_P - log_P_mid) / self.transition_width))

        # Blend: e = blend × e_thermal
        return blend * e_thermal


class RadialBinaryFraction(eqx.Module):
    """Radially varying binary fraction.

    Implements spatially varying binary fraction following the power-law model:

        f_b(r) = fb0 × (1 + A × (r/r_scale)^(-α))

    where:
        - A > 0: core-enhanced (more binaries in center)
        - A < 0: core-depleted (fewer binaries in center)
        - A = 0: constant binary fraction everywhere

    The result is clipped to [0, 1] to ensure valid binary fractions.

    References:
        Raghavan et al. (2010) ApJS 190, 1 - Solar neighborhood binary census
        Sana et al. (2012) Science 337, 444 - O-star binary fraction
        Moe & Di Stefano (2017) ApJS 230, 15 - Binary statistics review

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


# =============================================================================
# Mass-Dependent Binary Prescriptions
# =============================================================================


@dataclass(frozen=True)
class MassDependentBinaryConfig:
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
    high_mass_eccentricity: MoeEccentricity


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
    key_period, key_ecc = jax.random.split(key)

    # Sample from both distributions (always sample both for JIT)
    key_p_low, key_p_high, key_e_low, key_e_high = jax.random.split(key_period, 4)

    # Sample periods from both distributions
    periods_low = config.low_mass_period.sample(key_p_low, N)
    periods_high = config.high_mass_period.sample(key_p_high, N)

    # Sample eccentricities from both distributions
    # Note: MoeEccentricity depends on period, so we need to handle carefully
    # For now, sample eccentricities based on high-mass periods for high-mass stars
    ecc_low = config.low_mass_eccentricity.sample(key_e_low, N)
    ecc_high = config.high_mass_eccentricity.sample(periods_high, key_e_high)

    # Route based on mass threshold
    is_high_mass = masses >= config.m_break

    # Select appropriate distribution
    periods = jnp.where(is_high_mass, periods_high, periods_low)
    eccentricities = jnp.where(is_high_mass, ecc_high, ecc_low)

    return periods, eccentricities

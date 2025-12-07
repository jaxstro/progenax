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

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


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

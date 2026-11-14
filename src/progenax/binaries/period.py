"""Binary orbital-period distributions.

Period samplers for binary populations (period in days):

- :class:`LogUniformPeriod` — Öpik's law, p(log P) = const.
- :class:`LogNormalPeriod`  — Duquennoy & Mayor (1991) solar-type Gaussian in log P.
- :class:`SanaOBPeriod`     — Sana et al. (2012) O/B power law p(log P) ∝ (log P)^π.

References:
    Öpik (1924) Publ. Obs. Astron. Tartu 25 — log-uniform period distribution.
    Duquennoy & Mayor (1991) A&A 248, 485 §7.3 — log P̄=4.8, σ_logP=2.3 (days).
    Sana et al. (2012) Science 337, 444 — π = -0.55 ± 0.22.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from progenax.numerics import power_integral_stable, power_ppf_stable


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
        return 10.0**log_P

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
        return 10.0**log_P


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
        return 10.0**log_P

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
        """Inverse CDF via inverse error function (differentiable on (0,1)).

        u is clamped to (1e-12, 1-1e-12) so the gradient stays finite at the open
        boundary, where erfinv(±1) -> ±inf (a log-normal has unbounded support).
        """
        u_safe = jnp.clip(u, 1e-12, 1.0 - 1e-12)
        z = jnp.sqrt(2.0) * jax.scipy.special.erfinv(2.0 * u_safe - 1.0)
        log_P = self.mu_log_P + self.sigma_log_P * z
        return 10.0**log_P


class SanaOBPeriod(eqx.Module):
    """Sana+2012 period distribution for O/B stars.

    Power-law distribution in log-space:
        p(log P) ∝ (log P)^(-0.55)

    for log P in [0.15, 3.5] (P in days).

    This corresponds to shorter periods than solar-type binaries,
    consistent with observations of massive star binaries.

    Reference:
        Sana et al. (2012) Science 337, 444 - O-star binary survey. The intrinsic
        period distribution (their Fig. 2, π = -0.55 ± 0.22) runs from P ~ 1.4 d
        (log P ~ 0.15) to ~9 yr (log P = 3.5).

    Parameters:
        log_P_min: Minimum log10(P/days) (default: 0.15 = ~1.4 days; Sana 2012 Fig.2).
            Must be > 0: the model p(logP) ∝ (logP)^power is undefined for logP <= 0
            (P <= 1 day), since that raises a non-positive base to a fractional power.
        log_P_max: Maximum log10(P/days) (default: 3.5 = ~3162 days ~ 9 yr)
        power: Power-law index (default: -0.55 from Sana+2012)
    """

    log_P_min: float = 0.15
    log_P_max: float = 3.5
    power: float = -0.55

    def __post_init__(self):
        # Domain precondition: log_P_min must be > 0 (the power law is on log P,
        # undefined for logP <= 0). Fail fast on concrete configs (the normal case)
        # with a clear error instead of a cryptic ZeroDivisionError inside sample/ppf;
        # skip under tracing, where the bound is virtually never a differentiated value.
        if isinstance(self.log_P_min, (int, float)) and self.log_P_min <= 0.0:
            raise ValueError(
                f"SanaOBPeriod.log_P_min must be > 0 (got {self.log_P_min}): "
                f"p(logP) ∝ (logP)^power is undefined for logP <= 0 (P <= 1 day)."
            )

    def _inv_cdf_log_P(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Quantiles of p(x) proportional to x^power, x=log10(P/days).

        Stable power integrals retain the logarithmic limit and its power
        derivative at power=-1. That limit is logarithmic in x; Opik's law
        is uniform in x and corresponds to power=0.
        """
        a = self.log_P_min
        b = self.log_P_max
        exponent = self.power + 1.0
        integral = power_integral_stable(a, b, exponent)
        return power_ppf_stable(a, u * integral, exponent)

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample n periods [days] via inverse-transform sampling."""
        u = jax.random.uniform(key, (n,))
        return 10.0 ** self._inv_cdf_log_P(u)

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Inverse CDF: period [days] for quantile(s) u in [0, 1]."""
        return 10.0 ** self._inv_cdf_log_P(u)

    def cdf(self, P: Float[Array, "..."]) -> Float[Array, "..."]:
        """CDF F(P) of the truncated power law p(log P) proportional to (log P)^power."""
        x = jnp.log10(P)
        a = self.log_P_min
        b = self.log_P_max
        exponent = self.power + 1.0
        integral = power_integral_stable(a, jnp.clip(x, a, b), exponent)
        return jnp.clip(integral / power_integral_stable(a, b, exponent), 0.0, 1.0)

    def pdf(self, P: Float[Array, "..."]) -> Float[Array, "..."]:
        """PDF p(P): the density in P of p(log P) ∝ (log P)^power on [a, b]."""
        x = jnp.log10(P)
        alpha = self.power
        a = self.log_P_min
        b = self.log_P_max
        in_range = (x >= a) & (x <= b)

        # Z = integral_a^b x^power dx, including its smooth power=-1 limit.
        Z = power_integral_stable(a, b, alpha + 1.0)
        # p(x) = x^alpha / Z (x>0 on [a,b]); convert to density in P: dx = dP/(P ln10).
        x_pos = jnp.where(in_range, x, 1.0)
        p_x = jnp.power(x_pos, alpha) / Z
        p_P = p_x / (P * jnp.log(10.0))
        return jnp.where(in_range, p_P, 0.0)


__all__ = ["LogUniformPeriod", "LogNormalPeriod", "SanaOBPeriod"]

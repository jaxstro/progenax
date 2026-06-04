"""Binary orbital-eccentricity distributions.

Eccentricity samplers for binary populations:

- :class:`ThermalEccentricity` — f(e) = 2e (Ambartsumian energy-only / Heggie 1975).
- :class:`UniformEccentricity` — uniform on [e_min, e_max].
- :class:`MoeEccentricity`     — smooth period-dependent circular→thermal heuristic.

References:
    Ambartsumian (1937); Jeans (1919); Heggie (1975) MNRAS 173, 729 — thermal f(e)=2e.
    Duquennoy & Mayor (1991) A&A 248, 485 §6.1/§7.2 — P<10d circular, P>1000d thermal,
        circularization period P_circ ≈ 11.6 d.
    Moe & Di Stefano (2017) ApJS 230, 15 — period-dependent eccentricity statistics.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


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
        key: PRNGKeyArray,
        periods: Float[Array, "N"],
    ) -> Float[Array, "N"]:
        """Sample eccentricities given periods.

        Args:
            key: JAX random key
            periods: Orbital periods [days] (shape N,)

        Returns:
            Eccentricities (shape N,), period-dependent

        Note:
            Unlike the unconditional samplers (key, n), this is period-CONDITIONAL
            and takes (key, periods); n is implied by ``periods.shape``.
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


__all__ = ["ThermalEccentricity", "UniformEccentricity", "MoeEccentricity"]

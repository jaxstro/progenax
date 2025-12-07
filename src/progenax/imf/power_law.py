"""
N-segment power-law IMF implementation.

Supports Kroupa (2001), Salpeter (1955), and custom piecewise power-law IMFs.
Features exact analytical CDF/PPF for efficiency and differentiability.
"""

from __future__ import annotations
from typing import List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class PowerLawIMF(eqx.Module):
    """
    N-segment piecewise power-law IMF.

    The PDF is:
        ξ(m) ∝ m^(-α_i) for m ∈ [b_{i-1}, b_i]

    where α_i are the exponents and b_i are the breakpoints.
    Continuity is enforced at breakpoints.

    Attributes:
        m_min: Minimum mass [M_sun]
        m_max: Maximum mass [M_sun]
        breakpoints: Mass breakpoints (excluding m_min, m_max)
        exponents: Power-law exponents for each segment
        _continuity_factors: Multiplicative factors for PDF continuity
        _segment_integrals: Integral of PDF over each segment
        _segment_cumprob: Cumulative probability at segment boundaries

    Example:
        >>> imf = PowerLawIMF.kroupa()  # Kroupa (2001)
        >>> masses = imf.sample(key, 1000)
    """

    m_min: float
    m_max: float
    breakpoints: Tuple[float, ...]
    exponents: Tuple[float, ...]
    _continuity_factors: Float[Array, "n_segments"]
    _segment_integrals: Float[Array, "n_segments"]
    _segment_cumprob: Float[Array, "n_segments+1"]

    def __init__(
        self,
        exponents: List[float],
        breakpoints: List[float],
        m_min: float = 0.01,
        m_max: float = 100.0,
    ):
        """
        Create power-law IMF.

        Args:
            exponents: Power-law exponents for each segment
            breakpoints: Mass breakpoints between segments
            m_min: Minimum mass [M_sun]
            m_max: Maximum mass [M_sun]
        """
        n_segments = len(exponents)
        if len(breakpoints) != n_segments - 1:
            raise ValueError(
                f"Number of breakpoints ({len(breakpoints)}) must equal "
                f"number of exponents - 1 ({n_segments - 1})"
            )

        self.m_min = m_min
        self.m_max = m_max
        self.breakpoints = tuple(breakpoints)
        self.exponents = tuple(exponents)

        # Compute continuity factors and integrals
        bounds = jnp.array([m_min] + list(breakpoints) + [m_max])
        alphas = jnp.array(exponents)

        # Continuity factors: C_i such that PDF is continuous at breakpoints
        cont = jnp.ones(n_segments)
        for i in range(1, n_segments):
            factor = bounds[i] ** (alphas[i] - alphas[i - 1])
            cont = cont.at[i].set(cont[i - 1] * factor)

        # Segment integrals
        def segment_integral(i):
            a = alphas[i]
            lo, hi = bounds[i], bounds[i + 1]
            e = 1.0 - a
            return jnp.where(
                jnp.abs(e) < 1e-12,
                cont[i] * jnp.log(hi / lo),
                cont[i] * (hi**e - lo**e) / e,
            )

        integrals = jax.vmap(segment_integral)(jnp.arange(n_segments))
        Z = jnp.sum(integrals)

        # Cumulative probabilities
        cumprob = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(integrals / Z)])

        self._continuity_factors = cont
        self._segment_integrals = integrals
        self._segment_cumprob = cumprob

    @classmethod
    def kroupa(cls, m_min: float = 0.01, m_max: float = 100.0) -> "PowerLawIMF":
        """
        Kroupa (2001) IMF.

        Three-segment power law:
            α = 0.3 for 0.01 ≤ m < 0.08 M_sun
            α = 1.3 for 0.08 ≤ m < 0.50 M_sun
            α = 2.3 for 0.50 ≤ m ≤ 100 M_sun

        References:
            Kroupa (2001), MNRAS, 322, 231
        """
        return cls(
            exponents=[0.3, 1.3, 2.3],
            breakpoints=[0.08, 0.5],
            m_min=m_min,
            m_max=m_max,
        )

    @classmethod
    def salpeter(cls, m_min: float = 0.1, m_max: float = 100.0) -> "PowerLawIMF":
        """
        Salpeter (1955) IMF.

        Single power law: α = 2.35

        References:
            Salpeter (1955), ApJ, 121, 161
        """
        return cls(
            exponents=[2.35],
            breakpoints=[],
            m_min=m_min,
            m_max=m_max,
        )

    @classmethod
    def custom(
        cls,
        exponents: List[float],
        breakpoints: List[float],
        m_min: float = 0.01,
        m_max: float = 100.0,
    ) -> "PowerLawIMF":
        """Create custom power-law IMF."""
        return cls(exponents, breakpoints, m_min, m_max)

    def _logpdf_unnorm(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Unnormalized log-PDF."""
        bounds = jnp.array([self.m_min] + list(self.breakpoints) + [self.m_max])
        alphas = jnp.array(self.exponents)

        # Find segment for each mass
        idx = jnp.searchsorted(bounds[1:], m)
        idx = jnp.clip(idx, 0, len(self.exponents) - 1)

        # Log-PDF: log(C_i * m^(-alpha_i))
        return jnp.log(self._continuity_factors[idx]) - alphas[idx] * jnp.log(m)

    def _cdf_unnorm(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Unnormalized CDF."""
        bounds = jnp.array([self.m_min] + list(self.breakpoints) + [self.m_max])
        alphas = jnp.array(self.exponents)
        n_segments = len(self.exponents)

        # Find segment
        idx = jnp.searchsorted(bounds[1:], m)
        idx = jnp.clip(idx, 0, n_segments - 1)

        # For scalar idx, need to handle summing correctly
        # Cumulative integral up to start of this segment
        def sum_before_idx(idx_val):
            """Sum all segment integrals before index idx_val."""
            mask = jnp.arange(n_segments) < idx_val
            return jnp.sum(jnp.where(mask, self._segment_integrals, 0.0))

        # Handle both scalar and array idx
        is_scalar = jnp.ndim(idx) == 0
        if is_scalar:
            cum_before = sum_before_idx(idx)
        else:
            cum_before = jax.vmap(sum_before_idx)(idx)

        # Integral within segment
        lo = bounds[idx]
        a = alphas[idx]
        e = 1.0 - a
        within_segment = jnp.where(
            jnp.abs(e) < 1e-12,
            self._continuity_factors[idx] * jnp.log(m / lo),
            self._continuity_factors[idx] * (m**e - lo**e) / e,
        )

        return cum_before + within_segment

    def logpdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Normalized log-PDF."""
        Z = jnp.sum(self._segment_integrals)
        return self._logpdf_unnorm(m) - jnp.log(Z)

    def cdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Normalized CDF."""
        Z = jnp.sum(self._segment_integrals)
        return self._cdf_unnorm(m) / Z

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Inverse CDF (exact analytical)."""
        bounds = jnp.array([self.m_min] + list(self.breakpoints) + [self.m_max])
        alphas = jnp.array(self.exponents)

        # Find segment
        idx = jnp.searchsorted(self._segment_cumprob[1:], u)
        idx = jnp.clip(idx, 0, len(self.exponents) - 1)

        # Rescale u within segment
        u0 = self._segment_cumprob[idx]
        u1 = self._segment_cumprob[idx + 1]
        ur = (u - u0) / (u1 - u0 + 1e-30)

        # Inverse within segment
        lo = bounds[idx]
        hi = bounds[idx + 1]
        a = alphas[idx]
        e = 1.0 - a

        return jnp.where(
            jnp.abs(e) < 1e-12,
            lo * jnp.exp(ur * jnp.log(hi / lo)),
            (ur * (hi**e - lo**e) + lo**e) ** (1.0 / e),
        )

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample n masses."""
        u = jax.random.uniform(key, (n,))
        return self.ppf(u)

    def mean_mass(self) -> float:
        """Mean mass (numerical integration)."""
        m_grid = jnp.linspace(self.m_min, self.m_max, 1000)
        pdf_grid = jnp.exp(self.logpdf(m_grid))
        return jnp.trapezoid(m_grid * pdf_grid, m_grid)

    def inverse_cdf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Alias for ppf (legacy compatibility)."""
        return self.ppf(u)


# Convenience function
def prepare_imf_samples(N: int, key: PRNGKeyArray) -> Float[Array, "N"]:
    """
    Prepare uniform samples for reparameterized IMF sampling.

    Args:
        N: Number of samples
        key: JAX random key

    Returns:
        Uniform samples in [0, 1]
    """
    return jax.random.uniform(key, (N,))


# Estimation utilities
def estimate_N_max_for_M_total(
    m_total: float,
    imf: PowerLawIMF,
    safety_factor: float = 2.0,
) -> int:
    """Estimate N_max for M_total mode sampling."""
    mean_mass = imf.mean_mass()
    return int(m_total / mean_mass * safety_factor) + 100


def estimate_pool_size(m_total: float, imf: PowerLawIMF) -> int:
    """Alias for estimate_N_max_for_M_total."""
    return estimate_N_max_for_M_total(m_total, imf)


__all__ = [
    "PowerLawIMF",
    "prepare_imf_samples",
    "estimate_N_max_for_M_total",
    "estimate_pool_size",
]

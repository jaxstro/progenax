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
            # exp_safe double-where: keep BOTH branches finite so the VJP is
            # finite at exactly alpha=1 (bare where backprops 0*NaN — audit R10).
            # Same pattern as imf/differentiable.py power_integral.
            e_safe = jnp.where(jnp.abs(e) < 1e-12, 1.0, e)
            return jnp.where(
                jnp.abs(e) < 1e-12,
                cont[i] * jnp.log(hi / lo),
                cont[i] * (hi**e_safe - lo**e_safe) / e_safe,
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

        Kroupa (2001) Eq. 2 lists four segments with α₂=2.3 (0.5–1.0) and α₃=2.3 (≥1.0);
        since those two slopes are equal, merging them into one ≥0.5 segment is exact.
        IMFParams.kroupa() keeps the explicit 4-segment form for inference.

        References:
            Kroupa (2001), MNRAS, 322, 231 — Eq. 2 (α₀=0.3, α₁=1.3, α₂=α₃=2.3)
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
        # exp_safe double-where (audit R10): finite VJP at exactly alpha=1.
        e_safe = jnp.where(jnp.abs(e) < 1e-12, 1.0, e)
        within_segment = jnp.where(
            jnp.abs(e) < 1e-12,
            self._continuity_factors[idx] * jnp.log(m / lo),
            self._continuity_factors[idx] * (m**e_safe - lo**e_safe) / e_safe,
        )

        return cum_before + within_segment

    def logpdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Normalized log-PDF. Returns -inf outside [m_min, m_max]."""
        m_arr = jnp.asarray(m)
        in_domain = (m_arr >= self.m_min) & (m_arr <= self.m_max)
        Z = jnp.sum(self._segment_integrals)
        lp = self._logpdf_unnorm(m_arr) - jnp.log(Z)
        return jnp.where(in_domain, lp, -jnp.inf)

    def cdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Normalized CDF. Returns 0 below m_min, 1 above m_max."""
        m_arr = jnp.asarray(m)
        Z = jnp.sum(self._segment_integrals)
        # Clamp m to valid range before computing CDF
        m_clamped = jnp.clip(m_arr, self.m_min, self.m_max)
        raw = self._cdf_unnorm(m_clamped) / Z
        return jnp.where(
            m_arr <= self.m_min,
            0.0,
            jnp.where(m_arr >= self.m_max, 1.0, raw),
        )

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
        # exp_safe double-where (audit R10): e_safe substitutes for e in EVERY
        # power/division of the inverse-CDF branch (including the **(1/e) outer
        # exponent), so the VJP is finite at exactly alpha=1.
        e_safe = jnp.where(jnp.abs(e) < 1e-12, 1.0, e)

        return jnp.where(
            jnp.abs(e) < 1e-12,
            lo * jnp.exp(ur * jnp.log(hi / lo)),
            (ur * (hi**e_safe - lo**e_safe) + lo**e_safe) ** (1.0 / e_safe),
        )

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample n masses."""
        u = jax.random.uniform(key, (n,))
        return self.ppf(u)

    def mean_mass(self) -> float:
        """Exact mean mass E[m] for the piecewise power law (analytic, differentiable).

        E[m] = (Σ_i ∫ m·ξ_i dm) / (Σ_i ∫ ξ_i dm), where ξ_i(m) = C_i·m^(-α_i) on
        segment i. The denominator is the stored segment integrals (with exponent
        1-α_i); the numerator carries one extra power of m (exponent 2-α_i). This
        replaces a linear-grid trapezoid that under-resolved the steep low-mass spike.
        """
        bounds = jnp.array([self.m_min] + list(self.breakpoints) + [self.m_max])
        alphas = jnp.array(self.exponents)
        cont = self._continuity_factors

        def first_moment(i):
            a = alphas[i]
            lo, hi = bounds[i], bounds[i + 1]
            e = 2.0 - a  # antiderivative exponent of m·m^(-a) = m^(1-a)
            # exp_safe double-where (audit R10): here the removable singularity
            # is at alpha=2 (e=0); the alpha=1 NaN in mean_mass flows through Z
            # (segment_integral, fixed above), but guard this branch too.
            e_safe = jnp.where(jnp.abs(e) < 1e-12, 1.0, e)
            return jnp.where(
                jnp.abs(e) < 1e-12,
                cont[i] * jnp.log(hi / lo),
                cont[i] * (hi**e_safe - lo**e_safe) / e_safe,
            )

        moments = jax.vmap(first_moment)(jnp.arange(len(self.exponents)))
        Z = jnp.sum(self._segment_integrals)
        return jnp.sum(moments) / Z

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

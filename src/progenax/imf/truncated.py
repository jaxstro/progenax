"""
TruncatedIMF wrapper for hard mass truncation.

Wraps any IMF to enforce strict mass bounds [m_min, m_max].
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .base import BaseIMF


class TruncatedIMF(eqx.Module):
    """
    Wrapper for hard mass truncation of any IMF.

    Takes an existing IMF and enforces strict bounds [m_min, m_max].
    Renormalizes the PDF over the truncated domain.

    Attributes:
        inner: The wrapped IMF
        m_min: Minimum mass [M_sun]
        m_max: Maximum mass [M_sun]

    Example:
        >>> from progenax.imf import ChabrierIMF, TruncatedIMF
        >>> chabrier = ChabrierIMF()
        >>> truncated = TruncatedIMF(chabrier, m_min=0.08, m_max=150.0)
        >>> masses = truncated.sample(key, 1000)
    """

    inner: BaseIMF
    m_min: float
    m_max: float

    def __init__(self, inner: BaseIMF, m_min: float, m_max: float):
        """
        Create truncated IMF.

        Args:
            inner: IMF to wrap
            m_min: New minimum mass [M_sun]
            m_max: New maximum mass [M_sun]

        Raises:
            ValueError: If m_min >= m_max after clamping to inner IMF bounds
        """
        self.inner = inner
        self.m_min = max(m_min, inner.m_min)
        self.m_max = min(m_max, inner.m_max)

        if self.m_min >= self.m_max:
            raise ValueError(
                f"Invalid truncation bounds: m_min={self.m_min:.3f} >= m_max={self.m_max:.3f} "
                f"after clamping to inner IMF bounds [{inner.m_min:.3f}, {inner.m_max:.3f}]"
            )

    @property
    def _cdf_min(self) -> float:
        """CDF at m_min."""
        return self.inner.cdf(jnp.asarray(self.m_min))

    @property
    def _cdf_max(self) -> float:
        """CDF at m_max."""
        return self.inner.cdf(jnp.asarray(self.m_max))

    def logpdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Normalized log-PDF over truncated domain."""
        log_norm = jnp.log(self._cdf_max - self._cdf_min + 1e-30)
        return self.inner.logpdf(m) - log_norm

    def cdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """CDF rescaled to [0, 1] over truncated domain."""
        raw_cdf = self.inner.cdf(m)
        return (raw_cdf - self._cdf_min) / (self._cdf_max - self._cdf_min + 1e-30)

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Inverse CDF over truncated domain."""
        u_scaled = self._cdf_min + u * (self._cdf_max - self._cdf_min)
        return self.inner.ppf(u_scaled)

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample from truncated distribution."""
        u = jax.random.uniform(key, (n,))
        return self.ppf(u)

    def mean_mass(self) -> float:
        """Mean mass over truncated domain."""
        m_grid = jnp.linspace(self.m_min, self.m_max, 1000)
        pdf_grid = jnp.exp(self.logpdf(m_grid))
        return jnp.trapezoid(m_grid * pdf_grid, m_grid)


__all__ = ["TruncatedIMF"]

"""
TruncatedIMF wrapper for hard mass truncation.

Wraps any IMF to enforce strict mass bounds [m_min, m_max].
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .base import IMFProtocol


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

    inner: IMFProtocol
    m_min: float
    m_max: float

    def __init__(self, inner: IMFProtocol, m_min: float, m_max: float):
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
        """Normalized log-PDF. Returns -inf outside [m_min, m_max]."""
        m_arr = jnp.asarray(m)
        in_domain = (m_arr >= self.m_min) & (m_arr <= self.m_max)
        log_norm = jnp.log(self._cdf_max - self._cdf_min + 1e-30)
        lp = self.inner.logpdf(m_arr) - log_norm
        return jnp.where(in_domain, lp, -jnp.inf)

    def cdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """CDF rescaled to [0, 1]. Returns 0 below m_min, 1 above m_max."""
        m_arr = jnp.asarray(m)
        raw_cdf = self.inner.cdf(m_arr)
        cdf_trunc = (raw_cdf - self._cdf_min) / (self._cdf_max - self._cdf_min + 1e-30)
        return jnp.where(
            m_arr <= self.m_min,
            0.0,
            jnp.where(m_arr >= self.m_max, 1.0, cdf_trunc),
        )

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Inverse CDF (percent point function) over the truncated domain.

        Rescales u in [0, 1] to the inner IMF's CDF interval [F(m_min), F(m_max)] and
        defers to the inner IMF's ``ppf``, so the result lies in [m_min, m_max].

        Args:
            u: Quantiles in [0, 1] (any broadcastable shape).

        Returns:
            Masses [M_sun] in [m_min, m_max], same shape as ``u``.

        Differentiability:
            Reverse-mode differentiable on (0, 1) (the affine rescale plus the inner
            IMF's differentiable ``ppf``).
        """
        u_scaled = self._cdf_min + u * (self._cdf_max - self._cdf_min)
        return self.inner.ppf(u_scaled)

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample from truncated distribution."""
        u = jax.random.uniform(key, (n,))
        return self.ppf(u)

    def mean_mass(self) -> float:
        """Mean mass over the truncated domain via a LOG-spaced trapezoid.

        Log-spacing resolves a steep low-mass spike that a linear grid of the same size
        under-resolves (see BaseIMF.mean_mass).
        """
        m_grid = jnp.exp(jnp.linspace(jnp.log(self.m_min), jnp.log(self.m_max), 4000))
        pdf_grid = jnp.exp(self.logpdf(m_grid))
        return jnp.trapezoid(m_grid * pdf_grid, m_grid)


__all__ = ["TruncatedIMF"]

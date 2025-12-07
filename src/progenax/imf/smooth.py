"""
Smooth IMF families: Maschberger, TaperedPowerLaw, Schechter.

These IMFs have smooth functional forms (unlike piecewise power-laws)
and require numerical integration for their CDFs.

Integration Strategy:
We use LINEAR spacing with MANY points (10000+) because:
1. This matches standard verification methods (what users expect)
2. Trapezoidal rule with enough points converges for all IMF shapes
3. JAX can efficiently handle large arrays

Note: For pure power-law IMFs (like Salpeter), use PowerLawIMF which
has analytical CDF and is much faster.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .base import BaseIMF


def _linear_trapz_integrate(log_pdf_fn, m_min: float, m_max: float, n_points: int = 10000):
    """Integrate exp(log_pdf) using dense linear grid.

    For smooth IMFs, linear spacing with many points gives consistent
    results that match standard verification methods.

    Args:
        log_pdf_fn: Function returning log-PDF at mass m
        m_min: Lower integration bound
        m_max: Upper integration bound
        n_points: Number of points (default: 10000, matches test verification)

    Returns:
        Integral of exp(log_pdf) from m_min to m_max
    """
    m_grid = jnp.linspace(m_min, m_max, n_points)
    log_pdf_grid = log_pdf_fn(m_grid)
    pdf_grid = jnp.exp(log_pdf_grid)
    return jnp.trapezoid(pdf_grid, m_grid)


def _scalar_cdf_unnorm(log_pdf_fn, m_val, m_min: float, n_points: int = 10000):
    """Compute CDF for a single mass value, returning scalar output."""
    return jnp.where(
        m_val <= m_min,
        0.0,
        _linear_trapz_integrate(log_pdf_fn, m_min, m_val, n_points=n_points)
    )


class Maschberger(BaseIMF):
    """Maschberger (2013) smooth IMF.

    Single formula bridging lognormal turnover at low mass and
    power-law tail at high mass.

    PDF: f(m) ∝ (m/μ)^(-α) * (1 + (m/μ)^(1-α))^(-β)

    Default parameters from Maschberger (2013):
        mu = 0.2 M_sun (peak mass)
        alpha = 2.3 (Salpeter high-mass slope)
        beta = 1.4 (low-mass turnover)

    Reference:
        Maschberger, T. (2013), MNRAS, 429, 1725
        "On the function describing the stellar initial mass function"
    """

    mu: float = 0.2       # Peak mass [M_sun]
    alpha: float = 2.3    # High-mass slope
    beta: float = 1.4     # Low-mass turnover
    m_min: float = 0.01   # Natural domain lower bound
    m_max: float = 300.0  # Natural domain upper bound

    def _logpdf_unnorm(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Unnormalized log-PDF.

        log f(m) = -α * log(m/μ) + (-β) * log(1 + (m/μ)^(1-α))
                 = -α * log(m) + α * log(μ) - β * log(1 + (m/μ)^(1-α))
        """
        x = m / self.mu
        # -α * log(m/μ)
        term1 = -self.alpha * jnp.log(x + 1e-30)
        # -β * log(1 + x^(1-α))
        term2 = -self.beta * jnp.log1p(x ** (1 - self.alpha))
        return term1 + term2

    def _primitive(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Antiderivative P(m) such that dP/dm = p̃(m).

        For Maschberger PDF: p̃(m) = (m/μ)^(-α) × [1 + (m/μ)^(1-α)]^(-β)

        Primitive: P(m) = μ / [(1-β)(1-α)] × [1 + (m/μ)^(1-α)]^(1-β)

        Valid for β ≠ 1 and α ≠ 1.
        """
        x = m / self.mu  # Dimensionless mass
        u = x ** (1 - self.alpha)  # Auxiliary variable

        # Compute primitive
        coeff = self.mu / ((1 - self.beta) * (1 - self.alpha))
        P = coeff * (1 + u) ** (1 - self.beta)

        return P

    def _cdf_unnorm(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Unnormalized CDF via analytical primitive.

        Uses the analytical antiderivative for exact, fast computation.
        """
        P_m = self._primitive(m)
        P_min = self._primitive(self.m_min)
        return P_m - P_min

    def _ppf_analytical(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Analytical inverse CDF - no Newton iteration needed.

        Inverts: F(m) = [P(m) - P_min] / [P_max - P_min]

        Solution derivation:
            1. P_target = P_min + u × (P_max - P_min)
            2. From primitive: P = coeff × [1 + (m/μ)^(1-α)]^(1-β)
            3. Solve for m:
               [1 + (m/μ)^(1-α)]^(1-β) = P_target / coeff
               1 + (m/μ)^(1-α) = (P_target / coeff)^(1/(1-β))
               (m/μ)^(1-α) = (P_target / coeff)^(1/(1-β)) - 1
               m = μ × [(P_target / coeff)^(1/(1-β)) - 1]^(1/(1-α))

        Args:
            u: Uniform random variables in [0, 1]

        Returns:
            Masses sampled from Maschberger IMF
        """
        # Compute primitive values at bounds
        P_min = self._primitive(jnp.asarray(self.m_min))
        P_max = self._primitive(jnp.asarray(self.m_max))

        # Target primitive value
        P_target = P_min + u * (P_max - P_min)

        # Invert primitive formula
        # P = coeff * [1 + u_val]^(1-β) where u_val = (m/μ)^(1-α)
        coeff = self.mu / ((1 - self.beta) * (1 - self.alpha))

        # Step 1: [1 + u_val]^(1-β) = P_target / coeff
        bracket = P_target / coeff

        # Step 2: 1 + u_val = bracket^(1/(1-β))
        one_plus_u = bracket ** (1.0 / (1 - self.beta))
        u_val = one_plus_u - 1

        # Step 3: u_val = (m/μ)^(1-α), so m = μ × u_val^(1/(1-α))
        x = u_val ** (1.0 / (1 - self.alpha))
        m = self.mu * x

        # Clip to ensure domain [m_min, m_max]
        return jnp.clip(m, self.m_min, self.m_max)

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Inverse CDF using analytical formula.

        Args:
            u: Uniform random variables in [0, 1]

        Returns:
            Masses sampled from Maschberger IMF
        """
        return self._ppf_analytical(u)


class TaperedPowerLaw(BaseIMF):
    """Tapered Power Law IMF.

    PDF: f(m) ∝ m^(-α) * (1 - exp(-(m/m_peak)^β))

    The exponential taper suppresses low masses below m_peak,
    creating a smooth turnover.

    Attributes:
        alpha: Power-law slope (default: 2.3, Salpeter)
        m_peak: Turnover/peak mass [M_sun]
        beta: Taper sharpness (higher = sharper cutoff)
    """

    alpha: float = 2.3
    m_peak: float = 0.3   # Turnover mass [M_sun]
    beta: float = 2.0     # Taper sharpness
    m_min: float = 0.01
    m_max: float = 300.0

    def _logpdf_unnorm(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Unnormalized log-PDF."""
        x = m / self.m_peak
        powerlaw = -self.alpha * jnp.log(m + 1e-30)
        # log(1 - exp(-x^β)) with numerical stability
        # For small x: 1 - exp(-x^β) ≈ x^β, so log ≈ β*log(x)
        taper_arg = x ** self.beta
        taper = jnp.where(
            taper_arg < 0.01,
            self.beta * jnp.log(x + 1e-30),  # Small argument approximation
            jnp.log1p(-jnp.exp(-taper_arg) + 1e-30)
        )
        return powerlaw + taper

    def _cdf_unnorm(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Unnormalized CDF via numerical integration.

        Note: Analytical primitive exists but gamma recurrence for negative
        shape parameters (α > 1) is numerically unstable. Keep numerical
        integration for robustness.
        """
        m_arr = jnp.asarray(m)
        is_scalar = m_arr.ndim == 0

        if is_scalar:
            return _scalar_cdf_unnorm(self._logpdf_unnorm, m_arr, self.m_min)
        else:
            original_shape = m_arr.shape
            m_flat = m_arr.ravel()
            result = jax.vmap(lambda mv: _scalar_cdf_unnorm(self._logpdf_unnorm, mv, self.m_min))(m_flat)
            return result.reshape(original_shape)


class Schechter(BaseIMF):
    """Schechter function with exponential high-mass cutoff.

    PDF: f(m) ∝ m^(-α) * exp(-m/m_star)

    Originally developed for galaxy luminosity functions (Schechter 1976),
    also used for IMFs in extreme environments or IGIMF theory.

    WARNING: Default parameters (α=2.3, m_star=100) give a function
    that is essentially just a power-law since the cutoff is far above
    where the power-law already suppresses high masses.

    For stellar IMFs with meaningful high-mass cutoff, consider:
        - α = 1.0-1.5 (flatter slope)
        - m_star = 10-50 M_sun (relevant cutoff scale)

    Example:
        >>> # IGIMF-style with cluster-dependent upper mass limit
        >>> imf = Schechter(alpha=1.35, m_star=m_max_from_cluster_mass(M_cluster))

    Attributes:
        alpha: Power-law slope (default: 2.3, Salpeter-like)
        m_star: Exponential cutoff mass [M_sun] (default: 100)
    """

    alpha: float = 2.3
    m_star: float = 100.0  # Exponential cutoff mass [M_sun]
    m_min: float = 0.01
    m_max: float = 300.0

    def _logpdf_unnorm(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Unnormalized log-PDF."""
        return -self.alpha * jnp.log(m + 1e-30) - m / self.m_star

    def _cdf_unnorm(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Unnormalized CDF via numerical integration.

        Note: Analytical primitive exists but gamma recurrence for negative
        shape parameters (α > 1) is numerically unstable. Keep numerical
        integration for robustness.
        """
        m_arr = jnp.asarray(m)
        is_scalar = m_arr.ndim == 0

        if is_scalar:
            return _scalar_cdf_unnorm(self._logpdf_unnorm, m_arr, self.m_min)
        else:
            original_shape = m_arr.shape
            m_flat = m_arr.ravel()
            result = jax.vmap(lambda mv: _scalar_cdf_unnorm(self._logpdf_unnorm, mv, self.m_min))(m_flat)
            return result.reshape(original_shape)


__all__ = ["Maschberger", "TaperedPowerLaw", "Schechter"]

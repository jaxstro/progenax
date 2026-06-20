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

import jax.numpy as jnp
from jaxtyping import Array, Float

from .base import BaseIMF


def _shared_grid_cdf_unnorm(
    log_pdf_fn, m, m_min: float, m_max: float, n_points: int = 4000
):
    """Unnormalized CDF int_{m_min}^m pdf dm' via one shared cumulative-trapezoid grid.

    A single (log-spaced) grid is evaluated once, then

        cdf = cumsum( 0.5 (f_i + f_{i+1}) dm_i ),   f = pdf = exp(log_pdf) >= 0,

    is interpolated to the query masses ``m``. Each increment is non-negative, so the
    CDF is **monotone by construction** to machine precision -- no per-upper-limit
    re-gridding (the old approach re-gridded [m_min, m_val] for every query, which is
    O(N * n_points) and gave ~1e-4 non-monotonic wiggle over the steep m^-alpha spike).
    Cost is O(n_points + N), and it is smooth/differentiable (cumsum + interp, no
    argmax/argsort). Log-spacing concentrates nodes on the low-mass spike. Queries below
    m_min map to 0; above m_max to the total integral.

    Works for any ``m`` shape (jnp.interp preserves it).
    """
    grid = jnp.exp(jnp.linspace(jnp.log(m_min), jnp.log(m_max), n_points))
    pdf = jnp.exp(log_pdf_fn(grid))
    dgrid = jnp.diff(grid)
    cdf_grid = jnp.concatenate(
        [jnp.zeros(1, dtype=pdf.dtype), jnp.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * dgrid)]
    )
    return jnp.interp(jnp.asarray(m), grid, cdf_grid, left=0.0, right=cdf_grid[-1])


class Maschberger(BaseIMF):
    """Maschberger (2013) smooth IMF.

    Single formula bridging lognormal turnover at low mass and
    power-law tail at high mass.

    PDF: f(m) ∝ (m/μ)^(-α) * (1 + (m/μ)^(1-α))^(-β)

    Default parameters from Maschberger (2013) Table 1 (canonical single-star IMF):
        mu = 0.2 M_sun (scale parameter; the pdf peak is near here)
        alpha = 2.3 (high-mass slope; Kroupa/Chabrier canonical, not Salpeter's 2.35)
        beta = 1.4 (low-mass turnover; gives effective low-mass slope γ=α+β(1-α)=0.48)

    Note on m_max: Maschberger (2013) Table 1 adopts the fiducial upper limit
    m_u = 150 M_sun; these limits "are only needed for the normalization" (Table 1
    caption). progenax defaults to m_max = 300 M_sun to admit very massive stars;
    since the limit only sets the normalization constant, this is a convention choice,
    not a change to the IMF shape. Pass m_max=150.0 to reproduce the paper exactly.

    Reference:
        Maschberger, T. (2013), MNRAS, 429, 1725
        "On the function describing the stellar initial mass function"
    """

    mu: float = 0.2  # Scale parameter [M_sun] (Maschberger 2013 Table 1)
    alpha: float = 2.3  # High-mass slope (canonical; cf. Salpeter 2.35)
    beta: float = 1.4  # Low-mass turnover
    m_min: float = 0.01  # Lower limit (Maschberger 2013 fiducial m_l)
    m_max: float = (
        300.0  # Upper limit (paper fiducial m_u=150; 300 here, normalization-only)
    )

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
    creating a smooth turnover. Functional form: the tapered power law of
    Parravano, McKee & Hollenbach (2011) (cf. Maschberger 2013, Eq. 12).

    No closed-form CDF inverse: the CDF uses the shared cumulative-trapezoid grid
    and ``ppf`` is the inherited BaseIMF fixed-iteration Newton solver (unlike
    Maschberger, which has an analytic quantile). Differentiable and JIT-safe.

    Attributes:
        alpha: Power-law slope (default: 2.3, canonical high-mass)
        m_peak: Turnover/peak mass [M_sun]
        beta: Taper sharpness (higher = sharper cutoff)
    """

    alpha: float = 2.3
    m_peak: float = 0.3  # Turnover mass [M_sun]
    beta: float = 2.0  # Taper sharpness
    m_min: float = 0.01
    m_max: float = 300.0

    def _logpdf_unnorm(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Unnormalized log-PDF."""
        x = m / self.m_peak
        powerlaw = -self.alpha * jnp.log(m + 1e-30)
        # log(1 - exp(-x^β)) with numerical stability
        # For small x: 1 - exp(-x^β) ≈ x^β, so log ≈ β*log(x)
        taper_arg = x**self.beta
        taper = jnp.where(
            taper_arg < 0.01,
            self.beta * jnp.log(x + 1e-30),  # Small argument approximation
            jnp.log1p(-jnp.exp(-taper_arg) + 1e-30),
        )
        return powerlaw + taper

    def _cdf_unnorm(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Unnormalized CDF via the shared cumulative-trapezoid grid (monotone by
        construction). Analytical primitive exists but the gamma recurrence for
        negative shape parameters (alpha > 1) is numerically unstable; the shared-grid
        integral is robust, monotone, and differentiable.
        """
        return _shared_grid_cdf_unnorm(self._logpdf_unnorm, m, self.m_min, self.m_max)


class Schechter(BaseIMF):
    """Schechter function with exponential high-mass cutoff.

    PDF: f(m) ∝ m^(-α) * exp(-m/m_star)

    Originally developed for galaxy luminosity functions (Schechter 1976),
    also used for IMFs in extreme environments or IGIMF theory.

    No closed-form CDF inverse: the CDF uses the shared cumulative-trapezoid grid and
    ``ppf`` is the inherited BaseIMF fixed-iteration Newton solver. Differentiable/JIT-safe.

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
        """Unnormalized CDF via the shared cumulative-trapezoid grid (monotone by
        construction); see TaperedPowerLaw._cdf_unnorm.
        """
        return _shared_grid_cdf_unnorm(self._logpdf_unnorm, m, self.m_min, self.m_max)


__all__ = ["Maschberger", "TaperedPowerLaw", "Schechter"]

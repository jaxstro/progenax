"""
Chabrier (2003) IMF with lognormal + power-law components.

Implements the Chabrier (2003) single-star (disk) IMF:
- Lognormal component for m < 1 M☉
- Power-law tail (Chabrier 2003 Table 1 high-mass slope α=2.3) for m ≥ 1 M☉

Implements IMFProtocol for compatibility with BaseIMF framework and
TruncatedIMF wrapper.

Note: the default parameters (m_c=0.08, σ=0.69, A_ln=0.158) are the Chabrier
(2003) Table 1 *single-star* disk values (individual stars; binaries resolved
into components), NOT the system IMF (which has m_c≈0.22, σ≈0.57).

References:
    Chabrier (2003), PASP, 115, 763 - Table 1, single-star disk IMF
    Chabrier (2005), ASSL, 327, 41 - The Initial Mass Function 50 Years Later
"""

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxstro.numerics import newton_ppf
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray


class ChabrierIMF(eqx.Module):
    """Chabrier (2003) lognormal + power-law IMF.

    The Chabrier IMF has two components:
    - Lognormal: ξ(log m) = A_ln × exp[-(log m - log m_c)² / (2σ²)] for m < m_trans
    - Power-law: ξ(log m) = A_pl × m^(-α) for m ≥ m_trans

    Uses standard Chabrier (2003) parameters with A_pl computed for continuity
    at the transition mass m_trans = 1 M☉.

    Implements IMFProtocol for compatibility with BaseIMF framework.

    Attributes:
        m_min: Minimum mass [M☉] (default: 0.08 - hydrogen burning limit)
        m_max: Maximum mass [M☉] (default: 100)
        m_c: Characteristic mass for lognormal [M☉] (default: 0.08, Chabrier 2003)
        sigma: Width of lognormal in log-space (default: 0.69)
        alpha: High-mass slope for ξ(m)=dN/dm ∝ m^(-α) (default: 2.3, Chabrier 2003
            Table 1 high-mass tail x=1.3 ⇒ α=2.3; the original Salpeter slope is
            2.35, available via PowerLawIMF.salpeter())
        m_trans: Transition mass between lognormal and power-law (default: 1.0)
        A_ln: Lognormal coefficient (default: 0.158, Chabrier 2003 single-star disk IMF)
        A_pl: Power-law coefficient (computed for continuity at m_trans)

    Examples:
        >>> imf = ChabrierIMF()  # Default Chabrier (2003) single-star (disk) IMF
        >>> key = jax.random.PRNGKey(42)
        >>> masses = imf.sample(key, 1000)
        >>> print(f"Mean mass: {imf.mean_mass():.3f} M☉")  # ~0.61 M☉ over [0.08, 100]

    References:
        Chabrier (2003), PASP, 115, 763 - Table 1: single-star disk IMF coefficients
        Chabrier (2005), ASSL, 327, 41 - Review of IMF determinations
    """

    m_min: float = 0.08  # Hydrogen burning limit
    m_max: float = 100.0
    m_c: float = 0.08  # Chabrier (2003) Table 1 characteristic mass (0.079, rounded)
    sigma: float = 0.69  # Lognormal width (Chabrier 2003 Table 1, single-object disk)
    alpha: float = (
        2.3  # High-mass slope dN/dm ∝ m^-α (Chabrier 2003 Table 1: x=1.3 ⇒ α=2.3)
    )
    m_trans: float = 1.0  # Transition mass [M☉]
    A_ln: float = 0.158  # Chabrier (2003) lognormal coefficient

    @property
    def A_pl(self) -> Float[Array, ""]:
        """Power-law coefficient for continuity at m_trans.

        For continuity: ξ_ln(m_trans) = ξ_pl(m_trans)

        Using log₁₀-based lognormal:
            A_pl = ξ_ln(m_trans) × m_trans^α
        """
        m_t = jnp.asarray(self.m_trans)
        xi_ln_mt = self._lognormal_pdf_unnorm(m_t)
        return xi_ln_mt * jnp.power(m_t, self.alpha)

    def __check_init__(self):
        """Validate parameters."""
        # Note: m_c doesn't need to be in [m_min, m_max] - it's the lognormal peak
        # which can be below m_min (the distribution is truncated at m_min)
        if self.m_c <= 0 or self.m_c >= self.m_max:
            raise ValueError(
                f"Characteristic mass m_c ({self.m_c}) must be positive "
                f"and less than m_max ({self.m_max})"
            )
        if self.sigma <= 0:
            raise ValueError(f"Lognormal width sigma ({self.sigma}) must be positive")
        if self.alpha <= 0:
            raise ValueError(
                f"Power-law exponent alpha ({self.alpha}) must be positive"
            )
        if self.m_min >= self.m_trans:
            raise ValueError(
                f"m_min ({self.m_min}) must be less than m_trans ({self.m_trans})"
            )

    # ==========================================================================
    # Internal helpers for lognormal + power-law
    # ==========================================================================

    def _log10(self, m: ArrayLike) -> Float[Array, "..."]:
        """Safe log₁₀ computation."""
        return jnp.log10(jnp.maximum(m, 1e-30))

    def _lognormal_pdf_unnorm(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Unnormalized lognormal PDF in mass space (dN/dm).

        Chabrier (2003) for the single-star (disk) IMF:
        ξ(m) = A_ln / (m × ln 10) × exp[-(log₁₀ m - log₁₀ m_c)² / (2σ²)]

        The 1/(m × ln 10) is the Jacobian from log₁₀ space to mass space.
        """
        log10_m = self._log10(m)
        log10_mc = self._log10(self.m_c)

        # Gaussian in log₁₀ space
        quad = -((log10_m - log10_mc) ** 2) / (2.0 * self.sigma**2)

        # Jacobian: 1 / (m × ln 10)
        jacobian = 1.0 / (m * jnp.log(10.0) + 1e-30)

        return self.A_ln * jacobian * jnp.exp(quad)

    def _powerlaw_pdf_unnorm(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Unnormalized power-law PDF component with Chabrier coefficient.

        ξ_pl(m) = A_pl × m^(-α)

        Note: Chabrier (2003) defines ξ(log m) = A_pl × m^(-x) where x = α - 1.
        Converting: ξ(m) = ξ(log m) / m = A_pl × m^(-x-1) = A_pl × m^(-α).
        """
        return self.A_pl * (m + 1e-30) ** (-self.alpha)

    def _lognormal_integral(self, m_lo: ArrayLike, m_hi: ArrayLike) -> Float[Array, ""]:
        """Analytical integral of lognormal using log₁₀ form.

        For ξ(m) = A_ln / (m × ln 10) × exp[-(log₁₀ m - log₁₀ m_c)² / (2σ²)]:

        ∫ ξ(m) dm = A_ln × σ × √(π/2) × [erf(x_hi) - erf(x_lo)]

        where x = (log₁₀ m - log₁₀ m_c) / (σ × √2)

        Derivation: Let y = log₁₀ m, then dy = 1/(m ln 10) dm
        ∫ ξ(m) dm = ∫ A_ln × exp[-(y - y_c)²/(2σ²)] dy

        Substituting t = (y - y_c)/(σ√2):
        ∫ A_ln × exp(-t²) × σ√2 dt = A_ln × σ√2 × √π/2 × [erf(x_hi) - erf(x_lo)]
                                    = A_ln × σ × √(π/2) × [erf(x_hi) - erf(x_lo)]
        """
        m_lo = jnp.asarray(m_lo)
        m_hi = jnp.asarray(m_hi)

        log10_mc = self._log10(self.m_c)
        sqrt_2_sigma = jnp.sqrt(2.0) * self.sigma

        x_lo = (self._log10(m_lo) - log10_mc) / sqrt_2_sigma
        x_hi = (self._log10(m_hi) - log10_mc) / sqrt_2_sigma

        erf_lo = jax.scipy.special.erf(x_lo)
        erf_hi = jax.scipy.special.erf(x_hi)

        # Correct integral: A_ln × σ × √(π/2)
        return self.A_ln * self.sigma * jnp.sqrt(jnp.pi / 2.0) * (erf_hi - erf_lo)

    def _powerlaw_integral(self, m_lo: ArrayLike, m_hi: ArrayLike) -> Float[Array, ""]:
        """Integral of power-law component from m_lo to m_hi.

        For ξ_pl(m) = A_pl × m^(-α):
        ∫ξ_pl(m) dm = A_pl × [m^(1-α) / (1-α)] evaluated at bounds
        """
        e = 1.0 - self.alpha
        base_integral = jnp.where(
            jnp.abs(e) < 1e-12,
            jnp.log(m_hi / m_lo),
            (m_hi**e - m_lo**e) / e,
        )
        return self.A_pl * base_integral

    def _compute_normalization(
        self,
    ) -> Tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
        """Compute normalization constants.

        Returns:
            I_ln: Lognormal integral (with A_ln coefficient)
            I_pl: Power-law integral (with A_pl coefficient)
            Z: Total normalization (I_ln + I_pl)

        Note: Chabrier (2003) coefficients A_ln=0.158, A_pl=0.044 are now
        included in the integral functions, giving proper relative weighting.
        """
        m_ln_max = jnp.minimum(self.m_trans, self.m_max)
        m_pl_min = jnp.maximum(self.m_trans, self.m_min)

        # Lognormal integral (A_ln coefficient included)
        I_ln = self._lognormal_integral(self.m_min, m_ln_max)

        # Power-law integral (A_pl coefficient included)
        has_powerlaw = self.m_max > self.m_trans
        I_pl = jnp.where(
            has_powerlaw,
            self._powerlaw_integral(m_pl_min, self.m_max),
            0.0,
        )

        Z = I_ln + I_pl
        return I_ln, I_pl, Z

    # ==========================================================================
    # IMFProtocol methods
    # ==========================================================================

    def _logpdf_unnorm(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Unnormalized log-PDF (for TruncatedIMF compatibility).

        Returns log(ξ(m)) where ξ(m) uses Chabrier coefficients A_ln and A_pl.
        Note: the pdf VALUE is continuous at m_trans by construction — A_pl is set so
        ξ_ln(m_trans) = ξ_pl(m_trans) (see the A_pl property). Only the slope has a kink
        there (the two components have different log-slopes), matching Chabrier (2003),
        which joins the lognormal and Salpeter-tail continuously at 1 M☉.
        """
        is_lognormal = m < self.m_trans

        # Lognormal component (A_ln included in _lognormal_pdf_unnorm)
        ln_pdf = self._lognormal_pdf_unnorm(m)

        # Power-law component (A_pl included in _powerlaw_pdf_unnorm)
        pl_pdf = self._powerlaw_pdf_unnorm(m)

        pdf_unnorm = jnp.where(is_lognormal, ln_pdf, pl_pdf)
        return jnp.log(pdf_unnorm + 1e-30)

    def _cdf_unnorm(self, m: ArrayLike) -> Float[Array, "..."]:
        """Unnormalized CDF (for TruncatedIMF compatibility).

        Integrates the piecewise IMF from m_min to m.
        Coefficients A_ln and A_pl are included in the integral functions.
        """
        m_ln_max = jnp.minimum(self.m_trans, self.m_max)

        def cdf_unnorm_scalar(m_val):
            is_lognormal = m_val < self.m_trans

            # Lognormal CDF contribution (A_ln coefficient included)
            ln_cdf = self._lognormal_integral(self.m_min, jnp.minimum(m_val, m_ln_max))

            # Power-law CDF contribution (if m > m_trans, A_pl coefficient included)
            I_ln_full = self._lognormal_integral(self.m_min, m_ln_max)
            pl_cdf = jnp.where(
                m_val > self.m_trans,
                I_ln_full + self._powerlaw_integral(self.m_trans, m_val),
                ln_cdf,
            )

            return jnp.where(is_lognormal, ln_cdf, pl_cdf)

        m_arr = jnp.asarray(m)
        if m_arr.ndim == 0:
            return cdf_unnorm_scalar(m_arr)
        else:
            return jax.vmap(cdf_unnorm_scalar)(m_arr.ravel()).reshape(m_arr.shape)

    @property
    def _log_norm(self) -> Float[Array, ""]:
        """Log normalization constant."""
        _, _, Z = self._compute_normalization()
        return jnp.log(Z + 1e-30)

    def logpdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Normalized log-PDF. Returns -inf outside [m_min, m_max].

        Args:
            m: Mass values [M☉]

        Returns:
            Normalized log-PDF values
        """
        m_arr = jnp.asarray(m)
        in_domain = (m_arr >= self.m_min) & (m_arr <= self.m_max)
        lp = self._logpdf_unnorm(m_arr) - self._log_norm
        return jnp.where(in_domain, lp, -jnp.inf)

    def cdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Normalized CDF. Returns 0 below m_min, 1 above m_max.

        Args:
            m: Mass values [M☉]

        Returns:
            CDF values in [0, 1]
        """
        m_arr = jnp.asarray(m)
        _, _, Z = self._compute_normalization()
        raw = self._cdf_unnorm(m_arr) / (Z + 1e-30)
        return jnp.where(
            m_arr <= self.m_min,
            0.0,
            jnp.where(m_arr >= self.m_max, 1.0, raw),
        )

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Inverse CDF (percent point function) via Newton iteration.

        Uses custom initial guess optimized for lognormal+power-law shape.

        Args:
            u: Uniform samples in [0, 1]

        Returns:
            Mass values [M☉]
        """
        return self._ppf_newton_chabrier(u)

    def _ppf_newton_chabrier(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Newton solver with two-component initial guess.

        The Chabrier IMF has lognormal (m < 1) and power-law (m >= 1) components.
        We need separate initial guesses for each region:
        - For u < p_ln (lognormal region): use lognormal inverse
        - For u >= p_ln (power-law region): use power-law inverse
        """
        u = jnp.clip(jnp.asarray(u), 1e-10, 1.0 - 1e-10)

        # Compute probability in lognormal region
        I_ln, I_pl, Z = self._compute_normalization()
        p_ln = I_ln / Z  # Probability in lognormal (m < m_trans)

        # Lognormal initial guess parameters (using log₁₀)
        log10_mc = self._log10(self.m_c)
        sqrt_2_sigma = jnp.sqrt(2.0) * self.sigma

        # Lognormal erf bounds (within [m_min, m_trans])
        m_ln_max = jnp.minimum(self.m_trans, self.m_max)
        erf_min = jax.scipy.special.erf(
            (self._log10(self.m_min) - log10_mc) / sqrt_2_sigma
        )
        erf_max = jax.scipy.special.erf(
            (self._log10(m_ln_max) - log10_mc) / sqrt_2_sigma
        )

        # Lognormal initial guess: for u in [0, p_ln]
        # Scale u to [0, 1] within lognormal region
        u_ln_scaled = jnp.clip(u / p_ln, 0.0, 1.0)
        erf_target = erf_min + u_ln_scaled * (erf_max - erf_min)
        log10_m_ln = log10_mc + sqrt_2_sigma * jax.scipy.special.erfinv(erf_target)
        m0_ln = jnp.clip(jnp.power(10.0, log10_m_ln), self.m_min, m_ln_max)

        # Power-law initial guess: for u in [p_ln, 1]
        # Scale u to [0, 1] within power-law region
        u_pl_scaled = jnp.clip((u - p_ln) / (1.0 - p_ln + 1e-10), 0.0, 1.0)
        exp = 1.0 - self.alpha
        exp_safe = jnp.where(jnp.abs(exp) < 1e-10, 1e-10, exp)

        # Power-law inverse: m = [u_scaled * (m_max^exp - m_trans^exp) + m_trans^exp]^(1/exp)
        m_trans_exp = self.m_trans**exp_safe
        m_max_exp = self.m_max**exp_safe
        m0_pl = jnp.power(
            u_pl_scaled * (m_max_exp - m_trans_exp) + m_trans_exp, 1.0 / exp_safe
        )
        m0_pl = jnp.clip(m0_pl, self.m_trans, self.m_max)

        # Select initial guess based on which region u falls in
        is_lognormal = u < p_ln
        m0 = jnp.where(is_lognormal, m0_ln, m0_pl)

        # 30 fixed iterations (vs 20 in BaseIMF): the two-component lognormal+power-law
        # CDF has a sharp slope change at m_trans, so the worst-case starting guess needs
        # a few extra Newton steps to converge across the join. The Newton loop itself is
        # jaxstro's generic newton_ppf (JIT-safe, no while_loop); only the two-component
        # initial guess above is Chabrier-specific and stays local.
        return newton_ppf(
            u,
            self.cdf,
            x0=m0,
            lo=self.m_min,
            hi=self.m_max,
            pdf=lambda m: jnp.exp(self.logpdf(m)),
            n_iter=30,
            pdf_floor=1e-30,
        )

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample n masses via reparameterization trick.

        Args:
            key: JAX PRNG key
            n: Number of samples

        Returns:
            n mass samples [M☉]
        """
        u = jax.random.uniform(key, (n,))
        return self.ppf(u)

    def mean_mass(self) -> Float[Array, ""]:
        """Expected mass E[m] via a LOG-spaced trapezoid.

        Log-spacing is robust to a steep low-mass spike if m_min is lowered into the
        power-law-divergent regime; for the default lognormal (which turns over below
        m_c) it gives the same value as the old linear grid. Returns mean mass [M☉].
        """
        m_grid = jnp.exp(jnp.linspace(jnp.log(self.m_min), jnp.log(self.m_max), 4000))
        pdf_grid = jnp.exp(self.logpdf(m_grid))
        return jnp.trapezoid(m_grid * pdf_grid, m_grid)

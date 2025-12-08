"""Binary star mass functions with mass-ratio distributions.

Implements mass-ratio sampling for binary systems following the literature:

Primary References:
    - Moe & Di Stefano (2017) ApJS 230, 15
      "Mind Your Ps and Qs: The Interrelation between Period (P) and
      Mass-ratio (q) Distributions of Binary Stars"
      Key result: Mass-ratio distribution depends on primary mass and period.
      For solar-type primaries: excess of twins (q > 0.95).

    - Raghavan et al. (2010) ApJS 190, 1
      "A Survey of Stellar Families: Multiplicity of Solar-type Stars"
      Binary fraction ~46% for FGK dwarfs. Flat q distribution for wide binaries.

    - Sana et al. (2012) Science 337, 444
      "Binary Interaction Dominates the Evolution of Massive Stars"
      O-star binary fraction ~70%. Power-law q distribution with γ ≈ -0.1.

    - Duchêne & Kraus (2013) ARA&A 51, 269
      "Stellar Multiplicity"
      Comprehensive review of multiplicity across spectral types.

Mass Ratio Distributions:
    - FlatMassRatio: Uniform q ∈ [q_min, 1]
    - PowerLawMassRatio: p(q) ∝ q^γ (Sana+12: γ ≈ -0.1 for O-stars)
    - TwinPeakedMassRatio: Flat + Gaussian twin peak (Moe+17)
    - MoeDiStefano2017: Full mass-dependent model from Moe & Di Stefano (2017)

Binary Fraction Models:
    - ConstantBinaryFraction: f_bin = const (typical: 0.5)
    - MassDependentBinaryFraction: f_bin(M) following Moe+17 Table 13

Examples:
    >>> from progenax.imf.binary import BinaryIMF, FlatMassRatio
    >>> from progenax.imf import PowerLawIMF
    >>> import jax
    >>>
    >>> # Simple binary IMF with flat mass ratio
    >>> primary_imf = PowerLawIMF.kroupa()
    >>> q_dist = FlatMassRatio(q_min=0.1)
    >>> binary_imf = BinaryIMF(primary_imf, q_dist, binary_fraction=0.5)
    >>>
    >>> # Sample binary systems
    >>> key = jax.random.PRNGKey(42)
    >>> m1, m2, is_binary = binary_imf.sample_systems(key, n=1000)
"""

from __future__ import annotations

from typing import Callable, Protocol, Tuple, Union, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from .base import BaseIMF


# Type aliases for custom callables
BinaryFractionCallable = Callable[[Float[Array, "..."]], Float[Array, "..."]]
MassRatioSamplerCallable = Callable[
    [PRNGKeyArray, Float[Array, "n"]], Float[Array, "n"]
]


# =============================================================================
# Mass Ratio Distribution Protocol
# =============================================================================


@runtime_checkable
class MassRatioProtocol(Protocol):
    """Protocol for mass-ratio distributions.

    All mass-ratio distributions must implement:
        - pdf(q): Probability density at mass ratio q
        - cdf(q): Cumulative distribution up to q
        - ppf(u): Inverse CDF (percent point function)
        - sample(key, n): Generate n samples

    Note: q = M_secondary / M_primary ∈ [q_min, 1] by convention.
    """

    q_min: float

    def pdf(self, q: Float[Array, "..."]) -> Float[Array, "..."]:
        """Probability density function at mass ratio q."""
        ...

    def cdf(self, q: Float[Array, "..."]) -> Float[Array, "..."]:
        """Cumulative distribution function at mass ratio q."""
        ...

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Percent point function (inverse CDF)."""
        ...

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample n mass ratios."""
        ...


# =============================================================================
# Mass Ratio Distributions
# =============================================================================


class FlatMassRatio(eqx.Module):
    """Uniform mass-ratio distribution q ∈ [q_min, 1].

    Reference:
        Raghavan et al. (2010) ApJS 190, 1 - Fig. 16
        Wide solar-type binaries show approximately flat q distribution.

    Parameters:
        q_min: Minimum mass ratio (default: 0.1)
               Below ~0.1, companions become brown dwarfs.
    """

    q_min: float = 0.1

    def pdf(self, q: Float[Array, "..."]) -> Float[Array, "..."]:
        """Uniform PDF: p(q) = 1/(1 - q_min) for q ∈ [q_min, 1]."""
        norm = 1.0 / (1.0 - self.q_min)
        in_range = (q >= self.q_min) & (q <= 1.0)
        return jnp.where(in_range, norm, 0.0)

    def cdf(self, q: Float[Array, "..."]) -> Float[Array, "..."]:
        """Uniform CDF: F(q) = (q - q_min) / (1 - q_min)."""
        cdf_val = (q - self.q_min) / (1.0 - self.q_min)
        return jnp.clip(cdf_val, 0.0, 1.0)

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Inverse CDF: q = q_min + u × (1 - q_min)."""
        return self.q_min + u * (1.0 - self.q_min)

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample n mass ratios uniformly."""
        u = jax.random.uniform(key, (n,))
        return self.ppf(u)


class PowerLawMassRatio(eqx.Module):
    """Power-law mass-ratio distribution p(q) ∝ q^γ.

    Reference:
        Sana et al. (2012) Science 337, 444 - Eq. 3
        O-star binaries: γ ≈ -0.1 (nearly flat, slight preference for unequal)

        Moe & Di Stefano (2017) ApJS 230, 15 - Section 9.1
        Solar-type stars: γ ≈ 0 to +0.3 depending on period

    Parameters:
        gamma: Power-law exponent (default: 0.0 = flat)
               γ < 0: Prefers unequal masses
               γ > 0: Prefers equal masses
               γ = 0: Flat distribution
        q_min: Minimum mass ratio (default: 0.1)

    Note:
        For γ = -1, the distribution is singular at q = 0.
        Use q_min > 0 to avoid singularity.
    """

    gamma: float = 0.0
    q_min: float = 0.1

    def _norm(self) -> float:
        """Normalization constant for power-law."""
        g = self.gamma
        q0 = self.q_min
        # ∫_{q_min}^1 q^γ dq = (1^{γ+1} - q_min^{γ+1}) / (γ+1) for γ ≠ -1
        # For γ = -1: ∫ q^{-1} dq = log(1) - log(q_min) = -log(q_min)

        def log_case():
            return -jnp.log(q0)

        def power_case():
            # Add small epsilon to avoid division by zero during tracing
            g_plus_1 = g + 1.0
            g_plus_1_safe = jnp.where(jnp.abs(g_plus_1) < 1e-10, 1e-10, g_plus_1)
            return (1.0 - q0 ** (g + 1.0)) / g_plus_1_safe

        return jax.lax.cond(
            jnp.abs(g + 1.0) < 1e-10,
            log_case,
            power_case,
        )

    def pdf(self, q: Float[Array, "..."]) -> Float[Array, "..."]:
        """Power-law PDF: p(q) = q^γ / Z."""
        norm = self._norm()
        pdf_unnorm = q**self.gamma
        in_range = (q >= self.q_min) & (q <= 1.0)
        return jnp.where(in_range, pdf_unnorm / norm, 0.0)

    def cdf(self, q: Float[Array, "..."]) -> Float[Array, "..."]:
        """Power-law CDF."""
        g = self.gamma
        q0 = self.q_min
        norm = self._norm()

        def cdf_neq_m1(q_val):
            # ∫_{q_min}^q t^γ dt = (q^{γ+1} - q_min^{γ+1}) / (γ+1)
            integral = (q_val ** (g + 1.0) - q0 ** (g + 1.0)) / (g + 1.0)
            return integral / norm

        def cdf_eq_m1(q_val):
            # ∫_{q_min}^q t^{-1} dt = log(q) - log(q_min) = log(q/q_min)
            integral = jnp.log(q_val / q0)
            return integral / norm

        # Vectorize over q
        def cdf_scalar(q_val):
            raw = jax.lax.cond(
                jnp.abs(g + 1.0) < 1e-10,
                lambda: cdf_eq_m1(q_val),
                lambda: cdf_neq_m1(q_val),
            )
            return jnp.clip(raw, 0.0, 1.0)

        # Handle scalar vs array
        if jnp.ndim(q) == 0:
            return cdf_scalar(q)
        return jax.vmap(cdf_scalar)(q.ravel()).reshape(q.shape)

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Inverse CDF for power-law."""
        g = self.gamma
        q0 = self.q_min
        norm = self._norm()

        def ppf_neq_m1(u_val):
            # F(q) = (q^{γ+1} - q_min^{γ+1}) / ((γ+1) × Z) = u
            # q^{γ+1} = u × (γ+1) × Z + q_min^{γ+1}
            # q = [u × (γ+1) × Z + q_min^{γ+1}]^{1/(γ+1)}
            inner = u_val * (g + 1.0) * norm + q0 ** (g + 1.0)
            return inner ** (1.0 / (g + 1.0))

        def ppf_eq_m1(u_val):
            # F(q) = log(q/q_min) / Z = u
            # log(q/q_min) = u × Z
            # q = q_min × exp(u × Z)
            return q0 * jnp.exp(u_val * norm)

        def ppf_scalar(u_val):
            return jax.lax.cond(
                jnp.abs(g + 1.0) < 1e-10,
                lambda: ppf_eq_m1(u_val),
                lambda: ppf_neq_m1(u_val),
            )

        if jnp.ndim(u) == 0:
            return ppf_scalar(u)
        return jax.vmap(ppf_scalar)(u.ravel()).reshape(u.shape)

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample n mass ratios from power-law."""
        u = jax.random.uniform(key, (n,))
        return self.ppf(u)


class TwinPeakedMassRatio(eqx.Module):
    """Mass-ratio distribution with excess of twins (q ≈ 1).

    Reference:
        Moe & Di Stefano (2017) ApJS 230, 15 - Section 9.2
        "For solar-type primaries at intermediate periods (P = 10-200 days),
        we measure an excess fraction of twins F_twin ≈ 0.1 above the
        baseline power-law distribution."

        Lucy (2006) A&A 457, 629 - First systematic study of twin excess

    Model:
        p(q) = (1 - f_twin) × q^γ / Z_pl + f_twin × N(q | μ=1, σ_twin)

        where Z_pl is the power-law normalization.

    Parameters:
        gamma: Power-law exponent for baseline (default: 0.0)
        f_twin: Fraction of systems that are twins (default: 0.1)
        sigma_twin: Width of twin peak (default: 0.03)
                   Moe+17 suggest σ ≈ 0.02-0.05
        q_min: Minimum mass ratio (default: 0.1)
    """

    gamma: float = 0.0
    f_twin: float = 0.1
    sigma_twin: float = 0.03
    q_min: float = 0.1

    def _powerlaw_norm(self) -> float:
        """Normalization for power-law component."""
        g = self.gamma
        q0 = self.q_min

        def log_case():
            return -jnp.log(q0)

        def power_case():
            # Add small epsilon to avoid division by zero during tracing
            g_plus_1 = g + 1.0
            g_plus_1_safe = jnp.where(jnp.abs(g_plus_1) < 1e-10, 1e-10, g_plus_1)
            return (1.0 - q0 ** (g + 1.0)) / g_plus_1_safe

        return jax.lax.cond(
            jnp.abs(g + 1.0) < 1e-10,
            log_case,
            power_case,
        )

    def _gaussian_norm(self) -> float:
        """Normalization for truncated Gaussian at q=1."""
        # Gaussian centered at μ=1, truncated to [q_min, 1]
        # Full Gaussian integral from -∞ to ∞ is 1
        # Truncated integral: Φ((1-1)/σ) - Φ((q_min-1)/σ) = 0.5 - Φ((q_min-1)/σ)
        z_min = (self.q_min - 1.0) / self.sigma_twin
        cdf_min = 0.5 * (1.0 + jax.scipy.special.erf(z_min / jnp.sqrt(2.0)))
        return 0.5 - cdf_min  # Probability mass in [q_min, 1]

    def pdf(self, q: Float[Array, "..."]) -> Float[Array, "..."]:
        """Twin-peaked PDF."""
        # Power-law component
        pl_norm = self._powerlaw_norm()
        pl_pdf = q**self.gamma / pl_norm

        # Gaussian twin peak (centered at q=1)
        gauss_norm = self._gaussian_norm()
        gauss_pdf = (
            jnp.exp(-0.5 * ((q - 1.0) / self.sigma_twin) ** 2)
            / (self.sigma_twin * jnp.sqrt(2.0 * jnp.pi))
            / gauss_norm
        )

        # Combined
        combined = (1.0 - self.f_twin) * pl_pdf + self.f_twin * gauss_pdf
        in_range = (q >= self.q_min) & (q <= 1.0)
        return jnp.where(in_range, combined, 0.0)

    def cdf(self, q: Float[Array, "..."]) -> Float[Array, "..."]:
        """Twin-peaked CDF."""
        g = self.gamma
        q0 = self.q_min
        pl_norm = self._powerlaw_norm()
        gauss_norm = self._gaussian_norm()

        def cdf_scalar(q_val):
            # Power-law CDF
            pl_integral = jax.lax.cond(
                jnp.abs(g + 1.0) < 1e-10,
                lambda: jnp.log(q_val / q0),
                lambda: (q_val ** (g + 1.0) - q0 ** (g + 1.0)) / (g + 1.0),
            )
            pl_cdf = pl_integral / pl_norm

            # Gaussian CDF (truncated)
            z_val = (q_val - 1.0) / self.sigma_twin
            z_min = (q0 - 1.0) / self.sigma_twin
            cdf_val = 0.5 * (1.0 + jax.scipy.special.erf(z_val / jnp.sqrt(2.0)))
            cdf_min = 0.5 * (1.0 + jax.scipy.special.erf(z_min / jnp.sqrt(2.0)))
            gauss_cdf = (cdf_val - cdf_min) / gauss_norm

            combined = (1.0 - self.f_twin) * pl_cdf + self.f_twin * gauss_cdf
            return jnp.clip(combined, 0.0, 1.0)

        if jnp.ndim(q) == 0:
            return cdf_scalar(q)
        return jax.vmap(cdf_scalar)(q.ravel()).reshape(q.shape)

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Inverse CDF via Newton iteration."""

        def ppf_scalar(u_val):
            # Initial guess: linear interpolation
            m0 = self.q_min + u_val * (1.0 - self.q_min)

            def newton_step(_, m):
                f_m = self.cdf(m) - u_val
                df_m = self.pdf(m)
                df_m = jnp.maximum(df_m, 1e-10)  # Avoid division by zero
                m_new = m - f_m / df_m
                return jnp.clip(m_new, self.q_min, 1.0)

            return jax.lax.fori_loop(0, 20, newton_step, m0)

        if jnp.ndim(u) == 0:
            return ppf_scalar(u)
        return jax.vmap(ppf_scalar)(u.ravel()).reshape(u.shape)

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample via inverse transform."""
        u = jax.random.uniform(key, (n,))
        return self.ppf(u)


class MoeDiStefano2017(eqx.Module):
    """Mass-dependent mass-ratio distribution from Moe & Di Stefano (2017).

    Reference:
        Moe & Di Stefano (2017) ApJS 230, 15
        Table 10: Intrinsic mass-ratio distributions

    This implements the full mass-dependent model where:
        - γ (power-law exponent) depends on primary mass
        - f_twin (twin excess) depends on primary mass and period
        - The distribution transitions smoothly across mass ranges

    Model (simplified, period-averaged):
        - M1 < 0.8 Msun: γ ≈ 0.4, f_twin ≈ 0.05 (M-dwarfs)
        - 0.8 < M1 < 1.2 Msun: γ ≈ 0.3, f_twin ≈ 0.10 (Solar-type)
        - 1.2 < M1 < 3.5 Msun: γ ≈ 0.0, f_twin ≈ 0.08 (A/F stars)
        - M1 > 3.5 Msun: γ ≈ -0.5, f_twin ≈ 0.03 (OB stars)

    Parameters:
        q_min: Minimum mass ratio (default: 0.1)
        sigma_twin: Width of twin peak (default: 0.03)
    """

    q_min: float = 0.1
    sigma_twin: float = 0.03

    def _gamma_of_mass(self, m1: Float[Array, "..."]) -> Float[Array, "..."]:
        """Power-law exponent as function of primary mass (Moe+17 Table 10)."""
        # Piecewise linear interpolation
        gamma = jnp.where(
            m1 < 0.8,
            0.4,  # M-dwarfs
            jnp.where(
                m1 < 1.2,
                0.3,  # Solar-type
                jnp.where(
                    m1 < 3.5,
                    0.0,  # A/F stars
                    -0.5,  # OB stars
                ),
            ),
        )
        return gamma

    def _ftwin_of_mass(self, m1: Float[Array, "..."]) -> Float[Array, "..."]:
        """Twin excess fraction as function of primary mass (Moe+17 Table 10)."""
        f_twin = jnp.where(
            m1 < 0.8,
            0.05,  # M-dwarfs
            jnp.where(
                m1 < 1.2,
                0.10,  # Solar-type (highest twin excess)
                jnp.where(
                    m1 < 3.5,
                    0.08,  # A/F stars
                    0.03,  # OB stars (lowest twin excess)
                ),
            ),
        )
        return f_twin

    def sample_given_primary(
        self, key: PRNGKeyArray, m1: Float[Array, "n"]
    ) -> Float[Array, "n"]:
        """Sample mass ratios given primary masses.

        This is the key method: q distribution depends on M1.

        Args:
            key: JAX random key
            m1: Primary masses (n,)

        Returns:
            Mass ratios q ∈ [q_min, 1] with shape (n,)
        """
        n = m1.shape[0]
        key1, key2, key3 = jax.random.split(key, 3)

        # Get mass-dependent parameters
        gamma = self._gamma_of_mass(m1)
        f_twin = self._ftwin_of_mass(m1)

        # Decide which component: power-law or twin
        is_twin = jax.random.uniform(key1, (n,)) < f_twin

        # Sample power-law component
        u_pl = jax.random.uniform(key2, (n,))

        def sample_powerlaw(gamma_val, u_val):
            """Sample from power-law q^gamma."""
            q0 = self.q_min

            def neq_m1():
                g = gamma_val
                norm = (1.0 - q0 ** (g + 1.0)) / (g + 1.0)
                inner = u_val * (g + 1.0) * norm + q0 ** (g + 1.0)
                return inner ** (1.0 / (g + 1.0))

            def eq_m1():
                norm = -jnp.log(q0)
                return q0 * jnp.exp(u_val * norm)

            return jax.lax.cond(jnp.abs(gamma_val + 1.0) < 1e-10, eq_m1, neq_m1)

        q_powerlaw = jax.vmap(sample_powerlaw)(gamma, u_pl)

        # Sample twin component (truncated Gaussian centered at q=1, truncated to [q_min, 1])
        # Use inverse CDF sampling for correct distribution
        z_min = (self.q_min - 1.0) / self.sigma_twin
        z_max = 0.0  # (1.0 - 1.0) / sigma = 0
        # CDF values at boundaries
        cdf_min = 0.5 * (1.0 + jax.scipy.special.erf(z_min / jnp.sqrt(2.0)))
        cdf_max = 0.5  # Φ(0) = 0.5
        # Sample uniform in [cdf_min, cdf_max], then inverse CDF
        u_twin = jax.random.uniform(key3, (n,))
        u_scaled = cdf_min + u_twin * (cdf_max - cdf_min)
        # Inverse CDF: Φ^(-1)(u) = √2 * erfinv(2u - 1)
        z_samples = jnp.sqrt(2.0) * jax.scipy.special.erfinv(2.0 * u_scaled - 1.0)
        q_twin = 1.0 + z_samples * self.sigma_twin

        # Select based on component
        q = jnp.where(is_twin, q_twin, q_powerlaw)
        return q

    def pdf_given_primary(
        self, q: Float[Array, "..."], m1: float
    ) -> Float[Array, "..."]:
        """PDF of mass ratio given primary mass."""
        gamma = self._gamma_of_mass(jnp.asarray(m1))
        f_twin = self._ftwin_of_mass(jnp.asarray(m1))

        # Power-law normalization
        g = gamma
        q0 = self.q_min
        pl_norm = jax.lax.cond(
            jnp.abs(g + 1.0) < 1e-10,
            lambda: -jnp.log(q0),
            lambda: (1.0 - q0 ** (g + 1.0)) / (g + 1.0),
        )
        pl_pdf = q**g / pl_norm

        # Gaussian twin peak
        z_min = (q0 - 1.0) / self.sigma_twin
        cdf_min = 0.5 * (1.0 + jax.scipy.special.erf(z_min / jnp.sqrt(2.0)))
        gauss_norm = 0.5 - cdf_min
        gauss_pdf = (
            jnp.exp(-0.5 * ((q - 1.0) / self.sigma_twin) ** 2)
            / (self.sigma_twin * jnp.sqrt(2.0 * jnp.pi))
            / gauss_norm
        )

        combined = (1.0 - f_twin) * pl_pdf + f_twin * gauss_pdf
        in_range = (q >= q0) & (q <= 1.0)
        return jnp.where(in_range, combined, 0.0)


# =============================================================================
# Binary Fraction Models
# =============================================================================


class ConstantBinaryFraction(eqx.Module):
    """Constant binary fraction independent of mass.

    Reference:
        Raghavan et al. (2010) ApJS 190, 1
        Overall multiplicity fraction ~46% for solar-type stars.

    Parameters:
        f_bin: Binary fraction (default: 0.5)
    """

    f_bin: float = 0.5

    def __call__(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Return binary fraction (constant)."""
        return jnp.full_like(m, self.f_bin)


class MassDependentBinaryFraction(eqx.Module):
    """Mass-dependent binary fraction from Moe & Di Stefano (2017).

    Reference:
        Moe & Di Stefano (2017) ApJS 230, 15 - Table 13
        "Close Binary Fraction as Function of Primary Mass"

    Model (period-integrated companion frequency):
        - M < 0.1 Msun: f_bin ≈ 0.22 (VLM/brown dwarfs)
        - 0.1 < M < 0.5 Msun: f_bin ≈ 0.26 (M-dwarfs)
        - 0.5 < M < 1.0 Msun: f_bin ≈ 0.44 (K/G-dwarfs)
        - 1.0 < M < 2.0 Msun: f_bin ≈ 0.50 (F/A-stars)
        - 2.0 < M < 5.0 Msun: f_bin ≈ 0.60 (B-stars)
        - 5.0 < M < 10 Msun: f_bin ≈ 0.80 (early B)
        - M > 10 Msun: f_bin ≈ 0.90 (O-stars)

    Note: These are companion frequencies, not strict binary fractions.
    Higher-order multiples (triples, etc.) are common at high masses.
    """

    def __call__(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Return binary fraction as function of mass."""
        return jnp.where(
            m < 0.1,
            0.22,
            jnp.where(
                m < 0.5,
                0.26,
                jnp.where(
                    m < 1.0,
                    0.44,
                    jnp.where(
                        m < 2.0,
                        0.50,
                        jnp.where(
                            m < 5.0,
                            0.60,
                            jnp.where(m < 10.0, 0.80, 0.90),
                        ),
                    ),
                ),
            ),
        )


# =============================================================================
# Binary IMF Class
# =============================================================================


class BinaryIMF(eqx.Module):
    """Initial Mass Function for binary star populations.

    Combines a primary-mass IMF with a mass-ratio distribution and
    binary fraction model to generate complete binary populations.

    Default Configuration (Moe & Di Stefano 2017):
        - Mass-ratio distribution: MoeDiStefano2017 (mass-dependent γ and twin excess)
        - Binary fraction: MassDependentBinaryFraction (increases with primary mass)

    This default follows the comprehensive observational constraints from
    Moe & Di Stefano (2017) ApJS 230, 15, which represents the most complete
    characterization of binary statistics to date.

    Reference:
        Kroupa (1995) MNRAS 277, 1507 - IMF-consistent binary populations
        Moe & Di Stefano (2017) ApJS 230, 15 - Modern comprehensive model
        Raghavan et al. (2010) ApJS 190, 1 - Solar-type binary statistics
        Sana et al. (2012) Science 337, 444 - O-star binary statistics

    Parameters:
        primary_imf: IMF for primary stars (must implement BaseIMF interface)
        q_distribution: Mass-ratio distribution. Options:
            - MoeDiStefano2017() [DEFAULT] - Mass-dependent from Moe+17
            - FlatMassRatio() - Uniform q ∈ [q_min, 1]
            - PowerLawMassRatio(gamma) - p(q) ∝ q^γ
            - TwinPeakedMassRatio() - Flat + Gaussian twin peak
            - Custom callable: f(key, m1) -> q array
        binary_fraction: Binary fraction model. Options:
            - MassDependentBinaryFraction() [DEFAULT] - From Moe+17 Table 13
            - ConstantBinaryFraction(f_bin) - Constant fraction
            - float - Shorthand for ConstantBinaryFraction
            - Custom callable: f(m) -> f_bin array

    Examples:
        >>> from progenax.imf import PowerLawIMF
        >>> from progenax.imf.binary import BinaryIMF
        >>>
        >>> # Default: full Moe+17 mass-dependent model
        >>> imf = BinaryIMF(primary_imf=PowerLawIMF.kroupa())
        >>> # Equivalent to:
        >>> # imf = BinaryIMF(
        >>> #     primary_imf=PowerLawIMF.kroupa(),
        >>> #     q_distribution=MoeDiStefano2017(),
        >>> #     binary_fraction=MassDependentBinaryFraction(),
        >>> # )
        >>>
        >>> # Simple constant binary fraction
        >>> imf = BinaryIMF(
        ...     primary_imf=PowerLawIMF.kroupa(),
        ...     binary_fraction=0.5,  # 50% binaries
        ... )
        >>>
        >>> # Custom mass-ratio sampler
        >>> def my_q_sampler(key, m1):
        ...     # Custom logic: q depends on m1
        ...     return jax.random.uniform(key, m1.shape, minval=0.3, maxval=1.0)
        >>> imf = BinaryIMF(
        ...     primary_imf=PowerLawIMF.kroupa(),
        ...     q_distribution=my_q_sampler,
        ... )
        >>>
        >>> # Custom binary fraction function
        >>> def my_f_bin(m):
        ...     # Custom: 40% for low mass, 80% for high mass
        ...     return jnp.where(m < 1.0, 0.4, 0.8)
        >>> imf = BinaryIMF(
        ...     primary_imf=PowerLawIMF.kroupa(),
        ...     binary_fraction=my_f_bin,
        ... )
    """

    primary_imf: BaseIMF
    q_distribution: Union[
        MassRatioProtocol,
        MoeDiStefano2017,
        MassRatioSamplerCallable,
        None,
    ] = None
    binary_fraction: Union[
        ConstantBinaryFraction,
        MassDependentBinaryFraction,
        BinaryFractionCallable,
        float,
        None,
    ] = None

    def _get_q_distribution(
        self,
    ) -> Union[MassRatioProtocol, MoeDiStefano2017, MassRatioSamplerCallable]:
        """Get mass-ratio distribution, defaulting to MoeDiStefano2017."""
        if self.q_distribution is None:
            return MoeDiStefano2017()
        return self.q_distribution

    def _get_binary_fraction_model(
        self,
    ) -> Union[
        ConstantBinaryFraction,
        MassDependentBinaryFraction,
        BinaryFractionCallable,
        float,
    ]:
        """Get binary fraction model, defaulting to MassDependentBinaryFraction."""
        if self.binary_fraction is None:
            return MassDependentBinaryFraction()
        return self.binary_fraction

    def _get_binary_fraction(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Get binary fraction for given masses."""
        f_bin_model = self._get_binary_fraction_model()
        if isinstance(f_bin_model, float):
            return jnp.full_like(m, f_bin_model)
        return f_bin_model(m)

    def sample_primaries(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample n primary masses from the IMF."""
        return self.primary_imf.sample(key, n)

    def sample_mass_ratios(
        self, key: PRNGKeyArray, m1: Float[Array, "n"]
    ) -> Float[Array, "n"]:
        """Sample mass ratios given primary masses.

        For MoeDiStefano2017, the distribution depends on primary mass.
        For other distributions, it's independent.
        For custom callables, passes (key, m1) directly.
        """
        n = m1.shape[0]
        q_dist = self._get_q_distribution()

        # Check if it's a callable (custom sampler)
        if callable(q_dist) and not isinstance(
            q_dist,
            (
                MoeDiStefano2017,
                FlatMassRatio,
                PowerLawMassRatio,
                TwinPeakedMassRatio,
            ),
        ):
            return q_dist(key, m1)
        elif isinstance(q_dist, MoeDiStefano2017):
            return q_dist.sample_given_primary(key, m1)
        else:
            return q_dist.sample(key, n)

    def sample_systems(
        self, key: PRNGKeyArray, n: int
    ) -> Tuple[Float[Array, "n"], Float[Array, "n"], Bool[Array, "n"]]:
        """Sample n stellar systems (singles + binaries).

        Returns:
            Tuple of:
                - m1: Primary masses (n,)
                - m2: Secondary masses (n,) - 0 for single stars
                - is_binary: Boolean mask (n,) - True for binaries
        """
        key1, key2, key3 = jax.random.split(key, 3)

        # Sample primary masses
        m1 = self.sample_primaries(key1, n)

        # Decide which are binaries
        f_bin = self._get_binary_fraction(m1)
        is_binary = jax.random.uniform(key2, (n,)) < f_bin

        # Sample mass ratios
        q = self.sample_mass_ratios(key3, m1)

        # Compute secondary masses (0 for singles)
        m2 = jnp.where(is_binary, m1 * q, 0.0)

        return m1, m2, is_binary

    def sample_all_masses(
        self, key: PRNGKeyArray, n: int
    ) -> Tuple[Float[Array, "..."], Bool[Array, "n"]]:
        """Sample n systems and return all stellar masses.

        Returns:
            Tuple of:
                - masses: All stellar masses (flattened, primaries + secondaries)
                - is_binary: Boolean mask indicating which systems were binaries

        Note: The returned masses array has length n + n_binary where
        n_binary is the number of binary systems.
        """
        m1, m2, is_binary = self.sample_systems(key, n)

        # Concatenate primary and secondary masses
        # Filter out m2 = 0 (singles)
        all_masses = jnp.concatenate([m1, m2[is_binary]])

        return all_masses, is_binary

    def mean_system_mass(self) -> float:
        """Expected total mass per system.

        Returns:
            E[M_total] = E[M1] × (1 + f_bin × E[q])

        Note: This assumes binary fraction and q are independent of M1.
        For mass-dependent models, this is approximate.
        """
        mean_m1 = self.primary_imf.mean_mass()
        f_bin_model = self._get_binary_fraction_model()
        q_dist = self._get_q_distribution()

        # Get average binary fraction (approximate for mass-dependent)
        if isinstance(f_bin_model, float):
            avg_f_bin = f_bin_model
        else:
            # Sample estimate
            test_masses = jnp.logspace(-1, 2, 100)
            avg_f_bin = jnp.mean(self._get_binary_fraction(test_masses))

        # Get average q (approximate for mass-dependent)
        if isinstance(q_dist, MoeDiStefano2017):
            # Assume average q ≈ 0.5 for mass-dependent case
            avg_q = 0.5
        elif hasattr(q_dist, "q_min"):
            # For power-law or flat: E[q] ≈ (1 + q_min) / 2 for flat
            avg_q = (1.0 + q_dist.q_min) / 2.0
        else:
            avg_q = 0.5

        return mean_m1 * (1.0 + avg_f_bin * avg_q)

    def binary_fraction_overall(self) -> float:
        """Return overall binary fraction.

        For mass-dependent models, this integrates over the IMF.
        """
        f_bin_model = self._get_binary_fraction_model()
        if isinstance(f_bin_model, float):
            return f_bin_model
        else:
            # Sample estimate: integrate f_bin(M) × IMF(M) dM
            key = jax.random.PRNGKey(0)
            m_samples = self.primary_imf.sample(key, 10000)
            f_bin_samples = self._get_binary_fraction(m_samples)
            return float(jnp.mean(f_bin_samples))

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def moe2017(cls, primary_imf: BaseIMF) -> "BinaryIMF":
        """Create BinaryIMF with full Moe & Di Stefano (2017) model.

        This is the default configuration, but this factory method makes
        it explicit and documents the reference.

        Reference:
            Moe & Di Stefano (2017) ApJS 230, 15
            "Mind Your Ps and Qs: The Interrelation between Period (P) and
            Mass-ratio (q) Distributions of Binary Stars"

        Args:
            primary_imf: IMF for primary stars

        Returns:
            BinaryIMF with mass-dependent q and f_bin
        """
        return cls(
            primary_imf=primary_imf,
            q_distribution=MoeDiStefano2017(),
            binary_fraction=MassDependentBinaryFraction(),
        )

    @classmethod
    def simple(
        cls,
        primary_imf: BaseIMF,
        binary_fraction: float = 0.5,
        q_min: float = 0.1,
    ) -> "BinaryIMF":
        """Create BinaryIMF with simple constant parameters.

        Good for quick tests or when detailed binary statistics aren't needed.

        Reference:
            Raghavan et al. (2010) ApJS 190, 1 - f_bin ≈ 0.46 for FGK stars
            Duchêne & Kraus (2013) ARA&A 51, 269 - General review

        Args:
            primary_imf: IMF for primary stars
            binary_fraction: Constant binary fraction (default: 0.5)
            q_min: Minimum mass ratio (default: 0.1)

        Returns:
            BinaryIMF with flat q distribution and constant f_bin
        """
        return cls(
            primary_imf=primary_imf,
            q_distribution=FlatMassRatio(q_min=q_min),
            binary_fraction=binary_fraction,
        )

    @classmethod
    def massive_stars(
        cls,
        primary_imf: BaseIMF,
        gamma: float = -0.1,
        binary_fraction: float = 0.7,
    ) -> "BinaryIMF":
        """Create BinaryIMF tuned for massive star populations.

        Reference:
            Sana et al. (2012) Science 337, 444
            "Binary Interaction Dominates the Evolution of Massive Stars"
            O-stars: f_bin ≈ 0.69, γ ≈ -0.1 (slight preference for unequal)

        Args:
            primary_imf: IMF for primary stars
            gamma: Power-law exponent for q distribution (default: -0.1)
            binary_fraction: Binary fraction (default: 0.7)

        Returns:
            BinaryIMF configured for OB star populations
        """
        return cls(
            primary_imf=primary_imf,
            q_distribution=PowerLawMassRatio(gamma=gamma, q_min=0.1),
            binary_fraction=binary_fraction,
        )


__all__ = [
    "MassRatioProtocol",
    "FlatMassRatio",
    "PowerLawMassRatio",
    "TwinPeakedMassRatio",
    "MoeDiStefano2017",
    "ConstantBinaryFraction",
    "MassDependentBinaryFraction",
    "BinaryIMF",
]

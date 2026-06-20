"""Mass-ratio distributions for binary populations.

Split from the former monolithic ``binary.py`` (file-length limit). Public API is
unchanged: these symbols remain importable from ``progenax.imf`` and
``progenax.imf.binary``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

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
        """Uniform PDF: p(q) = 1/(1 - q_min) for q ∈ [q_min, 1], else 0.

        Args:
            q: Mass ratio q = M_secondary / M_primary (any broadcastable shape).

        Returns:
            Probability density at q (dimensionless), same shape as ``q``; 0 outside
            [q_min, 1].
        """
        norm = 1.0 / (1.0 - self.q_min)
        in_range = (q >= self.q_min) & (q <= 1.0)
        return jnp.where(in_range, norm, 0.0)

    def cdf(self, q: Float[Array, "..."]) -> Float[Array, "..."]:
        """Uniform CDF: F(q) = (q - q_min) / (1 - q_min), clipped to [0, 1].

        Args:
            q: Mass ratio (any broadcastable shape).

        Returns:
            Cumulative probability in [0, 1], same shape as ``q``.
        """
        cdf_val = (q - self.q_min) / (1.0 - self.q_min)
        return jnp.clip(cdf_val, 0.0, 1.0)

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Inverse CDF (percent point function): q = q_min + u × (1 - q_min).

        Args:
            u: Quantiles in [0, 1] (any broadcastable shape).

        Returns:
            Mass ratios q in [q_min, 1], same shape as ``u``.
        """
        return self.q_min + u * (1.0 - self.q_min)

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample n mass ratios uniformly."""
        u = jax.random.uniform(key, (n,))
        return self.ppf(u)


class PowerLawMassRatio(eqx.Module):
    """Power-law mass-ratio distribution p(q) ∝ q^γ.

    Reference:
        Sana et al. (2012) Science 337, 444 - main text & Fig. 1 (κ ≈ -0.1; Science Report, no numbered eqs)
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
        """Power-law PDF: p(q) = q^γ / Z for q ∈ [q_min, 1], else 0.

        Z is the normalization over [q_min, 1] (:meth:`_norm`).

        Args:
            q: Mass ratio q = M_secondary / M_primary (any broadcastable shape).

        Returns:
            Probability density at q (dimensionless), same shape as ``q``; 0 outside
            [q_min, 1].
        """
        norm = self._norm()
        pdf_unnorm = q**self.gamma
        in_range = (q >= self.q_min) & (q <= 1.0)
        return jnp.where(in_range, pdf_unnorm / norm, 0.0)

    def cdf(self, q: Float[Array, "..."]) -> Float[Array, "..."]:
        """Power-law CDF F(q) = ∫_{q_min}^q t^γ dt / Z, clipped to [0, 1].

        The γ = -1 (logarithmic) case is handled with a divide-safe branch
        (:func:`jax.lax.cond`).

        Args:
            q: Mass ratio (scalar or array).

        Returns:
            Cumulative probability in [0, 1], same shape as ``q``.
        """
        g = self.gamma
        q0 = self.q_min
        norm = self._norm()

        def cdf_neq_m1(q_val):
            # ∫_{q_min}^q t^γ dt = (q^{γ+1} - q_min^{γ+1}) / (γ+1)
            # Guard the denominator: lax.cond traces this branch even at γ=-1, so 1/(γ+1)
            # must stay finite (the eq-branch result is selected there).
            g_plus_1 = g + 1.0
            g_plus_1_safe = jnp.where(jnp.abs(g_plus_1) < 1e-10, 1.0, g_plus_1)
            integral = (q_val**g_plus_1 - q0**g_plus_1) / g_plus_1_safe
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
        """Inverse CDF (percent point function) for the power-law q^γ.

        Analytically inverts :meth:`cdf`; the γ = -1 (logarithmic) case uses a
        divide-safe branch (:func:`jax.lax.cond`).

        Args:
            u: Quantiles in [0, 1] (scalar or array).

        Returns:
            Mass ratios q in [q_min, 1], same shape as ``u``.
        """
        g = self.gamma
        q0 = self.q_min
        norm = self._norm()

        def ppf_neq_m1(u_val):
            # F(q) = (q^{γ+1} - q_min^{γ+1}) / ((γ+1) × Z) = u
            # q^{γ+1} = u × (γ+1) × Z + q_min^{γ+1}
            # q = [u × (γ+1) × Z + q_min^{γ+1}]^{1/(γ+1)}
            # Guard the reciprocal exponent: lax.cond traces this branch even at γ=-1,
            # where 1/(γ+1) would be a (Python) division by zero; the eq-branch (log)
            # result is selected there, so this dead value only needs to stay finite.
            g_plus_1 = g + 1.0
            g_plus_1_safe = jnp.where(jnp.abs(g_plus_1) < 1e-10, 1.0, g_plus_1)
            inner = u_val * g_plus_1 * norm + q0**g_plus_1
            return inner ** (1.0 / g_plus_1_safe)

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

        Lucy (2006) A&A 457, 629 - systematic statistical study of the twin
        excess at q≈1 (the strong twin hypothesis itself traces to Lucy & Ricco 1979)

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

    def _gaussian_norm(self) -> Float[Array, ""]:
        """Normalization for truncated Gaussian at q=1."""
        # Gaussian centered at μ=1, truncated to [q_min, 1]
        # Full Gaussian integral from -∞ to ∞ is 1
        # Truncated integral: Φ((1-1)/σ) - Φ((q_min-1)/σ) = 0.5 - Φ((q_min-1)/σ)
        z_min = (self.q_min - 1.0) / self.sigma_twin
        cdf_min = 0.5 * (1.0 + jax.scipy.special.erf(z_min / jnp.sqrt(2.0)))
        return 0.5 - cdf_min  # Probability mass in [q_min, 1]

    def pdf(self, q: Float[Array, "..."]) -> Float[Array, "..."]:
        """Twin-peaked PDF: (1 - f_twin) power-law q^γ + f_twin truncated Gaussian at q=1.

        Both components are normalized over [q_min, 1] before mixing.

        Args:
            q: Mass ratio q = M_secondary / M_primary (any broadcastable shape).

        Returns:
            Probability density at q (dimensionless), same shape as ``q``; 0 outside
            [q_min, 1].
        """
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
        """Twin-peaked CDF: the f_twin mixture of the power-law and truncated-Gaussian
        CDFs, clipped to [0, 1].

        Args:
            q: Mass ratio (scalar or array).

        Returns:
            Cumulative probability in [0, 1], same shape as ``q``.
        """
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
        """Inverse CDF (percent point function) via fixed-count Newton iteration.

        The twin-peaked CDF has no closed-form inverse, so :meth:`cdf` is inverted by a
        fixed 20-step Newton iteration (:func:`jax.lax.fori_loop`, clamped to
        [q_min, 1]), keeping the map differentiable.

        Args:
            u: Quantiles in [0, 1] (scalar or array).

        Returns:
            Mass ratios q in [q_min, 1], same shape as ``u``.
        """

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

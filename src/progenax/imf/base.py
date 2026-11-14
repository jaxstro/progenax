"""
Base infrastructure for Initial Mass Functions (IMFs).

Provides the core protocol and abstract base class for all IMF
implementations, enabling differentiable sampling with multiple modes.

The Newton solver for inverse CDF uses fixed iterations with automatic
differentiation, enabling gradients w.r.t. both samples and IMF parameters.
"""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxstro.numerics.rootfinding import newton_ppf
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray

# ============================================================================
# IMFProtocol - Runtime-checkable Protocol for Type-Safe Composition
# ============================================================================


@runtime_checkable
class IMFProtocol(Protocol):
    """Interface for scalar stellar-mass distributions used by TruncatedIMF.

    Mass arguments, bounds and returned masses use solar masses. Probability
    normalization counts stars, rather than weighting them by mass.

    Attributes:
        m_min: Lower mass bound [M_sun].
        m_max: Upper mass bound [M_sun].

    Runtime protocol checks test attribute presence. Numerical accuracy, parameter
    domains and differentiation limits belong to each concrete implementation.
    """

    m_min: float
    m_max: float

    def logpdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Evaluate the natural log of the normalized number density in mass.

        Args:
            m: Masses [M_sun], scalar or array, within [m_min, m_max].

        Returns:
            Log-density values with the shape of ``m``. Exponentiating gives a
            probability density per solar mass, normalized on [m_min, m_max].

        Notes:
            Out-of-range masking depends on the implementation. TruncatedIMF
            explicitly returns negative infinity outside its mass bounds.
        """
        ...

    def cdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Evaluate the normalized cumulative number fraction at mass m.

        Args:
            m: Masses [M_sun], scalar or array, within [m_min, m_max].

        Returns:
            Dimensionless cumulative probabilities with the shape of ``m``.

        Notes:
            Out-of-range handling depends on the implementation. TruncatedIMF
            returns zero below its lower bound and one above its upper bound.
        """
        ...

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Map cumulative number fractions to masses using the inverse CDF.

        Args:
            u: Dimensionless quantiles in [0, 1], scalar or array.

        Returns:
            Masses [M_sun] with the shape of ``u``.

        Notes:
            Consult the concrete implementation for inversion precision and
            differentiation limits. Endpoint handling can include clipping;
            supplying zero or one need not return the exact mass bound.
        """
        ...

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Draw n masses from the normalized number distribution.

        Args:
            key: JAX random key. Reusing a key with the same implementation,
                parameters and count repeats the draw.
            n: Nonnegative sample count; static under JIT because it sets the
                output shape.

        Returns:
            Mass array of shape (n,) [M_sun]. The realized total mass varies.

        Notes:
            The sampling algorithm and its pathwise differentiation limits
            belong to the concrete implementation.
        """
        ...

    def mean_mass(self) -> Float[Array, ""]:
        """Return the probability-weighted expected stellar mass.

        Returns:
            Scalar expected mass [M_sun], integrating mass times the normalized
            number density over [m_min, m_max].

        Notes:
            Analytic and numerical implementations have their own precision
            and parameter-domain limits; the expectation requires a finite
            first moment.
        """
        ...


# ============================================================================
# PPF Newton Solver
# ============================================================================


def _ppf_newton(imf: "BaseIMF", u: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Inverse CDF via fixed Newton iteration.

    Thin wrapper over jaxstro's generic ``newton_ppf`` solver: supplies the
    IMF's CDF, a log-space initial guess, and the [m_min, m_max] bounds.
    The Newton loop (fixed iterations, safe division, clip-to-bounds) lives
    in jaxstro.numerics.rootfinding.newton_ppf — JIT-safe, no convergence
    loops. Initial guess uses linear interpolation in log-mass space.

    Gradients flow through all iterations via automatic differentiation,
    enabling differentiation w.r.t. both u and IMF parameters.

    Args:
        imf: IMF instance
        u: Uniform samples in [0, 1]

    Returns:
        Mass values m such that CDF(m) ≈ u
    """
    # Initial guess: linear interpolation in log-mass space
    log_m_min = jnp.log(jnp.maximum(imf.m_min, 1e-10))
    log_m_max = jnp.log(imf.m_max)
    log_m0 = log_m_min + u * (log_m_max - log_m_min)
    m0 = jnp.exp(log_m0)

    # pdf = exp(logpdf); explicit form matches the original (avoids AD of cdf).
    return newton_ppf(
        u,
        imf.cdf,
        x0=m0,
        lo=imf.m_min,
        hi=imf.m_max,
        pdf=lambda m: jnp.exp(imf.logpdf(m)),
        n_iter=20,
        pdf_floor=1e-30,
    )


# ============================================================================
# BaseIMF Abstract Class
# ============================================================================


class BaseIMF(eqx.Module):
    """
    Abstract base for all IMFs with shared PPF solver and sampling modes.

    Provides automatic normalization, Newton PPF solver with custom gradients,
    and four sampling modes (N, M_total, M_total_packed, fixed-N).

    Subclasses must implement:
        - _logpdf_unnorm(m): Unnormalized log-PDF (shape function)
        - _cdf_unnorm(m): Unnormalized CDF (integral of unnormalized PDF)

    The base class handles normalization automatically via _log_norm property.

    Attributes:
        m_min: Minimum mass [M_sun] (default: 0.0)
        m_max: Maximum mass [M_sun] (default: inf)
    """

    m_min: float = 0.0
    m_max: float = jnp.inf

    @abstractmethod
    def _logpdf_unnorm(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Unnormalized log-PDF (shape function). Override in subclass."""
        ...

    @abstractmethod
    def _cdf_unnorm(self, m: ArrayLike) -> Float[Array, "..."]:
        """Unnormalized CDF. Override in subclass."""
        ...

    @property
    def _log_norm(self) -> Float[Array, ""]:
        """Log normalization constant over [m_min, m_max]."""
        F_min = self._cdf_unnorm(self.m_min)
        F_max = self._cdf_unnorm(self.m_max)
        return jnp.log(F_max - F_min + 1e-30)

    def logpdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Normalized log-PDF, with negative infinity outside the mass support.

        Support endpoints can be nonsmooth. Outside values are evaluated at a
        valid interior mass before masking, keeping inactive logarithms finite.
        """
        m = jnp.asarray(m)
        in_support = (m >= self.m_min) & (m <= self.m_max)
        interior = self.m_min + jnp.where(
            jnp.isfinite(self.m_max), 0.5 * (self.m_max - self.m_min), 1.0
        )
        safe_m = jnp.where(in_support, m, interior)
        value = self._logpdf_unnorm(safe_m) - self._log_norm
        masked = jnp.where(in_support, value, -jnp.inf)
        return jnp.where(jnp.isnan(m) | jnp.isnan(value), jnp.nan, masked)

    def cdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Normalized CDF, zero below the support and one above it."""
        m = jnp.asarray(m)
        F_min = self._cdf_unnorm(self.m_min)
        F_max = self._cdf_unnorm(self.m_max)
        bounded_m = jnp.clip(m, self.m_min, self.m_max)
        value = (self._cdf_unnorm(bounded_m) - F_min) / (F_max - F_min + 1e-30)
        masked = jnp.where(m <= self.m_min, 0.0, jnp.where(m >= self.m_max, 1.0, value))
        return jnp.where(jnp.isnan(value), jnp.nan, masked)

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Inverse CDF via Newton solver."""
        return _ppf_newton(self, u)

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample n masses via reparameterization trick."""
        u = jax.random.uniform(key, (n,))
        return self.ppf(u)

    def mean_mass(self) -> Float[Array, ""]:
        """Expected mass E[m] via a LOG-spaced trapezoid (override for analytic forms).

        Log-spacing concentrates nodes on the steep low-mass region; a linear grid of
        the same size badly under-resolves an m^-alpha spike near m_min (e.g. Schechter
        with m_min=0.01 was ~5x off on a linear grid). 4000 log-nodes converge to <<1%.
        """
        m_grid = jnp.exp(jnp.linspace(jnp.log(self.m_min), jnp.log(self.m_max), 4000))
        m_grid = m_grid.at[0].set(self.m_min).at[-1].set(self.m_max)
        pdf_grid = jnp.exp(self.logpdf(m_grid))
        return jnp.trapezoid(m_grid * pdf_grid, m_grid)

    # ========================================================================
    # Sampling Modes
    # ========================================================================

    def sample_n(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """N mode: exactly n masses, M_total varies."""
        return self.sample(key, n)

    def sample_m_total(
        self,
        key: PRNGKeyArray,
        m_total: float,
        n_max: int | None = None,
    ) -> tuple[Float[Array, "n_max"], Float[Array, ""]]:
        """
        M_total mode (simple): hard cutoff, NOT differentiable w.r.t. m_total.

        Samples masses until cumulative sum exceeds m_total, then pads with zeros.

        Args:
            key: JAX PRNG key
            m_total: Target total mass
            n_max: Maximum number of particles (default: auto-estimate from mean mass)

        Returns:
            masses_padded: Padded mass array of length n_max
            n_live: Number of live particles (JAX scalar array, not Python int)
        """
        if n_max is None:
            n_max = int(m_total / self.mean_mass() * 2.0) + 100

        u = jax.random.uniform(key, (n_max,))
        masses_all = self.ppf(u)
        cumsum = jnp.cumsum(masses_all)

        n_live = jnp.searchsorted(cumsum, m_total) + 1
        n_live = jnp.minimum(n_live, n_max)
        mask = jnp.arange(n_max) < n_live
        masses_padded = jnp.where(mask, masses_all, 0.0)

        return masses_padded, n_live

    def sample_m_total_packed(
        self,
        key: PRNGKeyArray,
        m_total: float,
        n_max: int,
    ) -> tuple[Float[Array, "n_max"], Float[Array, ""]]:
        """
        M_total mode (packed): differentiable via fractional boundary mass.

        Uses fractional mass for boundary particle to hit m_total exactly.
        """
        u = jax.random.uniform(key, (n_max,))
        masses_all = self.ppf(u)
        cumsum = jnp.cumsum(masses_all)

        over_threshold = cumsum >= m_total
        k = jnp.argmax(over_threshold)
        k = jnp.where(jnp.any(over_threshold), k, n_max - 1)

        cumsum_prev = jnp.where(k > 0, cumsum[k - 1], 0.0)
        deficit = m_total - cumsum_prev
        fractional_mass = jnp.clip(deficit, 0.0, masses_all[k])

        indices = jnp.arange(n_max)
        masses_packed = jnp.where(
            indices < k,
            masses_all,
            jnp.where(indices == k, fractional_mass, 0.0),
        )

        n_live_float = k + fractional_mass / (masses_all[k] + 1e-30)

        return masses_packed, n_live_float

    def sample_fixed_n(
        self,
        key: PRNGKeyArray,
        n: int,
        m_total: float,
        max_steps: int = 50,
    ) -> Float[Array, "n"]:
        """
        Fixed-N mode: n masses whose total APPROXIMATES m_total via quantile
        stretching (the realized total carries the residual of the final random
        jitter — it is not exact).

        Uses stratified quantile sampling with a stretch factor q* (<= 1) solved
        to hit the target total mass. Because the stretch is one-sided (q* <= 1)
        AND stratification truncates the heavy tail, the achievable total is
        capped at sum(ppf((i+0.5)/n)) — a ceiling that sits a few percent BELOW
        n*E[m]. A target ABOVE that ceiling is UNREACHABLE: eager (concrete)
        inputs now raise ValueError instead of silently undershooting (audit R5:
        target 500 Msun, n=1000 used to return ~349 Msun with no warning). Traced
        m_total cannot be checked under jit/grad — the caller owns reachability.
        Differentiable w.r.t. m_total.
        """
        u_base = (jnp.arange(n) + 0.5) / n

        # Reachability guard (eager inputs only). The one-sided stretch caps the
        # total at sum(ppf(u_base)); a target above it silently undershot before.
        try:
            m_ceiling = float(jnp.sum(self.ppf(u_base)))
            if float(m_total) > m_ceiling:
                raise ValueError(
                    f"m_total={float(m_total):.6g} Msun is unreachable for n={n}: "
                    f"the stratified-quantile ceiling is {m_ceiling:.6g} Msun. "
                    f"Increase n, lower m_total, or use sample_m_total()."
                )
        except (
            jax.errors.ConcretizationTypeError,
            jax.errors.TracerArrayConversionError,
        ):
            pass  # traced m_total / params: caller owns reachability

        q_star = self._solve_q_for_m_total(u_base, m_total, max_steps)

        u_random = jax.random.uniform(key, (n,))
        u_stretched = u_base * q_star + u_random * (1 - q_star) / n
        u_stretched = jnp.clip(u_stretched, 0.0, 1.0)

        return self.ppf(u_stretched)

    def _solve_q_for_m_total(
        self,
        u_base: Float[Array, "n"],
        m_total: float,
        max_steps: int = 50,
    ) -> Float[Array, ""]:
        """Newton solver for quantile stretch factor."""
        m_total_arr = jnp.asarray(m_total)

        def total_mass_from_q(q):
            return jnp.sum(self.ppf(u_base * q))

        M_max = total_mass_from_q(1.0)
        M_min = total_mass_from_q(0.1)
        q0 = jnp.clip(
            0.1 + 0.9 * (m_total_arr - M_min) / (M_max - M_min + 1e-12),
            0.05,
            0.99,
        )

        def newton_step(_, q):
            M_q = total_mass_from_q(q)
            residual = M_q - m_total_arr
            eps = 1e-6
            dMdq = (total_mass_from_q(q + eps) - M_q) / eps + 1e-12
            q_new = jnp.clip(q - 0.8 * residual / dMdq, 1e-6, 1.0 - 1e-6)
            return q_new

        return jax.lax.fori_loop(0, max_steps, newton_step, q0)


__all__ = ["IMFProtocol", "BaseIMF", "_ppf_newton"]

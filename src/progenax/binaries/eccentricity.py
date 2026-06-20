"""Binary orbital-eccentricity distributions.

Eccentricity samplers for binary populations:

- :class:`ThermalEccentricity` — f(e) = 2e (Ambartsumian energy-only / Heggie 1975).
- :class:`UniformEccentricity` — uniform on [e_min, e_max].
- :class:`MoeEccentricity`     — smooth period-dependent circular→thermal heuristic.

References:
    Ambartsumian (1937); Jeans (1919); Heggie (1975) MNRAS 173, 729 — thermal f(e)=2e.
    Duquennoy & Mayor (1991) A&A 248, 485 §6.1/§7.2 — P<10d circular, P>1000d thermal,
        circularization period P_circ ≈ 11.6 d.
    Moe & Di Stefano (2017) ApJS 230, 15 — period-dependent eccentricity statistics.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class ThermalEccentricity(eqx.Module):
    """Thermal eccentricity distribution f(e) = 2e.

    Arises from energy equipartition in dynamically relaxed systems.
    CDF: F(e) = e²  =>  PPF: e = √u

    Reference:
        Heggie (1975) MNRAS 173, 729
        Jeans (1919) "Problems of Cosmogony and Stellar Dynamics"

    Parameters:
        e_max: Maximum eccentricity (default: 0.99, avoids singularity)
    """

    e_max: float = 0.99

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample n eccentricities from f(e) = 2e."""
        u = jax.random.uniform(key, (n,))
        return self.e_max * jnp.sqrt(u)

    def pdf(self, e: Float[Array, "..."]) -> Float[Array, "..."]:
        """PDF: p(e) = 2e / e_max²."""
        in_range = (e >= 0.0) & (e <= self.e_max)
        return jnp.where(in_range, 2.0 * e / self.e_max**2, 0.0)

    def cdf(self, e: Float[Array, "..."]) -> Float[Array, "..."]:
        """CDF: F(e) = e² / e_max²."""
        cdf_val = (e / self.e_max) ** 2
        return jnp.clip(cdf_val, 0.0, 1.0)

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Inverse CDF: e = e_max × √u (differentiable on (0,1)).

        u is clamped to (1e-12, 1) so the gradient stays finite at u=0, where
        d/du √u = 1/(2√u) -> ∞.
        """
        u_safe = jnp.clip(u, 1e-12, 1.0)
        return self.e_max * jnp.sqrt(u_safe)


class UniformEccentricity(eqx.Module):
    """Uniform eccentricity distribution.

    Simple uniform distribution, useful for circular-dominated populations.

    Parameters:
        e_min: Minimum eccentricity (default: 0.0)
        e_max: Maximum eccentricity (default: 0.9)
    """

    e_min: float = 0.0
    e_max: float = 0.9

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample n eccentricities uniformly."""
        u = jax.random.uniform(key, (n,))
        return self.e_min + u * (self.e_max - self.e_min)

    def pdf(self, e: Float[Array, "..."]) -> Float[Array, "..."]:
        """Uniform PDF."""
        in_range = (e >= self.e_min) & (e <= self.e_max)
        return jnp.where(in_range, 1.0 / (self.e_max - self.e_min), 0.0)

    def cdf(self, e: Float[Array, "..."]) -> Float[Array, "..."]:
        """Uniform CDF."""
        cdf_val = (e - self.e_min) / (self.e_max - self.e_min)
        return jnp.clip(cdf_val, 0.0, 1.0)

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Inverse CDF."""
        return self.e_min + u * (self.e_max - self.e_min)


class LogisticThermalEccentricity(eqx.Module):
    """Smooth circular->thermal eccentricity heuristic (period-conditional).

    A differentiable heuristic that interpolates a thermal f(e)=2e distribution
    from near-circular at short P to fully thermal at long P, via a logistic
    blend in log10(P):  e = blend(P) * e_max * sqrt(u),  u ~ U(0,1).

    This captures the QUALITATIVE three-period structure of solar-type
    eccentricities measured by Duquennoy & Mayor (1991) (§6.1/§7.2): P < ~10 d
    tidally circularized (e ~ 0; their circularization period P_circ ~ 11.6 d),
    P > ~1000 d approaching thermal f(e)=2e (Ambartsumian 1937), with a smooth
    transition between. The logistic midpoint defaults to the geometric mean of
    P_circ=10 d and P_thermal=1000 d (log10 P = 2.0).

    It is NOT Moe & Di Stefano's (2017) f(e) ∝ e^η(P, M1) law — for that, use
    :class:`MoeEccentricity`. This class is a smooth, mass-independent surrogate
    motivated by the tidal-circularization physics (Zahn 1977).

    References:
        Duquennoy & Mayor (1991) A&A 248, 485 §6.1/§7.2 - three-period e model.
        Ambartsumian (1937); Heggie (1975) MNRAS 173, 729 - thermal f(e)=2e.
        Zahn (1977) A&A 57, 383 - tidal circularization.

    Parameters:
        P_circ: Circularization period [days] (default: 10.0; DM91 ~11.6 d)
        P_thermal: Thermalization period [days] (default: 1000.0; DM91 wide onset)
        e_max: Maximum eccentricity (default: 0.99)
        transition_width: Width of transition region in log10(P) (default: 0.5)
    """

    P_circ: float = 10.0
    P_thermal: float = 1000.0
    e_max: float = 0.99
    transition_width: float = 0.5

    def _blend(self, periods: Float[Array, "N"]) -> Float[Array, "N"]:
        """Logistic blend factor in [0, 1]: ~0 (circular) -> ~1 (thermal)."""
        log_P = jnp.log10(periods)
        log_P_mid = jnp.log10(jnp.sqrt(self.P_circ * self.P_thermal))
        return 1.0 / (1.0 + jnp.exp(-(log_P - log_P_mid) / self.transition_width))

    def sample(
        self,
        key: PRNGKeyArray,
        periods: Float[Array, "N"],
        masses: Float[Array, "N"] | None = None,
    ) -> Float[Array, "N"]:
        """Sample eccentricities given periods (mass-independent; masses ignored).

        Args:
            key: JAX random key
            periods: Orbital periods [days] (shape N,)
            masses: Ignored; accepted so this is drop-in interchangeable with the
                mass-conditional :class:`MoeEccentricity` in mass-dependent configs.

        Returns:
            Eccentricities (shape N,), period-dependent.
        """
        u = jax.random.uniform(key, periods.shape)
        e_thermal = self.e_max * jnp.sqrt(u)
        return self._blend(periods) * e_thermal

    def pdf(
        self,
        e: Float[Array, "..."],
        periods: Float[Array, "N"],
        masses: Float[Array, "N"] | None = None,
    ) -> Float[Array, "..."]:
        """Conditional pdf p(e | P): thermal scaled to [0, blend*e_max].

        With e = c*sqrt(u), c = blend(P)*e_max, the conditional density is
        p(e|P) = 2e/c^2 for 0 <= e <= c, else 0.
        """
        c = self._blend(periods) * self.e_max
        in_range = (e >= 0.0) & (e <= c)
        return jnp.where(in_range, 2.0 * e / c**2, 0.0)


class MoeEccentricity(eqx.Module):
    """Moe & Di Stefano (2017) eccentricity distribution p(e) ∝ e^η(logP, M1).

    Faithful implementation of Moe & Di Stefano (2017) §9.2 (Figure 36): the
    eccentricity follows a power law p(e) ∝ e^η on 0 <= e <= e_max, with the
    slope η a function of orbital period AND primary mass:

        Eq. 17 (late-type, 0.8 < M1 < 3 Msun):  η = 0.6 - 0.7 / (logP - 0.5)
        Eq. 18 (early-type, M1 > 7 Msun):       η = 0.9 - 0.2 / (logP - 0.5)

    with a linear interpolation in M1 for 3 <= M1 <= 7 Msun. η = 0 is uniform
    (<e> = 0.5); η = 1 is thermal f(e) = 2e (<e> = 2/3). Eqs. 17-18 themselves
    drive short-period (logP ≲ 1) massive binaries to η < 0 (tidal
    circularization); solar-type binaries asymptote to η ≈ 0.5 at long P;
    intermediate-period massive binaries reach η ≈ 0.8 (near-thermal). For short periods where η <= -1 (e^η non-normalizable),
    the orbit is tidally circularized and this returns e ≈ 0 (Moe notes η is "not
    well defined" for logP ≲ 1).

    Sampling uses the inverse CDF e = e_max(P) * u^(1/(η+1)); as η -> -1+ the
    exponent diverges so e -> 0 (circular) by construction.

    The upper limit is the period-dependent Roche-lobe ceiling (their Eq. 3),

        e_max(P) = 1 - (P / 2 days)^(-2/3)   for P > 2 d,

    clipped to [0, e_max], so the components do not overflow their Roche lobes at
    periapsis (e.g. e_max(10 d) ≈ 0.66, e_max(100 d) ≈ 0.93); P <= 2 d -> circular.
    The ``e_max`` field is the long-period numerical ceiling (avoids the e -> 1
    singularity), reached only where the Roche relation itself approaches 1.

    Reference:
        Moe & Di Stefano (2017) ApJS 230, 15, §9.2 Eqs. 17-18, Eq. 3, Fig. 36.
        Sana et al. (2012) Science 337, 444 - short-period O-star binaries are
            eccentricity-poor; the precise slope is in their supplementary Table S3
            (not reproduced in the held main report).

    Parameters:
        e_max: Numerical eccentricity ceiling at long P (default: 0.99); the
            physical cap is the period-dependent Roche relation (Eq. 3).
    """

    e_max: float = 0.99

    def e_max_of_period(
        self,
        periods: Float[Array, "N"],
    ) -> Float[Array, "N"]:
        """Roche-lobe eccentricity ceiling e_max(P) (Moe & Di Stefano 2017 Eq. 3).

        e_max(P) = 1 - (P / 2 days)^(-2/3) for P > 2 d, clipped to [0, e_max]. The
        ``jnp.maximum(periods, 2.0)`` keeps the fractional power away from P=0 so
        the gradient stays finite (below 2 d the cap is 0 regardless).

        Args:
            periods: Orbital periods [days] (shape N,).

        Returns:
            Period-dependent maximum eccentricity (shape N,), in [0, e_max].
        """
        P_safe = jnp.maximum(periods, 2.0)
        roche = 1.0 - (P_safe / 2.0) ** (-2.0 / 3.0)
        return jnp.clip(roche, 0.0, self.e_max)

    def eta(
        self,
        periods: Float[Array, "N"],
        masses: Float[Array, "N"],
    ) -> Float[Array, "N"]:
        """Power-law slope η(logP, M1) (Moe & Di Stefano 2017 Eqs. 17-18)."""
        log_P = jnp.log10(periods)
        # logP - 0.5 -> 0 at logP = 0.5 (P ~ 3.16 d); below that the orbit is
        # tidally circularized. Floor the denominator so η -> very negative
        # (=> circular) there rather than blowing up with the wrong sign.
        denom = jnp.maximum(log_P - 0.5, 1e-3)
        eta_late = 0.6 - 0.7 / denom  # Eq. 17 (0.8 < M1 < 3 Msun)
        eta_early = 0.9 - 0.2 / denom  # Eq. 18 (M1 > 7 Msun)
        # Linear interpolation in M1 across 3-7 Msun (w=0 late, w=1 early).
        w = jnp.clip((masses - 3.0) / (7.0 - 3.0), 0.0, 1.0)
        return (1.0 - w) * eta_late + w * eta_early

    def sample(
        self,
        key: PRNGKeyArray,
        periods: Float[Array, "N"],
        masses: Float[Array, "N"],
    ) -> Float[Array, "N"]:
        """Sample e ∝ e^η(logP, M1) on [0, e_max] via inverse-CDF.

        Args:
            key: JAX random key
            periods: Orbital periods [days] (shape N,)
            masses: Primary masses [Msun] (shape N,) — sets η via Eqs. 17-18.

        Returns:
            Eccentricities (shape N,). η <= -1 (very short P) -> circular (e=0).
        """
        u = jax.random.uniform(key, periods.shape)
        eta = self.eta(periods, masses)
        # F(e) = (e/e_max(P))^(η+1) -> e = e_max(P) u^(1/(η+1)) for η > -1;
        # double-where guards η <= -1 (degenerate -> circular) so neither branch
        # produces NaN. e_max(P) is the Roche-lobe ceiling (Eq. 3).
        etap1 = eta + 1.0
        is_circular = etap1 <= 1e-6
        etap1_safe = jnp.where(is_circular, 1.0, etap1)
        e_max_eff = self.e_max_of_period(periods)
        e = e_max_eff * u ** (1.0 / etap1_safe)
        return jnp.where(is_circular, 0.0, e)

    def pdf(
        self,
        e: Float[Array, "..."],
        periods: Float[Array, "N"],
        masses: Float[Array, "N"],
    ) -> Float[Array, "..."]:
        """Conditional pdf p(e | logP, M1) = (η+1) e^η / e_max(P)^(η+1) on [0, e_max(P)].

        Support is the period-dependent Roche ceiling e_max(P) (Eq. 3). Broadcasts e
        against (periods, masses): result[..., i, j] is p(e_j | P_i, M_i).
        """
        eta = self.eta(periods, masses)
        etap1 = eta + 1.0
        etap1_safe = jnp.where(etap1 <= 1e-6, 1.0, etap1)
        e_max_eff = self.e_max_of_period(periods)[..., None]
        in_range = (e >= 0.0) & (e <= e_max_eff)
        e_pos = jnp.where(in_range, e, e_max_eff)
        p = (
            etap1_safe[..., None]
            * jnp.power(e_pos, eta[..., None])
            / e_max_eff ** etap1_safe[..., None]
        )
        p = jnp.where(in_range, p, 0.0)
        # Circular (η <= -1): mass at e=0, continuous density undefined -> 0.
        return jnp.where((etap1 <= 1e-6)[..., None], 0.0, p)


__all__ = [
    "ThermalEccentricity",
    "UniformEccentricity",
    "LogisticThermalEccentricity",
    "MoeEccentricity",
]

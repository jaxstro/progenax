"""Companion/orbit layer for `build_binary_cluster` (Batch 4k).

A :class:`CompanionModel` is the single owner of the binary statistics: given
primary masses it decides multiplicity (f_b -> is_binary) and samples the
companion properties (q -> m2, P -> a, e, isotropic orientation), all keyed on the
primary masses. Folding multiplicity in here (rather than a separate
``binary_fraction`` arg) keeps the model internally consistent — for Moe+2017,
f_b(m1) is *part of* the model, set by the very IMF masses we pull.

Two implementations:

- :class:`IndependentCompanions` — versatile marginals (any f_b x q x P x e);
  reproduces the period-averaged default used today (and by the *Confidently Wrong*
  mass-function model).
- :class:`MoeCompanions` — faithful Moe & Di Stefano (2017): Moe's own mass-dependent
  f_b plus the joint (P, q, e) interrelation; the *same* q sets m2, so the P-q
  coupling ("Mind your Ps and Qs") shows up self-consistently in the secondaries.

Both reuse `period_to_semimajor_axis` and `sample_isotropic_orientations`.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from .kepler_period import period_to_semimajor_axis
from .orientation import sample_isotropic_orientations


class CompanionElements(NamedTuple):
    """Per-system companion mass + relative-orbit elements (singles: m2=0, a sanitized).

    Plugs directly into `resolve_binary_components(..., a, e, inc, Omega, omega, M_anom)`.
    """

    m2: Float[Array, "N"]
    a: Float[Array, "N"]
    e: Float[Array, "N"]
    inc: Float[Array, "N"]
    Omega: Float[Array, "N"]
    omega: Float[Array, "N"]
    M_anom: Float[Array, "N"]


def _eval_binary_fraction(bf: Any, m1: Float[Array, "N"]) -> Float[Array, "N"]:
    """Binary fraction f_b(m1) — a callable model, or a scalar constant fraction.

    Dispatch on ``callable`` (not ``isinstance(float)``) so a jnp/traced scalar
    ``f_b`` broadcasts instead of crashing with a cryptic "object is not callable"
    (audit D1). NOTE on gradients: a scalar ``f_b`` feeds only the DISCRETE
    multiplicity draw ``is_binary = u < f_b`` downstream, whose pathwise gradient
    w.r.t. ``f_b`` is exactly zero — inferring ``f_b`` requires a likelihood, not
    a gradient through the sampler.
    """
    if callable(bf):
        return bf(m1)
    return jnp.broadcast_to(jnp.asarray(bf, dtype=m1.dtype), m1.shape)


def _sample_mass_ratio(
    q_dist: Any, key: PRNGKeyArray, m1: Float[Array, "N"]
) -> Float[Array, "N"]:
    """Sample q given primaries (duck-typed dispatch, mirrors BinaryIMF.sample_mass_ratios).

    - mass-dependent (`sample_given_primary`, e.g. MoeDiStefano2017) -> q | m1
    - unconditional (`sample(key, n)`, e.g. FlatMassRatio/PowerLaw/Twin) -> q
    - custom callable f(key, m1) -> q
    """
    n = m1.shape[0]
    if hasattr(q_dist, "sample_given_primary"):
        return q_dist.sample_given_primary(key, m1)
    if hasattr(q_dist, "sample"):
        return q_dist.sample(key, n)
    if callable(q_dist):
        return q_dist(key, m1)
    raise TypeError(f"Unsupported mass-ratio distribution: {q_dist!r}")


class IndependentCompanions(eqx.Module):
    """Versatile companion model: independent f_b x q x P x e marginals.

    Multiplicity from `binary_fraction(m1)`; q from `q_distribution` (mass-dependent
    or unconditional); period and eccentricity sampled independently. Reproduces the
    period-averaged default of `build_binary_cluster` today. The mass-ratio is the
    single owner of q -> `m2 = m1 * q` (0 for singles).

    Entropy layout (the equivalence contract): ``split(key, 5)`` ->
    ``[is_binary, q, period, eccentricity, orientation]``.
    """

    binary_fraction: Any
    q_distribution: Any
    period_distribution: Any
    eccentricity_distribution: Any

    def sample(
        self,
        key: PRNGKeyArray,
        m1: Float[Array, "N"],
        *,
        G: float,
        day_in_time_units: float,
    ) -> Tuple[Bool[Array, "N"], CompanionElements]:
        """Draw (is_binary, companion elements) for each primary.

        Differentiability: the smooth channels (q, P, e) carry pathwise gradients
        at a fixed key, but multiplicity is a DISCRETE draw
        (``is_binary = u < f_b``) — its gradient w.r.t. the binary fraction is
        exactly zero. Infer ``f_b`` via a likelihood, not a gradient through this
        sampler (see ``DifferentiableBinaryModel`` for a smooth relaxation).
        """
        n = m1.shape[0]
        kb, kq, kP, ke, ko = jax.random.split(key, 5)

        f_bin = _eval_binary_fraction(self.binary_fraction, m1)
        is_binary = jax.random.uniform(kb, (n,)) < f_bin

        q = _sample_mass_ratio(self.q_distribution, kq, m1)
        m2 = jnp.where(is_binary, m1 * q, 0.0)

        P_days = self.period_distribution.sample(kP, n)
        a = period_to_semimajor_axis(P_days * day_in_time_units, m1 + m2, G)
        e = self.eccentricity_distribution.sample(ke, n)
        inc, Omega, omega, M_anom = sample_isotropic_orientations(ko, n)
        return is_binary, CompanionElements(m2, a, e, inc, Omega, omega, M_anom)


class MoeCompanions(eqx.Module):
    """Faithful Moe & Di Stefano (2017) companion model — the P-q-e interrelation.

    Multiplicity from Moe's own mass-dependent ``MassDependentBinaryFraction`` (no
    f_b supplied — it is set by the IMF masses); orbital parameters from the joint
    ``MoeJointOrbit`` (`logP ~ MoePeriod(M1)`, `q ~ MoeDiStefano2017Full(M1,P)`,
    `e ~ MoeEccentricity(P,M1)`). The same q sets ``m2 = m1 * q`` (self-consistent),
    so short-period binaries carry larger q than long-period ones.

    Reference: Moe & Di Stefano (2017) ApJS 230, 15 (full joint distribution).
    """

    q_min: float = 0.1

    def _joint(self):
        # Lazy imf import (imf -> binaries is lazy elsewhere; keep this import-cycle-safe).
        from ..imf.binary import MoeDiStefano2017Full, MoeJointOrbit, MoePeriod
        from .eccentricity import MoeEccentricity

        return MoeJointOrbit(
            period=MoePeriod(),
            massratio=MoeDiStefano2017Full(q_min=self.q_min),
            eccentricity=MoeEccentricity(),
        )

    def sample(
        self,
        key: PRNGKeyArray,
        m1: Float[Array, "N"],
        *,
        G: float,
        day_in_time_units: float,
    ) -> Tuple[Bool[Array, "N"], CompanionElements]:
        """Draw (is_binary, correlated Moe companion elements) per primary.

        Differentiability: the correlated (P, q, e) channels carry pathwise
        gradients at a fixed key, but multiplicity is a DISCRETE draw
        (``is_binary = u < f_b``) whose gradient w.r.t. the binary fraction is
        exactly zero — infer ``f_b`` via a likelihood, not through this sampler.
        """
        from ..imf.binary import MassDependentBinaryFraction

        n = m1.shape[0]
        kb, kj, ko = jax.random.split(key, 3)

        f_bin = MassDependentBinaryFraction()(m1)
        is_binary = jax.random.uniform(kb, (n,)) < f_bin

        P_days, q, e = self._joint().sample(kj, m1)
        m2 = jnp.where(is_binary, m1 * q, 0.0)
        a = period_to_semimajor_axis(P_days * day_in_time_units, m1 + m2, G)
        inc, Omega, omega, M_anom = sample_isotropic_orientations(ko, n)
        return is_binary, CompanionElements(m2, a, e, inc, Omega, omega, M_anom)


__all__ = ["CompanionElements", "IndependentCompanions", "MoeCompanions"]

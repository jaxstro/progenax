"""Soft-sigmoid dense-tail mask and mass-weighted tail fraction (spec §3.6).

Membership in the gravitationally-collapsing dense tail is a differentiable sigmoid
of the log-density relative to the transition, w = σ(κ(s − s_t)). The mass-weighted
fraction f_tail_actual = Σ w ρ / Σ ρ is the realized analogue of BM19's f_dense; in
the sharp limit κ→∞ it equals the mass fraction above s_t. Both are differentiable
in (s_t, κ), the hooks the calibration tunes.

JAX-native.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def tail_weights(
    s: Float[Array, "..."],
    s_t: Float[Array, ""],
    kappa: Float[Array, ""],
) -> Float[Array, "..."]:
    r"""Soft tail membership w = σ(κ(s − s_t)) ∈ (0,1).

    w(s_t) = 0.5; monotone increasing in s; κ→∞ recovers the hard indicator s > s_t.
    """
    return jax.nn.sigmoid(kappa * (s - s_t))


def f_tail_actual(
    s: Float[Array, "..."],
    rho: Float[Array, "..."],
    s_t: Float[Array, ""],
    kappa: Float[Array, ""],
) -> Float[Array, ""]:
    r"""Mass-weighted dense-tail fraction f_tail_actual = Σ w ρ / Σ ρ.

    The realized counterpart of BM19 f_dense (AC6 compares the two). Differentiable
    in (s_t, κ); raising s_t removes mass from the tail (∂/∂s_t < 0).
    """
    w = tail_weights(s, s_t, kappa)
    return jnp.sum(w * rho) / jnp.sum(rho)

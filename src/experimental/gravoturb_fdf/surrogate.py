"""Smooth differentiable Q(f_sub; σ_s, β) surrogate (spec §8 / P3.3).

The CW04 Q estimator and the categorical star sampling that precede it are both
non-differentiable. The differentiable interface to the f_sub→Q calibration is therefore
this smooth surrogate: a linear-in-features model fit (numpy, validation-side) to the
measured Q(f_sub; σ_s, β) grid and evaluated here in JAX so ``jax.grad`` flows.

Feature vector (order is the contract shared with fit_q_surrogate):
    [1, f, f², σ_s, β, f·σ_s, f·β]

``PERSISTED_COEFFS`` are fit from the production calibration grid (P3.2) and pinned here
as the small persisted artifact (no file I/O); regenerate with
``gravoturb_fdf.validation.calibration.fit_q_surrogate`` on a fresh grid.

JAX-native evaluation.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

N_FEATURES = 7


def surrogate_features(f_sub, sigma_s, beta, xp):
    r"""Feature vector for the surrogate (xp = jnp for eval / numpy for fitting).

    Returns shape ``(..., 7)`` stacking [1, f, f², σ_s, β, f·σ_s, f·β] along the last
    axis (the fit and the evaluation MUST use this same order).
    """
    f = xp.asarray(f_sub)
    s = xp.asarray(sigma_s)
    bt = xp.asarray(beta)
    ones = xp.ones_like(f * 1.0)
    return xp.stack([ones, f, f**2, s, bt, f * s, f * bt], axis=-1)


def q_surrogate(
    f_sub: Float[Array, "..."],
    sigma_s: Float[Array, "..."],
    beta: Float[Array, "..."],
    coeffs: Float[Array, " 7"],
) -> Float[Array, "..."]:
    r"""Differentiable Q ≈ features(f_sub, σ_s, β) · coeffs.

    Differentiable in all of (f_sub, σ_s, β); a well-fit surrogate has ∂Q/∂f_sub < 0.
    """
    feats = surrogate_features(f_sub, sigma_s, beta, jnp)
    return jnp.dot(feats, jnp.asarray(coeffs))


# Fit from the P3.2 production grid (64³, paired, 12 real, N⋆=500); regenerated
# 2026-06-05 from q_calibration_grid over σ_s∈{1.13,1.42,1.68}×β∈{3.0,3.5,4.0}, α=1.8.
# Fit RMS = 0.018. Order: [1, f, f², σ_s, β, f·σ_s, f·β].
PERSISTED_COEFFS = jnp.array(
    [0.813689, 0.212520, -0.080905, -0.021764, -0.028870, -0.092375, -0.041227]
)

"""Differentiable mass-segregation observables (JAX-native).

Mass segregation is commonly *measured* with non-differentiable estimators
(`compute_lambda_msr`: Allison et al. 2009, SciPy MST). Those break two ways under
autodiff: a hard top-k selection of the "massive" stars (``argsort``) and a
combinatorial spatial statistic (MST). This module provides smooth surrogates so
segregation can enter gradient-based / HMC inference, mirroring the surrogate +
calibration pattern of :mod:`progenax.diagnostics.q_approx` (CW04 Q).

Design: docs/plans/2026-06-09-differentiable-segregation-observable-design.md.

Shared kernel
-------------
Every star carries a smooth weight from a sigmoid soft mass-cut::

    w_i = sigmoid((m_i - m_cut) / tau)

mirroring the observer's choice of a "massive bin" defined by a mass/luminosity cut
``m_cut``. As ``tau -> 0`` this recovers the hard indicator ``1[m_i > m_cut]`` and the
observable reduces to its exact non-differentiable counterpart -- the central
validation route (Oracle 1).

All observables default to ``project_to_2d=True`` (observer-faithful: projected sky
positions), with a 3D flag for theory checks. Inputs are true masses for now; the
noisy mass proxy is a deferred data-realism layer (milestone B).
"""

from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = [
    "soft_mass_weights",
    "radial_concentration_approx",
]


def _project(positions: Float[Array, "N D"], project_to_2d: bool) -> Float[Array, "N d"]:
    """Project to the (x, y) sky plane when requested; pass through otherwise."""
    if positions.shape[1] == 2:
        return positions
    if positions.shape[1] != 3:
        raise ValueError(f"positions must be (N, 2) or (N, 3), got {positions.shape}")
    return positions[:, :2] if project_to_2d else positions


def soft_mass_weights(
    masses: Float[Array, "N"],
    m_cut: Float[Array, ""],
    tau: Float[Array, ""],
) -> Float[Array, "N"]:
    """Smooth soft mass-cut weights ``w_i = sigmoid((m_i - m_cut) / tau)``.

    The shared weighting kernel for every differentiable segregation observable. As
    ``tau -> 0`` the weights approach the hard indicator ``1[m_i > m_cut]``.

    Args:
        masses: Stellar masses ``(N,)``.
        m_cut: Mass cut defining the "massive" population (same units as ``masses``).
        tau: Softness scale; smaller is sharper. Must be > 0.

    Returns:
        Weights ``(N,)`` in the open interval ``(0, 1)``.
    """
    return jax.nn.sigmoid((masses - m_cut) / tau)


def radial_concentration_approx(
    positions: Float[Array, "N D"],
    masses: Float[Array, "N"],
    *,
    m_cut: Float[Array, ""],
    tau: Float[Array, ""],
    project_to_2d: bool = True,
    calibration: float = 1.0,
) -> Float[Array, ""]:
    """Mass-weighted radial-concentration segregation observable.

    Compares the (soft-mass-weighted) mean cluster-centric radius of the massive
    population to the unweighted mean radius of all stars::

        C = [ sum_i w_i r_i / sum_i w_i ]  /  [ mean_i r_i ]

    where ``r_i = |x_i - xbar_w|`` and ``xbar_w`` is the mass-weighted centroid.

    Interpretation:
        - ``C < 1``: massive stars more centrally concentrated (segregated).
        - ``C ~ 1``: no segregation.
        - ``C > 1``: inverse segregation.

    Smooth in positions and ``m_cut``; no graph, no ranking -- the cleanest-gradient
    member of the family. As ``tau -> 0`` it reduces to the exact mass-cut radial
    ratio.

    Args:
        positions: Positions ``(N, 3)`` or ``(N, 2)``.
        masses: Stellar masses ``(N,)``.
        m_cut: Mass cut for the massive population.
        tau: Soft mass-cut softness (> 0).
        project_to_2d: Use projected (x, y) positions (observer-faithful) if True.
        calibration: Multiplicative calibration factor (fit vs the exact oracle).

    Returns:
        Scalar concentration ``C``.
    """
    xy = _project(positions, project_to_2d)
    w = soft_mass_weights(masses, m_cut, tau)
    W = jnp.sum(w)

    # Mass-weighted centroid (segregation is measured about the massive population's
    # center; translation-equivariant).
    center = jnp.sum(w[:, None] * xy, axis=0) / W
    r = jnp.sqrt(jnp.sum((xy - center) ** 2, axis=1) + 1e-12)

    r_massive = jnp.sum(w * r) / W
    r_all = jnp.mean(r)
    return calibration * r_massive / (r_all + 1e-12)

"""Differentiable dispersion forward models for cluster kinematics.

Two packaged, reverse-mode-differentiable forward models that expose the
*equilibrium* velocity dispersion of a (potential, anisotropy) pair — a
property of the spatial ``profile`` (which owns rho, M, Phi), not the
stochastic velocity-DF sampler:

- :func:`jeans_dispersion` — the 3-D anisotropic Jeans solution
  (sigma_r / sigma_t / sigma_1d / beta) for an Osipkov-Merritt (1985)
  anisotropy radius ``r_a``.
- :func:`project_dispersion` — the OBSERVED line-of-sight and proper-motion
  dispersions (sigma_los / sigma_pm,R / sigma_pm,T / Sigma) via the
  Binney & Mamon (1982) projection integrals.

Physics references
------------------
- Binney & Tremaine (2008), *Galactic Dynamics* (2nd ed.), sec. 4.8.3 —
  anisotropic Jeans equation.
- Merritt, D. (1985), AJ 90, 1027 — Osipkov-Merritt anisotropy,
  beta(r) = r^2 / (r^2 + r_a^2); closed-form OM-Plummer dispersion oracle.
- Binney, J. & Mamon, G. A. (1982), MNRAS 200, 361 — line-of-sight
  projection of an anisotropic spherical model.

Units follow the caller-supplied ``G`` (progenax DEFAULT_UNITS is STELLAR:
Msun, pc, Myr). All public functions are reverse-mode differentiable.

NOTE (Phase 0 Task 1): scaffold only. The two NamedTuple return types and the
``_sigma_components`` helper are implemented; ``jeans_dispersion`` and
``project_dispersion`` are stubs raising ``NotImplementedError`` — their bodies
arrive in Tasks 2 and 5.
"""

from typing import NamedTuple, Optional

import jax.numpy as jnp
from jaxtyping import Array, Float


class DispersionProfile(NamedTuple):
    """3-D anisotropic Jeans dispersion evaluated at radii ``r``.

    Fields (all CGS-agnostic; velocities in the units implied by ``G``):

    - ``r``: radii at which the dispersion is evaluated.
    - ``sigma_r``: radial velocity dispersion sqrt(sigma_r^2).
    - ``sigma_t``: tangential (1-component) dispersion; for OM,
      sigma_t^2 = sigma_r^2 * r_a^2 / (r_a^2 + r^2).
    - ``sigma_1d``: 1-D (isotropic-equivalent) dispersion,
      sigma_1d^2 = (sigma_r^2 + 2 sigma_t^2) / 3.
    - ``beta``: Osipkov-Merritt anisotropy beta(r) = r^2 / (r^2 + r_a^2).
    """

    r: Float[Array, " n"]
    sigma_r: Float[Array, " n"]
    sigma_t: Float[Array, " n"]
    sigma_1d: Float[Array, " n"]
    beta: Float[Array, " n"]


class ProjectedDispersion(NamedTuple):
    """Observed (projected) dispersions at on-sky radii ``R`` (Binney & Mamon 1982).

    - ``R``: projected (on-sky) radii.
    - ``sigma_los``: line-of-sight dispersion (the RV channel).
    - ``sigma_pm_r``: on-sky radial proper-motion dispersion.
    - ``sigma_pm_t``: on-sky tangential proper-motion dispersion.
    - ``Sigma``: projected surface density (the projection normalisation).
    """

    R: Float[Array, " m"]
    sigma_los: Float[Array, " m"]
    sigma_pm_r: Float[Array, " m"]
    sigma_pm_t: Float[Array, " m"]
    Sigma: Float[Array, " m"]


def _sigma_components(
    sigma_r2: Float[Array, " n"],
    r: Float[Array, " n"],
    r_a: Optional[float],
):
    """Build (sigma_r, sigma_t, sigma_1d, beta) from sigma_r^2(r) under OM anisotropy.

    Osipkov-Merritt (Merritt 1985): beta(r) = r^2 / (r^2 + r_a^2), giving
    sigma_t^2 = sigma_r^2 * r_a^2 / (r_a^2 + r^2) and
    sigma_1d^2 = (sigma_r^2 + 2 sigma_t^2) / 3.

    Isotropic branch (``r_a is None``): beta = 0, sigma_t = sigma_r, so
    sigma_1d = sigma_r.

    Returns
    -------
    (sigma_r, sigma_t, sigma_1d, beta) : tuple of arrays shaped like ``r``.
    """
    sigma_r = jnp.sqrt(sigma_r2)
    if r_a is None:
        beta = jnp.zeros_like(r)
        sigma_t2 = sigma_r2
    else:
        r_a2 = jnp.asarray(r_a) ** 2
        beta = r**2 / (r**2 + r_a2)
        sigma_t2 = sigma_r2 * r_a2 / (r_a2 + r**2)
    sigma_t = jnp.sqrt(sigma_t2)
    sigma_1d = jnp.sqrt((sigma_r2 + 2.0 * sigma_t2) / 3.0)
    return sigma_r, sigma_t, sigma_1d, beta


def jeans_dispersion(profile, r_a, r, M, G) -> DispersionProfile:
    """3-D anisotropic Jeans dispersion of ``profile`` under OM ``r_a``.

    Stub (Phase 0 Task 1) — implemented in Task 2.
    """
    raise NotImplementedError(
        "jeans_dispersion is a Phase 0 Task 1 scaffold; physics lands in Task 2."
    )


def project_dispersion(profile, r_a, R, M, G) -> ProjectedDispersion:
    """Observed projected dispersions of ``profile`` via Binney & Mamon (1982).

    Stub (Phase 0 Task 1) — implemented in Task 5.
    """
    raise NotImplementedError(
        "project_dispersion is a Phase 0 Task 1 scaffold; physics lands in Task 5."
    )

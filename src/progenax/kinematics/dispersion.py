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

import jax.core

from progenax.numerics import cumulative_trapezoid


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


def jeans_sigma_r(
    r: Float[Array, " n"],
    rho: Float[Array, " ns"],
    M_enc: Float[Array, " ns"],
    s: Float[Array, " ns"],
    G,
    r_a: Optional[float] = None,
) -> Float[Array, " n"]:
    """Anisotropic-Jeans radial velocity dispersion sqrt(sigma_r^2) at radii ``r``.

    Solves the spherical anisotropic Jeans equation (Binney & Tremaine 2008,
    sec. 4.8.3; Merritt 1985, Eq. 15) for an Osipkov-Merritt anisotropy
    ``beta(r) = r^2 / (r^2 + r_a^2)``::

        rho sigma_r^2(r) = 1/(r^2 + r_a^2) * int_r^inf (s^2 + r_a^2) rho(s) G M(<s)/s^2 ds

    The isotropic limit (``r_a is None``) drops the ``(s^2 + r_a^2)`` weight and
    the ``1/(r^2 + r_a^2)`` prefactor (both -> 1).

    The outward integral ``I(s) = int_s^inf (...) ds`` is evaluated by a *reverse*
    cumulative trapezoid: flip the integrand, run :func:`cumulative_trapezoid`
    (which integrates from the leading edge), then flip back. ``rho(s)`` and
    ``I(s)`` are interpolated onto the query radii ``r`` with ``jnp.interp``.
    Fully reverse-mode differentiable; uniform ``s`` spacing assumed.

    Parameters
    ----------
    r : query radii.
    rho, M_enc, s : profile density, enclosed mass M(<s), and the (uniform) s-grid.
    G : gravitational constant (sets the velocity units).
    r_a : Osipkov-Merritt anisotropy radius, or ``None`` for isotropic.
    """
    ds = s[1] - s[0]
    s2 = s**2
    # Integrand g(s) = w(s) * rho(s) * G M(<s) / s^2, with the OM weight
    # w(s) = (s^2 + r_a^2) (isotropic: w == 1). Guard s^2 against the s=0 edge.
    if r_a is None:
        weight = jnp.ones_like(s)
    else:
        weight = s2 + jnp.asarray(r_a) ** 2
    integrand = weight * rho * G * M_enc / jnp.maximum(s2, 1e-30)
    # Reverse cumulative trapezoid -> I(s) = int_s^inf integrand ds.
    I_outward = jnp.flip(cumulative_trapezoid(jnp.flip(integrand), dx=ds))
    # Interpolate rho and the outward integral onto the query radii.
    rho_r = jnp.interp(r, s, rho)
    I_r = jnp.interp(r, s, I_outward)
    if r_a is None:
        prefactor = jnp.ones_like(r)
    else:
        prefactor = 1.0 / jnp.maximum(r**2 + jnp.asarray(r_a) ** 2, 1e-30)
    sigma_r2 = prefactor * I_r / jnp.maximum(rho_r, 1e-30)
    return jnp.sqrt(jnp.maximum(sigma_r2, 0.0))


def jeans_dispersion(profile, r_a, r, M, G) -> DispersionProfile:
    """3-D anisotropic Jeans dispersion of ``profile`` under OM ``r_a``.

    Returns the equilibrium ``(sigma_r, sigma_t, sigma_1d, beta)`` of the
    spatial ``profile`` (which owns rho, M, Phi) for an Osipkov-Merritt (1985)
    anisotropy radius ``r_a`` (``None`` -> isotropic). The radial dispersion is
    the anisotropic Jeans solution (:func:`jeans_sigma_r`); the tangential /
    1-D components and ``beta`` follow from OM (:func:`_sigma_components`).

    The enclosed mass ``M(<s)`` is a *quadrature of ``profile.density``*
    (builder-quality, no re-differentiated Psi): ``M_enc = M * cumtrap(rho s^2)
    / cumtrap_total(rho s^2)``. A fine radial s-grid runs to ``profile.r_t`` if
    present (King/EFF), else ``30 a`` (Plummer).

    Parameters
    ----------
    profile : spatial profile exposing ``density(r)`` (and optionally ``r_t``,
        ``a``).
    r_a : Osipkov-Merritt anisotropy radius, or ``None`` for isotropic.
    r : query radii (array-like; broadcast to at least 1-D).
    M : total mass normalising the enclosed-mass quadrature.
    G : gravitational constant (sets the velocity units).

    Returns
    -------
    DispersionProfile

    Notes
    -----
    For a *concrete* (non-traced) ``r_a`` the Plummer OM DF requires
    ``r_a >= 0.75 a`` (Merritt 1985, Eq. 46); a smaller value is unphysical
    (negative phase-space DF) and raises ``ValueError``. The check is eager,
    skipped under tracing so ``jax.grad``/``jax.jit`` over ``r_a`` still work,
    and gated on ``hasattr(profile, "a")`` so non-Plummer profiles (different
    validity domains) do not trip it.
    """
    # Eager r_a validity-domain guard (Plummer OM): mirror plummer_df.py:128.
    # Concrete = a Python scalar or a non-traced array; under tracing (jax.grad
    # / jax.jit over r_a) the value is a Tracer and the check is skipped so the
    # caller owns the r_a >= 0.75 a bound. Gated on hasattr(profile, "a") so
    # non-Plummer profiles (different validity domains) do not trip it.
    if (
        r_a is not None
        and not isinstance(r_a, jax.core.Tracer)
        and hasattr(profile, "a")
    ):
        a_val = float(profile.a)
        if float(r_a) < 0.75 * a_val:
            raise ValueError(
                f"Plummer Osipkov-Merritt dispersion requires r_a >= 0.75 a = "
                f"{0.75 * a_val:.4f} (Merritt 1985, Eq. 46); got {float(r_a)}. Smaller "
                f"r_a makes the phase-space DF negative (unphysical)."
            )

    r = jnp.atleast_1d(jnp.asarray(r))

    # Fine radial s-grid: out to the tidal radius if the profile has one,
    # else 30 scale radii (Plummer has no finite cutoff).
    r_max = getattr(profile, "r_t", None)
    if r_max is None:
        r_max = 30.0 * profile.a
    s = jnp.linspace(1e-4 * r_max, r_max, 4000)
    rho = profile.density(s)

    # Enclosed mass by quadrature of profile.density: M(<s) = M * cumtrap(rho s^2)
    # / cumtrap_total(rho s^2). Builder-quality, no re-differentiated Psi.
    cum = cumulative_trapezoid(rho * s**2, dx=s[1] - s[0])
    M_enc = M * cum / jnp.maximum(cum[-1], 1e-30)

    sigma_r = jeans_sigma_r(r, rho, M_enc, s, G, r_a=r_a)
    sr, st, s1d, beta = _sigma_components(sigma_r**2, r, r_a)
    return DispersionProfile(r=r, sigma_r=sr, sigma_t=st, sigma_1d=s1d, beta=beta)


def project_dispersion(profile, r_a, R, M, G) -> ProjectedDispersion:
    """Observed projected dispersions of ``profile`` via Binney & Mamon (1982).

    Stub (Phase 0 Task 1) — implemented in Task 5.
    """
    raise NotImplementedError(
        "project_dispersion is a Phase 0 Task 1 scaffold; physics lands in Task 5."
    )

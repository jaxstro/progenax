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
- Dejonghe, H. (1987), MNRAS 224, 13 — Plummer-family projected dispersion;
  the isotropic ``sigma_los^2(R) = (3 pi / 64) G M / sqrt(a^2 + R^2)`` oracle.

Units follow the caller-supplied ``G`` and ``M`` (progenax DEFAULT_UNITS is
STELLAR: Msun, pc, Myr): the returned dispersions are in whatever velocity unit
``sqrt(G M / length)`` implies, set entirely by the caller.

All public functions are differentiated with **reverse-mode** AD
(``jax.grad`` / ``jax.jacrev``) — that is what the OED Fisher uses and what the
grad-audit gate tests. For the *analytic-density* profiles (Plummer, EFF) the
quadratures here are plain ``jnp`` (``cumulative_trapezoid`` is a ``jnp.cumsum``,
no custom rule), so forward-mode (``jax.jacfwd`` / ``jax.jvp``) also works. The
forward-mode restriction is profile-specific: differentiating through the
*equilibrium-solver* profiles (King / Michie) hits ``custom_vjp`` ODE solvers
that define no ``jvp`` rule, so ``jacfwd`` through those raises. Reverse-mode
works for all profiles and is the supported/tested path.

Scope (Phase 0): spherical, single-population, mass-follows-light, Osipkov-Merritt
anisotropy. Rotation, non-sphericity, tracer != mass, native (non-OM) anisotropy,
and multi-population kinematics are tracked extensions — see the versatility
roadmap in ``docs/plans/2026-06-15-oed-dispersion-arc-design.md``.
"""

from typing import NamedTuple, Optional

import jax
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


def ftable_sigma_r_isotropic(
    E_grid: Float[Array, " n_e"],
    f_grid: Float[Array, " n_e"],
    Psi_r: Float[Array, ""],
    n_s: int = 512,
) -> Float[Array, ""]:
    """Isotropic radial dispersion sigma_r^2 from a tabulated DF speed second moment.

    At relative potential ``Psi_r = Psi(r)`` the (unnormalised) speed pdf of an
    *isotropic* ergodic model is ``p(s) ∝ s^2 f(Psi_r - s^2/2)`` on
    ``s in [0, sqrt(2 Psi_r)]`` (the energy is ``E = Psi - s^2/2 >= 0``). The speed
    second moment is ``<s^2> = int s^2 p(s) ds / int p(s) ds``, and for an isotropic
    velocity ellipsoid each component carries one third::

        sigma_r^2 = <s^2> / 3        (sigma_r = sigma_t = sigma_1d, isotropic)

    This is the DF-side cross-check of :func:`jeans_sigma_r`: a *different* code path
    (speed quadrature over the tabulated ``f``) for the same physical quantity. It is
    isotropic-only by design — the Osipkov-Merritt second moment is a 2-D (energy,
    angular-momentum) integral, out of scope here.

    The DF table ``(E_grid, f_grid)`` and ``Psi_r`` must share an energy scale
    (e.g. the dimensionless isotropic Plummer table ``f(E) ∝ E^(7/2)`` with
    ``Psi_r = 6 / sqrt(1 + r^2/a^2)``); the returned ``sigma_r^2`` is then in those
    same (Psi) units — rescale by ``sigma0^2`` for physical velocities. Differentiable
    trapezoid quadrature on a fixed ``n_s`` speed grid; ``f`` is interpolated with
    ``jnp.interp`` (which clamps off-grid, harmless here since the integrand support
    ``E in [0, Psi_r]`` lies inside the table when ``Psi_r <= E_grid[-1]``).

    Parameters
    ----------
    E_grid, f_grid : tabulated energy grid and DF values (any positive normalisation).
    Psi_r : relative potential at the query radius (same energy units as ``E_grid``).
    n_s : number of speed-quadrature points.
    """
    Psi_safe = jnp.maximum(Psi_r, 1e-30)
    s = jnp.linspace(0.0, jnp.sqrt(2.0 * Psi_safe), n_s)
    f_at = jnp.interp(Psi_r - s**2 / 2.0, E_grid, f_grid)
    p = jnp.maximum(s**2 * f_at, 0.0)  # speed pdf (unnormalised)
    num = jnp.trapezoid(s**2 * p, s)   # int s^2 p(s) ds
    den = jnp.trapezoid(p, s)          # int p(s) ds (normalisation)
    s2_mean = num / jnp.maximum(den, 1e-30)
    return s2_mean / 3.0


def jeans_dispersion(profile, r_a, r, M, G, n_s: int = 4000) -> DispersionProfile:
    """3-D anisotropic Jeans dispersion of ``profile`` under OM ``r_a``.

    Returns the equilibrium ``(sigma_r, sigma_t, sigma_1d, beta)`` of the
    spatial ``profile`` (which owns rho, M, Phi) for an Osipkov-Merritt (1985)
    anisotropy radius ``r_a`` (``None`` -> isotropic). The radial dispersion is
    the anisotropic Jeans solution (:func:`jeans_sigma_r`); the tangential /
    1-D components and ``beta`` follow from OM (:func:`_sigma_components`).

    Anisotropy model (IMPORTANT — read before using on a Michie/King profile).
    This function **imposes the Osipkov-Merritt anisotropy law**
    ``beta(r) = r^2 / (r^2 + r_a^2)`` on the *density* of ``profile``, regardless
    of the profile's own intrinsic anisotropy. Consequences:

    - **Plummer / EFF** are intrinsically isotropic, so layering OM on their
      density is the EXACT Osipkov-Merritt model — correct at all radii.
    - **Michie** is intrinsically anisotropic with its OWN (Michie-King)
      anisotropy law, which agrees with OM in the core but DIVERGES outward.
      ``jeans_dispersion(MichieProfile, r_a)`` is therefore "the Michie *density*
      under OM anisotropy", NOT the native Michie equilibrium — validated only in
      the inner region (r << r_t) where OM is a good model of the Michie law.

    Units follow the caller-supplied ``G`` and ``M``; the returned velocities are
    in whatever ``sqrt(G M / length)`` implies. Differentiated with **reverse-mode**
    AD (the supported/tested path); forward-mode also works for analytic-density
    profiles (Plummer/EFF) but not through the King/Michie equilibrium-solver
    profiles — see the module docstring.

    See also: a general-``beta(r)`` generalisation (native Michie/King anisotropy
    and arbitrary custom ``beta(r)`` via the integrating factor
    ``f(r) = exp(2 int beta(s)/s ds)``) is on the versatility roadmap
    (``docs/plans/2026-06-15-oed-dispersion-arc-design.md``).

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
    n_s : number of points on the fine radial s-grid (default 4000). The trapezoid
        quadrature is O(h^2 = (r_max/n_s)^2); exposed so a convergence study can
        refine it. Backward-compatible (keyword, default unchanged) and traced-safe
        (a static Python int, not a JAX value).

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
    # Concrete = a Python scalar or a non-traced array; under tracing the value
    # is a Tracer and the check is skipped so the caller owns the r_a >= 0.75 a
    # bound. Gated on hasattr(profile, "a") so non-Plummer profiles (different
    # validity domains) do not trip it. BOTH r_a AND profile.a must be concrete:
    # differentiating through r_h (jax.grad over a Plummer r_h) makes profile.a a
    # tracer even when r_a is a plain float, so float(profile.a) would raise a
    # ConcretizationTypeError — skip the eager guard in that case too.
    if (
        r_a is not None
        and not isinstance(r_a, jax.core.Tracer)
        and hasattr(profile, "a")
        and not isinstance(profile.a, jax.core.Tracer)
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
    s = jnp.linspace(1e-4 * r_max, r_max, n_s)
    rho = profile.density(s)

    # Enclosed mass by quadrature of profile.density: M(<s) = M * cumtrap(rho s^2)
    # / cumtrap_total(rho s^2). Builder-quality, no re-differentiated Psi.
    cum = cumulative_trapezoid(rho * s**2, dx=s[1] - s[0])
    M_enc = M * cum / jnp.maximum(cum[-1], 1e-30)

    sigma_r = jeans_sigma_r(r, rho, M_enc, s, G, r_a=r_a)
    sr, st, s1d, beta = _sigma_components(sigma_r**2, r, r_a)
    return DispersionProfile(r=r, sigma_r=sr, sigma_t=st, sigma_1d=s1d, beta=beta)


def project_dispersion(profile, r_a, R, M, G, n_u: int = 4000) -> ProjectedDispersion:
    """Observed projected dispersions of ``profile`` via Binney & Mamon (1982).

    Projects the 3-D anisotropic Jeans model (:func:`jeans_dispersion`) onto the
    sky along the line of sight, returning the OBSERVED dispersions at on-sky
    radii ``R`` (Binney & Mamon 1982, MNRAS 200, 361). With Osipkov-Merritt
    anisotropy ``beta(r) = r^2 / (r^2 + r_a^2)``::

        Sigma(R)            = 2 int_R^inf rho                       r/sqrt(r^2-R^2) dr
        Sigma sigma_los^2   = 2 int_R^inf (1 - beta R^2/r^2)  rho sr^2 r/sqrt(r^2-R^2) dr
        Sigma sigma_pmR^2   = 2 int_R^inf (1 - beta + beta R^2/r^2) rho sr^2 r/sqrt(...) dr
        Sigma sigma_pmT^2   = 2 int_R^inf (1 - beta)         rho sr^2 r/sqrt(r^2-R^2) dr

    where ``sr^2 = sigma_r^2(r)`` (radial), ``sigma_los`` is the line-of-sight
    (the RV channel), and ``sigma_pmR`` / ``sigma_pmT`` are the on-sky radial /
    tangential proper-motion (PM) channels. ``rho`` cancels in the sigma ratios; in
    ``Sigma`` it is the projected surface density in ``profile.density`` units.
    The isotropic limit (``r_a=None`` -> ``beta=0``) collapses all three kernels to
    1, so ``sigma_los = sigma_pm_r = sigma_pm_t``; anisotropy lives in the ratios.

    Isotropic-Plummer ``sigma_los`` oracle: for the isotropic Plummer member this
    projection has the closed form ``sigma_los^2(R) = (3 pi / 64) G M /
    sqrt(a^2 + R^2)`` (Dejonghe 1987, MNRAS 224, 13), used as the tight absolute
    validation anchor in ``test_dispersion_physics.py``.

    Singularity removal (load-bearing, keeps it differentiable): the
    ``1/sqrt(r^2-R^2)`` pole at ``r=R`` is removed ANALYTICALLY by the
    substitution ``r^2 = R^2 + u^2`` (so ``r dr / sqrt(r^2-R^2) = du``)::

        int_R^inf g(r) r/sqrt(r^2-R^2) dr = int_0^sqrt(r_max^2-R^2) g(sqrt(R^2+u^2)) du

    so NO ``1/sqrt(r^2-R^2)`` is ever evaluated — only a smooth uniform-``u``
    trapezoid quadrature. ``sigma_r^2(r)`` and ``beta(r)`` come straight from
    :func:`jeans_dispersion` (not re-derived); ``rho`` from ``profile.density``.

    The outward extent ``r_max`` is ``profile.r_t`` if present (King/EFF), else
    ``30 * profile.a`` (Plummer; matches :func:`jeans_dispersion`). The per-``R``
    ``u``-integral is vmapped over the ``R`` array. Fully reverse-mode
    differentiable; ``jax.jit``-able.

    Parameters
    ----------
    profile : spatial profile exposing ``density(r)`` (and optionally ``r_t``, ``a``).
    r_a : Osipkov-Merritt anisotropy radius, or ``None`` for isotropic (beta=0).
    R : projected (on-sky) radii (array-like; broadcast to at least 1-D).
    M : total mass normalising the enclosed-mass quadrature (passed through to
        :func:`jeans_dispersion`).
    G : gravitational constant (sets the velocity units).
    n_u : number of points on the uniform ``u``-quadrature grid (default 4000).

    Returns
    -------
    ProjectedDispersion
    """
    R = jnp.atleast_1d(jnp.asarray(R))

    # Outward extent: tidal radius if finite, else 30 scale radii (matches jeans).
    r_max = getattr(profile, "r_t", None)
    if r_max is None:
        r_max = 30.0 * profile.a
    r_max = jnp.asarray(r_max)

    # Pull the outer integration radius a hair inside r_max. At r == r_max the
    # outward Jeans integral I(r_max) = 0 EXACTLY, so sigma_r^2 = 0 there and
    # jnp.sqrt has an infinite (NaN) gradient at that zero-measure endpoint —
    # which would poison jax.grad of the whole reduction. The endpoint carries
    # rho * sigma_r^2 -> 0, i.e. ~no weight, so stopping at r_edge = (1 - 1e-6)
    # r_max removes the NaN without changing the integral (mirrors how
    # jeans_dispersion starts its own s-grid at 1e-4 r_max, not 0).
    r_edge = (1.0 - 1e-6) * r_max

    def _los_quantities(R_i):
        # r^2 = R^2 + u^2 substitution: u in [0, sqrt(r_edge^2 - R^2)] (clip the
        # upper limit to be non-negative if R_i ever brushes r_edge).
        u_max = jnp.sqrt(jnp.maximum(r_edge**2 - R_i**2, 0.0))
        u = jnp.linspace(0.0, u_max, n_u)
        r = jnp.sqrt(R_i**2 + u**2)

        # rho, sigma_r^2, beta at the integration radii (sigma_r/beta straight
        # from the anisotropic Jeans solution — NOT re-derived here).
        rho = profile.density(r)
        dp = jeans_dispersion(profile, r_a, r, M, G)
        sigma_r2 = dp.sigma_r**2
        beta = dp.beta

        ratio = R_i**2 / jnp.maximum(r**2, 1e-30)  # R^2 / r^2
        w = rho * sigma_r2  # common rho*sigma_r^2 weight

        # B&M82 kernels (integrands in u; the 2x and 1/sqrt cancellation are
        # folded into the substitution -> trapezoid over u, times 2).
        Sigma = 2.0 * jnp.trapezoid(rho, u)
        S_los = 2.0 * jnp.trapezoid((1.0 - beta * ratio) * w, u)
        S_pmr = 2.0 * jnp.trapezoid((1.0 - beta + beta * ratio) * w, u)
        S_pmt = 2.0 * jnp.trapezoid((1.0 - beta) * w, u)
        return Sigma, S_los, S_pmr, S_pmt

    Sigma, S_los, S_pmr, S_pmt = jax.vmap(_los_quantities)(R)

    Sigma_safe = jnp.maximum(Sigma, 1e-30)
    sigma_los = jnp.sqrt(jnp.maximum(S_los / Sigma_safe, 0.0))
    sigma_pm_r = jnp.sqrt(jnp.maximum(S_pmr / Sigma_safe, 0.0))
    sigma_pm_t = jnp.sqrt(jnp.maximum(S_pmt / Sigma_safe, 0.0))

    return ProjectedDispersion(
        R=R,
        sigma_los=sigma_los,
        sigma_pm_r=sigma_pm_r,
        sigma_pm_t=sigma_pm_t,
        Sigma=Sigma,
    )

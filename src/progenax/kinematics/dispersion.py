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

# Default resolution of the master anisotropic-Jeans radial s-grid. Single source
# of truth shared by jeans_dispersion (its n_s default) and project_dispersion
# (which solves the Jeans equation once at this resolution, independent of n_u).
_JEANS_N_S_DEFAULT = 4000

# Algebraic compactification of the semi-infinite Plummer outward Jeans integral
# (Task C). The Plummer profile has no finite cutoff, so a uniform s-grid must be
# truncated (was 30a), leaving an outward-growing tail bias (~8.6e-4 rel. at r=20a).
# Map s in [s_min, inf) to t in [_T_MIN, _T_MAX] via s = a t/(1-t): t->1 captures
# s->inf, so the full tail is integrated on a finite, uniform-t grid. _T_MAX is held
# a hair below 1 (s stays finite, ~1e6 a) and _T_MIN a hair above 0 (s_min ~ 1e-4 a).
_T_MIN = 1e-4
_T_MAX = 1.0 - 1e-6


def _safe_sqrt(x):
    """sqrt with an exact-0 forward value and a finite (0) gradient at x==0 (avoids the
    inf-derivative of sqrt at 0 that poisons jacrev over a radial grid past r_t)."""
    pos = x > 0.0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, x, 1.0)), 0.0)


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
    beta: Optional[Float[Array, " n"]] = None,
):
    """Build (sigma_r, sigma_t, sigma_1d, beta) from sigma_r^2(r).

    Two anisotropy sources:

    - **Explicit ``beta`` array (general-beta path):** when ``beta`` is given,
      ``sigma_t^2 = (1 - beta) sigma_r^2`` and ``sigma_1d^2 = (sigma_r^2 +
      2 sigma_t^2)/3`` directly (the arbitrary-beta(r) integrating-factor path of
      :func:`jeans_dispersion`'s ``beta_fn``). ``r_a`` is ignored in this branch.
    - **OM / isotropic closed form (``beta`` is None):** Osipkov-Merritt
      (Merritt 1985) ``beta(r) = r^2 / (r^2 + r_a^2)``, giving
      ``sigma_t^2 = sigma_r^2 * r_a^2 / (r_a^2 + r^2)``; isotropic
      (``r_a is None``) -> ``beta = 0``, ``sigma_t = sigma_r``.

    Returns
    -------
    (sigma_r, sigma_t, sigma_1d, beta) : tuple of arrays shaped like ``r``.
    """
    sigma_r = _safe_sqrt(sigma_r2)
    if beta is not None:
        sigma_t2 = (1.0 - beta) * sigma_r2
    elif r_a is None:
        beta = jnp.zeros_like(r)
        sigma_t2 = sigma_r2
    else:
        r_a2 = jnp.asarray(r_a) ** 2
        beta = r**2 / (r**2 + r_a2)
        sigma_t2 = sigma_r2 * r_a2 / (r_a2 + r**2)
    sigma_t = _safe_sqrt(sigma_t2)
    sigma_1d = _safe_sqrt((sigma_r2 + 2.0 * sigma_t2) / 3.0)
    return sigma_r, sigma_t, sigma_1d, beta


def _log_factor(beta_fn, s):
    """Log integrating factor ``F(s) = 2 int beta(s)/s ds`` (general-beta path).

    The anisotropic Jeans equation uses the integrating factor
    ``f(r) = exp(2 int beta(s)/s ds)``; only the RATIO ``f(s)/f(r)`` enters
    ``sigma_r^2``, so the (arbitrary) lower-limit constant of the indefinite
    integral cancels and we may build the cumulative integral from the grid's
    leading edge. Returned MAX-SUBTRACTED (``F - max F``) so the downstream
    ``exp(F)`` never overflows: the subtracted constant is the same in numerator
    (folded into the integrand) and denominator (the ``exp(-(F(r)-Fmax))``
    prefactor), so it cancels exactly.

    Integration is done in ``ln(s)`` — the exact change of variable
    ``2 beta(s)/s ds = 2 beta(s) d(ln s)`` — rather than in the master grid's own
    variable. This is the SAME integral on BOTH branches (uniform-s for finite
    ``r_t``; compactified uniform-t for Plummer): ``ln(s)`` is smooth across both
    grids, whereas the raw ``2 beta/s`` integrand (or its ``jac``-folded t form
    ``2 beta/((1-t) t)``) has a ``1/(1-t)`` blow-up at the Plummer grid's outer
    edge (``t -> 1``) that a uniform-dx trapezoid resolves poorly — a single
    contaminated last panel that corrupts ``max F``. Because ``beta`` is bounded
    and ``ln(s)`` is well-spaced, the ``ln(s)`` trapezoid reproduces the analytic
    OM factor ``ln(s^2 + r_a^2)`` to ~1e-6 at every radius (the OM-reduction
    proof). A variable-spacing trapezoid is used (the compactified ``ln(s)`` is
    non-uniform); fully reverse-mode differentiable.

    Returns ``(F_shifted, Fmax)`` with ``F_shifted = F - Fmax``. ``Fmax`` is the
    subtracted constant; callers fold it away (``F_shifted, _ = _log_factor(...)``)
    and it is returned only for testability (inspecting the absolute log factor).
    """
    beta_s = beta_fn(s)
    ln_s = jnp.log(jnp.maximum(s, 1e-30))
    # Variable-spacing cumulative trapezoid of 2*beta in ln(s) (leading zero).
    dx = jnp.diff(ln_s)
    panel = 0.5 * (beta_s[1:] + beta_s[:-1]) * dx
    F = 2.0 * jnp.concatenate([jnp.zeros(1, dtype=panel.dtype), jnp.cumsum(panel)])
    Fmax = jnp.max(F)
    return F - Fmax, Fmax


def _jeans_tables(profile, r_a, M, G, n_s: int, beta_fn=None):
    """R-independent master tables for the anisotropic-Jeans ``sigma_r``.

    Builds, ONCE, the radial grid ``s`` and the quantities the per-radius
    ``sigma_r^2`` needs: the density ``rho(s)`` and the outward integral
    ``I_outward(s) = int_s^inf f(s) rho(s) G M(<s) / s^2 ds``. The integrating
    factor ``f(s)`` takes one of two forms:

    - **OM / isotropic (``beta_fn is None``):** ``f = s^2 + r_a^2`` (isotropic
      ``f == 1``) — the analytic Osipkov-Merritt weight, with the matching
      ``1/(r^2 + r_a^2)`` prefactor applied in :func:`_sigma_r2_from_tables`.
      BIT-IDENTICAL to the pre-D1 path.
    - **General-beta (``beta_fn`` given):** ``f(s) = exp(2 int beta(s)/s ds)``,
      built NUMERICALLY in max-subtracted log form ``exp(F(s) - Fmax)`` (see
      :func:`_log_factor`); the matching prefactor is ``exp(-(F(r) - Fmax))``.

    This is the same construction the old per-call
    :func:`jeans_dispersion` radial-dispersion path did, factored out so
    :func:`project_dispersion` can solve the Jeans equation exactly once and
    interpolate per projected radius ``R`` (instead of a fresh master solve
    inside every vmapped ``_los_quantities``).

    The enclosed mass ``M(<s)`` is a quadrature of ``profile.density``:
    ``M_enc = M * cumtrap(rho s^2) / cumtrap_total(rho s^2)``.

    Two grids, branched on ``profile.r_t``:

    - **Finite ``r_t`` (King / EFF):** a uniform ``s`` grid on ``[1e-4 r_t, r_t]``
      (the density has a finite cutoff, so no compactification is needed).
    - **No ``r_t`` (Plummer; Task C):** the semi-infinite domain ``[s_min, inf)`` is
      mapped to ``t in [_T_MIN, _T_MAX]`` by ``s = a t/(1-t)`` (Jacobian
      ``ds/dt = a/(1-t)^2``). Both the ``M(<s)`` quadrature and the outward integral
      ``I_outward`` are evaluated in UNIFORM ``t`` with the Jacobian folded into the
      integrand — preserving ``cumulative_trapezoid``'s uniform-``dx`` contract — so
      the full Plummer tail is captured (killing the old 30a truncation bias). On the
      general-beta path the ``F = 2 int beta/s ds`` cumulative integral is done in
      ``ln(s)`` (the exact ``2 beta/s ds = 2 beta d(ln s)`` change of variable), which
      is smooth on BOTH grids and avoids the t-grid edge singularity (see
      :func:`_log_factor`).

    Both ``s`` grids are monotone increasing, so the downstream ``jnp.interp(r, s,
    ...)`` works (the compactified ``s`` is non-uniform but still sorted). Fully
    reverse-mode differentiable.

    Returns ``(s, rho, I_outward, F_shifted)`` — ``F_shifted`` is ``None`` on the
    OM/isotropic path and ``F(s) - Fmax`` on the general-beta path; pass these
    straight to :func:`_sigma_r2_from_tables`.
    """
    r_max = getattr(profile, "r_t", None)

    if r_max is None:
        # Plummer (no finite cutoff): algebraic compactification s = a t/(1-t),
        # integrate in uniform t with the Jacobian ds/dt = a/(1-t)^2 folded in so
        # cumulative_trapezoid still sees a uniform dx grid (here dt).
        a = profile.a
        t = jnp.linspace(_T_MIN, _T_MAX, n_s)
        s = a * t / (1.0 - t)
        jac = a / (1.0 - t) ** 2  # ds/dt
        rho = profile.density(s)
        dt = t[1] - t[0]
        s2 = s**2

        # M(<s) = M * cumtrap(rho s^2 ds) / total, in t: integrand carries `jac`.
        cum = cumulative_trapezoid(rho * s2 * jac, dx=dt)
        M_enc = M * cum / jnp.maximum(cum[-1], 1e-30)

        if beta_fn is not None:
            # f(s) = exp(F-Fmax); F = 2 int beta d(ln s) (smooth on the t-grid).
            F_shifted, _ = _log_factor(beta_fn, s)
            f_weight = jnp.exp(F_shifted)
        elif r_a is None:
            f_weight = jnp.ones_like(s)
            F_shifted = None
        else:
            f_weight = s2 + jnp.asarray(r_a) ** 2
            F_shifted = None
        # I(s) = int_s^inf f rho G M(<s)/s^2 ds; in t the integrand carries `jac`.
        integrand = f_weight * rho * G * M_enc / jnp.maximum(s2, 1e-30) * jac
        I_outward = jnp.flip(cumulative_trapezoid(jnp.flip(integrand), dx=dt))
        return s, rho, I_outward, F_shifted

    # Finite r_t (King / EFF): uniform s grid, no compactification needed.
    s = jnp.linspace(1e-4 * r_max, r_max, n_s)
    rho = profile.density(s)
    ds = s[1] - s[0]

    # Enclosed mass by quadrature of profile.density (builder-quality).
    cum = cumulative_trapezoid(rho * s**2, dx=ds)
    M_enc = M * cum / jnp.maximum(cum[-1], 1e-30)

    # Integrand g(s) = f(s) * rho(s) * G M(<s) / s^2; on the OM path the weight is
    # the analytic w(s) = (s^2 + r_a^2) (isotropic: w == 1). Guard s^2 at s=0.
    s2 = s**2
    if beta_fn is not None:
        F_shifted, _ = _log_factor(beta_fn, s)
        f_weight = jnp.exp(F_shifted)
    elif r_a is None:
        f_weight = jnp.ones_like(s)
        F_shifted = None
    else:
        f_weight = s2 + jnp.asarray(r_a) ** 2
        F_shifted = None
    integrand = f_weight * rho * G * M_enc / jnp.maximum(s2, 1e-30)
    # Reverse cumulative trapezoid -> I(s) = int_s^inf integrand ds.
    I_outward = jnp.flip(cumulative_trapezoid(jnp.flip(integrand), dx=ds))
    return s, rho, I_outward, F_shifted


def _sigma_r2_from_tables(r, s, rho, I_outward, r_a, F_shifted=None):
    """Anisotropic-Jeans ``sigma_r^2(r)`` from the master tables (no master solve).

    Interpolates ``rho`` and ``I_outward`` (built by :func:`_jeans_tables`) onto
    the query radii ``r`` and applies the integrating-factor prefactor::

        sigma_r^2(r) = prefactor(r) * I_outward(r) / rho(r)

    Two prefactors:

    - **OM / isotropic (``F_shifted is None``):** ``1/(r^2 + r_a^2)`` (isotropic:
      1) — the analytic Osipkov-Merritt prefactor. BIT-IDENTICAL to pre-D1.
    - **General-beta (``F_shifted`` given):** ``exp(-(F(r) - Fmax))`` =
      ``1/f(r)`` (the max-subtraction cancels with the same constant folded into
      ``I_outward``'s integrand), interpolating ``F - Fmax`` onto ``r``.

    This is exactly the tail of the anisotropic-Jeans ``sigma_r`` solve (same
    operations: interpolate + prefactor), minus the (now R-independent)
    integrand/``I_outward`` build done in :func:`_jeans_tables`.
    """
    rho_r = jnp.interp(r, s, rho)
    I_r = jnp.interp(r, s, I_outward)
    if F_shifted is not None:
        prefactor = jnp.exp(-jnp.interp(r, s, F_shifted))
    elif r_a is None:
        prefactor = jnp.ones_like(r)
    else:
        prefactor = 1.0 / jnp.maximum(r**2 + jnp.asarray(r_a) ** 2, 1e-30)
    return prefactor * I_r / jnp.maximum(rho_r, 1e-30)


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

    This is the DF-side cross-check of the anisotropic-Jeans ``sigma_r`` solve
    (:func:`_jeans_tables` / :func:`_sigma_r2_from_tables`): a *different* code path
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


def jeans_dispersion(
    profile, r_a, r, M, G, n_s: int = _JEANS_N_S_DEFAULT, beta_fn=None
) -> DispersionProfile:
    """3-D anisotropic Jeans dispersion of ``profile`` under OM ``r_a``.

    Returns the equilibrium ``(sigma_r, sigma_t, sigma_1d, beta)`` of the
    spatial ``profile`` (which owns rho, M, Phi) for an Osipkov-Merritt (1985)
    anisotropy radius ``r_a`` (``None`` -> isotropic). The radial dispersion is
    the anisotropic Jeans solution (master tables from :func:`_jeans_tables`,
    evaluated by :func:`_sigma_r2_from_tables`); the tangential / 1-D components
    and ``beta`` follow from OM (:func:`_sigma_components`).

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

    General-``beta(r)`` anisotropy (Tier A, Phase 0.5 D1). Pass a callable
    ``beta_fn(r) -> beta`` to use an ARBITRARY anisotropy profile (native
    Michie/King anisotropy, custom ``beta(r)``) instead of the OM law, via the
    general integrating factor ``f(r) = exp(2 int beta(s)/s ds)``::

        rho sigma_r^2(r) = (1/f(r)) int_r^inf f(s) rho(s) G M(<s)/s^2 ds
        sigma_t^2 = (1 - beta) sigma_r^2 ;  sigma_1d^2 = (sigma_r^2 + 2 sigma_t^2)/3

    ``f`` is built numerically (max-subtracted log form for stability); the OM
    special case ``f = r^2 + r_a^2`` (``beta = r^2/(r^2 + r_a^2)``) is the analytic
    default and is BIT-PRESERVED when ``beta_fn is None``. When ``beta_fn`` is
    given, ``r_a`` is unused and the Plummer-OM ``r_a >= 0.75 a`` validity guard
    (an OM-only domain) is skipped. Note: the ``beta_fn`` path carries an O(h^2)
    error from the numerically-built integrating factor (the trapezoid
    ``F = 2 int beta/s ds``), whereas the analytic-OM default has none.

    The enclosed mass ``M(<s)`` is a *quadrature of ``profile.density``*
    (builder-quality, no re-differentiated Psi): ``M_enc = M * cumtrap(rho s^2)
    / cumtrap_total(rho s^2)``. A fine radial s-grid runs to ``profile.r_t`` if
    present (King/EFF); for Plummer (no finite cutoff) the semi-infinite domain is
    algebraically compactified via ``s = a t/(1-t)`` (:func:`_jeans_tables`), so the
    full outward tail is integrated (no truncation bias).

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
    beta_fn : optional callable ``beta_fn(r) -> beta`` for general anisotropy. When
        given, the numerical integrating-factor path is used (``r_a`` ignored);
        ``None`` (default) keeps the bit-preserved analytic OM/isotropic path.

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
        beta_fn is None
        and r_a is not None
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

    # One master anisotropic-Jeans solve (R-independent tables), then evaluate
    # sigma_r^2 at the query radii. Bit-identical to the old inlined
    # rho/M_enc/s -> I_outward path — same grid, integrand, I_outward,
    # prefactor, and interpolation, just factored into reusable helpers.
    s, rho, I_outward, F_shifted = _jeans_tables(profile, r_a, M, G, n_s, beta_fn=beta_fn)
    sigma_r2 = jnp.maximum(
        _sigma_r2_from_tables(r, s, rho, I_outward, r_a, F_shifted=F_shifted), 0.0
    )
    beta = beta_fn(r) if beta_fn is not None else None
    sr, st, s1d, beta = _sigma_components(sigma_r2, r, r_a, beta=beta)
    return DispersionProfile(r=r, sigma_r=sr, sigma_t=st, sigma_1d=s1d, beta=beta)


def df_moment_dispersion(
    df, r, M, G, n_w: int = 256, n_alpha: int = 128
) -> DispersionProfile:
    """Exact second-moment dispersion of the anisotropic Michie-King DF (Tier B).

    Computes ``(sigma_r, sigma_t, sigma_1d, beta)`` by directly integrating the
    Michie (1963) velocity DF's second velocity moments at each radius — the
    *native* Michie equilibrium, correct at ALL radii (unlike
    :func:`jeans_dispersion` with an OM ``r_a``, which imposes the OM anisotropy
    law and is valid only in the inner region where OM approximates the Michie
    law). This is the DF-side truth that the Michie sampler draws from.

    Physics (verified vs ``michie_df.py`` / ``_michie_beta_oracle``; Michie 1963,
    King 1966). With dimensionless speed ``u = v/sigma``, at radius ``r``::

        W(r)  = interp(r/r_c, xi_grid, psi_grid, left=W0, right=0), clamped >= 0
        s     = r/r_a
        sigma = sqrt(G M / (9 r_c mu))     (self-consistent Michie velocity scale)
        f~(u_r, u_t) = exp(-s^2 u_t^2/2) [exp(W - (u_r^2+u_t^2)/2) - 1]
                       on the bound region u_r^2 + u_t^2 <= 2W.

    The bound region is integrated by a fixed-domain POLAR quadrature that needs
    NO boundary mask (fully differentiable): ``u_r = w cos(alpha)``,
    ``u_t = w sin(alpha)`` with ``w in [0, sqrt(2W)]`` and ``alpha in [0, pi]``
    (the energy bound becomes exactly the ``w`` upper limit). The velocity-space
    measure ``d^3u = 2 pi u_t du_r du_t = 2 pi w^2 sin(alpha) dw dalpha`` — the
    ``2 pi`` (and any constant) cancels in the moment RATIOS, so the integration
    weight is simply ``w^2 sin(alpha)``::

        <u_r^2> = int f~ (w cos a)^2 w^2 sin a / int f~ w^2 sin a
        <u_t^2> = int f~ (w sin a)^2 w^2 sin a / int f~ w^2 sin a

    Then ``sigma_r = sigma sqrt(<u_r^2>)``, ``sigma_t = sigma sqrt(<u_t^2>/2)``
    (since ``<v_t^2> = 2 sigma_t^2``), and::

        beta     = 1 - sigma_t^2/sigma_r^2 = 1 - <u_t^2>/(2 <u_r^2>)
        sigma_1d = sqrt((sigma_r^2 + 2 sigma_t^2)/3).

    The ``beta`` here matches ``_michie_beta_oracle``'s ``1 - ut2/(2 ur2)`` exactly.
    A fixed ``w_hat in [0,1]`` grid is scaled per-``r`` by ``sqrt(2W)`` (clamped
    so ``W <= 0`` outside the system gives ``sigma_r = sigma_t -> 0`` naturally —
    the integrand vanishes); the per-``r`` 2-D moment is ``jax.vmap``ed over ``r``.
    Fully reverse-mode differentiable; ``jnp.trapezoid`` (integrate ``alpha`` then
    ``w``). Zero new deps.

    Parameters
    ----------
    df : :class:`~progenax.kinematics.MichieVelocityDF` exposing ``W0``, ``r_c``,
        ``r_a``, ``xi_grid``, ``psi_grid``, ``mu``.
    r : query radii (array-like; broadcast to at least 1-D).
    M : total mass setting the self-consistent velocity scale ``sigma``.
    G : gravitational constant (sets the velocity units).
    n_w, n_alpha : quadrature resolutions in ``w`` (speed) and ``alpha`` (angle).

    Returns
    -------
    DispersionProfile
    """
    r = jnp.atleast_1d(jnp.asarray(r))

    sigma = jnp.sqrt(G * M / (9.0 * df.r_c * df.mu))

    # Fixed reference grids (shared across radii). w_hat in [0,1] is scaled by
    # sqrt(2W) per-r; alpha in [0, pi]. sin(alpha) supplies the polar measure.
    w_hat = jnp.linspace(0.0, 1.0, n_w)
    alpha = jnp.linspace(0.0, jnp.pi, n_alpha)
    sin_a = jnp.sin(alpha)
    cos_a = jnp.cos(alpha)

    def _moments_at_r(r_i):
        W = jnp.interp(r_i / df.r_c, df.xi_grid, df.psi_grid, left=df.W0, right=0.0)
        W = jnp.maximum(W, 0.0)
        s = r_i / df.r_a
        # wmax = sqrt(2W); clamp W against 0 so wmax stays finite (the integrand
        # below -> 0 as W -> 0, so the moments tend to 0 outside the system).
        wmax = jnp.sqrt(2.0 * jnp.maximum(W, 1e-30))
        w = w_hat * wmax  # (n_w,) speed grid for this radius

        # 2-D grids: rows = w, cols = alpha. u_r = w cos a, u_t = w sin a.
        WW = w[:, None]
        SA = sin_a[None, :]
        CA = cos_a[None, :]
        u_r = WW * CA
        u_t = WW * SA
        u2 = u_r**2 + u_t**2  # = w^2 (<= 2W by construction)

        # f~(u_r, u_t) on the bound region (the w upper-limit IS the energy bound,
        # so no mask). exp(W - u2/2) - 1 >= 0 there; clamp tiny negatives from
        # round-off at the w=wmax edge.
        f_tilde = jnp.exp(-(s**2) * u_t**2 / 2.0) * jnp.maximum(
            jnp.exp(W - u2 / 2.0) - 1.0, 0.0
        )
        weight = w[:, None] ** 2 * SA  # polar measure w^2 sin(alpha)
        base = f_tilde * weight

        # Integrate alpha (axis=1) then w (axis=0).
        norm = jnp.trapezoid(jnp.trapezoid(base, alpha, axis=1), w)
        num_r = jnp.trapezoid(jnp.trapezoid(base * u_r**2, alpha, axis=1), w)
        num_t = jnp.trapezoid(jnp.trapezoid(base * u_t**2, alpha, axis=1), w)
        norm_safe = jnp.maximum(norm, 1e-300)
        return num_r / norm_safe, num_t / norm_safe

    ur2_mean, ut2_mean = jax.vmap(_moments_at_r)(r)

    sigma_r2 = sigma**2 * ur2_mean
    sigma_t2 = 0.5 * sigma**2 * ut2_mean  # <v_t^2> = 2 sigma_t^2

    # NaN-safe outer sqrt: at/beyond r_t the clamped W -> 0 makes sigma_*2 == 0
    # exactly, where sqrt has derivative 1/(2 sqrt(0)) = inf -> inf*0 = NaN on the
    # backward pass. A single beyond-r_t point in a jacrev/grad over a radial grid
    # would otherwise poison the entire OED Fisher to NaN. Module-level _safe_sqrt
    # (cf. michie_df.py:50,60): forward is bit-exact 0 at 0 and the gradient there
    # is a finite 0; a no-op at interior radii where the argument is strictly positive.
    sigma_r = _safe_sqrt(sigma_r2)
    sigma_t = _safe_sqrt(sigma_t2)
    sigma_1d = _safe_sqrt((sigma_r2 + 2.0 * sigma_t2) / 3.0)
    beta = 1.0 - ut2_mean / (2.0 * jnp.maximum(ur2_mean, 1e-300))
    return DispersionProfile(
        r=r, sigma_r=sigma_r, sigma_t=sigma_t, sigma_1d=sigma_1d, beta=beta
    )


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

        int_R^inf g(r) r/sqrt(r^2-R^2) dr = int_0^Umax g(sqrt(R^2+u^2)) du

    so NO ``1/sqrt(r^2-R^2)`` is ever evaluated — only a smooth trapezoid quadrature
    in ``u`` (``Umax = sqrt(r_t^2 - R^2)`` for finite ``r_t``; ``Umax = inf`` for
    Plummer, see below). ``sigma_r^2(r)`` is read from the ONE master Jeans solve —
    :func:`_jeans_tables` is built once (hoisted out of the per-``R`` vmap) and
    :func:`_sigma_r2_from_tables` interpolates it per integration radius — and
    ``beta(r)`` is the closed-form OM law; ``rho`` from ``profile.density``.

    The outward ``u``-grid is branched on ``profile.r_t``:

    - **Finite ``r_t`` (King / EFF):** a uniform ``u`` grid on
      ``[0, sqrt(r_t^2 - R^2)]`` (the density has a hard cutoff, so the integral is
      naturally finite). UNCHANGED.
    - **No ``r_t`` (Plummer):** the ``u``-integral is SEMI-INFINITE. The grid is
      algebraically compactified, ``u = u_c tau/(1-tau)`` with ``u_c = profile.a`` and
      ``tau in [0, _T_MAX]`` (Jacobian ``du/dtau = u_c/(1-tau)^2`` folded into every
      trapezoid), so the full Plummer tail is integrated. This removes the former
      ``30a`` ``u``-truncation, which left an ``n_u``-independent tail floor of
      ~1.6e-4 (rel.) in ``sigma_los`` at outer ``R``; the residual is now pure
      O(h^2). (Mirrors the :func:`_jeans_tables` Task-C master-grid compactification.)

    The per-``R`` ``u``-integral is vmapped over the ``R`` array. Fully reverse-mode
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

    # Outward extent / u-grid strategy, branched on profile.r_t:
    #   - Finite r_t (King / EFF): the density has a hard cutoff, so the outward
    #     u-integral is naturally finite. Use a uniform u in [0, u_max] grid to the
    #     edge (UNCHANGED). r_edge is pulled a hair inside r_t so that the I(r)=0
    #     endpoint (where sigma_r^2=0 -> jnp.sqrt has NaN gradient) is excluded.
    #   - Plummer (no r_t): the tail is semi-infinite. The OLD code truncated the
    #     u-grid at u_max = sqrt((30a)^2 - R^2), leaving an n_u-INDEPENDENT tail
    #     floor of ~1.6e-4 (rel.) in sigma_los at outer R. We instead compactify
    #     u in [0, inf) -> tau in [0, _T_MAX] via u = u_c tau/(1-tau) (u_c = a),
    #     Jacobian du/dtau = u_c/(1-tau)^2, integrating in uniform tau so the full
    #     Plummer tail is captured. (Same family as the _jeans_tables Task-C grid.)
    r_t = getattr(profile, "r_t", None)
    is_plummer = r_t is None
    if is_plummer:
        u_c = jnp.asarray(profile.a)  # compactification scale for the u-grid
        r_max = jnp.asarray(30.0 * profile.a)  # only feeds _jeans_tables guard below
    else:
        r_max = jnp.asarray(r_t)
    # Pull the outer integration radius a hair inside r_max (finite-r_t branch). At
    # r == r_max the outward Jeans integral I(r_max) = 0 EXACTLY, so sigma_r^2 = 0
    # there and jnp.sqrt has an infinite (NaN) gradient at that zero-measure
    # endpoint — which would poison jax.grad of the whole reduction. The endpoint
    # carries rho * sigma_r^2 -> 0, i.e. ~no weight, so stopping at r_edge =
    # (1 - 1e-6) r_max removes the NaN without changing the integral (mirrors how
    # jeans_dispersion starts its own s-grid at 1e-4 r_max, not 0).
    r_edge = (1.0 - 1e-6) * r_max

    # ONE master anisotropic-Jeans solve, hoisted out of the per-R vmap. The old
    # code called jeans_dispersion (a full master solve) inside every vmapped
    # _los_quantities; here the R-independent tables (s, rho_tab, I_outward) are
    # built once and sigma_r^2 is interpolated per integration radius via
    # _sigma_r2_from_tables. This is pure code-motion: bit-identical operations
    # to the old per-R jeans_dispersion(...).sigma_r**2 path. The master Jeans
    # grid uses jeans_dispersion's own default n_s (4000) — independent of the
    # u-quadrature resolution n_u, exactly as the old per-R call did.
    s, rho_tab, I_outward, _F_shifted = _jeans_tables(profile, r_a, M, G, _JEANS_N_S_DEFAULT)

    if r_a is None:
        r_a2 = None
    else:
        r_a2 = jnp.asarray(r_a) ** 2

    def _los_quantities(R_i):
        # r^2 = R^2 + u^2 substitution removes the 1/sqrt(r^2-R^2) pole at r=R, so
        # we integrate a smooth g(sqrt(R^2+u^2)) du. Two u-grids:
        if is_plummer:
            # Plummer: compactify the SEMI-INFINITE u in [0, inf). u = u_c tau/(1-tau)
            # maps tau in [0, _T_MAX] -> u in [0, ~u_c 1e6], du/dtau = u_c/(1-tau)^2.
            # Integrate in UNIFORM tau (jnp.trapezoid sees a uniform grid) with the
            # Jacobian du/dtau folded into every integrand: int f du = int f (du/dtau) dtau.
            grid = jnp.linspace(0.0, _T_MAX, n_u)  # tau
            u = u_c * grid / (1.0 - grid)
            jac = u_c / (1.0 - grid) ** 2  # du/dtau
        else:
            # Finite r_t (King / EFF): uniform u in [0, sqrt(r_edge^2 - R^2)] to the
            # cutoff (clip non-negative if R_i brushes r_edge). du/du = 1 (no Jacobian).
            u_max = jnp.sqrt(jnp.maximum(r_edge**2 - R_i**2, 0.0))
            grid = jnp.linspace(0.0, u_max, n_u)  # u
            u = grid
            jac = jnp.ones_like(u)
        r = jnp.sqrt(R_i**2 + u**2)

        # rho (kernel weight) at the integration radii from the exact density;
        # sigma_r^2/beta straight from the master Jeans tables (interpolated),
        # NOT a fresh per-R re-solve.
        rho = profile.density(r)
        sigma_r2 = jnp.maximum(_sigma_r2_from_tables(r, s, rho_tab, I_outward, r_a), 0.0)
        if r_a is None:
            beta = jnp.zeros_like(r)
        else:
            beta = r**2 / (r**2 + r_a2)

        ratio = R_i**2 / jnp.maximum(r**2, 1e-30)  # R^2 / r^2
        w = rho * sigma_r2  # common rho*sigma_r^2 weight

        # B&M82 kernels (integrands in u; the 2x and 1/sqrt cancellation are folded
        # into the substitution -> trapezoid over the integration variable `grid`,
        # times 2). For the Plummer (tau) grid `jac` carries du/dtau; for finite
        # r_t (u grid) jac == 1, so this reduces to the unchanged uniform-u quadrature.
        Sigma = 2.0 * jnp.trapezoid(rho * jac, grid)
        S_los = 2.0 * jnp.trapezoid((1.0 - beta * ratio) * w * jac, grid)
        S_pmr = 2.0 * jnp.trapezoid((1.0 - beta + beta * ratio) * w * jac, grid)
        S_pmt = 2.0 * jnp.trapezoid((1.0 - beta) * w * jac, grid)
        return Sigma, S_los, S_pmr, S_pmt

    Sigma, S_los, S_pmr, S_pmt = jax.vmap(_los_quantities)(R)

    Sigma_safe = jnp.maximum(Sigma, 1e-30)
    sigma_los = _safe_sqrt(S_los / Sigma_safe)
    sigma_pm_r = _safe_sqrt(S_pmr / Sigma_safe)
    sigma_pm_t = _safe_sqrt(S_pmt / Sigma_safe)

    return ProjectedDispersion(
        R=R,
        sigma_los=sigma_los,
        sigma_pm_r=sigma_pm_r,
        sigma_pm_t=sigma_pm_t,
        Sigma=Sigma,
    )

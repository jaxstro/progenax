r"""Bonnor-Ebert sphere: the pressure-confined isothermal gas envelope (ADR-0062, 0066).

A Bonnor-Ebert sphere is the isothermal Lane-Emden solution truncated at a finite radius
by a confining external pressure. Unlike a polytrope it has **no zero** -- the density
falls forever -- so its edge is a physical input, not a property of the solution.

Scaling. With ``r = xi r_0`` and ``r_0 = c_s / sqrt(4 pi G rho_c)``, the enclosed mass is
``M(xi) = 4 pi rho_c r_0^3 m(xi)`` for ``m = xi^2 psi'``. Writing the confining pressure
as ``P_ext = rho_c e^{-psi} c_s^2`` makes the central density cancel:

    M = (1 / sqrt(4 pi)) * c_s^4 / (G^{3/2} P_ext^{1/2}) * [xi^2 psi' e^{-psi/2}]

The bracket -- call it ``m_BE(xi)`` -- therefore carries all the physics of the mass at
fixed confining pressure. It **rises to a single maximum and then falls**, and that
turning point *is* the critical Bonnor-Ebert sphere. :func:`critical_sphere` locates it
from our own ODE; the familiar literature numbers (xi ~ 6.451, contrast ~ 14.04,
coefficient ~ 1.18) are cross-checks on that derivation, never inputs.

Parametrization (ADR-0066). Because ``m_BE`` is non-monotonic, ``mu_BE = M/M_crit``
inverts only on the stable branch. So ``xi_max`` is the **stored primary** and ``mu_BE``
/ ``contrast`` are **derived diagnostics**; :meth:`BonnorEbertProfile.from_critical_ratio`
offers the physics-facing constructor and *raises* rather than silently choosing a branch.
Turbulent clouds are not equilibria, so formally-supercritical *shapes* stay reachable
through ``xi_max``.

The profile is normalised to **unit total mass**, so ``density`` integrates to 1 over the
sphere. Only the radial shape matters to the envelope layer, but a definite normalisation
makes ``mass_enclosed`` and the ``r_h`` inversion well posed.
"""

import functools
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jaxstro.numerics.interpolation import monotone_cubic_interp
from jaxstro.numerics.rootfinding import monotone_inverse_interp
from jaxtyping import Array, Float

from gravoturb.profiles._scaling import half_mass_xi, interp_flat
from gravoturb.profiles.lane_emden import solve_isothermal

# Domain for the critical-point search. The maximum sits near xi ~ 6.45, so 30 is
# comfortably past it while keeping the solve cheap.
_CRIT_XI_MAX = 30.0
_CRIT_N_POINTS = 40_000

_SQRT_4PI = float(jnp.sqrt(4.0 * jnp.pi))


class CriticalSphere(NamedTuple):
    """The critical Bonnor-Ebert sphere, derived from the isothermal ODE.

    Attributes:
        xi_crit: dimensionless radius maximising ``m_BE``.
        contrast: centre-to-edge density ratio there, ``e^{psi(xi_crit)}``.
        m_be_max: the maximum of ``m_BE = xi^2 psi' e^{-psi/2}``.
        mass_coefficient: ``m_be_max / sqrt(4 pi)`` -- the coefficient in
            ``M_BE = coeff * c_s^4 / (G^{3/2} P_ext^{1/2})``.
        xi: the search grid.
        m_be: ``m_BE`` on that grid.
    """

    xi_crit: Float[Array, ""]
    contrast: Float[Array, ""]
    m_be_max: Float[Array, ""]
    mass_coefficient: Float[Array, ""]
    xi: Float[Array, " n"]
    m_be: Float[Array, " n"]

    def m_be_at(self, xi_query) -> Float[Array, "..."]:
        """Interpolate ``m_BE`` at an arbitrary dimensionless radius."""
        return monotone_cubic_interp(self.xi, self.m_be, jnp.asarray(xi_query))


@functools.lru_cache(maxsize=4)
def critical_sphere(
    xi_search_max: float = _CRIT_XI_MAX, n_points: int = _CRIT_N_POINTS
) -> CriticalSphere:
    """Locate the critical Bonnor-Ebert sphere by maximising ``m_BE``.

    Cached: the result is a set of universal constants, so the solve runs once per
    process rather than once per profile construction.

    Note:
        The maximum is located by ``argmax`` on a fine grid. That is deliberate and
        safe *here* -- these are constants, never part of a gradient path. Anything
        differentiable (``mu_BE``, the ``r_h`` inversion) uses interpolation instead.
    """
    sol = solve_isothermal(xi_max=xi_search_max, n_points=n_points)
    m_be = sol.m * jnp.exp(-sol.y / 2.0)
    idx = jnp.argmax(m_be)
    return CriticalSphere(
        xi_crit=sol.xi[idx],
        contrast=jnp.exp(sol.y[idx]),
        m_be_max=m_be[idx],
        mass_coefficient=m_be[idx] / _SQRT_4PI,
        xi=sol.xi,
        m_be=m_be,
    )


class BonnorEbertProfile(eqx.Module):
    """Pressure-truncated isothermal sphere, normalised to unit total mass.

    Satisfies the duck-type the gravoturb chain requires: ``density(r)`` for the
    envelope layer and ``r_h`` for the magnetic chain.

    Args:
        r_h: half-mass radius, in the same length units as ``GeometrySpec.box_size``
            (traced/differentiable).
        xi_max: dimensionless truncation radius -- the stored shape primary (STATIC).
        n_points: ODE grid size (STATIC).

    Attributes:
        r_0: physical scale radius, back-solved so the half-mass radius is ``r_h``.
        rho_c: central density (set by the unit-mass normalisation).
        r_edge: outer truncation radius, ``xi_max * r_0``.
        total_mass: 1.0 by construction; kept explicit so callers need not assume it.
    """

    r_h: Float[Array, ""]
    xi_max: float = eqx.field(static=True)
    n_points: int = eqx.field(static=True)

    xi_grid: Float[Array, " n"]
    psi_grid: Float[Array, " n"]
    m_grid: Float[Array, " n"]

    r_0: Float[Array, ""]
    rho_c: Float[Array, ""]
    r_edge: Float[Array, ""]
    total_mass: Float[Array, ""]

    def __init__(self, r_h, xi_max: float = 6.0, n_points: int = 2000):
        if float(xi_max) <= 0.0:
            raise ValueError(f"xi_max must be positive, got {xi_max}")
        if float(r_h) <= 0.0:
            raise ValueError(f"r_h must be positive, got {r_h}")

        self.xi_max = float(xi_max)
        self.n_points = int(n_points)
        self.r_h = jnp.asarray(r_h, dtype=float)

        sol = solve_isothermal(xi_max=self.xi_max, n_points=self.n_points)
        self.xi_grid, self.psi_grid, self.m_grid = sol.xi, sol.y, sol.m

        # Half-mass inversion: bracket with the linear monotone inverse, which is only
        # first-order accurate, then refine against the PCHIP m(xi) (ADR-0067).
        m_total = sol.m[-1]
        xi_h = half_mass_xi(sol.xi, sol.m, context=f"xi_max={self.xi_max}")

        self.r_0 = self.r_h / xi_h
        self.r_edge = self.xi_max * self.r_0
        self.total_mass = jnp.asarray(1.0)
        # M_tot = 4 pi rho_c r_0^3 m(xi_max) = 1
        self.rho_c = 1.0 / (4.0 * jnp.pi * self.r_0**3 * m_total)

    def density(self, r: Float[Array, "..."]) -> Float[Array, "..."]:
        """Mass density at radius ``r``; exactly zero beyond ``r_edge``."""
        r = jnp.asarray(r)
        xi = r / self.r_0
        xi_clipped = jnp.clip(xi, self.xi_grid[0], self.xi_grid[-1])
        psi = interp_flat(self.xi_grid, self.psi_grid, xi_clipped)
        return jnp.where(xi <= self.xi_max, self.rho_c * jnp.exp(-psi), 0.0)

    def mass_enclosed(self, r: Float[Array, "..."]) -> Float[Array, "..."]:
        """Mass interior to ``r``, saturating at ``total_mass`` beyond ``r_edge``."""
        r = jnp.asarray(r)
        xi = jnp.clip(r / self.r_0, self.xi_grid[0], self.xi_grid[-1])
        m = interp_flat(self.xi_grid, self.m_grid, xi)
        return 4.0 * jnp.pi * self.rho_c * self.r_0**3 * m

    @property
    def contrast(self) -> Float[Array, ""]:
        """Centre-to-edge density ratio ``rho_c / rho_edge`` (derived diagnostic)."""
        return jnp.exp(self.psi_grid[-1])

    @property
    def mu_BE(self) -> Float[Array, ""]:
        """Mass in units of the critical Bonnor-Ebert mass (derived diagnostic).

        ``mu_BE = 1`` marks the critically stable sphere -- the thermal counterpart of
        ``mu_phi = 1`` in the magnetic layer.
        """
        crit = critical_sphere()
        m_be = self.m_grid[-1] * jnp.exp(-self.psi_grid[-1] / 2.0)
        return m_be / crit.m_be_max

    def characteristic_radius(self) -> Float[Array, ""]:
        """Half-mass radius (the ``SpatialProfile`` characteristic scale)."""
        return self.r_h

    @classmethod
    def from_critical_ratio(cls, mu_BE: float, r_h, n_points: int = 2000):
        """Build the subcritical sphere whose mass is ``mu_BE`` times the critical mass.

        Raises:
            ValueError: if ``mu_BE`` is outside ``(0, 1]``. Above 1 the inverse is
                genuinely ambiguous -- ``m_BE`` is non-monotonic -- so this refuses
                rather than silently selecting a branch (ADR-0066). Pass ``xi_max``
                directly if a formally-supercritical *shape* is what you want.
        """
        mu = float(mu_BE)
        if not (0.0 < mu <= 1.0):
            raise ValueError(
                f"mu_BE must lie in (0, 1]; got {mu}. The Bonnor-Ebert mass is "
                "non-monotonic in xi_max, so mu_BE inverts uniquely only on the stable "
                "branch. For a supercritical shape, construct with xi_max directly."
            )
        crit = critical_sphere()
        stable = crit.xi <= crit.xi_crit
        xi_stable = crit.xi[stable]
        m_be_stable = crit.m_be[stable]
        xi_max = monotone_inverse_interp(xi_stable, m_be_stable, mu * crit.m_be_max)
        return cls(r_h=r_h, xi_max=float(xi_max), n_points=n_points)

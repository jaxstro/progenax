r"""Polytropic gas envelope, ``P = K rho^gamma`` (ADR-0062, ADR-0065).

Where a Bonnor-Ebert sphere needs an external pressure to have an edge, a polytrope of
index ``n < 5`` **self-truncates**: ``theta`` reaches zero at a finite ``xi_1``, and the
density vanishes continuously there. That different termination rule is why this is a
separate class rather than a parameter setting of the isothermal solver.

The stored knob is the **adiabatic index** ``gamma`` -- the thing an equation of state
actually gives you -- with ``n = 1/(gamma - 1)`` derived. Making ``gamma`` a first-class,
traced parameter is the point: it turns the equation of state from a hardcoded assumption
into a *testable* one.

Finite radius requires ``n < 5``, i.e. ``gamma > 1.2``. At or below that the sphere is
infinite in extent and would need Bonnor-Ebert-style pressure truncation, so the
constructor refuses and says so rather than quietly cutting the profile off somewhere.

Like :class:`~gravoturb.profiles.bonnor_ebert.BonnorEbertProfile`, this is normalised to
unit total mass and exposes ``density(r)`` + ``r_h`` for the gravoturb chain.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from gravoturb.profiles._scaling import half_mass_xi, interp_flat, require
from gravoturb.profiles.lane_emden import polytrope_xi1, solve_polytrope

# gamma = 1.2 is exactly n = 5, the onset of infinite extent.
GAMMA_MIN = 1.2


class PolytropeProfile(eqx.Module):
    """Self-truncating polytropic sphere, normalised to unit total mass.

    Args:
        r_h: half-mass radius (traced/differentiable).
        gamma: adiabatic index; must exceed :data:`GAMMA_MIN` for a finite radius.
        n_points: ODE grid size (STATIC).

    Attributes:
        n: polytropic index ``1/(gamma - 1)``.
        xi_1: dimensionless first zero of ``theta`` -- the outer edge.
        r_0: physical scale radius, back-solved so the half-mass radius is ``r_h``.
        rho_c: central density (set by the unit-mass normalisation).
        r_edge: outer radius, ``xi_1 * r_0``.
        total_mass: 1.0 by construction.
    """

    r_h: Float[Array, ""]
    gamma: Float[Array, ""]
    n_points: int = eqx.field(static=True)

    n: Float[Array, ""]
    xi_1: Float[Array, ""]

    xi_grid: Float[Array, " n"]
    theta_grid: Float[Array, " n"]
    m_grid: Float[Array, " n"]

    r_0: Float[Array, ""]
    rho_c: Float[Array, ""]
    r_edge: Float[Array, ""]
    total_mass: Float[Array, ""]

    def __init__(self, r_h, gamma, n_points: int = 2000):
        require(
            jnp.asarray(gamma) > GAMMA_MIN,
            (
                f"gamma must exceed {GAMMA_MIN} for a finite radius; got {gamma}. "
                f"gamma <= {GAMMA_MIN} is n >= 5, where the polytrope has infinite extent "
                "and needs pressure truncation. For the isothermal limit (gamma -> 1) use "
                "BonnorEbertProfile, which truncates on external pressure by construction."
            ),
        )
        require(jnp.asarray(r_h) > 0.0, f"r_h must be positive, got {r_h}")

        self.n_points = int(n_points)
        self.r_h = jnp.asarray(r_h, dtype=float)
        self.gamma = jnp.asarray(gamma, dtype=float)
        self.n = 1.0 / (self.gamma - 1.0)

        # Locate the edge by event detection (differentiable in n), then tabulate on it.
        self.xi_1 = polytrope_xi1(self.n)
        sol = solve_polytrope(self.n, xi_max=self.xi_1, n_points=self.n_points)
        self.xi_grid, self.theta_grid, self.m_grid = sol.xi, sol.y, sol.m

        m_total = sol.m[-1]
        xi_h = half_mass_xi(sol.xi, sol.m, sol.dm)

        self.r_0 = self.r_h / xi_h
        self.r_edge = self.xi_1 * self.r_0
        self.total_mass = jnp.asarray(1.0)
        self.rho_c = 1.0 / (4.0 * jnp.pi * self.r_0**3 * m_total)

    def density(self, r: Float[Array, "..."]) -> Float[Array, "..."]:
        """Mass density at radius ``r``; exactly zero beyond ``r_edge``."""
        r = jnp.asarray(r)
        xi = r / self.r_0
        xi_clipped = jnp.clip(xi, self.xi_grid[0], self.xi_grid[-1])
        theta = interp_flat(self.xi_grid, self.theta_grid, xi_clipped)
        rho = self.rho_c * jnp.maximum(theta, 0.0) ** self.n
        return jnp.where(xi <= self.xi_1, rho, 0.0)

    def mass_enclosed(self, r: Float[Array, "..."]) -> Float[Array, "..."]:
        """Mass interior to ``r``, saturating at ``total_mass`` beyond ``r_edge``."""
        r = jnp.asarray(r)
        xi = jnp.clip(r / self.r_0, self.xi_grid[0], self.xi_grid[-1])
        m = interp_flat(self.xi_grid, self.m_grid, xi)
        return 4.0 * jnp.pi * self.rho_c * self.r_0**3 * m

    def characteristic_radius(self) -> Float[Array, ""]:
        """Half-mass radius (the ``SpatialProfile`` characteristic scale)."""
        return self.r_h

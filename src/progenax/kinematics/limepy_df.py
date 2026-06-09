"""
General-g LIMEPY lowered-isothermal velocity distribution function (Equinox module).

Samples the LIMEPY DF (Gieles & Zocchi 2015) in detailed equilibrium with the
general-g potential. The local speed distribution at dimensionless potential W(r) is

    g(v) ∝ v^2 E_gamma(g, W - v^2/2s^2),   0 <= v <= v_esc = s sqrt(2W),

with E_gamma the lowered exponential (Eq. 2; E_gamma(g, y) = e^y P(g, y), or e^y for
g=0) and s the central velocity scale fixed self-consistently from the model,

    s^2 = G M_total / (9 r_c mu(W0, g)),   mu(W0, g) = int_0^{xi_t} rho_tilde(xi) xi^2 dxi,

the same King-radius (factor-of-9) nondimensionalization as KingVelocityDF (LIMEPY
uses King's r_s by construction). With this s the sampled cluster is in virial
equilibrium (Q = T/|V| = 0.5) WITHOUT any external rescale, for every truncation g.

At g=1, E_gamma(1, x) = e^x - 1 and this reduces exactly to KingVelocityDF.

Sampling uses a per-particle tabulated inverse-CDF (vmap'd, differentiable, no
while_loop) and isotropic directions. Isotropic only; anisotropy is a planned
extension (the Michie/OM J^2 term).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from progenax import defaults
from progenax.profiles.king import _find_tidal_radius
from progenax.profiles.limepy import (
    limepy_density_hat,
    lowered_exponential,
    solve_limepy_profile,
)

# Resolution of the per-particle speed inverse-CDF table.
_N_SPEED_GRID = 256


def _sample_unit_speed(
    key: PRNGKeyArray, W: Float[Array, ""], g: Float[Array, ""], n_u: int
) -> Float[Array, ""]:
    """Sample one normalized speed u ~ g(u) = u^2 E_gamma(g, W - u^2/2) on [0, sqrt(2W)].

    u is in units of s; the physical speed is s * u. Differentiable inverse-CDF
    sampling on a fixed-size grid. Returns 0 where W <= 0 (escape speed vanishes at
    the truncation radius). At g=1 the weight is u^2(exp(W - u^2/2) - 1), the King DF.
    """
    W_safe = jnp.maximum(W, 1e-12)
    u_grid = jnp.linspace(0.0, jnp.sqrt(2.0 * W_safe), n_u)
    weight = u_grid**2 * lowered_exponential(g, W_safe - u_grid**2 / 2.0)
    weight = jnp.maximum(weight, 0.0)

    du = u_grid[1] - u_grid[0]
    cdf = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (weight[1:] + weight[:-1])) * du])
    cdf = cdf / (cdf[-1] + 1e-30)

    q = jax.random.uniform(key)
    u = jnp.interp(q, cdf, u_grid)
    return jnp.where(W > 1e-6, u, 0.0)


class LIMEPYVelocityDF(eqx.Module):
    """General-g LIMEPY lowered-isothermal velocity DF (isotropic).

    A true equilibrium DF: velocities are drawn from the lowered exponential whose
    radial speed structure follows the LIMEPY potential W(r), with the central
    velocity scale s fixed self-consistently from (G, M_total, r_c, W0, g). The ICs
    are in virial equilibrium (Q = 0.5) without rescaling, and all stars are bound.
    Generalizes KingVelocityDF with a continuous truncation parameter g.

    Attributes:
        W0: Dimensionless central potential.
        g:  Truncation parameter (g=0 Woolley, g=1 King, g=2 Wilson).
        r_c: Core (King) radius [length units].
        r_t: Truncation radius [length units] (derived from W(xi) -> 0).
        xi_grid, psi_grid: ODE solution W(xi).
    """

    W0: Float[Array, ""]
    g: Float[Array, ""]
    r_c: Float[Array, ""]
    r_t: Float[Array, ""]
    xi_grid: Float[Array, "n_ode"]
    psi_grid: Float[Array, "n_ode"]

    def __init__(
        self,
        W0: float = 5.0,
        g: float = 1.0,
        r_c: float = 1.0,
        xi_max: float = 300.0,
        n_ode_points: int = 2000,
    ):
        self.W0 = jnp.asarray(W0, dtype=jnp.float64)
        self.g = jnp.asarray(g, dtype=jnp.float64)
        self.r_c = jnp.asarray(r_c, dtype=jnp.float64)
        xi_grid, psi_grid = solve_limepy_profile(W0, g, xi_max=xi_max, n_points=n_ode_points)
        self.xi_grid = xi_grid
        self.psi_grid = psi_grid
        self.r_t = self.r_c * _find_tidal_radius(xi_grid, psi_grid)

    def _s(self, M_total: Float[Array, ""], G: float) -> Float[Array, ""]:
        """Self-consistent central velocity scale s = sqrt(G M / (9 r_c mu(W0, g)))."""
        rho0 = limepy_density_hat(self.W0, self.g)
        rho_tilde = jnp.where(
            rho0 > 1e-300, limepy_density_hat(self.psi_grid, self.g) / rho0, 0.0
        )
        mu = jnp.trapezoid(rho_tilde * self.xi_grid**2, self.xi_grid)
        return jnp.sqrt(G * M_total / (9.0 * self.r_c * mu))

    def sample_velocities(
        self,
        positions: Float[Array, "N 3"],
        masses: Float[Array, "N"],
        key: PRNGKeyArray,
        G: float | None = None,
    ) -> Float[Array, "N 3"]:
        """Sample velocities from the general-g lowered DF.

        At each radius the speed is drawn from g(v) ∝ v^2 E_gamma(g, W(r) - v^2/2s^2)
        on [0, v_esc(r)] via a differentiable tabulated inverse-CDF; directions are
        isotropic. Differentiable in (W0, g, r_c) through the model.
        """
        if G is None:
            G = defaults.DEFAULT_UNITS.G

        N = positions.shape[0]
        M_total = jnp.sum(masses)
        radii = jnp.linalg.norm(positions, axis=1)

        W = jnp.interp(radii / self.r_c, self.xi_grid, self.psi_grid, left=self.W0, right=0.0)
        W = jnp.maximum(W, 0.0)
        s = self._s(M_total, G)

        key_speed, key_dir = jax.random.split(key)
        speed_keys = jax.random.split(key_speed, N)
        u = jax.vmap(lambda k, w: _sample_unit_speed(k, w, self.g, _N_SPEED_GRID))(speed_keys, W)
        speeds = s * u

        dirs = jax.random.normal(key_dir, shape=(N, 3))
        dirs = dirs / (jnp.linalg.norm(dirs, axis=1, keepdims=True) + 1e-30)
        return speeds[:, None] * dirs


__all__ = ["LIMEPYVelocityDF"]

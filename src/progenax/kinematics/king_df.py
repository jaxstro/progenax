"""
King (1966) lowered-Maxwellian velocity distribution function (Equinox module).

This samples the *true* King DF in detailed equilibrium with the King potential:

    f(E) ∝ exp(E/sigma^2) - 1,   E = psi(r) sigma^2 - v^2/2 > 0,

so the speed distribution at radius r is

    g(v) ∝ v^2 [ exp(psi(r) - v^2/2sigma^2) - 1 ],   0 <= v <= v_esc = sigma sqrt(2 psi(r)),

with psi(r) the dimensionless King potential (from the ODE) and sigma the central
velocity scale fixed self-consistently from the model,

    sigma^2 = G M_total / (9 r_c mu(W0)),   mu(W0) = int_0^{xi_t} rho_tilde(xi) xi^2 dxi

(King 1966; Binney & Tremaine 2008, Eq. 4.131; the factor of 9 follows the standard
nondimensionalization where r_c is the King core radius). With this sigma the sampled
cluster is in virial equilibrium (Q = T/|V| = 0.5) WITHOUT any external rescale.

Sampling uses a per-particle tabulated inverse-CDF (jnp.lax-free, vmap'd, differentiable;
no while_loop), and isotropic directions.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from progenax import defaults
from progenax.profiles.king import (
    king_lowered_maxwellian_density,
    solve_king_profile,
)

# Resolution of the per-particle speed inverse-CDF table.
_N_SPEED_GRID = 256


def _sample_unit_speed(key: PRNGKeyArray, W: Float[Array, ""], n_u: int) -> Float[Array, ""]:
    """Sample one normalized speed u ~ g(u) = u^2 (exp(W - u^2/2) - 1) on [0, sqrt(2W)].

    u is in units of sigma; the physical speed is sigma * u. Differentiable inverse-CDF
    sampling on a fixed-size grid. Returns 0 where W <= 0 (at/outside the tidal radius,
    where the escape speed vanishes).
    """
    W_safe = jnp.maximum(W, 1e-12)
    u_grid = jnp.linspace(0.0, jnp.sqrt(2.0 * W_safe), n_u)
    g = u_grid**2 * (jnp.exp(W_safe - u_grid**2 / 2.0) - 1.0)
    g = jnp.maximum(g, 0.0)  # numerically non-negative on [0, sqrt(2W)]

    du = u_grid[1] - u_grid[0]
    cdf = jnp.concatenate(
        [jnp.zeros(1), jnp.cumsum(0.5 * (g[1:] + g[:-1])) * du]
    )
    cdf = cdf / (cdf[-1] + 1e-30)

    q = jax.random.uniform(key)
    u = jnp.interp(q, cdf, u_grid)
    return jnp.where(W > 1e-6, u, 0.0)


class KingVelocityDF(eqx.Module):
    """
    King (1966) lowered-Maxwellian velocity distribution function.

    A true equilibrium DF: velocities are sampled from the lowered Maxwellian whose
    radial speed structure follows the King potential psi(r), with the central velocity
    scale sigma fixed self-consistently from (G, M_total, r_c, W0). The resulting ICs are
    in virial equilibrium (Q = 0.5) without external rescaling, and all particles are
    bound (v < v_esc(r)).

    Attributes:
        W0: King concentration parameter (dimensionless central potential)
        r_c: Core radius [length units] (the King core radius)
        r_t: Tidal radius [length units]
        xi_grid, psi_grid: ODE solution of the King model (xi = r/r_c, psi(xi))

    References:
        King (1966), AJ, 71, 64
        Binney & Tremaine (2008), "Galactic Dynamics", 2nd ed., Eq. 4.131
    """

    W0: Float[Array, ""]
    r_c: Float[Array, ""]
    r_t: Float[Array, ""]
    xi_grid: Float[Array, "n_ode"]
    psi_grid: Float[Array, "n_ode"]

    def __init__(
        self,
        W0: float = 5.0,
        r_c: float = 1.0,
        r_t: float = 10.0,
        xi_max: float = 300.0,
        n_ode_points: int = 2000,
    ):
        self.W0 = jnp.asarray(W0)
        self.r_c = jnp.asarray(r_c)
        self.r_t = jnp.asarray(r_t)
        xi_grid, psi_grid = solve_king_profile(W0, xi_max=xi_max, n_points=n_ode_points)
        self.xi_grid = xi_grid
        self.psi_grid = psi_grid

    def _sigma(self, M_total: Float[Array, ""], G: float) -> Float[Array, ""]:
        """Self-consistent central velocity scale sigma = sqrt(G M / (9 r_c mu(W0)))."""
        rho0 = king_lowered_maxwellian_density(self.W0)
        rho_tilde = jnp.where(
            rho0 > 1e-10, king_lowered_maxwellian_density(self.psi_grid) / rho0, 0.0
        )
        # mu(W0) = int rho_tilde xi^2 dxi (rho_tilde -> 0 beyond xi_t, so the full grid works)
        mu = jnp.trapezoid(rho_tilde * self.xi_grid**2, self.xi_grid)
        sigma_sq = G * M_total / (9.0 * self.r_c * mu)
        return jnp.sqrt(sigma_sq)

    def sample_velocities(
        self,
        positions: Float[Array, "N 3"],
        masses: Float[Array, "N"],
        key: PRNGKeyArray,
        G: float | None = None,
    ) -> Float[Array, "N 3"]:
        """
        Sample velocities from the King lowered-Maxwellian DF.

        At each radius r, the speed is drawn from g(v) ∝ v^2 [exp(psi(r) - v^2/2sigma^2) - 1]
        on [0, v_esc(r)] via a differentiable tabulated inverse-CDF; directions are isotropic.

        Args:
            positions: Particle positions (N, 3) [length units]
            masses: Particle masses (N,) [M_sun]
            key: JAX random key
            G: Gravitational constant. If None, uses progenax.DEFAULT_UNITS.G.

        Returns:
            Cartesian velocities (N, 3) [velocity units]
        """
        if G is None:
            G = defaults.DEFAULT_UNITS.G

        N = positions.shape[0]
        M_total = jnp.sum(masses)
        radii = jnp.linalg.norm(positions, axis=1)

        # Local dimensionless potential W(r) = psi(r) from the King ODE.
        W = jnp.interp(radii / self.r_c, self.xi_grid, self.psi_grid, left=self.W0, right=0.0)
        W = jnp.maximum(W, 0.0)

        sigma = self._sigma(M_total, G)

        key_speed, key_dir = jax.random.split(key)
        speed_keys = jax.random.split(key_speed, N)
        u = jax.vmap(lambda k, w: _sample_unit_speed(k, w, _N_SPEED_GRID))(speed_keys, W)
        speeds = sigma * u

        # Isotropic directions (normalized Gaussian vectors).
        dirs = jax.random.normal(key_dir, shape=(N, 3))
        dirs = dirs / (jnp.linalg.norm(dirs, axis=1, keepdims=True) + 1e-30)

        return speeds[:, None] * dirs


__all__ = ["KingVelocityDF"]

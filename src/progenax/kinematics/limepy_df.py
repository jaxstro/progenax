"""
General-g LIMEPY lowered-isothermal velocity distribution function (Equinox module).

Samples the LIMEPY DF (Gieles & Zocchi 2015) in detailed equilibrium with the
general-g potential, optionally with Michie/Osipkov-Merritt radial anisotropy.

Isotropic (r_a=None): the local speed distribution at dimensionless potential W(r) is

    g(v) ∝ v^2 E_gamma(g, W - v^2/2s^2),   0 <= v <= v_esc = s sqrt(2W),

with E_gamma the lowered exponential (Eq. 2). At g=1, E_gamma(1,x)=e^x-1 = the King DF.

Anisotropic (finite r_a): the DF carries the factor exp(-J^2/2 r_a^2 s^2). In the
speed-angle (u, theta) parametrization (theta = angle between v and r), the speed
marginal is g(u) ∝ u^2 E_gamma(g, W - u^2/2) T(s^2 u^2/2) (the anisotropic-density
integrand, T the bounded angle integral), and the conditional cos(theta)|u has weight
exp(-(s^2 u^2/2)(1 - cos^2 theta)). v_r = v cos(theta), v_t = v sin(theta) in a random
azimuthal direction perpendicular to r. This produces beta(r) ~ 0 in the core rising
outward. At g=1 it reduces to MichieVelocityDF.

The velocity scale s^2 = G M_total / (9 r_c mu) uses the King-radius (factor-of-9)
nondimensionalization with mu the (iso or aniso) dimensionless mass integral, so the
sampled cluster is in virial equilibrium (Q = T/|V| = 0.5) WITHOUT external rescale.

Sampling uses per-particle tabulated inverse-CDFs (vmap'd, differentiable, no
while_loop).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from progenax import defaults
from progenax.profiles.king import _find_tidal_radius
from progenax.profiles.limepy import (
    _aniso_density_scalar,
    _aniso_density_vec,
    _angle_integral_T,
    limepy_density_hat,
    lowered_exponential,
    solve_limepy_profile,
)

# Inverse-CDF table resolutions.
_N_SPEED_GRID = 256  # speed grid
_N_C = 128           # cos(theta) grid (anisotropic conditional)


def _sample_unit_speed(
    key: PRNGKeyArray, W: Float[Array, ""], g: Float[Array, ""], n_u: int
) -> Float[Array, ""]:
    """Isotropic: sample u ~ u^2 E_gamma(g, W - u^2/2) on [0, sqrt(2W)] (u = v/s)."""
    W_safe = jnp.maximum(W, 1e-12)
    u_grid = jnp.linspace(0.0, jnp.sqrt(2.0 * W_safe), n_u)
    weight = jnp.maximum(u_grid**2 * lowered_exponential(g, W_safe - u_grid**2 / 2.0), 0.0)
    du = u_grid[1] - u_grid[0]
    cdf = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (weight[1:] + weight[:-1])) * du])
    cdf = cdf / (cdf[-1] + 1e-30)
    u = jnp.interp(jax.random.uniform(key), cdf, u_grid)
    return jnp.where(W > 1e-6, u, 0.0)


def _sample_costheta_given_u(
    key: PRNGKeyArray, u: Float[Array, ""], s: Float[Array, ""], n_c: int
) -> Float[Array, ""]:
    """Anisotropic angular conditional: sample c = cos(theta) | u with weight
    exp(-(s^2 u^2/2)(1 - c^2)) via differentiable inverse-CDF (s = r/r_a, the
    per-star anisotropy parameter). The EXACT conditional step of
    `_sample_speed_angle`, shared with the table-accelerated sampler
    (`AnisoSpeedCDFTable` draws u; the angle stays exact -- cheap exp
    arithmetic, no special functions)."""
    beta_u = s**2 * u**2 / 2.0
    c_grid = jnp.linspace(-1.0, 1.0, n_c)
    w_c = jnp.maximum(jnp.exp(-beta_u * (1.0 - c_grid**2)), 0.0)
    dc = c_grid[1] - c_grid[0]
    cdf_c = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (w_c[1:] + w_c[:-1])) * dc])
    cdf_c = cdf_c / (cdf_c[-1] + 1e-30)
    return jnp.interp(jax.random.uniform(key), cdf_c, c_grid)


def _sample_speed_angle(
    key: PRNGKeyArray, W: Float[Array, ""], s: Float[Array, ""], g: Float[Array, ""],
    n_u: int, n_c: int,
):
    """Anisotropic: sample (u_r, u_t) in units of s from the Michie/OM-LIMEPY DF.

    Speed marginal g(u) ∝ u^2 E_gamma(g, W - u^2/2) T(s^2 u^2/2); conditional
    cos(theta)|u with weight exp(-(s^2 u^2/2)(1 - c^2)). Returns (u cos, u sin).
    Both via differentiable inverse-CDF. (0, 0) where W <= 0.
    """
    W_safe = jnp.maximum(W, 1e-12)
    u_grid = jnp.linspace(0.0, jnp.sqrt(2.0 * W_safe), n_u)
    E = lowered_exponential(g, W_safe - u_grid**2 / 2.0)
    beta = s**2 * u_grid**2 / 2.0
    m_u = jnp.maximum(u_grid**2 * E * _angle_integral_T(beta), 0.0)
    du = u_grid[1] - u_grid[0]
    cdf_u = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (m_u[1:] + m_u[:-1])) * du])
    cdf_u = cdf_u / (cdf_u[-1] + 1e-30)

    key_u, key_c = jax.random.split(key)
    u = jnp.interp(jax.random.uniform(key_u), cdf_u, u_grid)

    c = _sample_costheta_given_u(key_c, u, s, n_c)

    u_r = u * c
    u_t = u * jnp.sqrt(jnp.maximum(1.0 - c**2, 0.0))
    bound = W > 1e-6
    return jnp.where(bound, u_r, 0.0), jnp.where(bound, u_t, 0.0)


class LIMEPYVelocityDF(eqx.Module):
    """General-g LIMEPY lowered-isothermal velocity DF (isotropic or Michie/OM).

    A true equilibrium DF; the central velocity scale s is self-consistent from
    (G, M_total, r_c, W0, g, r_a), so ICs are virial (Q=0.5) without rescaling and
    all stars are bound. r_a=None is isotropic (generalizes KingVelocityDF); a finite
    r_a adds radial anisotropy (generalizes MichieVelocityDF). At g=1 it reduces to
    King (isotropic) / Michie-King (anisotropic).

    Attributes:
        W0, g, r_c: model parameters. r_a: anisotropy radius (inf = isotropic).
        r_t: truncation radius. xi_grid, psi_grid: ODE solution W(xi).
        mu: dimensionless mass integral int rho_tilde xi^2 dxi (sets s).
        is_aniso: static flag selecting the anisotropic sampler.
    """

    W0: Float[Array, ""]
    g: Float[Array, ""]
    r_c: Float[Array, ""]
    r_a: Float[Array, ""]
    r_t: Float[Array, ""]
    xi_grid: Float[Array, "n_ode"]
    psi_grid: Float[Array, "n_ode"]
    mu: Float[Array, ""]
    is_aniso: bool = eqx.field(static=True)

    def __init__(
        self,
        W0: float = 5.0,
        g: float = 1.0,
        r_c: float = 1.0,
        r_a: float | None = None,
        xi_max: float = 300.0,
        n_ode_points: int = 2000,
    ):
        is_aniso = r_a is not None
        self.W0 = jnp.asarray(W0, dtype=jnp.float64)
        self.g = jnp.asarray(g, dtype=jnp.float64)
        self.r_c = jnp.asarray(r_c, dtype=jnp.float64)
        self.r_a = jnp.asarray(jnp.inf if r_a is None else r_a, dtype=jnp.float64)
        ra_hat = None if r_a is None else r_a / r_c
        xi_grid, psi_grid = solve_limepy_profile(
            W0, g, ra_hat=ra_hat, xi_max=xi_max, n_points=n_ode_points
        )
        self.xi_grid = xi_grid
        self.psi_grid = psi_grid
        self.r_t = self.r_c * _find_tidal_radius(xi_grid, psi_grid)

        # mu = int rho_tilde xi^2 dxi, rho_tilde normalized to 1 at the centre.
        if is_aniso:
            rho0 = _aniso_density_scalar(self.W0, jnp.asarray(0.0), self.g)
            rho_tilde = _aniso_density_vec(psi_grid, xi_grid / ra_hat, self.g) / rho0
        else:
            rho0 = limepy_density_hat(self.W0, self.g)
            rho_tilde = jnp.where(rho0 > 1e-300, limepy_density_hat(psi_grid, self.g) / rho0, 0.0)
        self.mu = jnp.trapezoid(rho_tilde * xi_grid**2, xi_grid)
        self.is_aniso = is_aniso

    def _s(self, M_total: Float[Array, ""], G: float) -> Float[Array, ""]:
        """Self-consistent central velocity scale s = sqrt(G M / (9 r_c mu))."""
        return jnp.sqrt(G * M_total / (9.0 * self.r_c * self.mu))

    def sample_velocities(
        self,
        positions: Float[Array, "N 3"],
        masses: Float[Array, "N"],
        key: PRNGKeyArray,
        G: float | None = None,
    ) -> Float[Array, "N 3"]:
        """Sample velocities from the general-g lowered DF (iso or Michie/OM aniso).

        Differentiable in (W0, g, r_a, r_c) through the model.
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

        if self.is_aniso:
            s_local = radii / self.r_a  # = (r/r_c)/(r_a/r_c)
            u_r, u_t = jax.vmap(
                lambda k, w, sl: _sample_speed_angle(k, w, sl, self.g, _N_SPEED_GRID, _N_C)
            )(speed_keys, W, s_local)
            v_r, v_t = s * u_r, s * u_t
            # v_r along r_hat (signed); v_t in a random azimuth perpendicular to r_hat.
            r_hat = positions / (radii[:, None] + 1e-30)
            rand = jax.random.normal(key_dir, (N, 3))
            rand = rand - jnp.sum(rand * r_hat, axis=1, keepdims=True) * r_hat
            t_hat = rand / (jnp.linalg.norm(rand, axis=1, keepdims=True) + 1e-30)
            return v_r[:, None] * r_hat + v_t[:, None] * t_hat

        # Isotropic: speed * isotropic direction.
        u = jax.vmap(lambda k, w: _sample_unit_speed(k, w, self.g, _N_SPEED_GRID))(speed_keys, W)
        speeds = s * u
        dirs = jax.random.normal(key_dir, shape=(N, 3))
        dirs = dirs / (jnp.linalg.norm(dirs, axis=1, keepdims=True) + 1e-30)
        return speeds[:, None] * dirs


__all__ = ["LIMEPYVelocityDF"]

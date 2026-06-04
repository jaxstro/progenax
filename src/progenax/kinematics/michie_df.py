"""Michie-King anisotropic velocity DF (Michie 1963 anisotropy + King 1966 cutoff).

The DF f ∝ exp(-J^2/2 r_a^2 sigma^2) [exp(-E/sigma^2) - 1] is not a function of a single
integral, so (unlike Osipkov-Merritt) there is no stretch trick: velocities are drawn by a
2-D marginal-then-conditional inverse-CDF over (u_r, u_t), u = v/sigma. The self-consistent
sigma uses the same factor-of-9 relation as the King DF, with the anisotropic density.

See docs/website/99-bibliography/per-paper/michie-1963.md.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from progenax import defaults
from progenax.profiles.michie import michie_density, solve_michie_profile

_N_T = 128  # tangential inverse-CDF resolution
_N_R = 128  # radial inverse-CDF resolution


def _sample_ur_ut(key, W, s, n_t: int = _N_T, n_r: int = _N_R):
    """Sample one (u_r, u_t) from the Michie-King DF at dimensionless potential W, s=r/r_a.

    Marginal of u_t is the michie_density integrand; conditional u_r | u_t is the King
    lowered-Maxwellian [exp(W - (u_r^2+u_t^2)/2) - 1] on |u_r| < sqrt(2W - u_t^2). Both via
    differentiable inverse-CDF. Returns (0, 0) where W <= 0 (outside the system).
    """
    W_pos = jnp.maximum(W, 1e-12)
    ut_max = jnp.sqrt(2.0 * W_pos)

    # Marginal of u_t (= the rho_hat(W, s) integrand). Guard sqrt(2W - u_t^2) against its
    # zero endpoint (sqrt has infinite derivative at 0 -> NaN backward) with double-where.
    u_t = jnp.linspace(0.0, ut_max, n_t)
    arg = 2.0 * W_pos - u_t**2
    a = jnp.where(arg > 0.0, jnp.sqrt(jnp.where(arg > 0.0, arg, 1.0)), 0.0)
    inner = (
        jnp.exp(W_pos - u_t**2 / 2.0) * jnp.sqrt(2.0 * jnp.pi)
        * jax.scipy.special.erf(a / jnp.sqrt(2.0)) - 2.0 * a
    )
    m = jnp.maximum(u_t * jnp.exp(-(s**2) * u_t**2 / 2.0) * inner, 0.0)
    du = u_t[1] - u_t[0]
    cdf_t = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (m[1:] + m[:-1])) * du])
    cdf_t = cdf_t / (cdf_t[-1] + 1e-30)

    key_t, key_r = jax.random.split(key)
    ut = jnp.interp(jax.random.uniform(key_t), cdf_t, u_t)

    # Conditional u_r | u_t (same sqrt-endpoint guard).
    arg_ut = 2.0 * W_pos - ut**2
    a_ut = jnp.where(arg_ut > 0.0, jnp.sqrt(jnp.where(arg_ut > 0.0, arg_ut, 1.0)), 0.0)
    u_r = jnp.linspace(-a_ut, a_ut, n_r)
    g = jnp.maximum(jnp.exp(W_pos - (u_r**2 + ut**2) / 2.0) - 1.0, 0.0)
    dur = u_r[1] - u_r[0]
    cdf_r = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (g[1:] + g[:-1])) * dur])
    cdf_r = cdf_r / (cdf_r[-1] + 1e-30)
    ur = jnp.interp(jax.random.uniform(key_r), cdf_r, u_r)

    bound = W > 1e-6
    return jnp.where(bound, ur, 0.0), jnp.where(bound, ut, 0.0)


class MichieVelocityDF(eqx.Module):
    """Michie-King anisotropic velocity DF (radially anisotropic, lowered-Maxwellian).

    beta(r) ~ 0 at the centre, increasing outward; r_a -> infinity is the isotropic King
    DF. The velocity scale sigma is self-consistent (sigma^2 = G M / (9 r_c mu), mu the
    anisotropic dimensionless mass integral), so ICs are virial without external rescale.

    Attributes:
        W0, r_c, r_a: model parameters. xi_grid, psi_grid: Michie ODE solution.
        mu: int rho_hat(psi, xi/ra_hat) xi^2 dxi (sets sigma).

    References:
        Michie (1963), MNRAS 125, 127; King (1966), AJ 71, 64.
    """

    W0: Float[Array, ""]
    r_c: Float[Array, ""]
    r_a: Float[Array, ""]
    xi_grid: Float[Array, "n_ode"]
    psi_grid: Float[Array, "n_ode"]
    mu: Float[Array, ""]

    def __init__(
        self, W0: float = 7.0, r_c: float = 1.0, r_a: float = 10.0,
        xi_max: float = 800.0, n_ode_points: int = 3000,
    ):
        self.W0 = jnp.asarray(W0)
        self.r_c = jnp.asarray(r_c)
        self.r_a = jnp.asarray(r_a)
        xi_grid, psi_grid = solve_michie_profile(
            W0, r_a / r_c, xi_max=xi_max, n_points=n_ode_points
        )
        self.xi_grid = xi_grid
        self.psi_grid = psi_grid
        # mu = int rho_tilde xi^2 dxi (rho_tilde normalised to 1 at centre).
        rho0 = michie_density(W0, 0.0)
        s_grid = xi_grid / (r_a / r_c)
        rho_tilde = jax.vmap(michie_density)(psi_grid, s_grid) / rho0
        self.mu = jnp.trapezoid(rho_tilde * xi_grid**2, xi_grid)

    def sample_velocities(
        self, positions: Float[Array, "N 3"], masses: Float[Array, "N"],
        key: PRNGKeyArray, G: float | None = None,
    ) -> Float[Array, "N 3"]:
        """Sample velocities from the Michie-King DF via the 2-D (u_r, u_t) sampler."""
        if G is None:
            G = defaults.DEFAULT_UNITS.G

        N = positions.shape[0]
        M_total = jnp.sum(masses)
        radii = jnp.linalg.norm(positions, axis=1)

        W = jnp.interp(radii / self.r_c, self.xi_grid, self.psi_grid, left=self.W0, right=0.0)
        W = jnp.maximum(W, 0.0)
        s = radii / self.r_a
        sigma = jnp.sqrt(G * M_total / (9.0 * self.r_c * self.mu))

        key_speed, key_dir = jax.random.split(key)
        speed_keys = jax.random.split(key_speed, N)
        ur, ut = jax.vmap(lambda k, w, ss: _sample_ur_ut(k, w, ss))(speed_keys, W, s)
        v_r = sigma * ur
        v_t = sigma * ut

        # v_r along r_hat (signed), v_t in a random azimuthal direction perp to r_hat.
        r_hat = positions / (radii[:, None] + 1e-30)
        rand = jax.random.normal(key_dir, (N, 3))
        rand = rand - jnp.sum(rand * r_hat, axis=1, keepdims=True) * r_hat
        t_hat = rand / (jnp.linalg.norm(rand, axis=1, keepdims=True) + 1e-30)
        return v_r[:, None] * r_hat + v_t[:, None] * t_hat


__all__ = ["MichieVelocityDF"]

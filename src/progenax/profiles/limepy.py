# progenax/src/progenax/profiles/limepy.py
"""
General-g LIMEPY lowered-isothermal density (Gieles & Zocchi 2015).

This module generalizes the King (1966) lowered-Maxwellian to the continuous
truncation-parameter family of Gieles & Zocchi (2015, MNRAS 454, 576; "LIMEPY"),
of which Woolley (g=0), King (g=1), and Wilson (g=2) are integer members. The
single dimensionless density needed for the self-consistent Poisson solve is

    I_rho(W) = E_gamma(g + 3/2, W),

where E_gamma is the "lowered exponential" of their Eq. 2,

    E_gamma(a, x) = e^x P(a, x),   P(a, x) = gamma(a, x) / Gamma(a),

and P is the regularized lower incomplete gamma function (`jax.scipy.special.
gammainc`), which is differentiable in BOTH arguments — so the truncation index
g (entering as a = g + 3/2) carries gradients.

Index note (verified against the source, not assumed): the closed form uses
g + 3/2, established in the paper's Appendix B (Eqs. B4-B9) via the E_gamma
convolution identity (Eq. D11). The main-text Eqs. 8/11 print "g + 1/2"; this is
a typesetting slip relative to the appendix and the released `limepy` code. The
2018 erratum's corrected Eqs. 20/21 carry the same g + 3/2 normalization
(E_gamma(g + 3/2, W0)). Three independent confirmations agree:
  1. E_gamma(5/2, W) expands exactly to the King volume density
     e^W erf(sqrt W) - (2/sqrt pi) sqrt(W) (1 + 2W/3)  -> g=1 corner.
  2. The erratum normalization denominator is E_gamma(g + 3/2, W0).
  3. The convolution Eq. D11 with b = 3/2 lifts the index by exactly 3/2.

References:
    Gieles, M. & Zocchi, A. (2015), MNRAS, 454, 576 (Eqs. 1-9, App. B, D).
    Gieles & Zocchi (2018), MNRAS, 474, 3997 (erratum: Eqs. 20, 21, 41).
    King, I. R. (1966), AJ, 71, 64 (the g=1 corner).
"""

from typing import Tuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from progenax.profiles.king import _find_tidal_radius


def lowered_exponential(
    a: Float[Array, ""], W: Float[Array, "..."]
) -> Float[Array, "..."]:
    """The lowered exponential E_gamma(a, W) of Gieles & Zocchi (2015) Eq. 2:

        E_gamma(a, W) = e^W              for a = 0   (the Woolley branch),
                      = e^W P(a, W)      for a > 0,

    with P(a, W) = gammainc(a, W) the regularized lower incomplete gamma function.
    For W <= 0 the function is 0 (no stars above the escape energy). The argument
    is clamped to W >= 0 before `gammainc` so the backward pass never sees a
    negative argument.

    The density uses a = g + 3/2 >= 3/2 (always the a > 0 branch); the velocity DF
    uses a = g, which is 0 for the Woolley (g=0) corner — hence the a = 0 branch.

    Differentiable in both a and W (carries d/dg).
    """
    W_pos = jnp.where(W > 0.0, W, 0.0)
    # gammainc requires a > 0; clamp the index for the a=0 Woolley branch, then select.
    a_safe = jnp.where(a > 0.0, a, 1.0)
    val_pos_a = jnp.exp(W_pos) * jax.scipy.special.gammainc(a_safe, W_pos)
    val_zero_a = jnp.exp(W_pos)  # E_gamma(0, W) = e^W
    val = jnp.where(a > 0.0, val_pos_a, val_zero_a)
    return jnp.where(W > 0.0, val, 0.0)


def limepy_density_hat(
    W: Float[Array, "..."], g: Float[Array, ""]
) -> Float[Array, "..."]:
    """Unnormalized isotropic LIMEPY density I_rho(W) = E_gamma(g + 3/2, W).

    This is the dimensionless Poisson source (before normalization to the central
    value). At g=1 it equals the King volume density
    `king_lowered_maxwellian_density(W)` to float64 precision.

    Args:
        W: Dimensionless potential (scalar or array). W <= 0 returns 0.
        g: Truncation parameter (g=0 Woolley, g=1 King, g=2 Wilson; continuous).
           The density is finite-extent for g <= 3.5.

    Returns:
        I_rho(W), the unnormalized LIMEPY volume density. Differentiable in (W, g).
    """
    return lowered_exponential(g + 1.5, W)


# ==============================================================================
# Self-consistent Poisson solve (general-g; generalizes solve_king_profile)
# ==============================================================================


def _limepy_poisson_rhs(xi: float, y: Float[Array, "2"], args: tuple) -> Float[Array, "2"]:
    """RHS of the LIMEPY dimensionless Poisson equation (Gieles & Zocchi Eq. 5):

        d^2W/dxi^2 + (2/xi) dW/dxi = -9 rho_hat(W; g),

    with rho_hat normalized to 1 at the centre (W = W0) and xi = r/r_s the
    King-radius-scaled radius (the same factor-of-9 nondimensionalization as the
    King solver; LIMEPY uses King's r_s by construction). State y = [W, dW/dxi].

    At g=1 this is identical to `_king_poisson_rhs`. The xi=0 singularity is
    handled by L'Hopital (lim (2/xi)dW/dxi = 0 since dW/dxi(0)=0).
    """
    W0, g = args
    psi, dpsi_dxi = y[0], y[1]

    rho0 = limepy_density_hat(W0, g)
    rho_tilde = jnp.where(rho0 > 1e-300, limepy_density_hat(psi, g) / rho0, 0.0)

    d2psi_dxi2 = jnp.where(
        xi > 1e-6,
        -9.0 * rho_tilde - (2.0 / xi) * dpsi_dxi,
        -9.0 * rho_tilde,  # centre guard (dW/dxi(0)=0)
    )
    return jnp.array([dpsi_dxi, d2psi_dxi2])


def solve_limepy_profile(
    W0: float, g: float, xi_max: float = 300.0, n_points: int = 2000
) -> Tuple[Float[Array, "n_points"], Float[Array, "n_points"]]:
    """Solve the general-g LIMEPY Poisson equation with diffrax (Tsit5).

    Integrates W(xi) from the centre (W=W0, dW/dxi=0) outward to the truncation
    radius where W -> 0. Identical structure to `solve_king_profile`, with the
    continuous truncation parameter g (g=0 Woolley, g=1 King, g=2 Wilson). At
    g=1 it reproduces the King solution.

    JIT/grad-safe in (W0, g): n_points and xi_max are static (they set the output
    grid size); W0 and g may be tracers, and the ODE -> W(xi) path carries
    dW/dW0 and dW/dg (the latter through gammainc's a-derivative).

    Args:
        W0: Dimensionless central potential (King's W0 = Gieles & Zocchi's phi0).
        g: Truncation parameter (continuous; finite extent for g <= 3.5).
        xi_max: Max dimensionless radius for integration.
        n_points: Output grid size.

    Returns:
        (xi_grid, psi_grid): dimensionless radius and potential W(xi) (>= 0).
    """
    y0 = jnp.array([W0, 0.0])
    xi_span = (1e-6, xi_max)
    term = diffrax.ODETerm(_limepy_poisson_rhs)
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-10)
    saveat = diffrax.SaveAt(ts=jnp.linspace(xi_span[0], xi_span[1], n_points))

    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=xi_span[0],
        t1=xi_span[1],
        dt0=1e-4,
        y0=y0,
        args=(W0, g),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=100000,
    )
    xi_grid = solution.ts
    psi_grid = jnp.maximum(solution.ys[:, 0], 0.0)
    return xi_grid, psi_grid


# ==============================================================================
# LIMEPYProfile (SpatialProfile implementation; generalizes KingProfile)
# ==============================================================================


class LIMEPYProfile(eqx.Module):
    """General-g LIMEPY (lowered-isothermal) spherical density profile.

    Implements the SpatialProfile protocol. Generalizes `KingProfile` with a
    continuous truncation parameter g (g=0 Woolley, g=1 King, g=2 Wilson); at g=1
    it reproduces KingProfile. Isotropic (anisotropy is a planned extension).

    The CDF is precomputed at construction for inverse-transform position
    sampling, exactly as in KingProfile.

    Attributes:
        W0: Dimensionless central potential.
        g:  Truncation parameter (continuous; finite extent for g <= 3.5).
        r_c: Core (King) radius [length units].
        r_t: Truncation radius [length units], where W(r) -> 0.
        xi_grid, psi_grid: ODE solution W(xi) on a dimensionless grid.
        _r_grid, _cdf_grid: precomputed mass CDF for sampling.
    """

    W0: Float[Array, ""]
    g: Float[Array, ""]
    r_c: Float[Array, ""]
    r_t: Float[Array, ""]
    xi_grid: Float[Array, "n_points"]
    psi_grid: Float[Array, "n_points"]
    _r_grid: Float[Array, "n_grid"]
    _cdf_grid: Float[Array, "n_grid"]

    def __init__(self, W0, g, r_c, r_t, xi_grid, psi_grid, n_grid: int = 1000):
        W0_arr = jnp.asarray(W0, dtype=jnp.float64)
        g_arr = jnp.asarray(g, dtype=jnp.float64)
        r_c_arr = jnp.asarray(r_c, dtype=jnp.float64)
        r_t_arr = jnp.asarray(r_t, dtype=jnp.float64)
        xi_grid_arr = jnp.asarray(xi_grid, dtype=jnp.float64)
        psi_grid_arr = jnp.asarray(psi_grid, dtype=jnp.float64)

        r_grid = jnp.linspace(0.0, r_t_arr, n_grid)
        xi_local = r_grid / r_c_arr
        psi_vals = jnp.interp(xi_local, xi_grid_arr, psi_grid_arr, left=W0_arr, right=0.0)

        rho0 = limepy_density_hat(W0_arr, g_arr)
        rho_grid = jnp.where(
            rho0 > 1e-300, limepy_density_hat(psi_vals, g_arr) / rho0, 0.0
        )
        rho_grid = jnp.where(r_grid <= r_t_arr, rho_grid, 0.0)

        integrand = 4.0 * jnp.pi * r_grid**2 * rho_grid
        dr = r_grid[1] - r_grid[0]
        M_cum = jnp.concatenate([
            jnp.zeros(1, dtype=integrand.dtype),
            jnp.cumsum(0.5 * (integrand[1:] + integrand[:-1])) * dr,
        ])
        cdf_grid = M_cum / (M_cum[-1] + 1e-30)

        object.__setattr__(self, "W0", W0_arr)
        object.__setattr__(self, "g", g_arr)
        object.__setattr__(self, "r_c", r_c_arr)
        object.__setattr__(self, "r_t", r_t_arr)
        object.__setattr__(self, "xi_grid", xi_grid_arr)
        object.__setattr__(self, "psi_grid", psi_grid_arr)
        object.__setattr__(self, "_r_grid", r_grid)
        object.__setattr__(self, "_cdf_grid", cdf_grid)

    @classmethod
    def from_W0_rc(
        cls,
        W0: float,
        g: float,
        r_c: float,
        xi_max: float = 300.0,
        n_ode_points: int = 2000,
        n_grid: int = 1000,
    ) -> "LIMEPYProfile":
        """Self-consistent constructor: r_t derived from where W(xi) -> 0.

        JIT/grad-safe in (W0, g): the fixed (xi_max, n_ode_points) keep array sizes
        static, so W0 and g may be tracers and profile-shape functionals carry
        dW0/dg gradients (the argmax truncation-radius crossing has zero gradient,
        as in KingProfile — structural-shape inference is unaffected). For large g
        / high W0, pass a larger xi_max so the model integrates to W -> 0.
        """
        xi_grid, psi_grid = solve_limepy_profile(W0, g, xi_max=xi_max, n_points=n_ode_points)
        xi_t = _find_tidal_radius(xi_grid, psi_grid)
        r_t = r_c * xi_t
        return cls(W0=W0, g=g, r_c=r_c, r_t=r_t, xi_grid=xi_grid, psi_grid=psi_grid, n_grid=n_grid)

    def sample_positions(
        self, masses: Float[Array, "N"], key: PRNGKeyArray
    ) -> Float[Array, "N 3"]:
        """Sample isotropic positions via the precomputed inverse mass CDF."""
        N = len(masses)
        key, sub = jax.random.split(key)
        radii = self._sample_radii(sub, N)
        key, sub = jax.random.split(key)
        theta = jnp.arccos(1.0 - 2.0 * jax.random.uniform(sub, shape=(N,)))
        key, sub = jax.random.split(key)
        phi = 2.0 * jnp.pi * jax.random.uniform(sub, shape=(N,))
        x = radii * jnp.sin(theta) * jnp.cos(phi)
        y = radii * jnp.sin(theta) * jnp.sin(phi)
        z = radii * jnp.cos(theta)
        return jnp.stack([x, y, z], axis=1)

    def _sample_radii(self, key: PRNGKeyArray, N: int) -> Float[Array, "N"]:
        u = jax.random.uniform(key, shape=(N,))
        r_sampled = jnp.interp(u, self._cdf_grid, self._r_grid)
        return jnp.clip(r_sampled, 0.0, self.r_t)

    def density(self, r: Float[Array, "..."]) -> Float[Array, "..."]:
        """Unnormalized radial density rho(r)/rho_0 (isotropic LIMEPY)."""
        xi = r / self.r_c
        psi_vals = jnp.interp(xi, self.xi_grid, self.psi_grid, left=self.W0, right=0.0)
        rho0 = limepy_density_hat(self.W0, self.g)
        rho = jnp.where(rho0 > 1e-300, limepy_density_hat(psi_vals, self.g) / rho0, 0.0)
        return jnp.where(r <= self.r_t, rho, 0.0)

    def characteristic_radius(self) -> Float[Array, ""]:
        """Truncation radius r_t (the model's outer scale)."""
        return self.r_t


__all__ = [
    "lowered_exponential",
    "limepy_density_hat",
    "solve_limepy_profile",
    "LIMEPYProfile",
]

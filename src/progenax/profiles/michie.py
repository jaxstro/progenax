"""Michie-King anisotropic spherical model.

Combines Michie's (1963) exp(-J^2/2 r_a^2 sigma^2) radial-anisotropy term with King's
(1966) lowered-Maxwellian energy cutoff [exp(-E/sigma^2) - 1] -- the standard
"Michie-King" model (Gunn & Griffin 1979; LIMEPY). Unlike Osipkov-Merritt (which holds a
given density fixed and inverts for f), Michie specifies the DF and solves Poisson for a
new, self-consistent, more centrally-radial density.

See docs/website/99-bibliography/per-paper/michie-1963.md and
docs/plans/2026-06-03-michie-king-anisotropic-model-design.md.
"""

from typing import Tuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from progenax.profiles.king import _find_tidal_radius

_N_U = 256  # tangential-velocity quadrature resolution for the density integral


def michie_density(W, s, n_u: int = _N_U):
    """Dimensionless Michie-King volume density rho_hat(W, s), s = r / r_a.

    The velocity-space integral of f ∝ exp(-s^2 u_t^2/2) [exp(W - u^2/2) - 1] over the
    bound region u^2 < 2W (u = v/sigma). The radial (u_r) integral is closed-form, leaving
    a 1-D tangential (u_t) quadrature:

        rho_hat(W, s) = int_0^{sqrt(2W)} u_t e^{-s^2 u_t^2/2}
                        [ e^{W - u_t^2/2} sqrt(2 pi) erf(a/sqrt 2) - 2 a ] du_t,
        a = sqrt(2W - u_t^2).

    At s=0 this equals sqrt(2 pi) * king_lowered_maxwellian_density(W) (the isotropic
    limit). W <= 0 returns 0. Gradient-safe at W=0 (clamp before sqrt/exp).

    Args:
        W: Dimensionless potential (scalar). s: Dimensionless radius r/r_a (scalar).
        n_u: Tangential quadrature points.

    Returns:
        rho_hat(W, s), unnormalized (consistent constant across W, s).
    """
    W_pos = jnp.maximum(W, 1e-12)
    u_t = jnp.linspace(0.0, jnp.sqrt(2.0 * W_pos), n_u)
    # a = sqrt(2W - u_t^2) hits 0 at the endpoint; sqrt has infinite derivative there,
    # so guard with the double-where pattern (no sqrt(0) reaches the backward pass).
    arg = 2.0 * W_pos - u_t**2
    a = jnp.where(arg > 0.0, jnp.sqrt(jnp.where(arg > 0.0, arg, 1.0)), 0.0)
    inner = (
        jnp.exp(W_pos - u_t**2 / 2.0) * jnp.sqrt(2.0 * jnp.pi)
        * jax.scipy.special.erf(a / jnp.sqrt(2.0))
        - 2.0 * a
    )
    integrand = u_t * jnp.exp(-(s**2) * u_t**2 / 2.0) * inner
    rho = jnp.trapezoid(integrand, u_t)
    return jnp.where(W > 0.0, rho, 0.0)


def _michie_poisson_rhs(xi, y, args):
    """RHS of the Michie-King Poisson equation (anisotropic, radius-dependent).

    Same factor-of-9 nondimensionalisation as King (xi = r/r_c), but the density is the
    *anisotropic* rho_hat(psi, xi/ra_hat), normalised at the centre (s=0). ra_hat ->
    infinity reduces this to King's RHS exactly.
    """
    W0, ra_hat = args
    psi, dpsi_dxi = y[0], y[1]
    s = xi / ra_hat
    rho0 = michie_density(W0, 0.0)
    rho_tilde = jnp.where(rho0 > 1e-10, michie_density(psi, s) / rho0, 0.0)
    d2psi_dxi2 = jnp.where(
        xi > 1e-6,
        -9.0 * rho_tilde - (2.0 / xi) * dpsi_dxi,
        -9.0 * rho_tilde,  # center guard (dpsi/dxi(0)=0)
    )
    return jnp.array([dpsi_dxi, d2psi_dxi2])


def solve_michie_profile(
    W0: float, ra_hat: float, xi_max: float = 800.0, n_points: int = 3000
) -> Tuple[Float[Array, "n_points"], Float[Array, "n_points"]]:
    """Solve the Michie-King Poisson equation from the centre outward to psi -> 0.

    Args:
        W0: Central concentration (psi(0) = W0).
        ra_hat: Anisotropy radius in core-radius units, r_a / r_c. ra_hat -> infinity is
            the isotropic King limit. Below a W0-dependent threshold the radial orbits
            build a 1/r^2 density tail and the model has no finite tidal radius (infinite
            mass) -- a concrete-input call then raises ValueError.
        xi_max: Maximum dimensionless radius. Larger than King's default because
            anisotropic models are far more extended (xi_t up to several hundred).
        n_points: output grid size.

    Returns:
        xi_grid (= r/r_c), psi_grid (psi >= 0, truncated at the tidal radius).

    References:
        Michie (1963), MNRAS 125, 127 (Eq. 5.8); King (1966), AJ 71, 64.
    """
    y0 = jnp.array([W0, 0.0])
    xi_span = (1e-6, xi_max)
    term = diffrax.ODETerm(_michie_poisson_rhs)
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
        args=(W0, ra_hat),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=100000,
    )
    xi_grid = solution.ts
    psi_end = solution.ys[-1, 0]  # psi at xi_max (negative once truncated)
    psi_grid = jnp.maximum(solution.ys[:, 0], 0.0)

    # Non-truncation guard (concrete inputs only; skipped under tracing). If psi has not
    # crossed 0 by xi_max, the radial-orbit 1/r^2 tail prevents truncation -> no finite
    # tidal radius (infinite mass). Refuse such an over-anisotropic model.
    if isinstance(W0, (int, float)) and isinstance(ra_hat, (int, float)):
        if float(psi_end) > 1e-3 * W0:
            raise ValueError(
                f"Michie-King model with W0={W0}, r_a/r_c={ra_hat} does not truncate "
                f"within xi_max={xi_max} (psi(xi_max)={float(psi_end):.3f} > 0): the "
                f"anisotropy is too strong (no finite tidal radius / infinite mass). "
                f"Increase anisotropy_radius, or raise xi_max if the model is genuinely "
                f"this extended."
            )
    return xi_grid, psi_grid


_michie_density_vec = jax.vmap(michie_density)  # over (W, s) grids


class MichieProfile(eqx.Module):
    """Michie-King anisotropic spherical density profile (SpatialProfile).

    The self-consistent density of the Michie-King model (Michie 1963 anisotropy + King
    1966 cutoff). More centrally-radial and more extended than the isotropic King model;
    r_a -> infinity recovers King. Construct with ``from_W0_rc``.

    Attributes:
        W0: central concentration. r_c: core radius. r_a: anisotropy radius [length].
        r_t: tidal radius (derived). xi_grid, psi_grid: ODE solution. _r_grid, _cdf_grid:
        precomputed mass-CDF for inverse-transform position sampling.

    References:
        Michie (1963), MNRAS 125, 127; King (1966), AJ 71, 64.
    """

    W0: Float[Array, ""]
    r_c: Float[Array, ""]
    r_a: Float[Array, ""]
    r_t: Float[Array, ""]
    xi_grid: Float[Array, "n_points"]
    psi_grid: Float[Array, "n_points"]
    _r_grid: Float[Array, "n_grid"]
    _cdf_grid: Float[Array, "n_grid"]

    def __init__(self, W0, r_c, r_a, r_t, xi_grid, psi_grid, n_grid: int = 1000):
        W0_a = jnp.asarray(W0, dtype=jnp.float64)
        r_c_a = jnp.asarray(r_c, dtype=jnp.float64)
        r_a_a = jnp.asarray(r_a, dtype=jnp.float64)
        r_t_a = jnp.asarray(r_t, dtype=jnp.float64)
        xi_a = jnp.asarray(xi_grid, dtype=jnp.float64)
        psi_a = jnp.asarray(psi_grid, dtype=jnp.float64)

        # sqrt-stretched grid (r = r_t * u^2): concentrates points in the core,
        # which a linear grid under-resolves at high W0 (audit R4 — same fix as
        # KingProfile). Smooth in r_t -> differentiable.
        u_grid = jnp.linspace(0.0, 1.0, n_grid)
        r_grid = r_t_a * u_grid**2
        psi_vals = jnp.interp(r_grid / r_c_a, xi_a, psi_a, left=W0_a, right=0.0)
        s_vals = r_grid / r_a_a  # = (r/r_c) / (r_a/r_c)

        rho0 = michie_density(W0_a, 0.0)
        rho_vals = _michie_density_vec(psi_vals, s_vals)
        rho_grid = jnp.where(rho0 > 1e-10, rho_vals / rho0, 0.0)
        rho_grid = jnp.where(r_grid <= r_t_a, rho_grid, 0.0)

        # Non-uniform trapezoid: variable spacing -> weight by diff(r_grid).
        integrand = 4.0 * jnp.pi * r_grid**2 * rho_grid
        M_cum = jnp.concatenate([
            jnp.zeros(1, dtype=integrand.dtype),
            jnp.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * jnp.diff(r_grid)),
        ])
        cdf_grid = M_cum / (M_cum[-1] + 1e-30)

        object.__setattr__(self, "W0", W0_a)
        object.__setattr__(self, "r_c", r_c_a)
        object.__setattr__(self, "r_a", r_a_a)
        object.__setattr__(self, "r_t", r_t_a)
        object.__setattr__(self, "xi_grid", xi_a)
        object.__setattr__(self, "psi_grid", psi_a)
        object.__setattr__(self, "_r_grid", r_grid)
        object.__setattr__(self, "_cdf_grid", cdf_grid)

    @classmethod
    def from_W0_rc(
        cls, W0: float, r_c: float, r_a: float,
        xi_max: float = 800.0, n_ode_points: int = 3000, n_grid: int = 1000,
    ) -> "MichieProfile":
        """Self-consistent Michie-King profile; r_t derived from where psi -> 0.

        Raises ValueError (via solve_michie_profile) if r_a is too small for a finite
        tidal radius (the radial-orbit pathology).
        """
        xi_grid, psi_grid = solve_michie_profile(
            W0, r_a / r_c, xi_max=xi_max, n_points=n_ode_points
        )
        xi_t = _find_tidal_radius(xi_grid, psi_grid)
        return cls(W0, r_c, r_a, r_c * xi_t, xi_grid, psi_grid, n_grid=n_grid)

    def sample_positions(
        self, masses: Float[Array, "N"], key: PRNGKeyArray
    ) -> Float[Array, "N 3"]:
        """Inverse-transform sample positions from the Michie-King density (isotropic
        angular distribution; mass values set only N)."""
        N = len(masses)
        key_r, key_theta, key_phi = jax.random.split(key, 3)
        u = jax.random.uniform(key_r, shape=(N,))
        radii = jnp.interp(u, self._cdf_grid, self._r_grid)

        cos_theta = 1.0 - 2.0 * jax.random.uniform(key_theta, shape=(N,))
        theta = jnp.arccos(cos_theta)
        phi = 2.0 * jnp.pi * jax.random.uniform(key_phi, shape=(N,))
        x = radii * jnp.sin(theta) * jnp.cos(phi)
        y = radii * jnp.sin(theta) * jnp.sin(phi)
        z = radii * jnp.cos(theta)
        return jnp.stack([x, y, z], axis=1)

    def density(self, r: Float[Array, "..."]) -> Float[Array, "..."]:
        """Normalized volume density rho(r)/rho_0 (0 outside r_t)."""
        psi = jnp.interp(r / self.r_c, self.xi_grid, self.psi_grid, left=self.W0, right=0.0)
        rho0 = michie_density(self.W0, 0.0)
        rho = _michie_density_vec(jnp.atleast_1d(psi), jnp.atleast_1d(r / self.r_a)) / rho0
        rho = jnp.reshape(rho, jnp.shape(r))
        return jnp.where(r <= self.r_t, rho, 0.0)

    def characteristic_radius(self) -> Float[Array, ""]:
        return self.r_t


__all__ = ["michie_density", "solve_michie_profile", "MichieProfile"]

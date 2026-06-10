"""Michie-King anisotropic velocity DF (Michie 1963 anisotropy + King 1966 cutoff).

The DF f ∝ exp(-J^2/2 r_a^2 sigma^2) [exp(-E/sigma^2) - 1] is not a function of a single
integral, so (unlike Osipkov-Merritt) there is no stretch trick: velocities are drawn from
a 2-D joint over (u_r, u_t), u = v/sigma. By default the speed MARGINAL comes from one
precomputed inverse-CDF table per call (AnisoSpeedCDFTable at g=1 — the exact Michie =
LIMEPY(g=1, finite r_a) reduction) with the EXACT angular conditional cos(theta)|u
(weight exp(-(s^2 u^2/2)(1 - cos^2 theta)), identical to the u_t-marginal-then-u_r
factorization under u_t = u sin(theta)); speed_method="quadrature" retains the per-star
2-D marginal-then-conditional inverse-CDF as the oracle. The self-consistent sigma uses
the same factor-of-9 relation as the King DF, with the anisotropic density.

See docs/website/99-bibliography/per-paper/michie-1963.md.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from progenax import defaults
from progenax.kinematics._speed_kernels import (
    _N_C,
    _ORACLE_BATCH,
    _sample_costheta_given_u,
)
from progenax.numerics import inverse_cdf_draw
from progenax.profiles.king import _find_tidal_radius
from progenax.profiles.limepy_tables import AnisoSpeedCDFTable
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

    key_t, key_r = jax.random.split(key)
    ut = inverse_cdf_draw(m, u_t, jax.random.uniform(key_t))

    # Conditional u_r | u_t (same sqrt-endpoint guard).
    arg_ut = 2.0 * W_pos - ut**2
    a_ut = jnp.where(arg_ut > 0.0, jnp.sqrt(jnp.where(arg_ut > 0.0, arg_ut, 1.0)), 0.0)
    u_r = jnp.linspace(-a_ut, a_ut, n_r)
    g = jnp.maximum(jnp.exp(W_pos - (u_r**2 + ut**2) / 2.0) - 1.0, 0.0)
    ur = inverse_cdf_draw(g, u_r, jax.random.uniform(key_r))

    # MANDATORY bound guard: zero total weight clamps to grid[-1], not 0
    # (see numerics.inverse_cdf_draw docstring).
    bound = W > 1e-6
    return jnp.where(bound, ur, 0.0), jnp.where(bound, ut, 0.0)


class MichieVelocityDF(eqx.Module):
    """Michie-King anisotropic velocity DF (radially anisotropic, lowered-Maxwellian).

    beta(r) ~ 0 at the centre, increasing outward; r_a -> infinity is the isotropic King
    DF. The velocity scale sigma is self-consistent (sigma^2 = G M / (9 r_c mu), mu the
    anisotropic dimensionless mass integral), so ICs are virial without external rescale.

    Speed draws are TABLE-BACKED by default (speed_method="table"): the speed
    marginal comes from one precomputed inverse-CDF table per call,
    AnisoSpeedCDFTable.build(W0, p_box, g=1) — Michie IS the g=1 anisotropic
    LIMEPY model (E_gamma(1, x) = e^x - 1 exactly; the standing
    distributional proof is test_g1_aniso_matches_michie_velocity_df). The
    angular conditional cos(theta)|u stays EXACT (_sample_costheta_given_u,
    weight exp(-(s^2 u^2/2)(1 - cos^2 theta)) = exp(-s^2 u_t^2/2) under
    u_t = u sin(theta)) — only the speed marginal is tabulated.
    speed_method="quadrature" retains the exact per-star 2-D
    (u_t-marginal-then-u_r) inverse-CDF as the oracle (statistical agreement
    asserted in tests/unit/kinematics/test_michie_df.py::
    TestMichieTableRouting). Small-N note: the aniso table build (~160 ms
    per sample_velocities call) dominates below ~3k stars — batch repeated
    small draws into one call, or use the quadrature oracle there.

    Attributes:
        W0, r_c, r_a: model parameters. xi_grid, psi_grid: Michie ODE solution.
        mu: int rho_hat(psi, xi/ra_hat) xi^2 dxi (sets sigma).
        speed_method: static, "table" (default) or "quadrature" (exact oracle).

    References:
        Michie (1963), MNRAS 125, 127; King (1966), AJ 71, 64.
    """

    W0: Float[Array, ""]
    r_c: Float[Array, ""]
    r_a: Float[Array, ""]
    xi_grid: Float[Array, "n_ode"]
    psi_grid: Float[Array, "n_ode"]
    mu: Float[Array, ""]
    speed_method: str = eqx.field(static=True)

    def __init__(
        self, W0: float = 7.0, r_c: float = 1.0, r_a: float = 10.0,
        xi_max: float = 800.0, n_ode_points: int = 3000,
        speed_method: str = "table",
    ):
        if speed_method not in ("table", "quadrature"):
            raise ValueError(
                f"speed_method must be 'table' or 'quadrature', got {speed_method!r}"
            )
        self.speed_method = speed_method
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
        if self.speed_method == "table":
            # Mirror of LIMEPYVelocityDF's aniso table path at g=1 (the exact
            # Michie reduction): the speed MARGINAL comes from ONE precomputed
            # 3-D CDF table; the angular conditional cos(theta)|u stays EXACT
            # (_sample_costheta_given_u — the same Michie factor
            # exp(-s^2 u_t^2/2) reparametrized via u_t = u sin(theta)).
            # Box covers every star: W <= W0, p <= r_t/r_a (radii <= r_t);
            # the 1e-3 p floor guards the near-isotropic corner.
            p_box = jnp.maximum(
                self.r_c * _find_tidal_radius(self.xi_grid, self.psi_grid) / self.r_a,
                1e-3,
            )
            table = AnisoSpeedCDFTable.build(self.W0, p_box, jnp.asarray(1.0))
            ku_kc = jax.vmap(jax.random.split)(speed_keys)
            unif = jax.vmap(lambda kk: jax.random.uniform(kk))(ku_kc[:, 0])
            u_sp = jax.vmap(table.inverse)(W, s, unif)
            cos_t = jax.vmap(
                lambda kk, uu, pp: _sample_costheta_given_u(kk, uu, pp, _N_C)
            )(ku_kc[:, 1], u_sp, s)
            ur = u_sp * cos_t
            ut = u_sp * jnp.sqrt(jnp.maximum(1.0 - cos_t**2, 0.0))
        else:
            # Bounded-memory oracle: lax.map in chunks of _ORACLE_BATCH stars
            # (vmap within each chunk) instead of one eager vmap over all N.
            ur, ut = jax.lax.map(
                lambda kws: _sample_ur_ut(kws[0], kws[1], kws[2]),
                (speed_keys, W, s),
                batch_size=_ORACLE_BATCH,
            )
        v_r = sigma * ur
        v_t = sigma * ut

        # v_r along r_hat (signed), v_t in a random azimuthal direction perp to r_hat.
        r_hat = positions / (radii[:, None] + 1e-30)
        rand = jax.random.normal(key_dir, (N, 3))
        rand = rand - jnp.sum(rand * r_hat, axis=1, keepdims=True) * r_hat
        t_hat = rand / (jnp.linalg.norm(rand, axis=1, keepdims=True) + 1e-30)
        return v_r[:, None] * r_hat + v_t[:, None] * t_hat


__all__ = ["MichieVelocityDF"]

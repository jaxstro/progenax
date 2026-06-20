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

from progenax.kinematics._speed_kernels import (
    _N_C,
    _ORACLE_BATCH,
    _sample_costheta_given_u,
)
from progenax.numerics import inverse_cdf_draw
from progenax.profiles.king import _find_tidal_radius
from progenax.profiles.limepy import (
    _angle_integral_T,
    _aniso_density_scalar,
    _aniso_density_vec,
    limepy_density_hat,
    lowered_exponential,
    solve_limepy_profile,
)
from progenax.profiles.limepy_tables import AnisoSpeedCDFTable, SpeedCDFTable

# Inverse-CDF table resolutions. (_N_C, the cos(theta) grid of the shared
# angular conditional _sample_costheta_given_u, lives in _speed_kernels.)
_N_SPEED_GRID = 256  # speed grid


def _sample_unit_speed(
    key: PRNGKeyArray, W: Float[Array, ""], g: Float[Array, ""], n_u: int
) -> Float[Array, ""]:
    """Isotropic: sample u ~ u^2 E_gamma(g, W - u^2/2) on [0, sqrt(2W)] (u = v/s)."""
    W_safe = jnp.maximum(W, 1e-12)
    u_grid = jnp.linspace(0.0, jnp.sqrt(2.0 * W_safe), n_u)
    weight = jnp.maximum(
        u_grid**2 * lowered_exponential(g, W_safe - u_grid**2 / 2.0), 0.0
    )
    u = inverse_cdf_draw(weight, u_grid, jax.random.uniform(key))
    # MANDATORY bound guard: zero total weight clamps to grid[-1], not 0
    # (see numerics.inverse_cdf_draw docstring).
    return jnp.where(W > 1e-6, u, 0.0)


def _sample_speed_angle(
    key: PRNGKeyArray,
    W: Float[Array, ""],
    s: Float[Array, ""],
    g: Float[Array, ""],
    n_u: int,
    n_c: int,
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

    key_u, key_c = jax.random.split(key)
    u = inverse_cdf_draw(m_u, u_grid, jax.random.uniform(key_u))

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

    Speed draws are TABLE-BACKED by default (speed_method="table"): the speed
    marginal comes from the inverse-CDF table CACHED at construction
    (SpeedCDFTable / AnisoSpeedCDFTable, the same machinery
    MultiComponentCluster uses), and the whole draw runs through a jitted
    core, so live memory is O(N) instead of the eager per-star quadrature's
    O(N * 256 * 91) Poisson-sum buffers (measured at N=2e4 anisotropic:
    10.87 GB pre-tables; draw-chain peak +0.20 GB cached+jitted; warm draw
    0.015 s after the one-time ~0.5 s compile). The anisotropic angular
    conditional cos(theta)|u stays EXACT (_sample_costheta_given_u) -- only
    the speed marginal is tabulated. speed_method="quadrature" retains the
    exact per-star quadrature as the oracle (statistical agreement asserted
    in tests/unit/kinematics/test_limepy_df.py::TestLimepyTableRouting).
    Cost structure: CONSTRUCTION dominates (~2.6 s / ~1.8 GB peak for an
    anisotropic model: two ODE solves + the mu integral + the cached table
    build) -- reuse one DF instance for repeated draws; per-draw cost is
    then milliseconds at any N.

    Attributes:
        W0, g, r_c: model parameters. r_a: anisotropy radius (inf = isotropic).
        r_t: truncation radius. xi_grid, psi_grid: ODE solution W(xi).
        mu: dimensionless mass integral int rho_tilde xi^2 dxi (sets s).
        is_aniso: static flag selecting the anisotropic sampler.
        speed_method: static, "table" (default) or "quadrature" (exact oracle).
    """

    W0: Float[Array, ""]
    g: Float[Array, ""]
    r_c: Float[Array, ""]
    r_a: Float[Array, ""]
    r_t: Float[Array, ""]
    xi_grid: Float[Array, "n_ode"]
    psi_grid: Float[Array, "n_ode"]
    mu: Float[Array, ""]
    speed_table: SpeedCDFTable | AnisoSpeedCDFTable | None
    is_aniso: bool = eqx.field(static=True)
    speed_method: str = eqx.field(static=True)

    def __init__(
        self,
        W0: float = 5.0,
        g: float = 1.0,
        r_c: float = 1.0,
        r_a: float | None = None,
        xi_max: float = 300.0,
        n_ode_points: int = 2000,
        speed_method: str = "table",
    ):
        if speed_method not in ("table", "quadrature"):
            raise ValueError(
                f"speed_method must be 'table' or 'quadrature', got {speed_method!r}"
            )
        self.speed_method = speed_method
        is_aniso = r_a is not None
        self.W0 = jnp.asarray(W0, dtype=jnp.float64)
        self.g = jnp.asarray(g, dtype=jnp.float64)
        self.r_c = jnp.asarray(r_c, dtype=jnp.float64)
        self.r_a = jnp.asarray(jnp.inf if r_a is None else r_a, dtype=jnp.float64)
        ra_hat = None if r_a is None else r_a / r_c
        xi_grid, psi_grid, psi_raw = solve_limepy_profile(
            W0, g, ra_hat=ra_hat, xi_max=xi_max, n_points=n_ode_points
        )
        self.xi_grid = xi_grid
        self.psi_grid = psi_grid
        # Feed UNCLAMPED psi_raw so d(r_t)/dW0 flows (the clamp zeros the crossing
        # node's gradient). Forward r_t is the interpolated crossing.
        self.r_t = self.r_c * _find_tidal_radius(xi_grid, psi_raw)

        # mu = int rho_tilde xi^2 dxi, rho_tilde normalized to 1 at the centre.
        if is_aniso:
            rho0 = _aniso_density_scalar(self.W0, jnp.asarray(0.0), self.g)
            rho_tilde = _aniso_density_vec(psi_grid, xi_grid / ra_hat, self.g) / rho0
        else:
            rho0 = limepy_density_hat(self.W0, self.g)
            rho_tilde = jnp.where(
                rho0 > 1e-300, limepy_density_hat(psi_grid, self.g) / rho0, 0.0
            )
        self.mu = jnp.trapezoid(rho_tilde * xi_grid**2, xi_grid)
        self.is_aniso = is_aniso

        # Build the speed-CDF table ONCE here: it depends only on
        # (W0, r_t/r_a, g), all fixed at construction. Pre-fix it was rebuilt
        # inside EVERY sample_velocities call (~0.4 s + build transients per
        # call at W0=7); caching also removes the small-N per-call penalty.
        # Differentiable: the table leaves are functions of (W0, g, r_a), so
        # gradients flow through construction exactly as before.
        if speed_method == "table":
            if is_aniso:
                p_box = jnp.maximum(self.r_t / self.r_a, 1e-3)
                self.speed_table = AnisoSpeedCDFTable.build(self.W0, p_box, self.g)
            else:
                self.speed_table = SpeedCDFTable.build(self.W0, self.g)
        else:
            self.speed_table = None  # quadrature oracle needs no table

    def _s(self, M_total: Float[Array, ""], G: float) -> Float[Array, ""]:
        """Self-consistent central velocity scale s = sqrt(G M / (9 r_c mu))."""
        return jnp.sqrt(G * M_total / (9.0 * self.r_c * self.mu))

    def sample_velocities(
        self,
        positions: Float[Array, "N 3"],
        masses: Float[Array, "N"],
        key: PRNGKeyArray,
        G: float,
    ) -> Float[Array, "N 3"]:
        """Sample velocities from the general-g lowered DF (iso or Michie/OM aniso).

        Differentiable in (W0, g, r_a, r_c) through the model. Runs through a
        jitted core (`_sample_velocities_core`, the cluster/sampling.py
        pattern): the draw chain fuses under XLA instead of materializing
        every eager intermediate (measured at N=2e4 aniso: draw-chain peak
        +0.63 GB eager -> +0.13 GB jitted; warm call 0.35 -> 0.17 s).

        G is REQUIRED (explicit-units policy; e.g. ``STELLAR.G``).
        """
        return _sample_velocities_core(
            self, positions, masses, key, jnp.asarray(G, dtype=jnp.float64)
        )


@eqx.filter_jit
def _sample_velocities_core(
    df: LIMEPYVelocityDF,
    positions: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    key: PRNGKeyArray,
    G: Float[Array, ""],
) -> Float[Array, "N 3"]:
    """Jitted sampling core (one compiled unit per (N, engine-statics) combo).

    `df` is a pytree argument: its static fields (is_aniso, speed_method)
    select the branch at trace time; its array leaves (incl. the CACHED
    speed_table built at construction) are traced, so gradients w.r.t.
    (W0, g, r_a, r_c) flow unchanged. G enters as a traced array so changing
    G never recompiles.
    """
    N = positions.shape[0]
    M_total = jnp.sum(masses)
    radii = jnp.linalg.norm(positions, axis=1)
    W = jnp.interp(radii / df.r_c, df.xi_grid, df.psi_grid, left=df.W0, right=0.0)
    W = jnp.maximum(W, 0.0)
    s = df._s(M_total, G)

    key_speed, key_dir = jax.random.split(key)
    speed_keys = jax.random.split(key_speed, N)

    if df.is_aniso:
        s_local = radii / df.r_a  # = (r/r_c)/(r_a/r_c)
        if df.speed_method == "table":
            # The speed MARGINAL comes from the table CACHED at construction
            # (box covers every star: W <= W0, p <= r_t/r_a); the angular
            # conditional cos(theta)|u stays EXACT (_sample_costheta_given_u).
            ku_kc = jax.vmap(jax.random.split)(speed_keys)
            unif = jax.vmap(lambda kk: jax.random.uniform(kk))(ku_kc[:, 0])
            u_sp = jax.vmap(df.speed_table.inverse)(W, s_local, unif)
            cos_t = jax.vmap(
                lambda kk, uu, pp: _sample_costheta_given_u(kk, uu, pp, _N_C)
            )(ku_kc[:, 1], u_sp, s_local)
            u_r = u_sp * cos_t
            u_t = u_sp * jnp.sqrt(jnp.maximum(1.0 - cos_t**2, 0.0))
        else:
            # Bounded-memory oracle: lax.map in chunks of _ORACLE_BATCH stars
            # (vmap within each chunk) instead of one vmap over all N,
            # bounding the (chunk, n_u, ~91) E_gamma Poisson-sum buffers.
            u_r, u_t = jax.lax.map(
                lambda kws: _sample_speed_angle(
                    kws[0], kws[1], kws[2], df.g, _N_SPEED_GRID, _N_C
                ),
                (speed_keys, W, s_local),
                batch_size=_ORACLE_BATCH,
            )
        v_r, v_t = s * u_r, s * u_t
        # v_r along r_hat (signed); v_t in a random azimuth perpendicular to r_hat.
        r_hat = positions / (radii[:, None] + 1e-30)
        rand = jax.random.normal(key_dir, (N, 3))
        rand = rand - jnp.sum(rand * r_hat, axis=1, keepdims=True) * r_hat
        t_hat = rand / (jnp.linalg.norm(rand, axis=1, keepdims=True) + 1e-30)
        return v_r[:, None] * r_hat + v_t[:, None] * t_hat

    # Isotropic: speed * isotropic direction.
    if df.speed_method == "table":
        unif = jax.vmap(lambda kk: jax.random.uniform(kk))(speed_keys)
        u = jax.vmap(df.speed_table.inverse)(W, unif)
    else:
        # Bounded-memory oracle: lax.map in chunks of _ORACLE_BATCH stars.
        u = jax.lax.map(
            lambda kw: _sample_unit_speed(kw[0], kw[1], df.g, _N_SPEED_GRID),
            (speed_keys, W),
            batch_size=_ORACLE_BATCH,
        )
    speeds = s * u
    dirs = jax.random.normal(key_dir, shape=(N, 3))
    dirs = dirs / (jnp.linalg.norm(dirs, axis=1, keepdims=True) + 1e-30)
    return speeds[:, None] * dirs


__all__ = ["LIMEPYVelocityDF"]

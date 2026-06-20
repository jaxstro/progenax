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

Speed draws are table-backed by default (one precomputed inverse-CDF table per call,
`SpeedCDFTable` at g=1 — E_gamma(1, x) = e^x - 1 IS the King lowering, exactly);
`speed_method="quadrature"` retains the per-star tabulated inverse-CDF (vmap'd,
differentiable, no while_loop) as the exact oracle. Directions are isotropic.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from progenax.kinematics._speed_kernels import _ORACLE_BATCH
from progenax.numerics import inverse_cdf_draw
from progenax.profiles.king import (
    _auto_ode_domain,
    king_lowered_maxwellian_density,
    solve_king_profile,
)
from progenax.profiles.limepy_tables import SpeedCDFTable

# Resolution of the per-particle speed inverse-CDF table.
_N_SPEED_GRID = 256


def _sample_unit_speed(
    key: PRNGKeyArray, W: Float[Array, ""], n_u: int
) -> Float[Array, ""]:
    """Sample one normalized speed u ~ g(u) = u^2 (exp(W - u^2/2) - 1) on [0, sqrt(2W)].

    u is in units of sigma; the physical speed is sigma * u. Differentiable inverse-CDF
    sampling on a fixed-size grid. Returns 0 where W <= 0 (at/outside the tidal radius,
    where the escape speed vanishes).
    """
    W_safe = jnp.maximum(W, 1e-12)
    u_grid = jnp.linspace(0.0, jnp.sqrt(2.0 * W_safe), n_u)
    g = u_grid**2 * (jnp.exp(W_safe - u_grid**2 / 2.0) - 1.0)
    g = jnp.maximum(g, 0.0)  # numerically non-negative on [0, sqrt(2W)]

    u = inverse_cdf_draw(g, u_grid, jax.random.uniform(key))
    # MANDATORY bound guard: zero total weight clamps to grid[-1], not 0
    # (see numerics.inverse_cdf_draw docstring).
    return jnp.where(W > 1e-6, u, 0.0)


class KingVelocityDF(eqx.Module):
    """
    King (1966) lowered-Maxwellian velocity distribution function.

    A true equilibrium DF: velocities are sampled from the lowered Maxwellian whose
    radial speed structure follows the King potential psi(r), with the central velocity
    scale sigma fixed self-consistently from (G, M_total, r_c, W0). The resulting ICs are
    in virial equilibrium (Q = 0.5) without external rescaling, and all particles are
    bound (v < v_esc(r)).

    Speed draws are TABLE-BACKED by default (speed_method="table"): the
    inverse-CDF table is CACHED at construction (SpeedCDFTable.build(W0, g=1))
    and the draw runs through a jitted core — at g=1 the LIMEPY lowered
    exponential reduces EXACTLY to the King lowering, E_gamma(1, x) = e^x - 1
    (identity guarded to rtol 1e-12 in
    tests/unit/kinematics/test_king_df.py::TestKingTableRouting), so the
    table weight x^2 E_gamma(1, W(1-x^2)) IS the King speed weight
    u^2 (exp(W - u^2/2) - 1). speed_method="quadrature" retains the exact
    per-star 256-point quadrature as the oracle (statistical agreement
    asserted in TestKingTableRouting). Reuse one DF instance for repeated
    draws: construction (ODE solve + table) dominates; per-draw cost is then
    milliseconds at any N.

    Attributes:
        W0: King concentration parameter (dimensionless central potential)
        r_c: Core radius [length units] (the King core radius)
        xi_grid, psi_grid: ODE solution of the King model (xi = r/r_c, psi(xi))
        speed_method: static, "table" (default) or "quadrature" (exact oracle)

    References:
        King (1966), AJ, 71, 64
        Binney & Tremaine (2008), "Galactic Dynamics", 2nd ed., Eq. 4.131
    """

    W0: Float[Array, ""]
    r_c: Float[Array, ""]
    xi_grid: Float[Array, "n_ode"]
    psi_grid: Float[Array, "n_ode"]
    speed_table: SpeedCDFTable | None
    speed_method: str = eqx.field(static=True)

    def __init__(
        self,
        W0: float = 5.0,
        r_c: float = 1.0,
        xi_max: float | None = None,
        n_ode_points: int | None = None,
        speed_method: str = "table",
    ):
        if speed_method not in ("table", "quadrature"):
            raise ValueError(
                f"speed_method must be 'table' or 'quadrature', got {speed_method!r}"
            )
        self.speed_method = speed_method
        self.W0 = jnp.asarray(W0)
        self.r_c = jnp.asarray(r_c)
        # Auto-size the ODE domain from W0 (matches KingProfile.from_W0_rc) so the
        # matched DF stays self-consistent for high-concentration models.
        auto_xi_max, auto_n_points = _auto_ode_domain(W0)
        if xi_max is None:
            xi_max = auto_xi_max
        if n_ode_points is None:
            n_ode_points = auto_n_points
        # King DF uses the clamped psi (density/mu); r_t is not needed here, so
        # psi_raw is discarded.
        xi_grid, psi_grid, _ = solve_king_profile(
            W0, xi_max=xi_max, n_points=n_ode_points
        )
        self.xi_grid = xi_grid
        self.psi_grid = psi_grid
        # Table cached at construction (depends only on W0; g=1 is the exact
        # King reduction) -- pre-fix it was rebuilt every sample_velocities
        # call. Differentiable: the table leaves are functions of W0.
        self.speed_table = (
            SpeedCDFTable.build(self.W0, jnp.asarray(1.0))
            if speed_method == "table"
            else None
        )

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
        G: float,
    ) -> Float[Array, "N 3"]:
        """
        Sample velocities from the King lowered-Maxwellian DF.

        At each radius r, the speed is drawn from g(v) ∝ v^2 [exp(psi(r) - v^2/2sigma^2) - 1]
        on [0, v_esc(r)] via a differentiable tabulated inverse-CDF; directions are isotropic.

        Args:
            positions: Particle positions (N, 3) [length units]
            masses: Particle masses (N,) [M_sun]
            key: JAX random key
            G: Gravitational constant (REQUIRED, explicit-units policy; e.g. ``STELLAR.G``).

        Returns:
            Cartesian velocities (N, 3) [velocity units]
        """
        return _sample_velocities_core(
            self, positions, masses, key, jnp.asarray(G, dtype=jnp.float64)
        )


@eqx.filter_jit
def _sample_velocities_core(
    df: KingVelocityDF,
    positions: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    key: PRNGKeyArray,
    G: Float[Array, ""],
) -> Float[Array, "N 3"]:
    """Jitted sampling core (cluster/sampling.py pattern; see LIMEPYVelocityDF).

    `df` is a pytree argument: speed_method selects the branch at trace time;
    array leaves (incl. the CACHED speed_table) are traced, so gradients
    w.r.t. (W0, r_c, M) flow unchanged. G is traced (no recompile per value).
    """
    N = positions.shape[0]
    M_total = jnp.sum(masses)
    radii = jnp.linalg.norm(positions, axis=1)

    # Local dimensionless potential W(r) = psi(r) from the King ODE.
    W = jnp.interp(radii / df.r_c, df.xi_grid, df.psi_grid, left=df.W0, right=0.0)
    W = jnp.maximum(W, 0.0)

    sigma = df._sigma(M_total, G)

    key_speed, key_dir = jax.random.split(key)
    speed_keys = jax.random.split(key_speed, N)
    if df.speed_method == "table":
        # The table CACHED at construction replaces the per-star 256-point
        # quadrature. g=1 is the EXACT King reduction:
        # E_gamma(1, x) = e^x - 1 (see class docstring).
        unif = jax.vmap(lambda kk: jax.random.uniform(kk))(speed_keys)
        u = jax.vmap(df.speed_table.inverse)(W, unif)
    else:
        # Bounded-memory oracle: lax.map in chunks of _ORACLE_BATCH stars
        # (vmap within each chunk) instead of one vmap over all N.
        u = jax.lax.map(
            lambda kw: _sample_unit_speed(kw[0], kw[1], _N_SPEED_GRID),
            (speed_keys, W),
            batch_size=_ORACLE_BATCH,
        )
    speeds = sigma * u

    # Isotropic directions (normalized Gaussian vectors).
    dirs = jax.random.normal(key_dir, shape=(N, 3))
    dirs = dirs / (jnp.linalg.norm(dirs, axis=1, keepdims=True) + 1e-30)

    return speeds[:, None] * dirs


__all__ = ["KingVelocityDF"]

"""Eddington inversion + shared Osipkov-Merritt (Merritt 1985) sampling helpers.

Three differentiable, vmap/JIT-safe primitives reused by the velocity DFs:

- `eddington_invert`: the generic (dimensionless) Eddington inversion of a density
  rho(r) in a relative potential Psi(r), optionally Osipkov-Merritt anisotropic via
  the augmented density rho_Q = (1 + r^2/r_a^2) rho (Merritt 1985, Eqs. 9-11).

- `sample_speed_from_f_table`: draw a (dimensionless) speed s ~ s^2 f(Psi - s^2/2) on
  [0, sqrt(2 Psi)] via a tabulated inverse-CDF. Identical for isotropic and OM models;
  only the energy DF table (E_grid, f_grid) differs (augmented density for OM).
- `assign_om_directions`: turn per-particle speeds into Cartesian velocities. With
  anisotropy_radius=None this is isotropic (random 3-D direction); with r_a set it uses
  the "stretched isotropic" split that realises the Osipkov-Merritt anisotropy
  sigma_r^2/sigma_t^2 = 1 + r^2/r_a^2 exactly (Merritt 1985, Eq. 15):

      draw a global isotropic unit vector n and let y = position/r_a,
      h = sqrt(1 + y.y). Then v = s [n/h + y(y.n)/(h(h+1))].

  The radial eigenvalue is 1 and both tangential eigenvalues are 1/h.
  This representation is smooth at zero position for fixed s and finite r_a>0.

See docs/website/99-bibliography/per-paper/merritt-1985.md.
"""

import jax
import jax.numpy as jnp
from jaxstro.numerics import gauss_legendre_nodes
from jaxstro.numerics.sampling import inverse_cdf_draw
from jaxtyping import Array, Float, PRNGKeyArray


def eddington_invert(
    r_grid: Float[Array, "n_r"],
    rho_grid: Float[Array, "n_r"],
    drho_dr_grid: Float[Array, "n_r"],
    Psi_grid: Float[Array, "n_r"],
    dPsi_dr_grid: Float[Array, "n_r"],
    r_a=None,
    n_e: int = 1000,
    n_u: int = 2000,
):
    """Generic (dimensionless) Eddington inversion in a given relative potential.

    Returns (E_grid, f_grid): the isotropic ergodic DF of rho in Psi, or with
    r_a set the Osipkov-Merritt f(Q) via the augmented density
    rho_Q = (1 + r^2/r_a^2) rho (Merritt 1985; r_a=inf or None -> isotropic).
    Zero-point contract: Psi_grid must decrease outward with Psi_grid[-1] the
    truncation zero point (~0); the boundary term is drho/dPsi at the last grid
    point.
    Raw (unclamped) f: callers detect genuine negativity; the speed sampler
    clamps grid-level ringing at use. Extracted VERBATIM from the validated
    _eff_eddington_table (Phase 2a) -- the r->0 double-where dPsi guard and the
    u = sqrt(E - Psi) substitution are gradient-safety load-bearing.
    """
    # Osipkov-Merritt augmentation applied to the PASSED density derivative
    # (only drho/dr enters the inversion; Psi always comes from the TRUE density).
    # r_a=None or inf -> weight 1 -> isotropic. The finite/inf split via jnp.where
    # keeps a traced infinite r_a gradient-safe (no inf enters the graph).
    if r_a is None:
        drho_dr = drho_dr_grid
    else:
        r_a = jnp.asarray(r_a)
        finite = jnp.isfinite(r_a)
        ra_safe = jnp.where(finite, r_a, 1.0)
        w = jnp.where(finite, 1.0 + (r_grid / ra_safe) ** 2, 1.0)
        dw = jnp.where(finite, 2.0 * r_grid / ra_safe**2, 0.0)
        drho_dr = dw * rho_grid + w * drho_dr_grid

    # drho/dPsi = (drho/dr)/(dPsi/dr). At r->0 the enclosed mass ->0, so dPsi/dr->0
    # (exactly at index 0 when M(<r_grid[0])=0). A bare 0-denominator divide is finite
    # in the forward pass after the center fix below, but its BACKWARD pass is NaN
    # (0 * inf in the VJP), which kills grad w.r.t. the density parameters. Guard with
    # the double-where pattern so no inf/NaN ever enters the graph, then set the center
    # point from its neighbor (the ratio has a finite limit there).
    safe_dPsi_dr = jnp.where(dPsi_dr_grid == 0.0, 1.0, dPsi_dr_grid)
    drho_dPsi = jnp.where(dPsi_dr_grid == 0.0, 0.0, drho_dr / safe_dPsi_dr)
    drho_dPsi = drho_dPsi.at[0].set(drho_dPsi[1])
    d2rho_dPsi2 = jnp.gradient(drho_dPsi, Psi_grid)

    Psi0 = Psi_grid[0]
    Psi_asc = Psi_grid[::-1]
    d2_asc = d2rho_dPsi2[::-1]
    bnd = drho_dPsi[-1]  # d rho/d Psi at Psi=0 (truncation boundary term)

    # End just below Psi0: E=Psi0 is the singular central energy (the Eddington
    # integrand reaches r->0); central lookups clamp to f_grid[-1], and the w^2 factor
    # makes the w->0 (E->Psi0) contribution negligible for sampling.
    E_grid = jnp.linspace(1e-4 * Psi0, 0.999 * Psi0, n_e)

    def f_one(E):
        # u = sqrt(E - Psi): int_0^E g/sqrt(E-Psi) dPsi = 2 int_0^sqrt(E) g(E-u^2) du
        u = jnp.linspace(0.0, jnp.sqrt(E), n_u)
        g = jnp.interp(E - u**2, Psi_asc, d2_asc)
        return (2.0 * jnp.trapezoid(g, u) + bnd / jnp.sqrt(E)) / (
            jnp.sqrt(8.0) * jnp.pi**2
        )

    # Raw (unclamped) f(E): the speed sampler clamps the speed pdf at use, and the raw
    # values let callers detect a genuinely negative (unphysical) OM DF.
    f_grid = jax.vmap(f_one)(E_grid)
    return E_grid, f_grid


def _interp_uniform(x, grid, values):
    """Linear lookup on an ascending uniform grid with at least two nodes.

    Uses stored knot coordinates to preserve the generic interpolant and its
    right-interval convention. Not valid for arbitrary nonuniform grids.
    """
    index = jnp.clip(
        jnp.floor((x - grid[0]) * (grid.size - 1) / (grid[-1] - grid[0])).astype(
            jnp.int32
        ),
        0,
        grid.size - 2,
    )
    # Resolve roundoff at exact linspace knots using the actual stored grid.
    index = jnp.where(
        x >= grid[index + 1], jnp.minimum(index + 1, grid.size - 2), index
    )
    index = jnp.where(x < grid[index], jnp.maximum(index - 1, 0), index)
    y = values[index] + (x - grid[index]) * (values[index + 1] - values[index]) / (
        grid[index + 1] - grid[index]
    )
    return jnp.where(x < grid[0], values[0], jnp.where(x > grid[-1], values[-1], y))


_MOMENT_NODES, _MOMENT_WEIGHTS = gauss_legendre_nodes(4)


def _df_speed_second_moment(psi, energy, df):
    """Integrate the clipped, linearly interpolated energy DF's speed moment.

    Split at energy knots and DF zero crossings. On each speed interval,
    w**2 f(psi-w**2/2) and w**4 f(psi-w**2/2) are polynomials of degree at
    most six, integrated exactly by four-point Gauss quadrature. The energy
    grid must be uniform, as supplied by Engine B's Eddington constructor.
    """
    crossing = df[:-1] * df[1:] < 0
    slope_denominator = jnp.where(crossing, jnp.diff(df), 1.0)
    roots = energy[:-1] - df[:-1] * jnp.diff(energy) / slope_denominator
    roots = jnp.where(crossing, roots, energy[:-1])
    # Interleaving preserves order and gives fixed shapes under differentiation.
    knots = jnp.concatenate(
        (energy[:1], jnp.stack((roots, energy[1:]), axis=1).ravel())
    )
    e = jnp.concatenate(
        (jnp.array([psi]), jnp.clip(knots[::-1], 0, psi), jnp.array([0.0]))
    )
    delta = 2 * (psi - e)
    edges = jnp.where(delta > 0, jnp.sqrt(jnp.where(delta > 0, delta, 1.0)), 0.0)
    half = (edges[1:] - edges[:-1]) / 2
    speed = (edges[1:] + edges[:-1])[:, None] / 2 + half[:, None] * _MOMENT_NODES
    density = jnp.maximum(_interp_uniform(psi - speed**2 / 2, energy, df), 0.0)
    weight = half[:, None] * _MOMENT_WEIGHTS * density
    norm = jnp.sum(weight * speed**2)
    return jnp.sum(weight * speed**4) / jnp.where(norm > 0, norm, 1.0)


def sample_speed_from_f_table(
    key: PRNGKeyArray,
    Psi_r: Float[Array, ""],
    E_grid: Float[Array, "n_e"],
    f_grid: Float[Array, "n_e"],
    n_w: int = 256,
    *,
    uniform_energy_grid: bool = False,
) -> Float[Array, ""]:
    """Sample one speed s ~ s^2 f(Psi_r - s^2/2) on [0, sqrt(2 Psi_r)].

    Differentiable tabulated inverse-CDF on a fixed grid. Returns 0 where the local
    binding potential Psi_r <= ~0 (escape speed vanishes, e.g. at/outside r_t).

    Set the static flag uniform_energy_grid only for an ascending linspace grid
    with at least two nodes, as returned by eddington_invert. The default accepts
    nonuniform grids. Both paths use linear interpolation with clamped endpoints.

    The sqrt floor is 1e-12 times E_grid[-1]; the zero-speed cutoff is 1e-6
    times E_grid[-1]. These relative thresholds preserve energy-scale covariance.
    """
    E_scale = E_grid[-1]
    Psi_safe = jnp.maximum(Psi_r, 1e-12 * E_scale)
    w_grid = jnp.linspace(0.0, jnp.sqrt(2.0 * Psi_safe), n_w)
    interp = _interp_uniform if uniform_energy_grid else jnp.interp
    f_at = interp(Psi_r - w_grid**2 / 2.0, E_grid, f_grid)
    p = jnp.maximum(w_grid**2 * f_at, 0.0)
    s = inverse_cdf_draw(p, w_grid, jax.random.uniform(key))
    # MANDATORY bound guard: zero total weight clamps to grid[-1], not 0
    # (see numerics.inverse_cdf_draw docstring).
    return jnp.where(Psi_r > 1e-6 * E_scale, s, 0.0)


def assign_om_directions(
    key: PRNGKeyArray,
    positions: Float[Array, "N 3"],
    speeds: Float[Array, "N"],
    r_a,
) -> Float[Array, "N 3"]:
    """Assign Cartesian velocities to per-particle speeds.

    r_a is None      -> isotropic (random 3-D direction; byte-identical to the
                        legacy path).
    r_a scalar       -> Osipkov-Merritt stretched split realising
                        beta(r) = r^2/(r^2 + r_a^2).
    r_a (N,) array   -> PER-STAR anisotropy radii (Engine B: r_a_j[component_id]);
                        the stretch broadcasts radii / r_a elementwise. Non-finite
                        entries (inf = isotropic component) give stretch exactly 1
                        via the finite/inf double-where (gradient-safe: no inf
                        enters the graph). A scalar is broadcast to per-star shape
                        first, so scalar == array-of-same-scalar bit-identically.
    The OM branch applies a smooth Cartesian stretch to a global isotropic
    direction. For fixed speeds and finite positive r_a, position derivatives
    exist at the origin. The speed sampler has its own differentiation contract.
    """
    N = positions.shape[0]
    dirs = jax.random.normal(key, shape=(N, 3))
    dirs = dirs / (jnp.linalg.norm(dirs, axis=1, keepdims=True) + 1e-30)
    if r_a is None:
        return speeds[:, None] * dirs

    ra_arr = jnp.broadcast_to(jnp.asarray(r_a), (N,))
    finite = jnp.isfinite(ra_arr)
    ra_safe = jnp.where(finite, ra_arr, 1.0)
    inverse_ra = jnp.where(finite, 1.0 / ra_safe, 0.0)
    y = positions * inverse_ra[:, None]
    h = jnp.sqrt(1.0 + jnp.sum(y**2, axis=1))
    # Equivalent to b I + (1-b) r_hat r_hat^T, without division by radius.
    projection = jnp.sum(y * dirs, axis=1)
    directions = dirs / h[:, None] + y * (projection / (h * (h + 1)))[:, None]
    return speeds[:, None] * directions


__all__ = ["eddington_invert", "sample_speed_from_f_table", "assign_om_directions"]

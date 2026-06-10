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

      sample s isotropically in the stretched frame (cos theta ~ U[-1,1]),
      v_r = s cos theta along r_hat,   w_t = s sin theta,
      v_t = w_t / sqrt(1 + r^2/r_a^2)  in a random azimuthal direction perp to r_hat.

See docs/website/99-bibliography/per-paper/merritt-1985.md.
"""

import jax
import jax.numpy as jnp
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
    bnd = drho_dPsi[-1]                # d rho/d Psi at Psi=0 (truncation boundary term)

    # End just below Psi0: E=Psi0 is the singular central energy (the Eddington
    # integrand reaches r->0); central lookups clamp to f_grid[-1], and the w^2 factor
    # makes the w->0 (E->Psi0) contribution negligible for sampling.
    E_grid = jnp.linspace(1e-4 * Psi0, 0.999 * Psi0, n_e)

    def f_one(E):
        # u = sqrt(E - Psi): int_0^E g/sqrt(E-Psi) dPsi = 2 int_0^sqrt(E) g(E-u^2) du
        u = jnp.linspace(0.0, jnp.sqrt(E), n_u)
        g = jnp.interp(E - u**2, Psi_asc, d2_asc)
        return (2.0 * jnp.trapezoid(g, u) + bnd / jnp.sqrt(E)) / (jnp.sqrt(8.0) * jnp.pi**2)

    # Raw (unclamped) f(E): the speed sampler clamps the speed pdf at use, and the raw
    # values let callers detect a genuinely negative (unphysical) OM DF.
    f_grid = jax.vmap(f_one)(E_grid)
    return E_grid, f_grid


def sample_speed_from_f_table(
    key: PRNGKeyArray,
    Psi_r: Float[Array, ""],
    E_grid: Float[Array, "n_e"],
    f_grid: Float[Array, "n_e"],
    n_w: int = 256,
) -> Float[Array, ""]:
    """Sample one speed s ~ s^2 f(Psi_r - s^2/2) on [0, sqrt(2 Psi_r)].

    Differentiable tabulated inverse-CDF on a fixed grid. Returns 0 where the local
    binding potential Psi_r <= 0 (escape speed vanishes, e.g. at/outside r_t).
    """
    Psi_safe = jnp.maximum(Psi_r, 1e-12)
    w_grid = jnp.linspace(0.0, jnp.sqrt(2.0 * Psi_safe), n_w)
    f_at = jnp.interp(Psi_r - w_grid**2 / 2.0, E_grid, f_grid)
    p = jnp.maximum(w_grid**2 * f_at, 0.0)
    dw = w_grid[1] - w_grid[0]
    cdf = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (p[1:] + p[:-1])) * dw])
    cdf = cdf / (cdf[-1] + 1e-30)
    s = jnp.interp(jax.random.uniform(key), cdf, w_grid)
    return jnp.where(Psi_r > 1e-6, s, 0.0)


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
    """
    N = positions.shape[0]
    if r_a is None:
        dirs = jax.random.normal(key, shape=(N, 3))
        dirs = dirs / (jnp.linalg.norm(dirs, axis=1, keepdims=True) + 1e-30)
        return speeds[:, None] * dirs

    radii = jnp.linalg.norm(positions, axis=1)
    r_hat = positions / (radii[:, None] + 1e-30)
    k_cos, k_az = jax.random.split(key)

    cos_t = jax.random.uniform(k_cos, (N,), minval=-1.0, maxval=1.0)
    sin_t = jnp.sqrt(jnp.maximum(1.0 - cos_t**2, 0.0))
    ra_arr = jnp.broadcast_to(jnp.asarray(r_a), radii.shape)
    finite = jnp.isfinite(ra_arr)
    ra_safe = jnp.where(finite, ra_arr, 1.0)
    stretch = jnp.where(finite, jnp.sqrt(1.0 + (radii / ra_safe) ** 2), 1.0)

    v_r = speeds * cos_t                 # signed radial component (stretched frame)
    v_t = speeds * sin_t / stretch       # physical tangential magnitude (un-stretched)

    # Random azimuthal unit vector in the plane perpendicular to r_hat.
    rand = jax.random.normal(k_az, (N, 3))
    rand = rand - jnp.sum(rand * r_hat, axis=1, keepdims=True) * r_hat
    t_hat = rand / (jnp.linalg.norm(rand, axis=1, keepdims=True) + 1e-30)

    return v_r[:, None] * r_hat + v_t[:, None] * t_hat


__all__ = ["eddington_invert", "sample_speed_from_f_table", "assign_om_directions"]

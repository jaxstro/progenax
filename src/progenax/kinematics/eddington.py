"""Shared Osipkov-Merritt (Merritt 1985) sampling helpers.

Two differentiable, vmap/JIT-safe primitives reused by the anisotropic velocity DFs:

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

    r_a is None  -> isotropic (random 3-D direction; byte-identical to the legacy path).
    r_a is float -> Osipkov-Merritt stretched split realising beta(r)=r^2/(r^2+r_a^2).
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
    stretch = jnp.sqrt(1.0 + (radii / r_a) ** 2)

    v_r = speeds * cos_t                 # signed radial component (stretched frame)
    v_t = speeds * sin_t / stretch       # physical tangential magnitude (un-stretched)

    # Random azimuthal unit vector in the plane perpendicular to r_hat.
    rand = jax.random.normal(k_az, (N, 3))
    rand = rand - jnp.sum(rand * r_hat, axis=1, keepdims=True) * r_hat
    t_hat = rand / (jnp.linalg.norm(rand, axis=1, keepdims=True) + 1e-30)

    return v_r[:, None] * r_hat + v_t[:, None] * t_hat


__all__ = ["sample_speed_from_f_table", "assign_om_directions"]

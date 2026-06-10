# progenax/src/progenax/kinematics/_speed_kernels.py
"""Shared per-star speed-sampling kernels (single physics home).

`_sample_costheta_given_u` (+ its grid resolution `_N_C`) has three consumers
-- `kinematics/limepy_df.py`, `kinematics/michie_df.py`, and the jitted
cluster sampler `cluster/sampling.py` -- so it lives here exactly once
(2026-06-10 review consolidation). It is physics (the Michie/Osipkov-Merritt
angular conditional), not a numerics primitive, hence a kinematics module
rather than `progenax.numerics`.

`_ORACLE_BATCH` is the shared `jax.lax.map` chunk size for the quadrature
ORACLE branches (king/limepy/michie `speed_method="quadrature"`): chunking the
per-star vmap bounds the live E_gamma Poisson-sum buffers to
O(batch * n_u * ~91) floats instead of O(N * n_u * ~91) (~11 GB at N=2e4).
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from progenax.numerics import inverse_cdf_draw

_N_C = 128           # cos(theta) grid (anisotropic conditional)
_ORACLE_BATCH = 2048  # lax.map chunk size for the per-star quadrature oracles


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
    return inverse_cdf_draw(w_c, c_grid, jax.random.uniform(key))


__all__ = ["_N_C", "_ORACLE_BATCH", "_sample_costheta_given_u"]

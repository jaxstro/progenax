"""Shared numerical primitives.

The cumulative-trapezoid pass now lives in jaxstro
(``jaxstro.numerics.integration.cumulative_trapz``) as the ecosystem's single
source of truth, standardized on the dx-OUTSIDE ordering (pairwise average ->
cumsum -> scale by scalar dx once -> leading zero). ``cumulative_trapz`` is
imported here so the progenax call sites and the ``inverse_cdf_draw`` kernel
below import it from ``progenax.numerics``. The jaxstro signature
``cumulative_trapz(y, x=None, *, dx, axis)`` is keyword-compatible with every
progenax call site (all pass ``dx=``/``axis=``).

The former dx-INSIDE sites (``profiles/density_poisson.py``, ``profiles/api.py``,
``kinematics/eff_df.py``) multiplied dx inside the cumsum; against the dx-outside
form they agree only to ~1 ulp (measured: 124/257 elements differ, max rel. diff
8.9e-16), within their existing test budgets.

The trapezoid-CDF inverse-draw kernel (``inverse_cdf_draw``) stays local here.
Fully differentiable; no data-dependent shapes.
"""
import jax.numpy as jnp
from jaxstro.numerics.integration import cumulative_trapz
from jaxtyping import Array, Float


def inverse_cdf_draw(
    weight: Float[Array, "n"],
    grid: Float[Array, "n"],
    unif: Float[Array, ""],
    reg: float = 1e-30,
) -> Float[Array, ""]:
    """Differentiable inverse-CDF draw from an unnormalized weight on a uniform grid.

    Builds the trapezoid CDF of ``weight`` over ``grid`` (uniform spacing inferred
    from the first cell), normalizes with the ``+reg`` guard the sampling kernels
    use, and interpolates the quantile. For a zero total weight the ``+reg``
    guard yields a finite draw — ``jnp.interp`` against the all-zero CDF clamps
    to ``grid[-1]``, not NaN — so callers MUST keep their bound guard (e.g.
    ``where(W > 1e-6, u, 0.0)``). Scalar draw; ``jax.vmap`` over stars.
    """
    dx = grid[1] - grid[0]
    cdf = cumulative_trapz(weight, dx=dx)
    cdf = cdf / (cdf[-1] + reg)
    return jnp.interp(unif, cdf, grid)

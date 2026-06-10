"""Shared numerical primitives (single source of truth).

Consolidates two patterns the 2026-06-10 code review found duplicated:
the cumulative-trapezoid pass and the trapezoid-CDF inverse-draw kernel.
The op order (pairwise average -> cumsum -> leading zero, dx OUTSIDE the
cumsum) is exactly the dx-outside majority pattern, so those migrated call
sites are bit-identical: the 8 speed-CDF sites (``kinematics/king_df.py``,
``kinematics/limepy_df.py`` x3, ``kinematics/michie_df.py`` x2,
``kinematics/eddington.py``) plus ``cluster/multicomponent.py`` x2. The
dx-INSIDE sites (``profiles/density_poisson.py``, ``profiles/api.py``,
``kinematics/eff_df.py``) multiply dx inside the cumsum and agree only to
~1 ulp (measured: 124/257 elements differ, max rel. diff 8.9e-16); they are
gated by their own test budgets at migration time (Task 8), with the
pre-multiplied-weights + ``dx=1.0`` escape hatch if a budget moves.
Fully differentiable; no data-dependent shapes.
"""
import jax.numpy as jnp
from jaxtyping import Array, Float


def cumulative_trapezoid(
    y: Float[Array, "... n"],
    dx: float | Float[Array, ""],
    axis: int = -1,
) -> Float[Array, "... n"]:
    """Cumulative trapezoid integral with a leading zero, uniform spacing.

    out[..., k] = sum_{i<k} 0.5 * (y[..., i] + y[..., i+1]) * dx, out[..., 0] = 0.
    Same length as ``y`` along ``axis``.
    """
    y = jnp.moveaxis(y, axis, -1)
    inner = jnp.cumsum(0.5 * (y[..., 1:] + y[..., :-1]), axis=-1) * dx
    zero = jnp.zeros(y.shape[:-1] + (1,), dtype=inner.dtype)
    return jnp.moveaxis(jnp.concatenate([zero, inner], axis=-1), -1, axis)


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
    cdf = cumulative_trapezoid(weight, dx=dx)
    cdf = cdf / (cdf[-1] + reg)
    return jnp.interp(unif, cdf, grid)

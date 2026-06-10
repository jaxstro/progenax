"""Shared numerical primitives (single source of truth).

Consolidates two patterns the 2026-06-10 code review found duplicated:
the cumulative-trapezoid pass (5 inline copies: Poisson integrations in
``profiles/density_poisson.py``, ``profiles/api.py``, ``kinematics/eff_df.py``,
``cluster/multicomponent.py`` x2) and the trapezoid-CDF inverse-draw kernel
(8 speed/angle sampling sites). The op order (pairwise average -> cumsum ->
leading zero) is EXACTLY the inline pattern, so migrated call sites are
bit-identical. Fully differentiable; no data-dependent shapes.
"""
import jax.numpy as jnp
from jaxtyping import Array, Float


def cumulative_trapezoid(
    y: Float[Array, "... n"],
    dx: float,
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
    use (a zero total weight then draws grid[0] instead of NaN), and interpolates
    the quantile. Scalar draw; ``jax.vmap`` over stars.
    """
    dx = grid[1] - grid[0]
    cdf = cumulative_trapezoid(weight, dx=dx)
    cdf = cdf / (cdf[-1] + reg)
    return jnp.interp(unif, cdf, grid)

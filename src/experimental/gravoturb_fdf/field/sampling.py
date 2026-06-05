"""Tail/smooth star sampling from an FDF density field (spec §3.6).

Two categorical PMFs over field cells set where stars form:
  p_tail   ∝ w ρ   — the dense, gravitationally-collapsing tail (w = soft mask),
  p_smooth ∝ ρ     — the diffuse background.
N_tail = round(f_sub · N⋆) stars are drawn from p_tail, the remainder from p_smooth;
each star gets an independent sub-voxel uniform jitter so positions are continuous.

Categorical sampling is non-differentiable in the resulting positions (accepted,
spec §8 — the differentiable interface is the fitted Q(f_sub) surrogate in P3).
``f_sub`` and ``n_stars`` are static (set sample shapes); the field is traced.

JAX-native (jax.random).
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from gravoturb_fdf.field.tail import tail_weights


def sample_cell_indices(
    s: Float[Array, "nx ny nz"],
    s_t: Float[Array, ""],
    kappa: Float[Array, ""],
    f_sub: float,
    n_stars: int,
    key: jax.Array,
) -> tuple[Int[Array, " n_tail"], Int[Array, " n_smooth"]]:
    r"""Draw flat cell indices: N_tail from p_tail ∝ wρ, N_smooth from p_smooth ∝ ρ.

    Returns ``(tail_idx, smooth_idx)``. ``n_tail = round(f_sub · n_stars)`` (Python int
    so the categorical sample shapes are static).
    """
    rho = jnp.exp(s).ravel()
    w = tail_weights(s, s_t, kappa).ravel()
    p_tail = w * rho
    p_tail = p_tail / jnp.sum(p_tail)
    p_smooth = rho / jnp.sum(rho)

    n_tail = int(round(f_sub * n_stars))
    n_smooth = n_stars - n_tail
    n_cells = rho.size

    k_tail, k_smooth = jax.random.split(key)
    tail_idx = jax.random.choice(k_tail, n_cells, (n_tail,), replace=True, p=p_tail)
    smooth_idx = jax.random.choice(k_smooth, n_cells, (n_smooth,), replace=True, p=p_smooth)
    return tail_idx, smooth_idx


def cells_to_positions(
    indices: Int[Array, " n"],
    shape: tuple[int, int, int],
    key: jax.Array,
    box_size: float = 1.0,
) -> Float[Array, "n 3"]:
    r"""Map flat cell indices → continuous positions with sub-voxel uniform jitter.

    ``position = (ijk + U[0,1)^3) · dx`` with ``dx = box_size / n`` per axis (cubic
    grid). The jitter keeps each star strictly inside its own voxel.
    """
    nx, ny, nz = shape
    ijk = jnp.stack(jnp.unravel_index(indices, shape), axis=-1)  # (n, 3)
    jitter = jax.random.uniform(key, ijk.shape)
    dx = jnp.array([box_size / nx, box_size / ny, box_size / nz])
    return (ijk + jitter) * dx


def sample_positions(
    s: Float[Array, "nx ny nz"],
    s_t: Float[Array, ""],
    kappa: Float[Array, ""],
    f_sub: float,
    n_stars: int,
    key: jax.Array,
    box_size: float = 1.0,
) -> Float[Array, "n_stars 3"]:
    r"""Sample ``n_stars`` star positions (tail + smooth) from the FDF field.

    Returns positions in [0, box_size)^3. Non-differentiable in the positions
    (categorical sampling); Q is scale-invariant, so ``box_size`` is conventional.
    """
    k_idx, k_jit = jax.random.split(key)
    tail_idx, smooth_idx = sample_cell_indices(s, s_t, kappa, f_sub, n_stars, k_idx)
    all_idx = jnp.concatenate([tail_idx, smooth_idx])
    return cells_to_positions(all_idx, s.shape, k_jit, box_size)

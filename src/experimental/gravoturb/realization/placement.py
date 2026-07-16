"""Tail/smooth star sampling from an FDF density field (spec §3.6).

Two categorical PMFs over field cells set where stars form:
  p_tail   ∝ w ρ   — the dense, gravitationally-collapsing tail (w = soft mask),
  p_smooth ∝ ρ     — the diffuse background.
N_tail = round(f_sub · N⋆) stars are drawn from p_tail, the remainder from p_smooth;
each star gets an independent sub-voxel uniform jitter so positions are continuous.

Categorical sampling is non-differentiable in the resulting positions (accepted, spec §8 —
the differentiable interface is the analytic predicted-statistics inference layer in
``inference/`` (AC11-AC17), not a fitted Q surrogate).
``f_sub`` and ``n_stars`` are static (set sample shapes); the field is traced.

JAX-native (jax.random).
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

# ── dense-tail mask (merged from the pre-rename tail.py; bodies unchanged, spec §3.6) ──


def collapse_weights(
    s: Float[Array, "..."],
    s_t: Float[Array, ""],
    mask_sharpness: Float[Array, ""],
) -> Float[Array, "..."]:
    r"""Soft tail membership w = σ(κ(s − s_t)) ∈ (0,1).

    w(s_t) = 0.5; monotone increasing in s; κ→∞ recovers the hard indicator s > s_t.
    """
    return jax.nn.sigmoid(mask_sharpness * (s - s_t))


def f_tail_actual(
    s: Float[Array, "..."],
    rho: Float[Array, "..."],
    s_t: Float[Array, ""],
    mask_sharpness: Float[Array, ""],
) -> Float[Array, ""]:
    r"""Mass-weighted dense-tail fraction f_tail_actual = Σ w ρ / Σ ρ.

    The realized counterpart of BM19 f_dense (AC6 compares the two). Differentiable
    in (s_t, κ); raising s_t removes mass from the tail (∂/∂s_t < 0).
    """
    w = collapse_weights(s, s_t, mask_sharpness)
    return jnp.sum(w * rho) / jnp.sum(rho)


def sample_cic_counts(
    s: Float[Array, "n n n"],
    n_bar: Float[Array, ""],
    cell_size: int,
    key: jax.Array,
) -> Int[Array, "m m m"]:
    r"""Clean inhomogeneous-Poisson counts-in-cells from a log-density field ``s``.

    A true Poisson point process with intensity proportional to rho = exp(s): the count in a
    cubic cell is ``Poisson(n_bar * rho_cell)``, ``rho_cell`` = the cell-averaged mean-1 density
    (a sum of sub-cell Poissons is Poisson of the integral, so this is resolution-independent at
    the cell scale). This replaces the ``cloud_to_stars`` multinomial/with-replacement fine-cell
    sampler for CIC inference: that sampler's counts pick up a fixed-total + fine-cell-pile-up
    over-dispersion (an artifact, density/grid-dependent) that the cell-mean count model can't
    represent; the clean Poisson process matches ``count_distribution``. ``cell_size`` divides the
    grid; returns the ``(M, M, M)`` integer count grid, ``M = n // cell_size``."""
    rho_tilde = jnp.exp(s) / jnp.mean(jnp.exp(s))
    n = s.shape[0]
    m = n // cell_size
    rho_cell = rho_tilde.reshape(m, cell_size, m, cell_size, m, cell_size).mean(axis=(1, 3, 5))
    return jax.random.poisson(key, n_bar * rho_cell)


def sample_cell_indices(
    s: Float[Array, "nx ny nz"],
    s_t: Float[Array, ""],
    mask_sharpness: Float[Array, ""],
    f_sub: float,
    n_stars: int,
    key: jax.Array,
    s_density: Float[Array, "nx ny nz"] | None = None,
) -> tuple[Int[Array, " n_tail"], Int[Array, " n_smooth"]]:
    r"""Draw flat cell indices: N_tail from p_tail ∝ wρ, N_smooth from p_smooth ∝ ρ.

    Returns ``(tail_idx, smooth_idx)``. ``n_tail = round(f_sub · n_stars)`` (Python int
    so the categorical sample shapes are static).

    ``s_density`` (optional) is the log-density field used for the PLACEMENT PMF (ρ ∝
    e^{s_density}); ``s`` always defines the dense-tail mask against ``s_t``. With a spherical
    envelope, ``s_density = s_turb + ln ρ_env(r)`` (placement follows the centrally-concentrated
    total density) while the tail mask stays on ``s = s_turb`` (dense clumps = LOCAL
    overdensities, decoupled from the envelope). Default ``None`` ⇒ ``s_density = s`` (original
    behaviour: a featureless / periodic box).
    """
    s_place = s if s_density is None else s_density
    rho = jnp.exp(s_place).ravel()
    w = collapse_weights(s, s_t, mask_sharpness).ravel()
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
    mask_sharpness: Float[Array, ""],
    f_sub: float,
    n_stars: int,
    key: jax.Array,
    box_size: float = 1.0,
    s_density: Float[Array, "nx ny nz"] | None = None,
) -> Float[Array, "n_stars 3"]:
    r"""Sample ``n_stars`` star positions (tail + smooth) from the FDF field.

    Returns positions in [0, box_size)^3. Non-differentiable in the positions
    (categorical sampling); Q is scale-invariant, so ``box_size`` is conventional.
    ``s_density`` (optional) is the placement-density field (e.g. envelope-modulated
    ``s_turb + ln ρ_env``); ``s`` defines the dense-tail mask. See :func:`sample_cell_indices`.
    """
    k_idx, k_jit = jax.random.split(key)
    tail_idx, smooth_idx = sample_cell_indices(
        s, s_t, mask_sharpness, f_sub, n_stars, k_idx, s_density=s_density
    )
    all_idx = jnp.concatenate([tail_idx, smooth_idx])
    return cells_to_positions(all_idx, s.shape, k_jit, box_size)

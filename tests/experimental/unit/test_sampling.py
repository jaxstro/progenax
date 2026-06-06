"""Tail/smooth star sampling from an FDF field (spec §3.6).

Stars are drawn from two categorical PMFs over field cells: p_tail ∝ w ρ (the dense
collapsing tail) and p_smooth ∝ ρ (the diffuse background). N_tail = round(f_sub·N⋆)
come from p_tail, the rest from p_smooth; each gets a sub-voxel uniform jitter.
Categorical sampling is non-differentiable in positions (accepted, spec §8).
"""

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.experimental

MACH, B, ALPHA, KAPPA = 8.0, 0.5, 1.8, 6.0


def _field(seed=0, n=16):
    from gravoturb_fdf.field.field import gaussian_random_field, rank_copula_field

    g = gaussian_random_field((n, n, n), beta=3.5, key=jax.random.PRNGKey(seed))
    return rank_copula_field(g, MACH, B, ALPHA)


def test_sampling_counts_split():
    """N_tail = round(f_sub·N⋆); N_smooth = N⋆ − N_tail; total = N⋆."""
    from gravoturb_fdf.field.sampling import sample_cell_indices

    s = _field()
    from gravoturb_fdf.theory.bm19 import sigma_s_squared, transition_density

    s_t = float(transition_density(ALPHA, sigma_s_squared(MACH, B)))
    tail_idx, smooth_idx = sample_cell_indices(
        s, s_t, KAPPA, f_sub=0.3, n_stars=1000, key=jax.random.PRNGKey(1)
    )
    assert tail_idx.size == 300
    assert smooth_idx.size == 700
    assert tail_idx.size + smooth_idx.size == 1000


def test_tail_stars_in_denser_cells():
    """Tail-sampled cells have higher mean ρ than smooth-sampled cells."""
    from gravoturb_fdf.field.sampling import sample_cell_indices
    from gravoturb_fdf.theory.bm19 import sigma_s_squared, transition_density

    s = _field()
    rho = jnp.exp(s).ravel()
    s_t = float(transition_density(ALPHA, sigma_s_squared(MACH, B)))
    tail_idx, smooth_idx = sample_cell_indices(
        s, s_t, KAPPA, f_sub=0.3, n_stars=4000, key=jax.random.PRNGKey(2)
    )
    assert float(jnp.mean(rho[tail_idx])) > float(jnp.mean(rho[smooth_idx]))


def test_positions_within_box_and_count():
    """sample_positions returns (N⋆,3) inside [0,box_size)."""
    from gravoturb_fdf.field.sampling import sample_positions
    from gravoturb_fdf.theory.bm19 import sigma_s_squared, transition_density

    s = _field()
    s_t = float(transition_density(ALPHA, sigma_s_squared(MACH, B)))
    pos = sample_positions(
        s, s_t, KAPPA, f_sub=0.4, n_stars=500, key=jax.random.PRNGKey(3), box_size=2.0
    )
    assert pos.shape == (500, 3)
    assert jnp.all(pos >= 0.0) and jnp.all(pos < 2.0)


def test_subvoxel_jitter_keeps_star_in_its_cell():
    """floor(position/dx) recovers the sampled cell index (jitter stays sub-voxel)."""
    from gravoturb_fdf.field.sampling import cells_to_positions

    shape = (8, 8, 8)
    idx = jnp.array([0, 73, 511, 256])
    box = 1.0
    dx = box / 8
    pos = cells_to_positions(idx, shape, key=jax.random.PRNGKey(4), box_size=box)
    ijk = jnp.floor(pos / dx).astype(int)
    recovered = ijk[:, 0] * 64 + ijk[:, 1] * 8 + ijk[:, 2]
    assert jnp.array_equal(recovered, idx)


def test_sampling_deterministic_in_key():
    """Same key → identical positions."""
    from gravoturb_fdf.field.sampling import sample_positions
    from gravoturb_fdf.theory.bm19 import sigma_s_squared, transition_density

    s = _field()
    s_t = float(transition_density(ALPHA, sigma_s_squared(MACH, B)))
    kw = dict(s=s, s_t=s_t, kappa=KAPPA, f_sub=0.3, n_stars=200)
    a = sample_positions(**kw, key=jax.random.PRNGKey(9))
    b = sample_positions(**kw, key=jax.random.PRNGKey(9))
    assert jnp.allclose(a, b)

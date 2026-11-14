"""Approved angular-averaging prototype; production sampler unchanged."""

import jax
import jax.numpy as jnp

from progenax.cluster._radial_cdf import component_linear_quantile
from progenax.kinematics.eddington import sample_speed_from_f_table


def _engine_b_speed_moment(model, key, labels, weights, G):
    """Conditional speed moment with bounded reverse storage."""
    state = model.engine_b
    count = labels.size
    _, kr, _, ks, _ = jax.random.split(key, 5)
    radii = component_linear_quantile(
        jax.random.uniform(kr, (count,)), labels, model._r_grid, model._cdf_j
    )
    psi = jnp.interp(radii, model._r_grid, state.Psi_grid)
    keys = jax.random.split(ks, count)
    finite = jnp.isfinite(state.r_a_j[labels])
    ra = jnp.where(finite, state.r_a_j[labels], 1.0)
    factor = 1 / 3 + 2 / 3 * jnp.where(finite, 1 / (1 + (radii / ra) ** 2), 1.0)
    batch = 64
    pad = (-count) % batch
    arrays = jax.tree.map(
        lambda x: jnp.pad(x, [(0, pad)] + [(0, 0)] * (x.ndim - 1), mode="edge"),
        (keys, psi, labels, factor, weights),
    )
    arrays = (*arrays, jnp.arange(count + pad) < count)

    @jax.checkpoint
    def chunk(index):
        k, p, c, d, w, valid = jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(x, index * batch, batch), arrays
        )
        speed = jax.vmap(
            lambda kk, pp, cc: sample_speed_from_f_table(
                kk,
                pp,
                state.E_grid,
                state.f_j_grid[cc],
                n_w=1024,
                uniform_energy_grid=True,
            )
        )(k, p, c)
        return jnp.sum(jnp.where(valid, speed**2 * d * w, 0.0))

    return (
        G
        * jnp.sum(jax.lax.map(chunk, jnp.arange((count + pad) // batch)))
        / (4 * jnp.pi * state.mu)
    )

"""Retained isotropic Engine B: indexed radial lookup and streamed speed moment."""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from diagnose_engine_b_population_moments import model_at

from progenax.cluster._radial_cdf import component_linear_quantile
from progenax.kinematics.eddington import (
    assign_om_directions,
    sample_speed_from_f_table,
)

OUT = Path(
    "docs/reviews/plummer-om-artifacts/engine-b-streamed-moment-performance.json"
)


def moment(theta, key, *, execution="map", checkpoint=True, batch=64):
    """Retained isotropic prototype; checkpointed map is the selected execution.

    Alternate execution is retained only to reproduce recorded benchmarks.
    """
    if execution not in {"map", "fori", "chunk_map"}:
        raise ValueError("execution must be map, fori or chunk_map")
    return moment_from_model(
        model_at(theta), key, execution=execution, checkpoint=checkpoint, batch=batch
    )


def moment_from_model(m, key, *, execution="map", checkpoint=True, batch=64):
    """Same retained-model calculation, exposing the constructor boundary."""
    st = m.engine_b
    count = 100000
    labels = jnp.repeat(jnp.arange(2), 50000)
    pi = jax.nn.softmax(jnp.log(m.N_frac_j + 1e-30))
    expected_mass = m.m_j + 99999 * (pi @ m.m_j)

    def draw(k):
        _, kr, _, ks, kd = jax.random.split(k, 5)
        u = jax.random.uniform(kr, (count,))
        radii = component_linear_quantile(u, labels, m._r_grid, m._cdf_j)
        psi = jnp.interp(radii, m._r_grid, st.Psi_grid)
        keys = jax.random.split(ks, count)
        dirs = assign_om_directions(
            kd, jnp.zeros((count, 3)), jnp.ones(count), st.r_a_j[labels]
        )
        factor = pi[labels] * expected_mass[labels] / (50000 * 4 * jnp.pi * st.mu)
        # A static final partial batch retains every draw and ignores padding.
        pad = (-count) % batch
        inputs = jax.tree.map(
            lambda x: jnp.pad(x, [(0, pad)] + [(0, 0)] * (x.ndim - 1), mode="edge"),
            (keys, psi, labels, dirs, factor),
        )
        valid = jnp.arange(count + pad) < count

        def chunk_sum(chunk):
            k, p, c, d, f, mask = chunk
            speed = jax.vmap(
                lambda kk, pp, cc: sample_speed_from_f_table(
                    kk, pp, st.E_grid, st.f_j_grid[cc], n_w=1024
                )
            )(k, p, c)
            v2 = jnp.sum((speed[:, None] * d) ** 2, axis=1)
            return jnp.sum(jnp.where(mask, v2 * f, 0.0))

        if execution == "chunk_map":
            chunks = jax.tree.map(
                lambda x: x.reshape((-1, batch) + x.shape[1:]), (*inputs, valid)
            )
            kernel = jax.checkpoint(chunk_sum) if checkpoint else chunk_sum
            return jnp.sum(jax.lax.map(kernel, chunks))

        def body(i, total):
            chunk = jax.tree.map(
                lambda x: jax.lax.dynamic_slice_in_dim(x, i * batch, batch),
                (*inputs, valid),
            )
            return total + chunk_sum(chunk)

        if checkpoint:
            body = jax.checkpoint(body)
        if execution == "map":
            return jnp.sum(
                jax.lax.map(
                    lambda i: body(i, jnp.array(0.0)),
                    jnp.arange((count + pad) // batch),
                )
            )
        return jax.lax.fori_loop(0, (count + pad) // batch, body, jnp.array(0.0))

    return jnp.mean(jax.lax.map(draw, jax.random.split(key, 4)))


def main():
    assert jax.config.jax_enable_x64
    if OUT.exists():
        raise FileExistsError(OUT)
    theta = jnp.array([0.6, 2.0, 0.8])
    e = jnp.array([0.0, 0.0, 1.0])
    key = jax.random.PRNGKey(1)
    fn = jax.jit(
        lambda t, e, k: jax.jvp(
            lambda tt: moment(tt, k, execution="fori", checkpoint=False), (t,), (e,)
        )
    )
    times = []
    for _ in range(4):
        start = time.perf_counter()
        result = jax.block_until_ready(fn(theta, e, key))
        times.append(time.perf_counter() - start)
    baseline = json.loads(
        Path(
            "docs/reviews/plummer-om-artifacts/engine-b-moment-performance.json"
        ).read_text()
    )["modes"]["outer_jit"]["value_and_jvp"]
    np.testing.assert_allclose(np.asarray(result), baseline, rtol=1e-12, atol=1e-12)
    record = dict(
        scope="retained isotropic model only; same seed and draws; streamed scalar reduction batch64; full JIT core-scale JVP",
        first_call_seconds=times[0],
        warm_seconds=times[1:],
        value_and_jvp=np.asarray(result).tolist(),
        parity_pass=True,
        compiled_temporary_bytes=fn.lower(theta, e, key)
        .compile()
        .memory_analysis()
        .temp_size_in_bytes,
    )
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(record)


if __name__ == "__main__":
    main()

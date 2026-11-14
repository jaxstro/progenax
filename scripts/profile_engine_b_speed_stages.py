"""Exact reverse chain split: DF interpolation versus inverse-CDF sampling."""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from diagnose_engine_b_population_moments import model_at
from jaxstro.numerics.sampling import inverse_cdf_draw

from progenax.cluster._radial_cdf import component_linear_quantile

OUT = Path("docs/reviews/plummer-om-artifacts/engine-b-speed-stage-profile.json")


def tables(psi, E, f):
    w = jax.vmap(
        lambda p: jnp.linspace(0.0, jnp.sqrt(2 * jnp.maximum(p, 1e-12 * E[-1])), 1024)
    )(psi)
    values = jnp.interp(psi[:, None] - w**2 / 2, E, f)
    return w, jnp.maximum(w**2 * values, 0.0)


def draw(w, p, u):
    return jnp.sum(jax.vmap(inverse_cdf_draw)(p, w, u) ** 2)


def timed(fn, *args):
    f = jax.jit(fn)
    c = f.lower(*args).compile()
    assert c.memory_analysis().temp_size_in_bytes < 32_000_000_000
    result = jax.block_until_ready(c(*args))
    times = []
    for _ in range(10):
        t = time.perf_counter()
        result = jax.block_until_ready(c(*args))
        times.append(time.perf_counter() - t)
    return result, dict(
        seconds=times,
        median_seconds=float(np.median(times)),
        temporary_bytes=c.memory_analysis().temp_size_in_bytes,
    )


def main():
    assert jax.config.jax_enable_x64
    if OUT.exists():
        raise FileExistsError(OUT)
    m = model_at(jnp.array([0.6, 2.0, 0.8]))
    st = m.engine_b
    key = jax.random.split(jax.random.PRNGKey(1), 4)[0]
    _, kr, _, ks, _ = jax.random.split(key, 5)
    radial = jax.random.uniform(kr, (100000,))
    keys = jax.random.split(ks, 100000)
    record = {}
    for component, start in [(0, 0), (1, 50000)]:
        ids = jnp.full((64,), component)
        r = component_linear_quantile(
            radial[start : start + 64], ids, m._r_grid, m._cdf_j
        )
        psi = jnp.interp(r, m._r_grid, st.Psi_grid)
        E = st.E_grid
        f = st.f_j_grid[component]
        u = jax.vmap(jax.random.uniform)(keys[start : start + 64])
        wp, back = jax.vjp(tables, psi, E, f)
        covector = jax.grad(draw, argnums=(0, 1))(*wp, u)
        split, table_stats = timed(back, covector)
        _, cdf_stats = timed(jax.value_and_grad(draw, argnums=(0, 1)), *wp, u)
        full, full_stats = timed(
            jax.value_and_grad(
                lambda pp, ee, ff: draw(*tables(pp, ee, ff), u), argnums=(0, 1, 2)
            ),
            psi,
            E,
            f,
        )
        for a, b in zip(split, full[1]):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
        record[str(component)] = dict(
            table_pullback=table_stats,
            cdf_value_and_pullback=cdf_stats,
            full_value_and_pullback=full_stats,
            parity_pass=True,
        )
        OUT.write_text(json.dumps(record, indent=2) + "\n")
        print(component, record[str(component)], flush=True)


if __name__ == "__main__":
    main()

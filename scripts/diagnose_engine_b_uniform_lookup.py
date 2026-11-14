"""Reproduce the uniform-energy lookup benchmark using the adopted kernel."""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import profile_engine_b_reduction as prototype

from progenax.kinematics.eddington import sample_speed_from_f_table

OUT = Path("docs/reviews/plummer-om-artifacts/engine-b-uniform-energy-performance.json")


def speed(key, psi, E, f, n_w=1024):
    return sample_speed_from_f_table(key, psi, E, f, n_w=n_w, uniform_energy_grid=True)


def main():
    assert jax.config.jax_enable_x64
    if OUT.exists():
        raise FileExistsError(OUT)
    prototype.sample_speed_from_f_table = speed
    x = jnp.array([0.6, 2.0, 0.8])
    key = jax.random.PRNGKey(1)
    fn = jax.jit(jax.value_and_grad(lambda t: prototype.moment(t, key)))
    start = time.perf_counter()
    compiled = fn.lower(x).compile()
    compile_seconds = time.perf_counter() - start
    mem = compiled.memory_analysis().temp_size_in_bytes
    assert mem < 32_000_000_000
    ref = json.loads(
        Path(
            "docs/reviews/plummer-om-artifacts/engine-b-streamed-reverse.json"
        ).read_text()
    )
    record = dict(compile_seconds=compile_seconds, temporary_bytes=mem, seconds=[])
    for _ in range(3):
        start = time.perf_counter()
        v, g = jax.block_until_ready(compiled(x))
        record["seconds"].append(time.perf_counter() - start)
        np.testing.assert_allclose(v, ref["value"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(g, ref["reference_forward"], rtol=1e-12, atol=1e-12)
        record.update(value=float(v), gradient=g.tolist(), parity_pass=True)
        OUT.write_text(json.dumps(record, indent=2) + "\n")
        print(record, flush=True)


if __name__ == "__main__":
    main()

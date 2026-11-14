"""Exact chain-rule split of retained reverse gradient into construction/sampling."""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from diagnose_engine_b_population_moments import model_at
from profile_engine_b_reduction import moment_from_model

OUT = Path("docs/reviews/plummer-om-artifacts/engine-b-reverse-stage-cost.json")


def main():
    assert jax.config.jax_enable_x64
    if OUT.exists():
        raise FileExistsError(OUT)
    x = jnp.array([0.6, 2.0, 0.8])
    key = jax.random.PRNGKey(1)
    record = {}
    m, pull = jax.vjp(model_at, x)
    f = jax.jit(
        jax.value_and_grad(lambda model: moment_from_model(model, key), allow_int=True)
    )
    compiled = f.lower(m).compile()
    mem = compiled.memory_analysis().temp_size_in_bytes
    assert mem < 32_000_000_000
    times = []
    for _ in range(3):
        start = time.perf_counter()
        v, gm = jax.block_until_ready(compiled(m))
        times.append(time.perf_counter() - start)
    record["sampling_reverse"] = dict(temporary_bytes=mem, seconds=times)
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(record, flush=True)
    back = jax.jit(pull)
    times = []
    for _ in range(3):
        start = time.perf_counter()
        g = jax.block_until_ready(back(gm))[0]
        times.append(time.perf_counter() - start)
    record["constructor_pullback"] = dict(seconds=times)
    build = jax.jit(model_at)
    times = []
    for _ in range(3):
        start = time.perf_counter()
        jax.block_until_ready(build(x))
        times.append(time.perf_counter() - start)
    record["constructor_primal"] = dict(seconds=times)
    ref = json.loads(
        Path(
            "docs/reviews/plummer-om-artifacts/engine-b-streamed-reverse.json"
        ).read_text()
    )
    np.testing.assert_allclose(g, ref["reference_forward"], rtol=1e-12, atol=1e-12)
    record.update(value=float(v), gradient=g.tolist(), parity_pass=True)
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(record, flush=True)


if __name__ == "__main__":
    main()

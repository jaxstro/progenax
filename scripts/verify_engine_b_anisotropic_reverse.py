"""Reverse replay against the retained anisotropic campaign's seed-one JVPs."""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from qualify_engine_b_population_moments import anisotropic_moment

if __name__ == "__main__":
    assert jax.config.jax_enable_x64
    root = Path("docs/reviews/plummer-om-artifacts")
    output = root / "engine-b-anisotropic-reverse.json"
    if output.exists():
        raise FileExistsError(output)
    campaign = json.loads(
        (root / "engine-b-anisotropic-population-qualification.json").read_text()
    )
    assert campaign["status"] == "completed"
    reference = campaign["batches"][0]
    theta = jnp.array([0.6, 2.0, 0.8])
    start = time.perf_counter()
    compiled = (
        jax.jit(
            jax.value_and_grad(lambda t: anisotropic_moment(t, jax.random.PRNGKey(1)))
        )
        .lower(theta)
        .compile()
    )
    compile_seconds = time.perf_counter() - start
    temporary = compiled.memory_analysis().temp_size_in_bytes
    if temporary >= 32_000_000_000:
        raise RuntimeError(
            "Reverse execution exceeds the 32 GB temporary-storage guard"
        )
    start = time.perf_counter()
    value, gradient = jax.block_until_ready(compiled(theta))
    elapsed = time.perf_counter() - start
    np.testing.assert_allclose(value, reference["value"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        gradient, reference["derivative"], rtol=1e-12, atol=1e-12
    )
    record = dict(
        value=float(value),
        gradient=gradient.tolist(),
        compile_seconds=compile_seconds,
        first_execution_seconds=elapsed,
        temporary_bytes=temporary,
        parity_pass=True,
    )
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(record)

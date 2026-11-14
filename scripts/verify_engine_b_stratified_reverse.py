"""Reverse-mode replay of the stratified moment prototype's first batch."""

import hashlib
import json
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from prototype_engine_b_speed_average import _engine_b_speed_moment
from qualify_engine_b_anisotropy_radius import moment

from progenax.cluster import sampling

if __name__ == "__main__":
    assert jax.config.jax_enable_x64
    if not hasattr(sampling, "_engine_b_speed_moment"):
        raise RuntimeError(
            "Replay this historical experiment at its recorded Git revision"
        )
    root = Path("docs/reviews/plummer-om-artifacts")
    output = root / "engine-b-ra-stratified-reverse.json"
    if output.exists():
        raise FileExistsError(output)
    source = root / "engine-b-ra-stratified-qualification.json"
    campaign = json.loads(source.read_text())
    assert campaign["status"] == "completed"
    sampling._engine_b_speed_moment = partial(_engine_b_speed_moment, stratified=True)
    function = lambda r: moment(r, jax.random.PRNGKey(1))
    start = time.perf_counter()
    compiled = jax.jit(jax.value_and_grad(function)).lower(jnp.array(3.0)).compile()
    compile_seconds = time.perf_counter() - start
    temporary_bytes = compiled.memory_analysis().temp_size_in_bytes
    if temporary_bytes >= 32_000_000_000:
        raise RuntimeError(
            f"Reverse temporary storage exceeds 32 GB: {temporary_bytes}"
        )
    timings = []
    for _ in range(2):
        start = time.perf_counter()
        value, gradient = jax.block_until_ready(compiled(jnp.array(3.0)))
        timings.append(time.perf_counter() - start)
    expected = campaign["cases"][0]
    np.testing.assert_allclose(value, expected["value"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gradient, expected["gradient"], rtol=1e-12, atol=1e-12)
    result = dict(
        value=float(value),
        gradient=float(gradient),
        parity_pass=True,
        compile_seconds=compile_seconds,
        execution_seconds=timings,
        temporary_bytes=temporary_bytes,
        campaign_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)

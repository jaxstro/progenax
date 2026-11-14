"""Approved 1600-to-3200 speed refinement at 12000 radial nodes."""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from diagnose_engine_b_population_moments import df_speed_moment

if __name__ == "__main__":
    assert jax.config.jax_enable_x64
    root = Path("docs/reviews/plummer-om-artifacts")
    output = root / "engine-b-anisotropic-reference-3200.json"
    if output.exists():
        raise FileExistsError(output)
    previous = json.loads(
        (root / "engine-b-anisotropic-reference-convergence.json").read_text()
    )["cases"][-1]
    start = time.perf_counter()
    value, gradient = jax.value_and_grad(
        lambda t: df_speed_moment(
            t, n_w=3200, n_r=12000, r_a_j=jnp.array([3.0, jnp.inf])
        )
    )(jnp.array([0.6, 2.0, 0.8]))
    jax.block_until_ready((value, gradient))
    relative = abs(np.array(previous["gradient"]) - np.array(gradient)) / abs(
        np.array(gradient)
    )
    result = dict(
        n_w=3200,
        n_r=12000,
        value=float(value),
        gradient=gradient.tolist(),
        seconds=time.perf_counter() - start,
        relative_gradient_change=relative.tolist(),
        passed=(relative <= 0.003).tolist(),
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(result)
    if not all(result["passed"]):
        raise SystemExit("Reference refinement failed; evidence retained")

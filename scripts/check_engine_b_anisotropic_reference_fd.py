"""Retained anisotropic deterministic reference: AD versus central differences."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from diagnose_engine_b_population_moments import df_speed_moment

if __name__ == "__main__":
    assert jax.config.jax_enable_x64
    output = Path(
        "docs/reviews/plummer-om-artifacts/engine-b-anisotropic-reference-fd.json"
    )
    if output.exists():
        raise FileExistsError(output)
    evaluate = jax.jit(
        lambda t: df_speed_moment(
            t, n_w=1600, n_r=12000, r_a_j=jnp.array([3.0, jnp.inf])
        )
    )
    theta = jnp.array([0.6, 2.0, 0.8])
    ad = jax.grad(evaluate)(theta)
    fd = []
    for i in range(3):
        step = 1e-4 * float(theta[i])
        shift = step * jnp.eye(3)[i]
        fd.append(
            float((evaluate(theta + shift) - evaluate(theta - shift)) / (2 * step))
        )
    relative = abs(np.asarray(ad) - fd) / abs(np.asarray(fd))
    result = dict(
        ad=ad.tolist(),
        fd=fd,
        relative_error=relative.tolist(),
        criterion=0.003,
        passed=(relative <= 0.003).tolist(),
        x64=True,
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(result)
    if not all(result["passed"]):
        raise SystemExit("Reference AD/FD check failed; evidence retained")

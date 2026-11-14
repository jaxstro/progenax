"""Approved one-axis-at-a-time speed-table refinement; production unchanged."""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from diagnose_anisotropic_population_moments import THETA
from diagnose_anisotropic_table_moment import reference

OUT = Path("docs/reviews/plummer-om-artifacts/anisotropic-table-axes.json")


def main():
    assert jax.config.jax_enable_x64
    if OUT.exists():
        raise FileExistsError(OUT)
    ref = json.loads(
        Path(
            "docs/reviews/plummer-om-artifacts/population-moment-anisotropic-mapped-pilot.json"
        ).read_text()
    )["references"][-1]
    report = dict(
        scope="one-axis doubling; finest retained radial subdivision 4; exact inverse-table moment",
        theta=THETA.tolist(),
        df_reference=ref,
        cases=[],
    )
    for name, shape in [
        ("q", (192, 96, 384)),
        ("sqrtW", (384, 48, 384)),
        ("x", (192, 48, 768)),
    ]:
        fn = lambda t: reference(t, 4, n_W=shape[0], n_p=shape[1], n_x=shape[2])
        f = jax.jit(lambda t: jax.jvp(fn, (t,), (jnp.array([0.0, 0.0, 0.0, 1.0]),)))
        start = time.perf_counter()
        c = f.lower(THETA).compile()
        compile_seconds = time.perf_counter() - start
        print(name, "compiled", compile_seconds, flush=True)
        start = time.perf_counter()
        v, d = jax.block_until_ready(c(THETA))
        row = dict(
            axis=name,
            shape=shape,
            value=float(v),
            ra_derivative=float(d),
            relative_derivative_offset=float(d / ref["derivative"][-1] - 1),
            compile_seconds=compile_seconds,
            execute_seconds=time.perf_counter() - start,
            temp_bytes=c.memory_analysis().temp_size_in_bytes,
        )
        report["cases"].append(row)
        OUT.write_text(json.dumps(report, indent=2) + "\n")
        print(row, flush=True)
        jax.clear_caches()


if __name__ == "__main__":
    main()

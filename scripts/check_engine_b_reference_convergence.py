"""Approved independent speed and radial reference resolution diagnostics."""

import argparse
import functools
import hashlib
import json
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from diagnose_engine_b_population_moments import df_speed_moment

OUT = Path("docs/reviews/plummer-om-artifacts/engine-b-reference-convergence.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anisotropic", action="store_true")
    args = parser.parse_args()
    global OUT
    if args.anisotropic:
        OUT = OUT.with_name("engine-b-anisotropic-reference-convergence.json")
    assert jax.config.jax_enable_x64
    if OUT.exists():
        raise FileExistsError(OUT)
    record = dict(
        status="running",
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(__file__).with_name("diagnose_engine_b_population_moments.py"),
            ]
        },
        theta=[0.6, 2.0, 0.8],
        relative_gate=0.003,
        cases=[],
        scope=(
            "halo ra=3, isotropic core" if args.anisotropic else "isotropic halo/core"
        ),
    )

    def save():
        OUT.write_text(json.dumps(record, indent=2) + "\n")

    save()
    try:
        for n_w, n_r in [(400, 6000), (800, 6000), (1600, 6000), (1600, 12000)]:
            evaluate = functools.partial(
                df_speed_moment,
                n_w=n_w,
                n_r=n_r,
                r_a_j=jnp.array([3.0, jnp.inf]) if args.anisotropic else None,
            )
            start = time.perf_counter()
            value, gradient = jax.value_and_grad(evaluate)(jnp.array(record["theta"]))
            jax.block_until_ready((value, gradient))
            row = dict(
                n_w=n_w,
                n_r=n_r,
                value=float(value),
                gradient=np.asarray(gradient).tolist(),
                seconds=time.perf_counter() - start,
            )
            record["cases"].append(row)
            save()
            print(row, flush=True)
        comparisons = []
        for left, right in zip(record["cases"][:-1], record["cases"][1:]):
            relative = abs(
                np.array(left["gradient"]) - np.array(right["gradient"])
            ) / abs(np.array(right["gradient"]))
            comparisons.append(
                dict(
                    from_grid=[left["n_w"], left["n_r"]],
                    to_grid=[right["n_w"], right["n_r"]],
                    relative_gradient_change=relative.tolist(),
                    passed=(relative <= 0.003).tolist(),
                )
            )
        record["comparisons"] = comparisons
        record["status"] = "completed"
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()


if __name__ == "__main__":
    main()

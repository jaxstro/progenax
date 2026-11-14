"""Four-parameter derivative diagnostic for the deterministic Engine B candidate."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from diagnose_engine_b_deterministic_moments import moment
from diagnose_engine_b_knot_reference import reference
from diagnose_engine_b_population_moments import model_at

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-defaults", action="store_true")
    parser.add_argument("--isotropic", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64
    output = Path(
        "docs/reviews/plummer-om-artifacts/engine-b-deterministic-candidate.json"
    )
    if args.public_defaults:
        output = output.with_name("engine-b-deterministic-public-defaults.json")
    if args.isotropic:
        output = output.with_stem(output.stem + "-isotropic")
    if args.output is not None:
        output = args.output
    if output.exists():
        raise FileExistsError(output)
    theta = jnp.array([0.6, 2.0, 0.8] if args.isotropic else [0.6, 2.0, 0.8, 3.0])
    radius = lambda t: jnp.inf if args.isotropic else t[3]
    f = lambda t: moment(radius(t), 4, n_grid=4000, n_r=12000, theta=t[:3])
    if args.public_defaults:

        def f(t):
            model = model_at(t[:3], r_a_j=jnp.array([radius(t), jnp.inf]))
            return model.estimate_population_moments(
                jax.random.PRNGKey(0),
                100000,
                1.0,
                n_realizations=4,
                draws_per_component=50000,
            )["mean_speed_squared"]

    start = time.perf_counter()
    compiled = jax.jit(jax.value_and_grad(f)).lower(theta).compile()
    memory = compiled.memory_analysis().temp_size_in_bytes
    if memory >= 32_000_000_000:
        raise RuntimeError("32 GB temporary-memory guard")
    compile_seconds = time.perf_counter() - start
    timings = []
    for _ in range(2):
        start = time.perf_counter()
        value, gradient = jax.block_until_ready(compiled(theta))
        timings.append(time.perf_counter() - start)
    result = dict(
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        isotropic=args.isotropic,
        value=float(value),
        gradient=gradient.tolist(),
        temporary_bytes=memory,
        compile_seconds=compile_seconds,
        execution_seconds=timings,
        checks=[],
        status="running",
        scope="Retained halo/core model; public defaults"
        if args.public_defaults
        else "Deterministic prototype",
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)
    for index, name in enumerate(
        ["mass_fraction", "halo_half_mass_radius", "core_scale", "anisotropy_radius"]
    ):
        if index >= theta.size:
            break
        h = 1e-4 * max(1.0, float(theta[index]))
        dt = jnp.zeros_like(theta).at[index].set(h)
        vals = [
            float(reference(radius(t), nr=12000, theta=t[:3]))
            for t in (theta - dt, theta + dt)
        ]
        fd = (vals[1] - vals[0]) / (2 * h)
        relative = abs(float(gradient[index]) - fd) / abs(fd)
        result["checks"].append(
            dict(
                parameter=name,
                reference_fd=fd,
                relative_discrepancy=relative,
                pass_gate=relative <= 0.003,
            )
        )
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(result["checks"][-1], flush=True)
    result["status"] = "completed"
    result["all_pass"] = all(row["pass_gate"] for row in result["checks"])
    output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["all_pass"]:
        raise SystemExit("Derivative diagnostic failed; evidence retained")

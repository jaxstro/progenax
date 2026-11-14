"""Verify approved analytic radius averaging against retained independent evidence."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from diagnose_population_moments import THETA, build_model, conditional_sampled, timed

OUT = Path("docs/reviews/plummer-om-artifacts/population-moment-analytic-radius.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", action="store_true")
    args = parser.parse_args()
    global OUT
    if args.production:
        OUT = OUT.with_name("population-moment-production.json")
    assert jax.config.jax_enable_x64
    if OUT.exists():
        raise FileExistsError(f"Retain existing exposure: {OUT}")
    prior_path = OUT.with_name("population-moment-qualification.json")
    prior = json.loads(prior_path.read_text())
    report = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": hashlib.sha256(
            Path("scripts/diagnose_population_moments.py").read_bytes()
        ).hexdigest(),
        "prior_evidence_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        "theta": THETA.tolist(),
        "seed": 1,
        "x64": True,
        "scope": "same-seed estimator regression; not a new independent statistical campaign",
        "columns": [],
    }

    def save():
        OUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()

    def moments(theta):
        if not args.production:
            return conditional_sampled(
                theta, jax.random.PRNGKey(1), analytic_radius=True
            )[0]
        result = build_model(theta).estimate_population_moments(
            jax.random.PRNGKey(1),
            100000,
            G=1.0,
            n_realizations=4,
            draws_per_component=25000,
        )
        return jnp.stack(
            [
                result[name]
                for name in ("mean_mass", "mean_radius_squared", "mean_speed_squared")
            ]
        )

    report["production_api"] = args.production
    if args.production:
        report["production_source_sha256"] = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in map(
                Path,
                (
                    "src/progenax/cluster/sampling.py",
                    "src/progenax/cluster/_radial_cdf.py",
                    "src/progenax/cluster/multicomponent.py",
                ),
            )
        }

    def evaluate(tangent):
        return jax.jvp(moments, (THETA,), (tangent,))

    compiled = jax.jit(evaluate)
    for direction in range(3):
        (values, column), seconds = timed(
            f"analytic radius JVP {direction}", lambda: compiled(jnp.eye(3)[direction])
        )
        report["columns"].append({"derivatives": column.tolist(), "seconds": seconds})
        save()
    derivatives = np.stack([c["derivatives"] for c in report["columns"]], axis=1)
    reference = np.array(prior["references"][-1]["derivatives"])
    old = prior["batches"][0]
    checks = {
        "analytic_values": bool(
            np.allclose(
                np.asarray(values)[:2],
                prior["references"][-1]["values"][:2],
                rtol=1e-12,
                atol=1e-12,
            )
        ),
        "analytic_derivatives": bool(
            np.allclose(derivatives[:2], reference[:2], rtol=1e-12, atol=1e-12)
        ),
        "velocity_value_unchanged": bool(
            np.allclose(values[2], old["values"][2], rtol=1e-12, atol=1e-12)
        ),
        "velocity_derivatives_unchanged": bool(
            np.allclose(derivatives[2], old["derivatives"][2], rtol=1e-12, atol=1e-12)
        ),
    }
    report.update(
        status="completed",
        values=values.tolist(),
        derivatives=derivatives.tolist(),
        checks=checks,
        analytic_derivative_absolute_errors=abs(
            derivatives[:2] - reference[:2]
        ).tolist(),
    )
    save()
    print(json.dumps(report, indent=2), flush=True)
    if not all(checks.values()):
        raise AssertionError(checks)


if __name__ == "__main__":
    main()

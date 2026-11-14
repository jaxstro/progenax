"""Frozen retained-model population-moment qualification; retain every batch."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from diagnose_population_moments import (
    THETA,
    build_model,
    conditional_sampled,
    reference,
    timed,
)

OUT = Path("docs/reviews/plummer-om-artifacts/population-moment-qualification.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=(
            "retained",
            "W0-7",
            "g-0",
            "g-0.5",
            "delta-0",
            "delta-0-tight",
            "delta-0.6",
            "components-2",
        ),
        default="retained",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    global OUT
    theta = THETA.at[2].set(7.0) if args.case == "W0-7" else THETA
    g = {"g-0": 0.0, "g-0.5": 0.5}.get(args.case, 1.0)
    if args.case in ("delta-0", "delta-0-tight", "delta-0.6"):
        theta = theta.at[1].set(0.6 if args.case == "delta-0.6" else 0.0)
    n_components = 2 if args.case == "components-2" else 4
    fit_tol = 1e-10 if args.case == "delta-0-tight" else 1e-6
    n_iter = 60 if args.case == "delta-0-tight" else 30
    production = args.case != "retained"
    if production:
        OUT = OUT.with_name(f"population-moment-{args.case}.json")
        if args.case == "delta-0-tight":
            OUT = OUT.with_name("population-moment-delta-0-tight-identity.json")

    def reference_case(t, subdivisions):
        return reference(
            t,
            analytic=True,
            subdivisions=subdivisions,
            g=g,
            n_components=n_components,
            fit_tol=fit_tol,
            n_iter=n_iter,
        )

    assert jax.config.jax_enable_x64
    if OUT.exists() and not args.resume:
        raise FileExistsError(f"Retain existing exposure: {OUT}")
    report = {
        "status": "reference refinement running",
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path("scripts/diagnose_population_moments.py"))
        },
        "protocol": {
            "theta": theta.tolist(),
            "g": g,
            "case": args.case,
            "production_api": production,
            "n_components": n_components,
            "fit_tol": fit_tol,
            "n_iter": n_iter,
            "delta_fd": "forward_second_order"
            if args.case in ("delta-0", "delta-0-tight")
            else "central",
            "target_stars": 100000,
            "realizations_per_batch": 4,
            "batch_seeds": list(range(1, 9)),
            "reference_subdivisions": [1, 2, 4],
            "fd_steps": [1e-4, 1e-5],
            "deterministic_relative_gate": 0.003,
            "zero_radius_identity_gate": 1e-8 if args.case == "delta-0-tight" else None,
            "relative_discrepancy_gate": 0.01,
            "relative_standard_error_gate": 0.0025,
            "consistency_standard_errors": 3,
            "reference_error_estimate": "maximum of last refinement difference and both AD/FD differences; empirical, not a rigorous bound",
            "scope": "one isotropic Engine A case; first derivatives in alpha, delta, W0; no derivative in g",
        },
        "references": [],
        "batches": [],
    }

    def save():
        OUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    if args.resume:
        existing = json.loads(OUT.read_text())
        if existing["protocol"] != report["protocol"]:
            raise ValueError("Resume protocol differs from the frozen exposure")
        report = existing
        if not report.get("reference_pass") or len(report["batches"]) >= 8:
            raise ValueError(
                "Resume requires a passed reference and incomplete batches"
            )
        report.setdefault("resume_segments", []).append(
            {
                "revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "source_sha256": {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (
                        Path(__file__),
                        Path("src/progenax/cluster/_radial_cdf.py"),
                        Path("src/progenax/cluster/sampling.py"),
                    )
                },
                "retained_batches": len(report["batches"]),
                "reason": "indexed radial lookup; retained batch replay required before continuation",
            }
        )
        rd = np.asarray(report["references"][-1]["derivatives"])
        error = np.asarray(report["reference_error_estimate"])
    else:
        save()
        for level in (1, 2, 4):
            fn = lambda theta: reference_case(theta, level)
            evaluate = jax.jit(lambda theta: (fn(theta), jax.jacfwd(fn)(theta)))
            (value, derivative), seconds = timed(
                f"reference refinement {level}", lambda: evaluate(theta)
            )
            report["references"].append(
                {
                    "subdivisions": level,
                    "values": value.tolist(),
                    "derivatives": derivative.tolist(),
                    "seconds": seconds,
                }
            )
            save()
        ref = jax.jit(lambda theta: reference_case(theta, 4))
        rd = np.asarray(derivative)
        errors = [np.abs(rd - np.asarray(report["references"][-2]["derivatives"]))]
        for step in (1e-4, 1e-5):
            fd = np.asarray(
                jnp.stack(
                    [
                        (
                            (
                                -3 * ref(theta)
                                + 4 * ref(theta + step * e)
                                - ref(theta + 2 * step * e)
                            )
                            / (2 * step)
                            if args.case in ("delta-0", "delta-0-tight") and index == 1
                            else (ref(theta + step * e) - ref(theta - step * e))
                            / (2 * step)
                        )
                        for index, e in enumerate(jnp.eye(3))
                    ],
                    axis=1,
                )
            )
            report[f"reference_fd_{step}"] = fd.tolist()
            errors.append(np.abs(rd - fd))
            save()
        error = np.maximum.reduce(errors)
        reference_cells = error[1:] / np.abs(rd[1:]) <= 0.003
        if args.case == "delta-0-tight":
            # At zero segregation all components share the spatial profile.
            # Test the exact-zero IMF-slope derivative without dividing by it.
            scale = abs(float(theta[0]) / float(value[1]))
            candidates = [abs(rd[1, 0])]
            candidates.extend(
                abs(np.asarray(report[f"reference_fd_{h}"])[1, 0]) for h in (1e-4, 1e-5)
            )
            report["zero_radius_identity_scaled_derivatives"] = [
                float(x * scale) for x in candidates
            ]
            reference_cells[0, 0] = all(x * scale <= 1e-8 for x in candidates)
        reference_pass = bool(np.all(reference_cells))
        report["reference_error_estimate"] = error.tolist()
        report["reference_pass"] = reference_pass
        if not reference_pass:
            report["status"] = "reference gate failed; sampling not started"
            save()
            return

    def directional(theta, tangent, key):
        def moments(t):
            if not production:
                return conditional_sampled(t, key)[0]
            result = build_model(
                t, g=g, n_components=n_components, fit_tol=fit_tol, n_iter=n_iter
            ).estimate_population_moments(
                key,
                100000,
                G=1.0,
                n_realizations=4,
                draws_per_component=100000 // n_components,
            )
            return jnp.stack(
                [
                    result[name]
                    for name in (
                        "mean_mass",
                        "mean_radius_squared",
                        "mean_speed_squared",
                    )
                ]
            )

        return jax.jvp(moments, (theta,), (tangent,))

    evaluate = jax.jit(directional)
    report["status"] = "independent batches running"
    save()
    if args.resume:
        old = report["batches"][-1]
        columns = []
        for direction in range(3):
            (means, column), seconds = timed(
                f"resume parity direction {direction}",
                lambda: evaluate(
                    theta, jnp.eye(3)[direction], jax.random.PRNGKey(old["seed"])
                ),
            )
            columns.append(column)
        np.testing.assert_allclose(means, old["values"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            jnp.stack(columns, axis=1), old["derivatives"], rtol=1e-12, atol=1e-12
        )
        report["resume_segments"][-1]["retained_batch_parity_pass"] = True
        save()
    for seed in range(len(report["batches"]) + 1, 9):
        columns, times = [], []
        for direction in range(3):
            (means, column), seconds = timed(
                f"batch {seed} direction {direction}",
                lambda: evaluate(
                    theta, jnp.eye(3)[direction], jax.random.PRNGKey(seed)
                ),
            )
            columns.append(column)
            times.append(seconds)
        report["batches"].append(
            {
                "seed": seed,
                "values": means.tolist(),
                "derivatives": jnp.stack(columns, axis=1).tolist(),
                "direction_seconds": times,
            }
        )
        save()
    samples = np.asarray([b["derivatives"] for b in report["batches"]])
    mean = samples.mean(axis=0)
    se = samples.std(axis=0, ddof=1) / np.sqrt(8)
    discrepancy = np.abs(mean - rd)
    cells = (
        (discrepancy <= 0.01 * np.abs(rd))
        & (se <= 0.0025 * np.abs(rd))
        & (discrepancy <= 3 * se + error)
    )
    report.update(
        status="completed",
        derivative_mean=mean.tolist(),
        standard_error=se.tolist(),
        relative_discrepancy=(discrepancy / np.abs(rd)).tolist(),
        spatial_velocity_pass=cells[1:].tolist(),
        mean_mass_identity_absolute_error=discrepancy[0].tolist(),
    )
    if production:
        report["velocity_pass"] = cells[2].tolist()
        report["analytic_identity_pass"] = bool(
            np.allclose(samples[:, :2], rd[None, :2], rtol=1e-12, atol=1e-12)
        )
        report["analytic_values_pass"] = bool(
            np.allclose(
                np.asarray([b["values"][:2] for b in report["batches"]]),
                report["references"][-1]["values"][:2],
                rtol=1e-12,
                atol=1e-12,
            )
        )
        del report["spatial_velocity_pass"]
    save()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

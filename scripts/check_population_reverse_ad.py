"""Reverse-AD replay of the three accepted isotropic population-moment cases."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from diagnose_population_moments import THETA, build_model, timed

ROOT = Path("docs/reviews/plummer-om-artifacts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--delta-zero-tight", action="store_true")
    args = parser.parse_args()
    assert jax.config.jax_enable_x64
    output = ROOT / "population-moment-reverse-ad.json"
    if args.extended:
        output = ROOT / "population-moment-reverse-extended.json"
    if args.delta_zero_tight:
        output = ROOT / "population-moment-reverse-delta-0-tight.json"
    if output.exists():
        raise FileExistsError(output)
    report = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "x64": True,
        "seed": 1,
        "target_stars": 100000,
        "realizations": 4,
        "total_draws_per_realization": 100000,
        "derivative_rtol": 0.003,
        "derivative_atol": 1e-12,
        "scope": "default checkpointed reverse constructor AD versus retained forward AD; same physical cases and random inputs",
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path("scripts/diagnose_population_moments.py"))
        },
        "cases": [],
    }

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    cases = [
        ("retained", 5.0, 1.0, 0.4, 4),
        ("W0-7", 7.0, 1.0, 0.4, 4),
        ("g-0", 5.0, 0.0, 0.4, 4),
    ]
    if args.extended:
        cases = [
            ("g-0.5", 5.0, 0.5, 0.4, 4),
            ("delta-0", 5.0, 1.0, 0.0, 4),
            ("delta-0.6", 5.0, 1.0, 0.6, 4),
            ("components-2", 5.0, 1.0, 0.4, 2),
        ]
    if args.delta_zero_tight:
        cases = [("delta-0-tight-identity", 5.0, 1.0, 0.0, 4)]
    for name, w0, g, delta, n_components in cases:
        file = (
            "population-moment-production.json"
            if name == "retained"
            else f"population-moment-{name}.json"
        )
        record = json.loads((ROOT / file).read_text())
        if name != "retained" and not record["batches"]:
            report["cases"].append(
                {
                    "name": name,
                    "status": "not replayed: forward reference gate did not pass",
                }
            )
            save()
            continue
        expected = record if name == "retained" else record["batches"][0]
        theta = THETA.at[2].set(w0).at[1].set(delta)

        def objective(t, weight):
            result = build_model(
                t,
                g=g,
                n_components=n_components,
                adjoint=None,
                fit_tol=record.get("protocol", {}).get("fit_tol", 1e-6),
                n_iter=record.get("protocol", {}).get("n_iter", 30),
            ).estimate_population_moments(
                jax.random.PRNGKey(1),
                100000,
                G=1.0,
                n_realizations=4,
                draws_per_component=100000 // n_components,
            )
            values = jnp.stack(
                [
                    result[k]
                    for k in ("mean_mass", "mean_radius_squared", "mean_speed_squared")
                ]
            )
            return values @ weight, values

        evaluate = jax.jit(jax.value_and_grad(objective, has_aux=True))
        case = {
            "name": name,
            "g": g,
            "theta": theta.tolist(),
            "n_components": n_components,
            "draws_per_component": 100000 // n_components,
            "rows": [],
        }
        report["cases"].append(case)
        for row in range(3):
            ((_, values), derivative), seconds = timed(
                f"{name} reverse row {row}", lambda: evaluate(theta, jnp.eye(3)[row])
            )
            case["rows"].append({"derivative": derivative.tolist(), "seconds": seconds})
            save()
        matrix = np.asarray([r["derivative"] for r in case["rows"]])
        case.update(
            values=values.tolist(),
            values_pass=bool(
                np.allclose(values, expected["values"], rtol=1e-12, atol=1e-12)
            ),
            derivative_pass=np.isclose(
                matrix, expected["derivatives"], rtol=0.003, atol=1e-12
            ).tolist(),
            derivative_absolute_error=abs(
                matrix - np.asarray(expected["derivatives"])
            ).tolist(),
        )
        save()
        jax.clear_caches()
    report["status"] = "completed"
    save()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Full-constructor shape derivatives at retained single-component models."""

import argparse
import json
import subprocess
from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

from progenax import EFFProfile, KingProfile
from progenax.cluster.multicomponent import MultiComponentCluster


def evaluate(parameter, family, *, forward=False):
    profile = (
        EFFProfile(a=1.0, gamma=parameter, r_t=10.0)
        if family == "eff"
        else KingProfile.from_W0_rc(
            W0=parameter,
            r_c=1.0,
            adjoint=diffrax.ForwardMode() if forward else None,
        )
    )
    model = MultiComponentCluster.from_density_profiles(
        profiles=[profile], mass_fractions=jnp.array([1.0]), m_j=jnp.array([1.0])
    )
    result = model.estimate_population_moments(
        jax.random.PRNGKey(0),
        100000,
        1.0,
        n_realizations=1,
        draws_per_component=1,
    )
    return jnp.array([result["mean_radius_squared"], result["mean_speed_squared"]])


if __name__ == "__main__":
    assert jax.config.jax_enable_x64
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/reviews/plummer-om-artifacts/engine-b-shape-derivatives.json"
        ),
    )
    output = parser.parse_args().output
    if output.exists():
        raise FileExistsError(output)
    result = dict(
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        criterion=0.003,
        scope="Single-component EFF gamma=5 and King W0=5; public defaults; first reverse derivatives",
        cases=[],
    )
    for family in ("eff", "king"):
        try:
            f = lambda x: evaluate(x, family)
            compiled = jax.jit(jax.jacrev(f)).lower(5.0).compile()
            memory = compiled.memory_analysis().temp_size_in_bytes
            if memory >= 32_000_000_000:
                raise RuntimeError("32 GB temporary-memory guard")
            ad = np.asarray(jax.block_until_ready(compiled(5.0)))
            forward = np.asarray(
                jax.jit(jax.jacfwd(lambda x: evaluate(x, family, forward=True)))(5.0)
            )
            parity = bool(np.allclose(ad, forward, rtol=1e-10, atol=1e-10))
            primal = jax.jit(f)
            checks = []
            for h in (5e-4, 1.5e-4):
                fd = np.asarray((primal(5.0 + h) - primal(5.0 - h)) / (2 * h))
                discrepancy = np.abs(ad - fd) / np.maximum(np.abs(fd), 1e-30)
                checks.append(
                    dict(
                        step=h,
                        fd=fd.tolist(),
                        relative_discrepancy=discrepancy.tolist(),
                        pass_gate=bool(np.all(discrepancy <= 0.003)),
                    )
                )
            row = dict(
                family=family,
                derivative=ad.tolist(),
                forward_derivative=forward.tolist(),
                forward_reverse_pass=parity,
                temporary_bytes=memory,
                checks=checks,
                all_pass=parity and all(c["pass_gate"] for c in checks),
            )
        except Exception as error:
            row = dict(
                family=family, all_pass=False, error=f"{type(error).__name__}: {error}"
            )
        result["cases"].append(row)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(row, flush=True)
    result["all_pass"] = all(c["all_pass"] for c in result["cases"])
    output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["all_pass"]:
        raise SystemExit("Shape derivative qualification failed; evidence retained")

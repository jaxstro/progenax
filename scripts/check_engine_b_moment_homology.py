"""Physical length-scaling identities across Engine B profile families."""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from progenax import EFFProfile, KingProfile, PlummerProfile
from progenax.cluster.multicomponent import MultiComponentCluster

if __name__ == "__main__":
    assert jax.config.jax_enable_x64
    output = Path("docs/reviews/plummer-om-artifacts/engine-b-moment-homology.json")
    if output.exists():
        raise FileExistsError(output)
    king = KingProfile.from_W0_rc(W0=5.0, r_c=1.0)
    records = []
    for family in ["plummer", "eff", "king", "three_component_plummer"]:

        def evaluate(scale):
            anisotropy = None
            if family == "plummer":
                profiles = [PlummerProfile(r_h=scale)]
            elif family == "eff":
                profiles = [EFFProfile(a=scale, gamma=5.0, r_t=10 * scale)]
            elif family == "king":
                profiles = [
                    KingProfile(
                        W0=king.W0,
                        r_c=scale,
                        r_t=scale * king.r_t,
                        xi_grid=king.xi_grid,
                        psi_grid=king.psi_grid,
                    )
                ]
            else:
                profiles = [PlummerProfile(r_h=scale) for _ in range(3)]
                anisotropy = jnp.array([2 * scale, jnp.inf, jnp.inf])
            fractions = (
                jnp.array([0.2, 0.3, 0.5]) if len(profiles) == 3 else jnp.array([1.0])
            )
            masses = (
                jnp.array([0.3, 1.0, 3.0]) if len(profiles) == 3 else jnp.array([1.0])
            )
            model = MultiComponentCluster.from_density_profiles(
                profiles=profiles,
                mass_fractions=fractions,
                m_j=masses,
                r_a_j=anisotropy,
            )
            result = model.estimate_population_moments(
                jax.random.PRNGKey(0),
                100000,
                1.0,
                n_realizations=1,
                draws_per_component=1,
            )
            return jnp.array(
                [
                    result["mean_mass"],
                    result["mean_radius_squared"],
                    result["mean_speed_squared"],
                ]
            )

        start = time.perf_counter()
        value = jax.jit(evaluate)(jnp.array(1.0))
        forward = jax.jit(jax.jacfwd(evaluate))(jnp.array(1.0))
        reverse_compiled = jax.jit(jax.jacrev(evaluate)).lower(jnp.array(1.0)).compile()
        temporary = reverse_compiled.memory_analysis().temp_size_in_bytes
        if temporary >= 32_000_000_000:
            raise RuntimeError("32 GB temporary-memory guard")
        reverse = reverse_compiled(jnp.array(1.0))
        value, forward, reverse = jax.block_until_ready((value, forward, reverse))
        expected = np.array([0.0, 2 * float(value[1]), -float(value[2])])
        errors = np.abs(np.asarray(forward) - expected) / np.maximum(
            np.abs(expected), 1.0
        )
        parity = bool(np.allclose(forward, reverse, rtol=1e-12, atol=1e-12))
        passed = bool(
            np.all(np.isfinite(errors)) and np.max(errors) <= 0.003 and parity
        )
        records.append(
            dict(
                family=family,
                value=value.tolist(),
                forward=forward.tolist(),
                reverse=reverse.tolist(),
                expected=expected.tolist(),
                relative_errors=errors.tolist(),
                parity_pass=parity,
                pass_gate=passed,
                temporary_bytes=temporary,
                seconds=time.perf_counter() - start,
            )
        )
        output.write_text(
            json.dumps(dict(status="running", cases=records), indent=2) + "\n"
        )
        print(records[-1], flush=True)
    output.write_text(
        json.dumps(
            dict(
                status="completed",
                all_pass=all(r["pass_gate"] for r in records),
                cases=records,
                scope="Length homology; King dimensionless ODE held fixed; not concentration or arbitrary-domain qualification",
            ),
            indent=2,
        )
        + "\n"
    )
    if not all(r["pass_gate"] for r in records):
        raise SystemExit("Homology gate failed; evidence retained")

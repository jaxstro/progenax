"""Retained Engine B halo/core: deterministic mass/radius derivative diagnostic."""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from progenax import EFFProfile, PlummerProfile
from progenax.cluster._radial_cdf import inverse_cdf_second_moments
from progenax.cluster.multicomponent import MultiComponentCluster
from progenax.cluster.sampling import _estimate_population_moments

OUT = Path("docs/reviews/plummer-om-artifacts/engine-b-population-diagnostic.json")


def model_at(theta, *, n_r=6000, r_a_j=None, n_grid=1000):
    fraction, half_mass_radius, core_scale = theta
    return MultiComponentCluster.from_density_profiles(
        profiles=[
            PlummerProfile(r_h=half_mass_radius),
            EFFProfile(a=core_scale, gamma=5.0, r_t=9.0),
        ],
        mass_fractions=jnp.array([fraction, 1 - fraction]),
        m_j=jnp.array([0.5, 1.0]),
        n_r=n_r,
        n_grid=n_grid,
        r_a_j=r_a_j,
    )


def moments(theta):
    model = model_at(theta)
    pi = jax.nn.softmax(jnp.log(model.N_frac_j + 1e-30))
    return jnp.array(
        [pi @ model.m_j, pi @ inverse_cdf_second_moments(model._r_grid, model._cdf_j)]
    )


def speed_moment(theta):
    return _estimate_population_moments(
        model_at(theta), jax.random.PRNGKey(0), 100000, 1.0, 4, 50000
    )["mean_speed_squared"]


def df_speed_moment(theta, *, n_w=400, n_r=6000, sampler_table=False, r_a_j=None):
    """Existing Engine B hybrid reference with explicit diagnostic resolutions.

    Integrate the conditional DF moment against prescribed density, on the
    existing Poisson grid. Independent of the sampler's inverse speed CDF;
    shared potential and Eddington tables are not independently validated here.
    """
    model = model_at(theta, n_r=n_r, r_a_j=r_a_j)
    state = model.engine_b
    r = state.r_poisson
    potential = jnp.maximum(state.Psi_poisson, 1e-12)
    components = []
    for j in range(model.m_j.size):

        def local(psi):
            w = jnp.linspace(0.0, jnp.sqrt(2 * psi), n_w)
            f = jnp.maximum(
                jnp.interp(psi - w**2 / 2, state.E_grid, state.f_j_grid[j]), 0.0
            )
            if sampler_table:
                from jaxstro.numerics.integration import cumulative_trapz

                cdf = cumulative_trapz(w**2 * f, dx=w[1] - w[0])
                cdf = cdf / (cdf[-1] + 1e-30)
                value = inverse_cdf_second_moments(w, cdf[None, :])[0]
                return jnp.where(psi > 1e-6 * state.E_grid[-1], value, 0.0)
            return jnp.trapezoid(w**4 * f, w) / (jnp.trapezoid(w**2 * f, w) + 1e-300)

        v2 = jax.lax.map(local, potential, batch_size=32)
        finite = jnp.isfinite(state.r_a_j[j])
        ra = jnp.where(finite, state.r_a_j[j], 1.0)
        stretch2 = jnp.where(finite, 1 / (1 + (r / ra) ** 2), 1.0)
        v2 = v2 * (1 / 3 + 2 * stretch2 / 3)
        density = state.rho_j_poisson[j] * r**2
        components.append(jnp.trapezoid(density * v2, r) / jnp.trapezoid(density, r))
    pi = jax.nn.softmax(jnp.log(model.N_frac_j + 1e-30))
    mean_mass = pi @ model.m_j
    return jnp.sum(pi * jnp.stack(components) * (model.m_j + 99999 * mean_mass)) / (
        4 * jnp.pi * state.mu
    )


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--speed", action="store_true")
    modes.add_argument("--df", action="store_true")
    args = parser.parse_args()
    global OUT
    if args.speed:
        OUT = OUT.with_name("engine-b-population-speed-diagnostic.json")
    if args.df:
        OUT = OUT.with_name("engine-b-population-df-diagnostic.json")
    evaluate = df_speed_moment if args.df else speed_moment if args.speed else moments
    assert jax.config.jax_enable_x64
    if OUT.exists():
        raise FileExistsError(OUT)
    theta = jnp.array([0.6, 2.0, 0.8])
    record = dict(
        status="running",
        theta=theta.tolist(),
        parameters=["halo_mass_fraction", "halo_half_mass_radius", "core_scale"],
        scope=(
            "fixed-key velocity derivative diagnostic; no precision qualification"
            if args.speed
            else "deterministic DF speed reference; no convergence qualification"
            if args.df
            else "constructor-to-mass/radius diagnostic; no velocity qualification"
        ),
        reference="existing hybrid DF quadrature: 400 speed nodes, prescribed density on Poisson grid"
        if args.df
        else None,
        grids="unchanged constructor defaults",
        steps=[1e-4, 1e-5],
        exposure=dict(
            target_stars=100000,
            seed=0,
            realizations=4,
            draws_per_component=50000,
            G=1.0,
        )
        if args.speed
        else None,
        stages={},
    )

    def save():
        OUT.write_text(json.dumps(record, indent=2) + "\n")

    save()
    try:
        for name, function in [
            ("value", evaluate),
            ("forward", jax.jacfwd(evaluate)),
            ("reverse", jax.jacrev(evaluate)),
        ]:
            start = time.perf_counter()
            result = jax.block_until_ready(function(theta))
            record["stages"][name] = dict(
                value=np.asarray(result).tolist(), seconds=time.perf_counter() - start
            )
            save()
            print(name, record["stages"][name], flush=True)
        for h in record["steps"]:
            columns = [
                (evaluate(theta + h * e) - evaluate(theta - h * e)) / (2 * h)
                for e in jnp.eye(3)
            ]
            record["stages"][f"fd_{h}"] = np.asarray(
                jnp.stack(columns, axis=-1)
            ).tolist()
            save()
        record["status"] = "completed"
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()


if __name__ == "__main__":
    main()

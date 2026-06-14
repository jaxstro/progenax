"""End-to-end integration tests for the cluster-builder convenience API.

build_cluster orchestrates masses -> matched_velocity_df -> build_spatial_ic ->
optional tidal/rotation modifiers. build_spatial_ic applies the Q=0.5 virial scale by
default, so every profile family lands near virial equilibrium (Q = T/|V| ~ 0.5) regardless
of the unscaled DF's native virial ratio. These tests confirm the end-to-end equilibrium for
all 5 profiles plus jit/grad transparency through the builder.

The non-Plummer equilibria (EFF/King/Michie/LIMEPY) re-solve their ODEs at N=3000 and are
SLOW -> marked @pytest.mark.slow so the FAST GATE (-m "not slow") skips them. The Plummer
near-virial case and the small-N jit/grad smoke tests stay fast (unmarked).
"""

import jax
import jax.numpy as jnp
import pytest
from jaxstro.units import STELLAR

from progenax import (
    build_cluster,
    build_cluster_from_params,
    ClusterParams,
    PlummerProfile,
    EFFProfile,
    KingProfile,
    MichieProfile,
    LIMEPYProfile,
    compute_kinetic_energy,
    compute_potential_energy,
)

_K = jax.random.PRNGKey(3)


def _Q(ic):
    """Virial ratio Q = T / |V| (0.5 == equilibrium)."""
    T = compute_kinetic_energy(ic.velocities, ic.masses)
    V = compute_potential_energy(ic.positions, ic.masses, G=STELLAR.G)
    return float(T / jnp.abs(V))


@pytest.mark.parametrize(
    "profile",
    [
        # Plummer: fast (closed-form sampler) -> stays in the FAST GATE.
        PlummerProfile(r_h=1.0),
        # EFF/King/Michie/LIMEPY: ODE-backed equilibria at N=3000 -> SLOW.
        pytest.param(
            EFFProfile(a=1.0, gamma=5.0, r_t=12.0),  # gamma=5 ~ virial
            marks=pytest.mark.slow,
        ),
        pytest.param(
            KingProfile.from_W0_rc(W0=7.0, r_c=1.0),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            MichieProfile.from_W0_rc(W0=7.0, r_c=1.0, r_a=8.0),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            LIMEPYProfile.from_W0_rc(W0=5.0, g=1.0, r_c=1.0),
            marks=pytest.mark.slow,
        ),
    ],
)
def test_each_profile_builds_near_virial(profile):
    """build_cluster applies Q=0.5 virial scaling by default, so every profile family
    builds into near-virial equilibrium (Q = T/|V| ~ 0.5) with finite phase space."""
    ic = build_cluster(profile, masses=jnp.ones(3000), key=_K)
    measured_Q = _Q(ic)
    print(f"\n[virial] {type(profile).__name__}: measured Q = {measured_Q:.4f}")
    # NEVER loosen this tolerance: if a profile is outside abs=0.05 it is a physics bug
    # in the DF/virial-scale path, not a tolerance issue. (plan Batch 6 hard constraint)
    assert measured_Q == pytest.approx(0.5, abs=0.05), (
        f"{type(profile).__name__}: measured Q={measured_Q:.4f} outside 0.5 +/- 0.05"
    )
    assert jnp.all(jnp.isfinite(ic.positions))
    assert jnp.all(jnp.isfinite(ic.velocities))


def test_jit_through_build_cluster():
    """build_cluster is jit-transparent: positions are finite under jax.jit."""

    @jax.jit
    def f(r_h):
        return build_cluster(
            PlummerProfile(r_h=r_h), masses=jnp.ones(200), key=_K
        ).positions

    assert jnp.all(jnp.isfinite(f(1.0)))


def test_grad_through_build_cluster_from_params():
    """jax.grad flows through build_cluster_from_params (the theta -> ICResult inference
    map), even with tidal-truncation + rotation modifiers active, giving a finite,
    non-zero gradient in r_h."""
    m = jnp.ones(200)

    def loss(r_h):
        params = ClusterParams(
            profile=PlummerProfile(r_h=r_h), tidal_radius=3.0, rotation=0.1
        )
        ic = build_cluster_from_params(params, masses=m, key=_K)
        return jnp.mean(jnp.linalg.norm(ic.positions, axis=1))

    g = jax.grad(loss)(1.0)
    assert jnp.isfinite(g)
    assert abs(g) > 1e-6

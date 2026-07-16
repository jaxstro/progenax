"""Gravax seam test: a ClusterIC transfers to an N-body integration intact (audit C13).

Asserts the handoff contract — units (explicit G), COM centring, realized virial ratio,
finiteness, and energy conservation through a short symplectic integration. Small N and a
short t_end keep this a seconds-scale gate; the science-grade dynamical-memory runs live in
the validation scripts, not here.

Skips when gravax (or its deps) is not importable: gravax is a sibling workspace package,
not a declared progenax dependency.
"""

import jax
import jax.numpy as jnp
import pytest

gravax = pytest.importorskip("gravax")

from jaxstro.units import STELLAR  # noqa: E402
from progenax import (  # noqa: E402
    PlummerProfile,
    compute_kinetic_energy,
    compute_potential_energy,
)

from gravoturb_fdf.cluster import build_cluster_ic  # noqa: E402

pytestmark = [pytest.mark.experimental, pytest.mark.integration]

G = STELLAR.G


@pytest.fixture(scope="module")
def ic():
    return build_cluster_ic(
        jnp.ones(200),
        mach=8.0, b=0.5, alpha=1.8, beta=3.0,
        profile=PlummerProfile(r_h=0.5), beta_v=4.0,
        Q_target=0.5, f_sub=0.3, shape=(32, 32, 32),
        box_size=4.0, G=G, key=jax.random.PRNGKey(0),
    )


def test_ic_is_finite_and_com_centred(ic):
    assert bool(jnp.all(jnp.isfinite(ic.positions)))
    assert bool(jnp.all(jnp.isfinite(ic.velocities)))
    m = ic.masses
    com = jnp.sum(ic.positions * m[:, None], axis=0) / jnp.sum(m)
    vcom = jnp.sum(ic.velocities * m[:, None], axis=0) / jnp.sum(m)
    assert float(jnp.max(jnp.abs(com))) < 1e-10   # pc
    assert float(jnp.max(jnp.abs(vcom))) < 1e-10  # pc/Myr


def test_realized_virial_ratio_matches_target(ic):
    T = compute_kinetic_energy(ic.velocities, ic.masses)
    V = compute_potential_energy(ic.positions, ic.masses, G=G)
    assert abs(float(T / jnp.abs(V)) - 0.5) < 1e-6


def test_velocity_scale_is_physical(ic):
    # ~200 Msun inside ~0.5 pc: sigma ~ sqrt(GM/r) ~ 1.3 pc/Myr; median |v| must be
    # the right order of magnitude (unit-threading guard, not a physics assertion).
    speed = jnp.linalg.norm(ic.velocities, axis=1)
    med = float(jnp.median(speed))
    assert 0.1 < med < 10.0  # pc/Myr


def test_short_integration_conserves_energy(ic):
    from gravax import LeapfrogIntegrator, ParticleSystem

    system = ParticleSystem.from_velocities(
        positions=ic.positions, velocities=ic.velocities, masses=ic.masses,
        units=STELLAR, softening=0.02,
    )
    e0 = float(system.total_energy)
    final = LeapfrogIntegrator(dt=0.002).integrate(system, t_end=0.1)
    e1 = float(final.total_energy)
    assert jnp.isfinite(e1)
    # Clumpy ICs + fixed-dt leapfrog: a loose smoke tolerance (production uses
    # collisional integrators); catches unit / softening / handoff blunders.
    assert abs((e1 - e0) / e0) < 5e-3

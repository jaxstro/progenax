"""Gravax seam test: a TurbulentCloudIC transfers to an N-body integration intact (audit C13).

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

from gravoturb.cluster import build_cluster_ic  # noqa: E402

pytestmark = [pytest.mark.experimental, pytest.mark.integration]

G = STELLAR.G


@pytest.fixture(scope="module")
def ic():
    from gravoturb.specs import CloudSpec, CompositionSpec, GeometrySpec, VelocitySpec

    return build_cluster_ic(
        jnp.ones(200),
        cloud=CloudSpec(mach=8.0, b=0.5, alpha=1.8, beta=3.0),
        geometry=GeometrySpec(profile=PlummerProfile(r_h=0.5), box_size=4.0,
                              shape=(32, 32, 32)),
        velocity=VelocitySpec(beta_v=4.0, Q_target=0.5),
        composition=CompositionSpec(placement="two_population", f_sub=0.3),
        G=G, key=jax.random.PRNGKey(0),
    )


def test_ic_is_finite_and_com_centred(ic):
    assert bool(jnp.all(jnp.isfinite(ic.stars.positions)))
    assert bool(jnp.all(jnp.isfinite(ic.stars.velocities)))
    m = ic.stars.masses
    com = jnp.sum(ic.stars.positions * m[:, None], axis=0) / jnp.sum(m)
    vcom = jnp.sum(ic.stars.velocities * m[:, None], axis=0) / jnp.sum(m)
    assert float(jnp.max(jnp.abs(com))) < 1e-10   # pc
    assert float(jnp.max(jnp.abs(vcom))) < 1e-10  # pc/Myr


def test_realized_virial_ratio_matches_target(ic):
    T = compute_kinetic_energy(ic.stars.velocities, ic.stars.masses)
    V = compute_potential_energy(ic.stars.positions, ic.stars.masses, G=G)
    assert abs(float(T / jnp.abs(V)) - 0.5) < 1e-6


def test_velocity_scale_is_physical(ic):
    # ~200 Msun inside ~0.5 pc: sigma ~ sqrt(GM/r) ~ 1.3 pc/Myr; median |v| must be
    # the right order of magnitude (unit-threading guard, not a physics assertion).
    speed = jnp.linalg.norm(ic.stars.velocities, axis=1)
    med = float(jnp.median(speed))
    assert 0.1 < med < 10.0  # pc/Myr


def test_short_integration_conserves_energy(ic):
    from gravax import LeapfrogIntegrator, ParticleSystem

    system = ParticleSystem.from_velocities(
        positions=ic.stars.positions, velocities=ic.stars.velocities, masses=ic.stars.masses,
        units=STELLAR, softening=0.02,
    )
    e0 = float(system.total_energy)
    final = LeapfrogIntegrator(dt=0.002).integrate(system, t_end=0.1)
    e1 = float(final.total_energy)
    assert jnp.isfinite(e1)
    # Clumpy ICs + fixed-dt leapfrog: a loose smoke tolerance (production uses
    # collisional integrators); catches unit / softening / handoff blunders.
    assert abs((e1 - e0) / e0) < 5e-3


# ── Phase 2 (AC-IC8d): the physical velocity mode crosses the seam intact ──

@pytest.fixture(scope="module")
def ic_physical():
    from gravoturb.specs import CloudSpec, CompositionSpec, GeometrySpec, VelocitySpec

    return build_cluster_ic(
        jnp.ones(200),
        cloud=CloudSpec(mach=8.0, b=0.5, alpha=1.8, beta=3.0),
        geometry=GeometrySpec(profile=PlummerProfile(r_h=0.5), box_size=4.0,
                              shape=(32, 32, 32)),
        velocity=VelocitySpec(beta_v=4.0, mode="physical", c_s=0.2),  # km/s
        composition=CompositionSpec(placement="two_population", f_sub=0.3),
        G=G, units=STELLAR, key=jax.random.PRNGKey(0),
    )


def test_physical_mode_dispersion_survives_seam(ic_physical):
    """Field-first (Phase 4a): stars COM-centred with the EMERGENT dispersion inside
    the characterized inheritance band (the exact identity lives on the gas grid —
    AC-G7); the handoff contract for the physical mode."""
    ic = ic_physical
    m = ic.stars.masses
    vcom = jnp.sum(ic.stars.velocities * m[:, None], axis=0) / jnp.sum(m)
    assert float(jnp.max(jnp.abs(vcom))) < 1e-10
    sigma = float(jnp.sqrt(jnp.sum(m * jnp.sum(ic.stars.velocities**2, axis=1)) / jnp.sum(m)))
    target = 8.0 * 0.2 / STELLAR.velocity_scale_km_s  # η_v=1 → σ_g
    assert 0.4 < sigma / target < 1.1  # characterized band (see AC-IC8a re-scope)
    assert float(ic.ledger.Q_virial) > 0.0  # emergent, reported


def test_physical_mode_short_integration_conserves_energy(ic_physical):
    from gravax import LeapfrogIntegrator, ParticleSystem

    system = ParticleSystem.from_velocities(
        positions=ic_physical.stars.positions, velocities=ic_physical.stars.velocities,
        masses=ic_physical.stars.masses, units=STELLAR, softening=0.02,
    )
    e0 = float(system.total_energy)
    final = LeapfrogIntegrator(dt=0.002).integrate(system, t_end=0.1)
    e1 = float(final.total_energy)
    assert jnp.isfinite(e1)
    assert abs((e1 - e0) / e0) < 5e-3

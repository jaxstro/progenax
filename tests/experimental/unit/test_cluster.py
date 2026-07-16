"""End-to-end FDF cluster IC builder (Build 4 — the forward generative tool).

``build_cluster_ic`` composes the verified pieces into a complete N-body IC:
  turbulent BM19 field (β,ℳ,α)  →  spherical envelope (progenax SpatialProfile)
  →  star positions (placement ∝ ρ_total, tail mask on s_turb)
  →  coherent turbulent velocities (β_v)  →  virial-scaled to a chosen Q.

Returns positions/velocities/masses (physical units via explicit G) plus the realized
field (for diagnostics) and the realized virial ratio. Spherical + centrally concentrated
(envelope), substructured (turbulence), and dynamically prepared (chosen Q).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.experimental

SHAPE = (16, 16, 16)
BOX = 4.0  # pc


def _profile():
    from progenax import PlummerProfile

    return PlummerProfile(r_h=BOX / 8)  # r_h = 0.5 pc, tapers well inside the box


def _ic(n=600, Q_target=0.5, beta=3.5, key=0):
    from gravoturb.cluster import build_cluster_ic

    masses = jnp.ones(n)
    from jaxstro.units import STELLAR

    from gravoturb.specs import CloudSpec, CompositionSpec, GeometrySpec, VelocitySpec

    return build_cluster_ic(
        masses,
        cloud=CloudSpec(mach=8.0, b=0.5, alpha=1.8, beta=beta),
        geometry=GeometrySpec(profile=_profile(), box_size=BOX, shape=SHAPE),
        velocity=VelocitySpec(beta_v=4.0, Q_target=Q_target),
        composition=CompositionSpec(placement="two_population", f_sub=0.3),
        G=STELLAR.G,
        key=jax.random.PRNGKey(key),
    )


def test_build_cluster_ic_shapes_and_finite():
    """Returns positions/velocities (n,3) and masses (n,), all finite; n = len(masses)."""
    ic = _ic(n=600)
    assert ic.positions.shape == (600, 3)
    assert ic.velocities.shape == (600, 3)
    assert ic.masses.shape == (600,)
    assert np.all(np.isfinite(np.asarray(ic.positions)))
    assert np.all(np.isfinite(np.asarray(ic.velocities)))


def test_build_cluster_ic_is_spherical_and_concentrated():
    """Envelope makes the cluster centrally concentrated vs a uniform box, and COM-centered.

    Uses the MEDIAN radius (robust to the BM19 density tail, which physically broadens the
    cluster by scattering stars into clumps at larger radii — the pure-envelope median is ≈ r_h,
    turbulence broadens it but the cluster stays decisively more concentrated than uniform).
    """
    ic = _ic(n=3000)
    pos = np.asarray(ic.positions)
    com = pos.mean(axis=0)
    assert np.allclose(com, 0.0, atol=0.15)  # COM-centered
    med_r = float(np.median(np.linalg.norm(pos - com, axis=1)))
    # uniform-box baseline (same centered box) for an honest comparison
    u = (
        np.asarray(jax.random.uniform(jax.random.PRNGKey(99), (20000, 3))) * BOX
        - BOX / 2
    )
    med_r_uniform = float(np.median(np.linalg.norm(u, axis=1)))
    assert med_r < 0.78 * med_r_uniform  # clearly concentrated (worst seed ≈0.62)


def test_build_cluster_ic_achieves_target_Q():
    """Realized virial ratio Q = T/|V| matches the target (velocities + virial scaling wired)."""
    from jaxstro.units import STELLAR

    from progenax import compute_kinetic_energy, compute_potential_energy

    ic = _ic(n=600, Q_target=0.5)
    T = compute_kinetic_energy(ic.velocities, ic.masses)
    V = compute_potential_energy(ic.positions, ic.masses, G=STELLAR.G)
    Q = float(T / jnp.abs(V))
    assert abs(Q - 0.5) < 1e-2
    assert abs(float(ic.Q_virial) - 0.5) < 1e-2  # reported Q matches


def test_build_cluster_ic_carries_field_for_diagnostics():
    """The realized TurbulentField is returned (BM19 provenance: f_dense_realized defined on s_turb)."""
    ic = _ic(n=400)
    assert hasattr(ic.field, "f_dense_realized")
    assert np.isfinite(float(ic.field.f_dense_realized))
    assert float(ic.field.f_dense_realized) > 0.0

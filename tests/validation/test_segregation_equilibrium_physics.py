"""Physics validation: the equilibrium quality of mass-segregated cluster ICs.

Quantifies the central caveat of the lambda_seg blend (Phase 0 of the multi-mass
LIMEPY hardening), using per_group_virial_ratio to ask whether each MASS GROUP is
individually in virial balance — the test the global virial rescale hides.

Measured, honest story (these tests lock it):
  - The full-Baumgardt endpoint (lambda_seg=1) is a genuine equilibrium: every mass
    group has Q_j ~ 0.5 (the mass groups are clean energy shells).
  - The lambda_seg blend DEGRADES this: intermediate states are a phase-space chord,
    not an equilibrium, so the per-group drift roughly doubles relative to lambda=1 —
    even though the GLOBAL Q is rescaled to 0.5 throughout.
  - The drift is modest (bounded ~0.15), not catastrophic: the blend is an acceptable
    approximation, but NOT the first-principles partial equilibrium that the multi-mass
    LIMEPY family provides.
  - Regression: the blend never produces coincident stars / non-finite energy at any
    lambda (the orbit-reuse bug is fixed).

A small physical softening (~ mean inter-particle spacing) is used for the per-group
virial so it is not dominated by close-pair r.a noise (the scalar virial's known
sensitivity to close encounters); the function itself defaults to softening=0, where it
reproduces the global virial exactly (Clausius).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro.units import STELLAR
from progenax.cluster import (
    generate_cluster_ic, SpatialStructureParams, MassSegregationLayer,
)
from progenax.imf import PowerLawIMF
from progenax.dynamics import (
    mass_group_masks, per_group_virial_ratio, compute_virial_ratio,
    compute_potential_energy,
)

G = STELLAR.G
N_STARS = 800
N_GROUPS = 4
SOFT = 0.05  # ~ mean inter-particle spacing for r_h=1; suppresses close-pair noise
SEEDS = range(10)


def _cluster(lam, seed, N=N_STARS):
    return generate_cluster_ic(
        key=jax.random.PRNGKey(seed), N_stars=N, M_total=float(N), R_half=1.0,
        imf_params=PowerLawIMF.kroupa(),
        structure_params=SpatialStructureParams(
            base_profile="plummer",
            mass_segregation=MassSegregationLayer(lambda_seg=lam)),
        G=G,
    )


def _mean_group_drift(lam, seeds=SEEDS):
    """Seed-averaged max_j |Q_j - 0.5| (the per-group equilibrium error)."""
    drifts = []
    for s in seeds:
        cl = _cluster(lam, s)
        masks = mass_group_masks(cl.masses, n_groups=N_GROUPS)
        Qj = np.asarray(per_group_virial_ratio(
            cl.positions, cl.velocities, cl.masses, G=G, group_masks=masks, softening=SOFT))
        drifts.append(np.max(np.abs(Qj - 0.5)))
    return float(np.mean(drifts))


def test_full_baumgardt_endpoint_is_per_group_equilibrium():
    """lambda_seg=1 (full energy-ordered segregation) is a genuine equilibrium: each
    mass group, being a clean energy shell, is individually virial (Q_j ~ 0.5)."""
    drift = _mean_group_drift(1.0)
    assert drift < 0.08, f"full-Baumgardt should be a clean equilibrium, drift={drift:.3f}"


def test_blend_degrades_equilibrium_relative_to_baumgardt():
    """The lambda_seg=0.5 blend is NOT a true equilibrium: its per-group drift is
    substantially larger than the full-Baumgardt endpoint (measured ~2x). This is the
    quantified version of the page-2 caveat — the blend is a phase-space chord across
    the curved equilibrium manifold."""
    drift_mid = _mean_group_drift(0.5)
    drift_seg = _mean_group_drift(1.0)
    assert drift_mid > 1.5 * drift_seg, (
        f"blend should degrade equilibrium: mid={drift_mid:.3f} vs Baumgardt={drift_seg:.3f}")


def test_blend_drift_is_bounded_not_catastrophic():
    """The blend's per-group drift is modest (not a broken IC): bounded well below the
    fully-non-equilibrium regime across the whole lambda range. The blend is an
    acceptable approximation; the first-principles alternative is multi-mass LIMEPY."""
    worst = max(_mean_group_drift(lam) for lam in [0.0, 0.25, 0.5, 0.75, 1.0])
    assert worst < 0.15, f"per-group drift unexpectedly large ({worst:.3f}) — not just a chord"


def test_global_virial_is_half_for_all_lambda():
    """The generator rescales the GLOBAL virial to Q=0.5 at every lambda — which is
    exactly why a per-group diagnostic is needed: the global number looks healthy even
    when the blend is internally out of balance."""
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        Qg = np.mean([float(compute_virial_ratio(
            (c := _cluster(lam, s)).positions, c.velocities, c.masses, G=G))
            for s in range(3)])
        assert abs(Qg - 0.5) < 0.02, f"global Q at lambda={lam} is {Qg:.3f}"


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_blend_energy_finite_and_no_coincident_stars(lam):
    """Regression for the orbit-reuse bug: a realistic IMF at any lambda yields finite
    potential energy and strictly distinct positions (pre-fix, lambda=1 gave V=-inf)."""
    cl = _cluster(lam, 0)
    V = float(compute_potential_energy(cl.positions, cl.masses, G=G))
    assert np.isfinite(V), f"V not finite at lambda={lam}"
    d = jnp.linalg.norm(cl.positions[:, None] - cl.positions[None], axis=2)
    d = d + jnp.eye(cl.positions.shape[0]) * 1e9
    assert float(d.min()) > 1e-6, f"coincident stars at lambda={lam}"

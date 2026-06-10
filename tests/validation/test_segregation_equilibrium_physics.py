"""Physics validation: the PRIMORDIAL (energy-ordered) mass-segregation generator.

`energy_sorted_segregation` is a Baumgardt+2008-style primordial generator: it
assigns the most massive stars to the most bound orbits of an equilibrium pool.
The fully-ordered state is the one clean configuration — each mass group is an
energy shell of the parent equilibrium, so every group is individually virial
(Q_j ~ 0.5). These tests lock that property, the orbit-reuse regression (pre-fix,
steep IMFs produced coincident stars and V = -inf), and the segregation signal
itself (Lambda_MSR rises, massive stars more bound).

History: this file previously quantified the lambda_seg BLEND's equilibrium-error
budget. The blend was retired in the 2026-06 unified redesign (the differentiable
segregation knob is now MultiComponentCluster.from_mass_segregation's delta — a
true equilibrium at every value); the endpoint physics tests were ported here onto
the surviving primordial generator, recomposed from the protocol API.

A small physical softening (~ mean inter-particle spacing) is used for the
per-group virial so it is not dominated by close-pair noise; the global virial is
rescaled to exactly 0.5 by construction (rescale_velocities_to_virial).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro.units import STELLAR
from progenax import PlummerProfile, PlummerVelocityDF
from progenax.cluster import energy_sorted_segregation
from progenax.profiles.api import compute_profile_potential
from progenax.imf import PowerLawIMF
from progenax.dynamics import (
    mass_group_masks, per_group_virial_ratio, compute_virial_ratio,
    compute_potential_energy, rescale_velocities_to_virial,
)

G = STELLAR.G
N_STARS = 800
N_GROUPS = 4
POOL_FACTOR = 4
R_HALF = 1.0
SOFT = 0.05  # ~ mean inter-particle spacing for r_h=1; suppresses close-pair noise
SEEDS = range(10)


def _primordial_cluster(seed, N=N_STARS):
    """Fully-segregated primordial cluster from the protocol API.

    Pool: 4N equilibrium Plummer orbits (positions from the profile, velocities
    from the DF at the cluster's total mass); energy ordering uses the ANALYTIC
    Plummer potential — the same potential the DF equilibrates in, as the
    energy_sorted_segregation docstring requires. Finalized with COM removal and
    an exact global virial rescale (Q = 0.5), mirroring the retired generator.
    """
    k_m, k_pos, k_vel, k_seg = jax.random.split(jax.random.PRNGKey(seed), 4)
    masses = PowerLawIMF.kroupa().sample(k_m, N)
    M_total = float(N)
    masses = masses * (M_total / jnp.sum(masses))

    N_pool = POOL_FACTOR * N
    pool_masses = jnp.full(N_pool, M_total / N_pool)
    profile = PlummerProfile(r_h=R_HALF)
    df = PlummerVelocityDF(r_h=R_HALF)
    pos_pool = profile.sample_positions(pool_masses, k_pos)
    vel_pool = df.sample_velocities(pos_pool, pool_masses, k_vel, G=G)

    def potential_fn(p):
        return compute_profile_potential(p, "plummer", M_total, R_HALF, G)

    m, pos, vel = energy_sorted_segregation(k_seg, masses, pos_pool, vel_pool,
                                            potential_fn)

    x_com = jnp.sum(m[:, None] * pos, axis=0) / jnp.sum(m)
    v_com = jnp.sum(m[:, None] * vel, axis=0) / jnp.sum(m)
    pos, vel = pos - x_com, vel - v_com
    vel = rescale_velocities_to_virial(pos, vel, m, G=G, target_Q=0.5)
    return m, pos, vel, pos_pool


def _mean_group_drift(seeds=SEEDS):
    """Seed-averaged max_j |Q_j - 0.5| (the per-group equilibrium error)."""
    drifts = []
    for s in seeds:
        m, pos, vel, _ = _primordial_cluster(s)
        masks = mass_group_masks(m, n_groups=N_GROUPS)
        Qj = np.asarray(per_group_virial_ratio(
            pos, vel, m, G=G, group_masks=masks, softening=SOFT))
        drifts.append(np.max(np.abs(Qj - 0.5)))
    return float(np.mean(drifts))


@pytest.mark.slow
def test_primordial_full_segregation_is_per_group_equilibrium():
    """Full energy-ordered segregation is a genuine equilibrium: each mass group,
    being a clean energy shell of the parent Plummer equilibrium, is individually
    virial (Q_j ~ 0.5). Same 0.08 budget the lambda=1 endpoint carried before the
    blend's retirement."""
    drift = _mean_group_drift()
    assert drift < 0.08, f"primordial endpoint should be a clean equilibrium, drift={drift:.3f}"


def test_global_virial_is_half():
    """The finalization rescales the GLOBAL virial to Q = 0.5 exactly."""
    Qg = np.mean([float(compute_virial_ratio(
        (c := _primordial_cluster(s))[1], c[2], c[0], G=G)) for s in range(3)])
    assert abs(Qg - 0.5) < 0.02, f"global Q is {Qg:.3f}"


def test_energy_finite_and_no_coincident_stars():
    """Regression for the orbit-reuse bug: a realistic (steep) IMF yields finite
    potential energy and strictly distinct positions (pre-fix, full ordering gave
    coincident stars and V = -inf)."""
    m, pos, _, _ = _primordial_cluster(0)
    V = float(compute_potential_energy(pos, m, G=G))
    assert np.isfinite(V), "V not finite for the primordial cluster"
    d = jnp.linalg.norm(pos[:, None] - pos[None], axis=2)
    d = d + jnp.eye(pos.shape[0]) * 1e9
    assert float(d.min()) > 1e-6, "coincident stars in the primordial cluster"


def test_segregation_increases_lambda_msr():
    """The generator produces real mass segregation: Lambda_MSR of the segregated
    cluster exceeds the unsegregated pool's (ported from the retired
    test_cluster_ic.py onto the protocol-API composition)."""
    from progenax.diagnostics import compute_lambda_msr

    m, pos, _, pos_pool = _primordial_cluster(123)
    lam_seg, _ = compute_lambda_msr(np.asarray(pos), np.asarray(m), N_massive=20)
    # Unsegregated reference: the same masses on a random subset of pool orbits
    lam_unseg, _ = compute_lambda_msr(np.asarray(pos_pool[:m.shape[0]]),
                                      np.asarray(m), N_massive=20)
    assert lam_seg > lam_unseg, (
        f"energy ordering should raise Lambda_MSR: {lam_seg:.2f} vs {lam_unseg:.2f}")

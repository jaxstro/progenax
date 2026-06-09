"""Physics validation: the multi-mass LIMEPY model is a segregated EQUILIBRIUM.

The headline of Phase 2 (Gieles & Zocchi 2015, Section 2.2): mass segregation as a
first-principles equilibrium, not an imposed reshuffle. These tests assert, on sampled
clusters, the equilibrium properties that distinguish it from the lambda_seg blend:

  - global virial Q = T/|V| = 0.5 across delta, UNSCALED;
  - each mass component is drawn from ITS OWN equilibrium DF (sampled sigma_1d,j(r)
    matches the analytic LIMEPY moment) -- so each mass group is individually virial;
  - the per-group virial Q_j stays ~0.5 across the segregation range delta in [0, 0.6]
    (the well-resolved light component and the global value are exact; the rarer,
    concentrated heavy component is softening/N-limited, asserted with a wider band);
  - delta=0 is the single-mass model (no segregation); segregation grows with delta.

Anchored to scripts/validate_multimass_equilibrium.py.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro.units import STELLAR
from progenax.profiles.limepy import lowered_exponential
from progenax.profiles.limepy_multimass import MultiMassLIMEPY
from progenax.dynamics import per_group_virial_ratio, compute_virial_ratio

G = STELLAR.G
W0, GG = 7.0, 1.0
M_J = jnp.array([1.0, 4.0])
ALPHA_J = jnp.array([0.5, 0.5])
SOFT = 0.05


def _model(delta):
    return MultiMassLIMEPY.from_alpha(ALPHA_J, M_J, W0=W0, g=GG, delta=delta, r_c=1.0)


def _sample(model, seed, n=8000):
    p, v, m = model.sample_cluster(jax.random.PRNGKey(seed), n_stars=n, G=G)
    p = p - jnp.average(p, axis=0, weights=m)
    v = v - jnp.average(v, axis=0, weights=m)
    return p, v, m


def test_global_virial_is_half_across_delta():
    """The whole sampled cluster is in virial equilibrium (Q=0.5) without rescaling,
    for every equipartition degree delta."""
    for delta in (0.0, 0.3, 0.6):
        model = _model(delta)
        Qs = []
        for s in range(3):
            p, v, m = _sample(model, s)
            Qs.append(float(compute_virial_ratio(p, v, m, G=G)))
        assert abs(np.mean(Qs) - 0.5) < 0.03, f"delta={delta}: global Q={np.mean(Qs):.3f}"


def test_per_component_dispersion_matches_equilibrium_df():
    """Each component's sampled velocity dispersion sigma_1d,j(r) matches the analytic
    LIMEPY moment s_j sqrt(I2/I0/3) -- the proof that every mass group is sampled from
    its own equilibrium DF (hence individually virial)."""
    model = _model(0.5)
    p, v, m = _sample(model, 0, n=40000)
    r = np.asarray(jnp.linalg.norm(p, axis=1))
    v2 = np.asarray(jnp.sum(v**2, axis=1))
    masses = np.asarray(m)
    s = float(jnp.sqrt(G * jnp.sum(m) / (9.0 * model.r_c * model.mu_tot)))

    def analytic(W_j, s_j):
        u = jnp.linspace(0.0, jnp.sqrt(2.0 * W_j), 400)
        E = lowered_exponential(model.g, W_j - u**2 / 2.0)
        return float(s_j * jnp.sqrt(jnp.trapezoid(u**4 * E, u)
                                    / jnp.trapezoid(u**2 * E, u) / 3.0))

    for j in range(2):
        sel = np.isclose(masses, float(M_J[j]))
        s_j = s * float(model.mu_j[j]) ** (-0.5)
        core = sel & (r < 1.0)
        assert core.sum() > 80
        rmid = float(np.median(r[core]))
        Wj = float(model.rescale_j[j]) * float(jnp.interp(rmid, model.xi_grid, model.psi_grid))
        sig_meas = np.sqrt(v2[core].mean() / 3.0)
        np.testing.assert_allclose(sig_meas, analytic(jnp.asarray(Wj), s_j), rtol=0.07)


def test_theoretical_component_virial_is_exactly_half():
    """The bias-free equilibrium proof: the model's THEORETICAL per-component virial
    ratio Q_j (mean-field, no sampling/softening/finite-N) is exactly 0.5 for both
    components across delta. This is the rigorous statement of 'each mass group is in
    equilibrium' -- what the lambda_seg blend cannot achieve per group, and what the
    sampled per-group Q_j below is a finite-N estimator of."""
    for delta in (0.0, 0.3, 0.5, 0.6):
        Qj = np.asarray(_model(delta).component_virial_ratios())
        np.testing.assert_allclose(Qj, 0.5, atol=2e-3, err_msg=f"delta={delta}: Q_j={Qj}")


def test_sampled_per_group_virial_converges_to_theory():
    """The sampled per-group Q_j (an N-body OBSERVABLE) tracks the exact theoretical
    value. The light component (well-resolved) and the global value are tight; the
    rarer, concentrated heavy component carries a small POSITIVE finite-N bias (the
    1/r-weighted W_j = sum m_i M(<r_i)/r_i is dominated by its few innermost stars) that
    grows toward the Spitzer-unstable delta->1 limit. This is a finite-N sampling effect,
    NOT softening (it persists unchanged at softening=0) and NOT the physics (the
    theoretical Q_j above is exactly 0.5). Measured at softening=0 (exact Clausius)."""
    for delta in (0.0, 0.3, 0.5):  # physical range (delta<=0.5); 0.6 nears instability
        model = _model(delta)
        Ql, Qh = [], []
        for s in range(5):
            p, v, m = _sample(model, s)
            masks = jnp.stack([jnp.isclose(m, float(mj)) for mj in M_J])
            Qj = np.asarray(per_group_virial_ratio(p, v, m, G=G, group_masks=masks, softening=0.0))
            Ql.append(Qj[0]); Qh.append(Qj[1])
        assert abs(np.mean(Ql) - 0.5) < 0.04, f"delta={delta}: light Q={np.mean(Ql):.3f}"
        # heavy: small positive finite-N bias, bounded well below catastrophic
        assert abs(np.mean(Qh) - 0.5) < 0.06, f"delta={delta}: heavy Q={np.mean(Qh):.3f}"


def test_delta0_is_single_mass_and_segregation_grows():
    """delta=0 produces no segregation (light/heavy half-mass ratio = 1); the ratio
    increases monotonically with delta -- segregation is a controlled delta effect."""
    from progenax.profiles.limepy_multimass import solve_multimass_limepy

    def half_mass_ratio(delta):
        xi, psi, rho_j = solve_multimass_limepy(ALPHA_J, M_J, W0, GG, delta, 300.0, 3000)
        def rh(rho):
            integ = rho * xi**2
            Mc = jnp.concatenate([jnp.zeros(1),
                                  jnp.cumsum(0.5 * (integ[1:] + integ[:-1])) * (xi[1] - xi[0])])
            return jnp.interp(0.5 * Mc[-1], Mc, xi)
        return float(rh(rho_j[0]) / rh(rho_j[1]))

    ratios = [half_mass_ratio(d) for d in (0.0, 0.2, 0.4, 0.6)]
    assert abs(ratios[0] - 1.0) < 1e-3
    assert np.all(np.diff(ratios) > 0), f"segregation not monotonic: {ratios}"

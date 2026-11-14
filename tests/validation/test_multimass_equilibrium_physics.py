"""Retained Engine A virial, dispersion, anisotropy and segregation checks.

Finite sampled moments are compared with explicit budgets. These initial-condition
checks do not establish dynamical stability or accuracy over the full domain.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxstro.units import STELLAR

from progenax.cluster.multicomponent import MultiComponentCluster
from progenax.dynamics import compute_virial_ratio, per_group_virial_ratio
from progenax.profiles.limepy import lowered_exponential

pytestmark = pytest.mark.slow  # every test samples >=8000 stars / solves aniso ODEs

G = STELLAR.G
W0, GG = 7.0, 1.0
M_J = jnp.array([1.0, 4.0])
ALPHA_J = jnp.array([0.5, 0.5])


def _build_model(delta):
    return MultiComponentCluster.from_mass_segregation(
        alpha_j=ALPHA_J, m_j=M_J, W0=W0, g=GG, delta=delta, r_c=1.0
    )


@pytest.fixture(scope="module")
def isotropic_models():
    """Reuse immutable models without changing configurations or sample keys."""
    models = {}

    def get(delta):
        if delta not in models:
            models[delta] = _build_model(delta)
        return models[delta]

    return get


def _sample(model, seed, n=8000):
    ic = model.sample_cluster(jax.random.PRNGKey(seed), n_stars=n, G=G)
    p = ic.positions - jnp.average(ic.positions, axis=0, weights=ic.masses)
    v = ic.velocities - jnp.average(ic.velocities, axis=0, weights=ic.masses)
    return p, v, ic.masses


def test_global_virial_is_half_across_delta(isotropic_models):
    """Unscaled global virial ratio meets the retained three-seed budget."""
    for delta in (0.0, 0.3, 0.6):
        model = isotropic_models(delta)
        Qs = []
        for s in range(3):
            p, v, m = _sample(model, s)
            Qs.append(float(compute_virial_ratio(p, v, m, G=G)))
        assert abs(np.mean(Qs) - 0.5) < 0.03, (
            f"delta={delta}: global Q={np.mean(Qs):.3f}"
        )


def test_per_component_dispersion_matches_equilibrium_df(isotropic_models):
    """Core dispersions agree with each component DF at the retained 7% budget."""
    model = isotropic_models(0.5)
    p, v, m = _sample(model, 0, n=40000)
    r = np.asarray(jnp.linalg.norm(p, axis=1))
    v2 = np.asarray(jnp.sum(v**2, axis=1))
    masses = np.asarray(m)
    s = float(jnp.sqrt(G * jnp.sum(m) / (9.0 * model.r_c * model.mu_tot)))

    def analytic(W_j, s_j):
        u = jnp.linspace(0.0, jnp.sqrt(2.0 * W_j), 400)
        E = lowered_exponential(model.g, W_j - u**2 / 2.0)
        return float(
            s_j
            * jnp.sqrt(jnp.trapezoid(u**4 * E, u) / jnp.trapezoid(u**2 * E, u) / 3.0)
        )

    for j in range(2):
        sel = np.isclose(masses, float(M_J[j]))
        s_j = s * float(model.w_j[j])  # w_j = mu_j^(-delta), delta=0.5 here
        core = sel & (r < 1.0)
        assert core.sum() > 80
        rmid = float(np.median(r[core]))
        Wj = float(model.rescale_j[j]) * float(
            jnp.interp(rmid, model.xi_grid, model.psi_grid)
        )
        sig_meas = np.sqrt(v2[core].mean() / 3.0)
        np.testing.assert_allclose(sig_meas, analytic(jnp.asarray(Wj), s_j), rtol=0.07)


def test_theoretical_component_virial_is_exactly_half(isotropic_models):
    """Independent continuum component virials agree with one half within 2e-3."""
    for delta in (0.0, 0.3, 0.5, 0.6):
        Qj = np.asarray(isotropic_models(delta).component_virial_ratios())
        np.testing.assert_allclose(
            Qj, 0.5, atol=2e-3, err_msg=f"delta={delta}: Q_j={Qj}"
        )


def test_sampled_per_group_virial_converges_to_theory(isotropic_models):
    """Five-seed component virial means meet the retained 0.04/0.06 budgets."""
    for delta in (0.0, 0.3, 0.5):
        model = isotropic_models(delta)
        Ql, Qh = [], []
        for s in range(5):
            p, v, m = _sample(model, s)
            masks = jnp.stack([jnp.isclose(m, float(mj)) for mj in M_J])
            Qj = np.asarray(
                per_group_virial_ratio(p, v, m, G=G, group_masks=masks, softening=0.0)
            )
            Ql.append(Qj[0])
            Qh.append(Qj[1])
        assert abs(np.mean(Ql) - 0.5) < 0.04, (
            f"delta={delta}: light Q={np.mean(Ql):.3f}"
        )
        assert abs(np.mean(Qh) - 0.5) < 0.06, (
            f"delta={delta}: heavy Q={np.mean(Qh):.3f}"
        )


def _analytic_beta_meanfield(model, j, rr):
    """DF anisotropy beta_j(r) = 1 - <v_t^2>/(2<v_r^2>) by direct (u,c) quadrature of the
    LIMEPY phase-space weight u^2 E_gamma(g, W_j - u^2/2) exp(-(p_j^2/2) u^2 (1-c^2)) --
    the mean-field ground truth (no sampling) the sampled beta must reproduce."""
    psi = float(
        jnp.interp(
            rr / model.r_c, model.xi_grid, model.psi_grid, left=model.W0, right=0.0
        )
    )
    W_j = float(model.rescale_j[j]) * max(psi, 0.0)
    p = (rr / float(model.r_c)) / float(model.ra_hat_j[j])
    u = jnp.linspace(0.0, jnp.sqrt(2.0 * W_j), 400)
    c = jnp.linspace(-1.0, 1.0, 240)
    E = lowered_exponential(model.g, W_j - u**2 / 2.0)
    U, C = jnp.meshgrid(u, c, indexing="ij")
    w = U**2 * E[:, None] * jnp.exp(-(p**2 / 2.0) * U**2 * (1.0 - C**2))
    nr = jnp.trapezoid(jnp.trapezoid(w * U**2 * C**2, c, axis=1), u)
    nt = jnp.trapezoid(jnp.trapezoid(w * U**2 * (1.0 - C**2), c, axis=1), u)
    return float(1.0 - nt / (2.0 * nr))


def test_anisotropic_sampled_cluster_is_equilibrium_and_correctly_anisotropic():
    """Retain global virial and light-component anisotropy gates at fixed exposures."""
    r_a = 5.0
    for delta in (0.0, 0.3, 0.5):
        model = MultiComponentCluster.from_mass_segregation(
            alpha_j=ALPHA_J,
            m_j=M_J,
            W0=W0,
            g=GG,
            delta=delta,
            r_a=r_a,
            eta=0.0,
            r_c=1.0,
            xi_max=800.0,
            n_ode_points=3000,
        )
        Qs = []
        for s in range(3):
            p, v, m = _sample(model, s, n=20000)
            Qs.append(float(compute_virial_ratio(p, v, m, G=G)))
        assert abs(np.mean(Qs) - 0.5) < 0.04, (
            f"aniso delta={delta}: global Q={np.mean(Qs):.3f}"
        )

    # Anisotropy correctness at delta=0.4: sampled beta_light(r) matches the DF.
    model = MultiComponentCluster.from_mass_segregation(
        alpha_j=ALPHA_J,
        m_j=M_J,
        W0=W0,
        g=GG,
        delta=0.4,
        r_a=r_a,
        eta=0.0,
        r_c=1.0,
        xi_max=800.0,
        n_ode_points=3000,
    )
    p, v, m = _sample(model, 0, n=60000)
    r = np.asarray(jnp.linalg.norm(p, axis=1))
    r_hat = np.asarray(p) / (r[:, None] + 1e-30)
    v_r = np.sum(np.asarray(v) * r_hat, axis=1)
    v_t2 = np.sum(np.asarray(v) ** 2, axis=1) - v_r**2
    sel = np.isclose(np.asarray(m), float(M_J[0]))
    rt = float(model.r_t)
    for lo, hi in [(0.3 * rt, 0.5 * rt), (0.5 * rt, 0.7 * rt)]:
        b = sel & (r >= lo) & (r < hi)
        assert b.sum() > 400
        beta_meas = 1.0 - v_t2[b].mean() / (2.0 * (v_r[b] ** 2).mean())
        beta_pred = _analytic_beta_meanfield(model, 0, float(np.median(r[b])))
        assert abs(beta_meas - beta_pred) < 0.05, (
            f"r in [{lo:.1f},{hi:.1f}): sampled beta={beta_meas:.3f} vs DF {beta_pred:.3f}"
        )


def test_delta0_is_single_mass_and_segregation_grows():
    """delta=0 produces no segregation (light/heavy half-mass ratio = 1); the ratio
    increases monotonically with delta -- segregation is a controlled delta effect."""
    from progenax.profiles.limepy_multimass import solve_multimass_limepy

    def half_mass_ratio(delta):
        xi, psi, _, rho_j = solve_multimass_limepy(
            ALPHA_J, M_J, W0, GG, delta, 300.0, 3000
        )

        def rh(rho):
            integ = rho * xi**2
            Mc = jnp.concatenate(
                [
                    jnp.zeros(1),
                    jnp.cumsum(0.5 * (integ[1:] + integ[:-1]) * jnp.diff(xi)),
                ]
            )
            return jnp.interp(0.5 * Mc[-1], Mc, xi)

        return float(rh(rho_j[0]) / rh(rho_j[1]))

    ratios = [half_mass_ratio(d) for d in (0.0, 0.2, 0.4, 0.6)]
    assert abs(ratios[0] - 1.0) < 1e-3
    assert np.all(np.diff(ratios) > 0), f"segregation not monotonic: {ratios}"

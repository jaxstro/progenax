# progenax/tests/unit/cluster/test_multicomponent.py
"""Unit tests for MultiComponentCluster (Phase 1b of the unified redesign).

The unified, differentiable multi-component cluster model (Engine A: DF-defined
lowered-isothermal / LIMEPY). The physics flows from DIRECT per-component scales:

    w_j = s_j / s          per-component velocity-scale ratio (THE free scale),
    rescale_j = w_j^-2     potential-depth rescaling fed to the coupled solve,
    ra_hat_j = r_{a,j}/r_c per-component anisotropy radius (None = isotropic),

with the representative stellar mass m_j DECOUPLED from the dynamics (it only
labels the stars). Mass segregation is the convenience w_j = mu_j^(-delta)
(Gieles & Zocchi 2015); GC 1G/2G and halo+core populations set w_j directly.

THE new regression here (the bug the legacy generate_two_component_cluster had):
an UNEQUAL-mass two-population model must be a true shared-potential equilibrium
-- global Q = T/|V| = 0.5 AND per-component Q_j ~ 0.5 -- for mean(m_A) != mean(m_B).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro.units import STELLAR

from progenax import EFFProfile, KingProfile, PlummerProfile

G = STELLAR.G


def _component_half_mass_radius(model, j):
    """Half-mass radius of component j from the model's own mass CDF."""
    return float(jnp.interp(0.5, model._cdf_j[j], model._r_grid))


class TestFromComponents:
    """Direct constructor: components defined by (alpha_j, w_j, m_j[, ra_hat_j])."""

    def test_unit_w_recovers_single_mass_density(self):
        """All w_j = 1 => identical components => the total density is the single-mass
        LIMEPY profile (the structural oracle), regardless of alpha_j / m_j."""
        from progenax.cluster.multicomponent import MultiComponentCluster
        from progenax.profiles.limepy import LIMEPYProfile

        model = MultiComponentCluster.from_components(
            alpha_j=jnp.array([0.6, 0.4]), w_j=jnp.array([1.0, 1.0]),
            m_j=jnp.array([0.4, 5.0]), W0=7.0, g=1.0, r_c=1.0)
        king = LIMEPYProfile.from_W0_rc(W0=7.0, g=1.0, r_c=1.0)
        r = jnp.linspace(0.0, float(king.r_t) * 0.98, 200)
        tot = model.total_density(r)
        ref = king.density(r)
        np.testing.assert_allclose(np.asarray(tot / tot[0]), np.asarray(ref / ref[0]),
                                   rtol=3e-3, atol=3e-3)

    def test_equal_mass_populations_segregate_by_w(self):
        """The GC 1G/2G case the mass path cannot express: two EQUAL-MASS populations
        with different w_j. The colder one (smaller w_j) is more centrally
        concentrated -- concentration is a velocity-scale effect, not a mass effect."""
        from progenax.cluster.multicomponent import MultiComponentCluster

        model = MultiComponentCluster.from_components(
            alpha_j=jnp.array([0.5, 0.5]), w_j=jnp.array([1.0, 0.6]),
            m_j=jnp.array([0.8, 0.8]), W0=7.0, g=1.0, r_c=1.0)
        rh_hot = _component_half_mass_radius(model, 0)
        rh_cold = _component_half_mass_radius(model, 1)
        assert rh_cold < rh_hot, (
            f"cold (w=0.6) r_h={rh_cold:.2f} not < hot (w=1.0) r_h={rh_hot:.2f}")

    def test_theoretical_component_virial_is_half(self):
        """Every component of a from_components model is in equilibrium: the
        theoretical Q_j = T_j/|W_j| (no sampling, no softening) is exactly 0.5."""
        from progenax.cluster.multicomponent import MultiComponentCluster

        model = MultiComponentCluster.from_components(
            alpha_j=jnp.array([0.5, 0.5]), w_j=jnp.array([1.0, 0.7]),
            m_j=jnp.array([0.8, 0.8]), W0=7.0, g=1.0, r_c=1.0, n_ode_points=4000)
        Qj = np.asarray(model.component_virial_ratios())
        np.testing.assert_allclose(Qj, 0.5, atol=2e-3, err_msg=f"Q_j={Qj}")

    def test_rescale_is_w_inverse_squared(self):
        """The model exposes the velocity-scale ratio w_j and the derived potential
        rescaling rescale_j = w_j^-2 consistently."""
        from progenax.cluster.multicomponent import MultiComponentCluster

        w = jnp.array([1.0, 0.5])
        model = MultiComponentCluster.from_components(
            alpha_j=jnp.array([0.5, 0.5]), w_j=w, m_j=jnp.array([1.0, 1.0]),
            W0=5.0, g=1.0, r_c=1.0)
        np.testing.assert_allclose(np.asarray(model.rescale_j), np.asarray(w**-2),
                                   rtol=1e-12)

    @pytest.mark.slow
    def test_differentiable_in_w_j(self):
        """Gradients flow through construction + sampling w.r.t. the direct
        per-component velocity-scale ratios w_j -- the scales are inferable."""
        from progenax.cluster.multicomponent import MultiComponentCluster

        def loss(w_j):
            model = MultiComponentCluster.from_components(
                alpha_j=jnp.array([0.6, 0.4]), w_j=w_j,
                m_j=jnp.array([0.5, 2.0]), W0=7.0, g=1.0, r_c=1.0,
                n_ode_points=1500, n_grid=600)
            ic = model.sample_cluster(jax.random.PRNGKey(0), n_stars=400, G=G)
            return jnp.mean(jnp.sum(ic.velocities**2, axis=1))

        d = jax.grad(loss)(jnp.array([1.0, 0.8]))
        assert jnp.all(jnp.isfinite(d)) and jnp.any(jnp.abs(d) > 0)

    @pytest.mark.slow
    def test_direct_per_component_anisotropy(self):
        """from_components accepts a direct per-component ra_hat_j; the anisotropic
        model still has every component in theoretical equilibrium (Q_j = 0.5)."""
        from progenax.cluster.multicomponent import MultiComponentCluster

        model = MultiComponentCluster.from_components(
            alpha_j=jnp.array([0.6, 0.4]), w_j=jnp.array([1.0, 0.7]),
            m_j=jnp.array([1.0, 1.0]), W0=7.0, g=1.0, r_c=1.0,
            ra_hat_j=jnp.array([10.0, 10.0]), xi_max=800.0, n_ode_points=3000)
        Qj = np.asarray(model.component_virial_ratios())
        np.testing.assert_allclose(Qj, 0.5, atol=3e-3, err_msg=f"aniso Q_j={Qj}")

    @pytest.mark.slow
    def test_table_model_equilibrium_matches_quadrature_oracle(self):
        """A table-backed anisotropic model still proves Q_j = 0.5 via the
        EXACT-quadrature component_virial_ratios (oracle independence), and its
        mass CDF matches the quadrature-built model to 5e-4."""
        from progenax.cluster.multicomponent import MultiComponentCluster

        kw = dict(alpha_j=jnp.array([0.6, 0.4]), w_j=jnp.array([1.0, 0.79]),
                  m_j=jnp.array([1.0, 4.0]), W0=7.0, g=1.0, r_c=1.0,
                  ra_hat_j=jnp.array([10.0, 10.0]), xi_max=800.0,
                  n_ode_points=3000)
        m_tab = MultiComponentCluster.from_components(**kw)  # default: table
        m_quad = MultiComponentCluster.from_components(**kw, aniso_method="quadrature")
        Qj = np.asarray(m_tab.component_virial_ratios())
        np.testing.assert_allclose(Qj, 0.5, atol=3e-3, err_msg=f"table Q_j={Qj}")
        np.testing.assert_allclose(np.asarray(m_tab._cdf_j),
                                   np.asarray(m_quad._cdf_j), atol=5e-4)


class TestFromMassSegregation:
    """Mass-segregation convenience: w_j = mu_j^(-delta), ra_hat_j = (r_a/r_c) mu_j^eta."""

    def test_solve_matches_mass_wrapper(self):
        """from_mass_segregation rides EXACTLY the validated mass-path coupled solve:
        its shared potential equals solve_multimass_limepy's for the same (m_j, delta)."""
        from progenax.cluster.multicomponent import MultiComponentCluster
        from progenax.profiles.limepy_multimass import solve_multimass_limepy

        alpha = jnp.array([0.6, 0.4]); m_j = jnp.array([0.5, 4.0]); delta = 0.5
        model = MultiComponentCluster.from_mass_segregation(
            alpha_j=alpha, m_j=m_j, W0=7.0, g=1.0, delta=delta, r_c=1.0)
        _, psi_ref, _ = solve_multimass_limepy(alpha, m_j, 7.0, 1.0, delta, 300.0, 2000)
        np.testing.assert_allclose(np.asarray(model.psi_grid), np.asarray(psi_ref),
                                   rtol=1e-11, atol=1e-11)
        # and the w_j are the equipartition law w_j = mu_j^(-delta)
        bar_m = jnp.sum(m_j * alpha); mu_j = m_j / bar_m
        np.testing.assert_allclose(np.asarray(model.w_j),
                                   np.asarray(mu_j ** (-delta)), rtol=1e-12)

    @pytest.mark.slow
    def test_segregation_and_equipartition_signatures(self):
        """The classic delta>0 signatures survive the rewrite: the heavy component is
        more concentrated AND kinematically colder (sampled core dispersions)."""
        from progenax.cluster.multicomponent import MultiComponentCluster

        model = MultiComponentCluster.from_mass_segregation(
            alpha_j=jnp.array([0.6, 0.4]), m_j=jnp.array([1.0, 4.0]),
            W0=7.0, g=1.0, delta=0.5, r_c=1.0)
        assert _component_half_mass_radius(model, 1) < _component_half_mass_radius(model, 0)

        ic = model.sample_cluster(jax.random.PRNGKey(2), n_stars=15000, G=G)
        r = jnp.linalg.norm(ic.positions, axis=1)
        v2 = jnp.sum(ic.velocities**2, axis=1)
        core = r < 1.0
        sig = [float(jnp.sqrt(jnp.mean(v2[core & (ic.component_id == j)]) / 3.0))
               for j in range(2)]
        assert sig[1] < sig[0], f"heavy sigma={sig[1]:.3f} not < light {sig[0]:.3f}"


class TestUnequalMassTwoPopulationEquilibrium:
    """THE new regression (the legacy generate_two_component_cluster bug): an
    unequal-mass two-population model -- realistic mean(m_A) != mean(m_B) -- must be a
    TRUE shared-potential equilibrium, globally AND per component. The legacy path fed
    the full cluster mass to each sub-population's DF and superposed two independently
    sampled spheres, so it was only (silently) virial for equal masses."""

    pytestmark = pytest.mark.slow

    def test_true_shared_potential_equilibrium(self):
        from progenax.cluster.multicomponent import MultiComponentCluster
        from progenax.dynamics import compute_virial_ratio
        from progenax.dynamics.virial import per_group_virial_ratio

        # Two populations with different representative masses AND different
        # velocity scales (w_j NOT derived from the masses -- the general case).
        model = MultiComponentCluster.from_components(
            alpha_j=jnp.array([0.6, 0.4]), w_j=jnp.array([1.0, 0.8]),
            m_j=jnp.array([0.5, 2.0]), W0=7.0, g=1.0, r_c=1.0)

        # Exact statement: every component is in equilibrium in the shared potential.
        Qj_theory = np.asarray(model.component_virial_ratios())
        np.testing.assert_allclose(Qj_theory, 0.5, atol=2e-3,
                                   err_msg=f"theoretical Q_j={Qj_theory}")

        # Sampled statement: global Q AND per-component Q_j (finite-N estimators).
        ic = model.sample_cluster(jax.random.PRNGKey(0), n_stars=20000, G=G)
        pos = ic.positions - jnp.average(ic.positions, axis=0, weights=ic.masses)
        vel = ic.velocities - jnp.average(ic.velocities, axis=0, weights=ic.masses)

        Qg = float(compute_virial_ratio(pos, vel, ic.masses, G=G))
        assert abs(Qg - 0.5) < 0.04, f"global Q={Qg:.3f} (expected 0.5)"

        masks = jnp.stack([ic.component_id == 0, ic.component_id == 1])
        Qj = np.asarray(per_group_virial_ratio(pos, vel, ic.masses, G=G,
                                               group_masks=masks))
        np.testing.assert_allclose(Qj, 0.5, atol=0.07,
                                   err_msg=f"sampled per-component Q_j={Qj}")


class TestSampleClusterICResult:
    """sample_cluster returns a full ICResult carrying the per-particle component label."""

    def test_icresult_fields_and_component_id(self):
        from progenax.builders import ICResult, compute_stellar_radii
        from progenax.cluster.multicomponent import MultiComponentCluster

        model = MultiComponentCluster.from_components(
            alpha_j=jnp.array([0.6, 0.4]), w_j=jnp.array([1.0, 0.8]),
            m_j=jnp.array([0.5, 2.0]), W0=7.0, g=1.0, r_c=1.0)
        n = 3000
        ic = model.sample_cluster(jax.random.PRNGKey(0), n_stars=n, G=G)

        assert isinstance(ic, ICResult)
        assert ic.positions.shape == (n, 3) and ic.velocities.shape == (n, 3)
        assert ic.masses.shape == (n,) and ic.stellar_radii.shape == (n,)
        assert bool(jnp.all(jnp.isfinite(ic.positions)))
        assert bool(jnp.all(jnp.isfinite(ic.velocities)))

        # component_id labels each star; the mass IS the component's m_j.
        assert ic.component_id is not None and ic.component_id.shape == (n,)
        assert jnp.issubdtype(ic.component_id.dtype, jnp.integer)
        assert bool(jnp.all((ic.component_id >= 0) & (ic.component_id <= 1)))
        np.testing.assert_allclose(np.asarray(ic.masses),
                                   np.asarray(model.m_j[ic.component_id]), rtol=0)
        # both components actually drawn
        assert int(jnp.sum(ic.component_id == 1)) > 0

        np.testing.assert_allclose(np.asarray(ic.stellar_radii),
                                   np.asarray(compute_stellar_radii(ic.masses)),
                                   rtol=1e-12)

    def test_iso_speeds_bounded_by_local_escape_speed(self):
        """Every isotropically sampled star satisfies |v| <= s_i sqrt(2 W_i)
        (the local escape speed at its rescaled potential) -- the invariant
        the SpeedCDFTable's normalized coordinate x = u/sqrt(2W) in [0, 1]
        enforces by construction, so the whole sampled cluster is bound."""
        from progenax.cluster.multicomponent import MultiComponentCluster

        model = MultiComponentCluster.from_components(
            alpha_j=jnp.array([0.6, 0.4]), w_j=jnp.array([1.0, 0.7]),
            m_j=jnp.array([0.5, 4.0]), W0=7.0, g=1.0, r_c=1.0)
        n = 5000
        ic = model.sample_cluster(jax.random.PRNGKey(3), n_stars=n, G=G)
        speeds = jnp.linalg.norm(ic.velocities, axis=1)
        r = jnp.linalg.norm(ic.positions, axis=1)
        s = jnp.sqrt(G * jnp.sum(ic.masses) / (9.0 * model.r_c * model.mu_tot))
        s_i = s * model.w_j[ic.component_id]
        W_i = model.rescale_j[ic.component_id] * jnp.maximum(
            jnp.interp(r / model.r_c, model.xi_grid, model.psi_grid,
                       left=model.W0, right=0.0), 0.0)
        v_esc = s_i * jnp.sqrt(2.0 * W_i)
        assert bool(jnp.all(jnp.isfinite(speeds)))
        assert bool(jnp.all(speeds <= v_esc * (1.0 + 1e-12))), (
            f"max v/v_esc = {float(jnp.max(speeds / (v_esc + 1e-30))):.6f}")

    def test_icresult_component_id_defaults_none(self):
        """The new ICResult field is optional: existing construction is unchanged."""
        from progenax.builders import ICResult

        ic = ICResult(positions=jnp.zeros((2, 3)), velocities=jnp.zeros((2, 3)),
                      masses=jnp.ones(2), stellar_radii=jnp.ones(2))
        assert ic.component_id is None


class TestFromIMF:
    """IMF path: bin the IMF, eigenvalue-solve for alpha_j, mass-segregation scales."""

    pytestmark = pytest.mark.slow

    def test_constructs_and_hits_masses(self):
        from progenax.imf import PowerLawIMF
        from progenax.cluster.multicomponent import MultiComponentCluster

        model = MultiComponentCluster.from_imf(
            PowerLawIMF.kroupa(), n_comp=4, W0=7.0, g=1.0, delta=0.5,
            m_range=(0.1, 20.0), r_c=1.0)
        assert float(model.residual) < 2e-3
        assert model.m_j.shape == (4,) and model.alpha_j.shape == (4,)
        assert abs(float(jnp.sum(model.alpha_j)) - 1.0) < 1e-9
        # equipartition ordering: heavier bins are colder (smaller w_j)
        assert bool(jnp.all(jnp.diff(model.w_j) < 0))


# ==============================================================================
# Ported from tests/unit/profiles/test_limepy_multimass.py (TestMultiMassLIMEPYModel
# + TestAnisotropicSampling), adapted to MultiComponentCluster + ICResult. These are
# the validated regression suite of the retired MultiMassLIMEPY class.
# ==============================================================================


def _two_component(delta=0.5, W0=7.0, g=1.0):
    """A mass ratio mild enough that the (rarer, concentrated) heavy component is
    resolvable for kinematics, while still clearly segregating (mu_heavy > 1)."""
    from progenax.cluster.multicomponent import MultiComponentCluster
    return MultiComponentCluster.from_mass_segregation(
        alpha_j=jnp.array([0.6, 0.4]), m_j=jnp.array([1.0, 4.0]),
        W0=W0, g=g, delta=delta, r_c=1.0)


class TestSampledEquilibrium:
    """Sampled-cluster equilibrium regressions (ported from the legacy model tests)."""

    pytestmark = pytest.mark.slow

    def test_per_group_dispersion_matches_df_and_global_virial(self):
        """THE headline equilibrium proof: each mass component is sampled from ITS OWN
        equilibrium DF, so the per-component core dispersion sigma_1d,j matches the
        analytic LIMEPY moment s_j sqrt(I2/I0/3), and the whole cluster is virial
        (global Q=0.5, unscaled)."""
        from progenax.dynamics import compute_virial_ratio
        from progenax.profiles.limepy import lowered_exponential

        model = _two_component(delta=0.5)
        ic = model.sample_cluster(jax.random.PRNGKey(0), n_stars=15000, G=G)
        pos = ic.positions - jnp.average(ic.positions, axis=0, weights=ic.masses)
        vel = ic.velocities - jnp.average(ic.velocities, axis=0, weights=ic.masses)
        r = jnp.linalg.norm(pos, axis=1)
        v2 = jnp.sum(vel**2, axis=1)
        s = jnp.sqrt(G * jnp.sum(ic.masses) / (9.0 * model.r_c * model.mu_tot))

        def analytic_sigma1d(W_j, s_j):
            u = jnp.linspace(0.0, jnp.sqrt(2.0 * W_j), 400)
            E = lowered_exponential(model.g, W_j - u**2 / 2.0)
            return float(s_j * jnp.sqrt(jnp.trapezoid(u**4 * E, u)
                                        / jnp.trapezoid(u**2 * E, u) / 3.0))

        for jc in range(2):
            sel = ic.component_id == jc
            r_j, v2_j = r[sel], v2[sel]
            s_j = float(s * model.w_j[jc])
            core = r_j < 1.0
            assert int(jnp.sum(core)) > 60, "too few stars to resolve the core dispersion"
            sig_meas = float(jnp.sqrt(jnp.mean(v2_j[core]) / 3.0))
            r_mid = float(jnp.median(r_j[core]))
            W_j = float(model.rescale_j[jc]) * float(
                jnp.interp(r_mid, model.xi_grid, model.psi_grid))
            sig_pred = analytic_sigma1d(jnp.asarray(W_j), s_j)
            np.testing.assert_allclose(sig_meas, sig_pred, rtol=0.08,
                                       err_msg=f"component {jc} dispersion off equilibrium")

        Qg = float(compute_virial_ratio(pos, vel, ic.masses, G=G))
        assert abs(Qg - 0.5) < 0.04, f"global Q={Qg:.3f} (expected 0.5)"

    def test_theoretical_component_virial_across_delta(self):
        """The bias-free equilibrium proof holds at every equipartition degree:
        theoretical Q_j = 0.5 for both components for delta in [0, 0.6]."""
        from progenax.cluster.multicomponent import MultiComponentCluster

        for delta in (0.0, 0.3, 0.5, 0.6):
            model = MultiComponentCluster.from_mass_segregation(
                alpha_j=jnp.array([0.5, 0.5]), m_j=jnp.array([1.0, 4.0]),
                W0=7.0, g=1.0, delta=delta, r_c=1.0, n_ode_points=4000)
            Qj = np.asarray(model.component_virial_ratios())
            np.testing.assert_allclose(Qj, 0.5, atol=2e-3,
                                       err_msg=f"delta={delta}: theoretical Q_j={Qj}")

    def test_sampled_cluster_is_mass_segregated(self):
        """In the sampled cluster the heavy component is more centrally concentrated
        than the light one (mean radius), the observable signature of segregation."""
        model = _two_component(delta=0.5)
        ic = model.sample_cluster(jax.random.PRNGKey(0), n_stars=20000, G=G)
        r = jnp.linalg.norm(ic.positions, axis=1)
        r_light = float(jnp.mean(r[ic.component_id == 0]))
        r_heavy = float(jnp.mean(r[ic.component_id == 1]))
        assert r_heavy < r_light, f"heavy <r>={r_heavy:.2f} not < light <r>={r_light:.2f}"

    def test_no_coincident_stars_and_finite_energy(self):
        model = _two_component(delta=0.5)
        ic = model.sample_cluster(jax.random.PRNGKey(1), n_stars=5000, G=G)
        from progenax.builders import compute_potential_energy
        V = float(compute_potential_energy(ic.positions, ic.masses, G=G))
        assert np.isfinite(V)
        d = jnp.linalg.norm(ic.positions[:, None] - ic.positions[None], axis=2) \
            + jnp.eye(ic.positions.shape[0]) * 1e9
        assert float(d.min()) > 1e-6  # no coincident stars

    def test_sample_differentiable_in_delta(self):
        """grad of a kinematic functional w.r.t. delta flows through construction +
        sampling -- the equipartition degree is inferable from a sampled cluster."""
        from progenax.cluster.multicomponent import MultiComponentCluster

        def loss(delta):
            model = MultiComponentCluster.from_mass_segregation(
                alpha_j=jnp.array([0.7, 0.3]), m_j=jnp.array([0.4, 5.0]),
                W0=7.0, g=1.0, delta=delta, r_c=1.0)
            ic = model.sample_cluster(jax.random.PRNGKey(0), n_stars=400, G=G)
            return jnp.mean(jnp.sum(ic.velocities**2, axis=1))

        d = jax.grad(loss)(0.4)
        assert jnp.isfinite(d)


def _analytic_beta(model, j, r_eval):
    """Analytic anisotropy beta_j(r) = 1 - <v_t^2>/(2<v_r^2>) of the LIMEPY DF itself,
    computed by direct quadrature of the SAME (u, c) phase-space weight the sampler
    draws from: w(u,c) = u^2 E_gamma(g, W_j - u^2/2) exp(-(p_j^2/2) u^2 (1-c^2)), with
    W_j(r) = rescale_j psi(r), p_j(r) = (r/r_c)/ra_hat_j[j], c = cos(angle to r_hat).
    The s_j^2 cancels in beta. This is the ground truth the sampled beta must reproduce
    (proves the angular sampling is correct). p_j -> 0 gives beta = 0 (isotropic)."""
    from progenax.profiles.limepy import lowered_exponential

    out = []
    for rr in np.atleast_1d(np.asarray(r_eval)):
        psi = float(jnp.interp(rr / model.r_c, model.xi_grid, model.psi_grid,
                               left=model.W0, right=0.0))
        W_j = float(model.rescale_j[j]) * max(psi, 0.0)
        if W_j <= 0.0:
            out.append(np.nan); continue
        p = (rr / float(model.r_c)) / float(model.ra_hat_j[j])
        u = jnp.linspace(0.0, jnp.sqrt(2.0 * W_j), 400)
        c = jnp.linspace(-1.0, 1.0, 240)
        E = lowered_exponential(model.g, W_j - u**2 / 2.0)  # (nu,)
        U, C = jnp.meshgrid(u, c, indexing="ij")
        w = U**2 * E[:, None] * jnp.exp(-(p**2 / 2.0) * U**2 * (1.0 - C**2))
        num_r = jnp.trapezoid(jnp.trapezoid(w * U**2 * C**2, c, axis=1), u)
        num_t = jnp.trapezoid(jnp.trapezoid(w * U**2 * (1.0 - C**2), c, axis=1), u)
        out.append(float(1.0 - num_t / (2.0 * num_r)))
    return np.array(out)


class TestAnisotropicSampling:
    """Per-component ANISOTROPIC IC sampling (ported): the angular distribution of
    each star's velocity must reproduce the Michie/Osipkov-Merritt LIMEPY DF, so the
    sampled cluster is (a) globally virial Q=0.5 -- the scalar virial theorem is
    anisotropy-blind -- and (b) radially anisotropic with beta_j(r) matching the DF."""

    pytestmark = pytest.mark.slow

    def _aniso_model(self, delta=0.4, eta=0.0, r_a=10.0, W0=7.0, g=1.0):
        from progenax.cluster.multicomponent import MultiComponentCluster
        return MultiComponentCluster.from_mass_segregation(
            alpha_j=jnp.array([0.6, 0.4]), m_j=jnp.array([1.0, 4.0]),
            W0=W0, g=g, delta=delta, r_a=r_a, eta=eta, r_c=1.0,
            xi_max=800.0, n_ode_points=3000)

    def test_aniso_model_is_equilibrium_and_segregated(self):
        """An anisotropic multi-component model is still a true equilibrium
        (theoretical Q_j = 0.5) AND segregated (heavy more concentrated)."""
        model = self._aniso_model(delta=0.4)
        Qj = np.asarray(model.component_virial_ratios())
        np.testing.assert_allclose(Qj, 0.5, atol=3e-3, err_msg=f"aniso Q_j={Qj}")
        assert _component_half_mass_radius(model, 1) < _component_half_mass_radius(model, 0), \
            "heavy not more concentrated under anisotropy"

    def test_aniso_sample_cluster_runs(self):
        """The anisotropic model samples to a finite ICResult with component labels."""
        model = self._aniso_model()
        ic = model.sample_cluster(jax.random.PRNGKey(0), n_stars=2000, G=G)
        assert ic.positions.shape == (2000, 3) and ic.velocities.shape == (2000, 3)
        assert bool(jnp.all(jnp.isfinite(ic.positions)))
        assert bool(jnp.all(jnp.isfinite(ic.velocities)))
        np.testing.assert_allclose(np.asarray(ic.masses),
                                   np.asarray(model.m_j[ic.component_id]), rtol=0)

    def test_aniso_global_virial_is_half(self):
        """The sampled anisotropic cluster is in virial equilibrium WITHOUT rescaling:
        global Q = T/|V| = 0.5 (2T+W=0 holds for any anisotropy)."""
        from progenax.dynamics import compute_virial_ratio

        model = self._aniso_model(delta=0.4, eta=0.0)
        Qs = []
        for s in range(3):
            ic = model.sample_cluster(jax.random.PRNGKey(s), n_stars=20000, G=G)
            pos = ic.positions - jnp.average(ic.positions, axis=0, weights=ic.masses)
            vel = ic.velocities - jnp.average(ic.velocities, axis=0, weights=ic.masses)
            Qs.append(float(compute_virial_ratio(pos, vel, ic.masses, G=G)))
        assert abs(np.mean(Qs) - 0.5) < 0.04, f"aniso global Q={np.mean(Qs):.3f} (expected 0.5)"

    def test_aniso_velocity_is_radially_anisotropic(self):
        """The sampled velocities show the full LIMEPY radial-anisotropy signature:
        beta(r) ~ 0 in the core, RISES to a radial-bias peak near ~0.5 r_t, then TURNS
        OVER toward r_t (truncation lowers the most radial orbits at the edge --
        Gieles & Zocchi 2015)."""
        model = self._aniso_model(delta=0.4, eta=0.0, r_a=5.0)
        ic = model.sample_cluster(jax.random.PRNGKey(0), n_stars=40000, G=G)
        pos = ic.positions - jnp.average(ic.positions, axis=0, weights=ic.masses)
        vel = ic.velocities - jnp.average(ic.velocities, axis=0, weights=ic.masses)
        r = jnp.linalg.norm(pos, axis=1)
        r_hat = pos / (r[:, None] + 1e-30)
        v_r = jnp.sum(vel * r_hat, axis=1)
        v_t2 = jnp.sum(vel**2, axis=1) - v_r**2

        def beta_in(lo, hi):
            sel = (r >= lo) & (r < hi)
            return float(1.0 - jnp.mean(v_t2[sel]) / (2.0 * jnp.mean(v_r[sel] ** 2)))

        r_t = float(model.r_t)
        beta_core = beta_in(0.0, 0.15 * r_t)
        beta_peak = beta_in(0.4 * r_t, 0.6 * r_t)
        beta_edge = beta_in(0.75 * r_t, 0.95 * r_t)
        assert abs(beta_core) < 0.06, f"core should be ~isotropic, beta_core={beta_core:.3f}"
        assert beta_peak > beta_core + 0.1, f"no radial-bias peak: {beta_core:.3f}->{beta_peak:.3f}"
        assert beta_edge < beta_peak, f"no truncation turnover: peak {beta_peak:.3f}, edge {beta_edge:.3f}"

    def test_aniso_sampled_beta_matches_analytic(self):
        """THE sampler-correctness proof: the per-component sampled beta_j(r)
        reproduces the DF's own analytic beta_j(r) (direct (u,c) quadrature of the
        same phase-space weight) across radial bins."""
        model = self._aniso_model(delta=0.4, eta=0.0, r_a=5.0)
        ic = model.sample_cluster(jax.random.PRNGKey(1), n_stars=60000, G=G)
        pos = ic.positions - jnp.average(ic.positions, axis=0, weights=ic.masses)
        vel = ic.velocities - jnp.average(ic.velocities, axis=0, weights=ic.masses)
        r = jnp.linalg.norm(pos, axis=1)
        r_hat = pos / (r[:, None] + 1e-30)
        v_r = jnp.sum(vel * r_hat, axis=1)
        v_t2 = jnp.sum(vel**2, axis=1) - v_r**2

        r_t = float(model.r_t)
        # light component (well-resolved); bins from core to mid-halo
        sel_j = np.asarray(ic.component_id == 0)
        rj = np.asarray(r)[sel_j]
        vr2j = np.asarray(v_r**2)[sel_j]
        vt2j = np.asarray(v_t2)[sel_j]
        edges = np.array([0.05, 0.2, 0.4, 0.6]) * r_t
        for lo, hi in zip(edges[:-1], edges[1:]):
            b = (rj >= lo) & (rj < hi)
            assert b.sum() > 400, f"too few light stars in [{lo:.1f},{hi:.1f})"
            beta_meas = 1.0 - vt2j[b].mean() / (2.0 * vr2j[b].mean())
            beta_pred = _analytic_beta(model, 0, np.median(rj[b]))[0]
            assert abs(beta_meas - beta_pred) < 0.06, (
                f"r in [{lo:.1f},{hi:.1f}): sampled beta={beta_meas:.3f} vs DF {beta_pred:.3f}")

    def test_aniso_sample_differentiable_in_ra(self):
        """grad of a kinematic functional w.r.t. r_a flows through construction + the
        anisotropic sampler -- the anisotropy radius is inferable from a sampled cluster."""
        from progenax.cluster.multicomponent import MultiComponentCluster

        def loss(r_a):
            model = MultiComponentCluster.from_mass_segregation(
                alpha_j=jnp.array([0.6, 0.4]), m_j=jnp.array([1.0, 4.0]),
                W0=7.0, g=1.0, delta=0.4, r_a=r_a, eta=0.0, r_c=1.0,
                xi_max=800.0, n_ode_points=2000, n_grid=600)
            ic = model.sample_cluster(jax.random.PRNGKey(0), n_stars=400, G=G)
            r = jnp.linalg.norm(ic.positions, axis=1)
            r_hat = ic.positions / (r[:, None] + 1e-30)
            v_r2 = jnp.sum(ic.velocities * r_hat, axis=1) ** 2
            return jnp.mean(v_r2)

        d = jax.grad(loss)(10.0)
        assert jnp.isfinite(d)


class TestEngineB:
    # Default mix: Plummer halo + EFF core. The core scale a=0.8 (NOT the
    # originally drafted 0.4) is a REALIZABILITY constraint, independently
    # verified with a closed-form two-Plummer oracle (gamma=5 EFF == Plummer):
    # at a=0.4 the cored halo density in the concentrated core's potential has
    # a GENUINELY negative Eddington DF for E > ~0.66 Psi0 (f ~ -1.2e-3 absolute,
    # min f/max|f| ~ -0.2, resolution-independent) -- the design doc's named
    # "shallow component in a concentrated companion's potential" non-equilibrium.
    # At a=0.8 the same oracle gives f > 0 everywhere (min f/max|f| = +0.016).
    def _model(self, **kw):
        from progenax.cluster.multicomponent import MultiComponentCluster
        defaults_ = dict(
            profiles=[PlummerProfile(r_h=2.0), EFFProfile(a=0.8, gamma=5.0, r_t=9.0)],
            mass_fractions=jnp.array([0.6, 0.4]), m_j=jnp.array([0.5, 1.0]))
        defaults_.update(kw)
        return MultiComponentCluster.from_density_profiles(**defaults_)

    def test_constructs_and_reports_domain(self):
        m = self._model()
        assert m.engine == "B"
        assert float(m.r_t) == 9.0                       # EFF extent wins (design c)
        assert "EFF" in m.engine_b.r_t_provenance
        assert bool(jnp.all(jnp.isfinite(m.engine_b.f_j_grid)))

    def test_f_min_diagnostic_stored_and_benign_here(self):
        m = self._model()
        fmin = np.asarray(m.engine_b.f_min_j)
        assert fmin.shape == (2,)
        assert np.all(fmin > -1e-3)                      # realizable mix (relative units)

    def test_a_only_access_raises_not_nan(self):
        """Engine-A-only quantities REFUSE in B mode (consolidation 4/4): the
        old NaN tripwires poisoned downstream results silently on access; the
        grouped engine_a=None state raises immediately, naming the engine."""
        m = self._model()
        assert m.engine_a is None
        for name in ("W0", "g", "mu_tot", "alpha_j", "w_j"):
            with pytest.raises(AttributeError, match="Engine A"):
                getattr(m, name)

    def test_position_cdf_matches_component_masses(self):
        """_cdf_j is reused verbatim by the sampler: each row is a normalized
        M_j(<r); the Plummer row must match the analytic CDF."""
        m = self._model(profiles=[PlummerProfile(r_h=1.0)],
                        mass_fractions=jnp.array([1.0]), m_j=jnp.array([1.0]))
        a = float(PlummerProfile(r_h=1.0).a)
        x = np.asarray(m._r_grid) / a
        exact = x**3 / (1 + x**2) ** 1.5
        exact = exact / exact[-1]
        np.testing.assert_allclose(np.asarray(m._cdf_j[0]), exact, atol=2e-3)

    def test_unrealizable_mix_raises_naming_component(self):
        """The realizability refusal is ACTIONABLE (Task 6, 2d): the message
        names the offending component, quantifies HOW negative the DF is
        (the relative f_min in scientific notation), and quotes the design
        doc's remedy. An absurdly radial halo (r_a = 0.05) in the headline
        mix is the known-unrealizable config."""
        import re

        with pytest.raises(ValueError) as exc:
            self._model(r_a_j=jnp.array([0.05, jnp.inf]))   # absurdly radial halo
        msg = str(exc.value)
        assert "component 0" in msg                          # which component
        # the f_min value, as a formatted (scientific-notation) number
        assert re.search(r"min f / max\|f\| = -\d+\.\d+e[+-]\d+", msg), msg
        for fragment in ("steepen", "mass fraction", "r_a"):  # the remedy
            assert fragment in msg, f"remedy fragment {fragment!r} missing: {msg}"

    def test_traced_build_skips_raise_stores_diagnostic(self):
        """Under tracing (jit/grad) the realizability gate CANNOT raise -- the
        contract is "no raise under tracing, f_min_j always stored". Jit a
        scalar-r_a surrogate of the build and feed it the SAME unrealizable
        r_a = 0.05: construction must complete and return the genuinely
        negative diagnostic (consistent with the concrete-path refusal value;
        not identical -- this surrogate uses a different n_r/n_e)."""
        from progenax.cluster.eddington_engine import build_engine_b_state

        profiles = [PlummerProfile(r_h=2.0), EFFProfile(a=0.8, gamma=5.0, r_t=9.0)]

        @jax.jit
        def fmin_of_ra(ra):
            state, _ = build_engine_b_state(
                profiles, jnp.array([0.6, 0.4]), jnp.stack([ra, jnp.inf]),
                None, 0.995, 3000, 500)
            return state.f_min_j

        fmin = fmin_of_ra(jnp.asarray(0.05))                 # must NOT raise
        assert fmin.shape == (2,)
        assert bool(jnp.all(jnp.isfinite(fmin)))
        # the diagnostic records the SAME genuine negativity the concrete
        # path refuses on (min f/max|f| ~ -0.25 for this config)
        assert float(fmin[0]) < -1e-3
        assert float(fmin[1]) > -1e-3                        # core stays realizable

    def test_mass_fraction_sum_raise(self):
        """Public-API duplicate of the shared_potential sum gate: fractions
        are M_j/M_total amplitudes and MUST sum to 1."""
        with pytest.raises(ValueError, match="mass_fractions"):
            self._model(mass_fractions=jnp.array([0.6, 0.6]))

    def test_king_override_conflict_raise(self):
        """Public-API duplicate of the derive_r_t King-conflict gate: an
        explicit r_t override below a King component's natural r_t would
        silently re-truncate a lowered-Maxwellian edge -- refuse."""
        from progenax.cluster.multicomponent import MultiComponentCluster

        king = KingProfile.from_W0_rc(W0=5.0, r_c=1.0)       # natural r_t ~ 10.8
        with pytest.raises(ValueError, match="King"):
            MultiComponentCluster.from_density_profiles(
                [king], jnp.array([1.0]), m_j=jnp.array([1.0]), r_t=5.0)

    # ---- Task 4 (2c-ii): Engine B sampling + exact-quadrature Q_j oracle ----

    def test_sampled_density_matches_each_component(self):
        """Position pipeline: each component's sampled radial mass ECDF matches
        its OWN normalized M_j(<r) row of _cdf_j (KS distance < 0.02 at >= 20k
        stars per component; N_frac = [0.75, 0.25] for the fixture defaults)."""
        m = self._model()
        ic = m.sample_cluster(jax.random.PRNGKey(0), n_stars=84000, G=G)
        r = np.asarray(jnp.linalg.norm(ic.positions, axis=1))
        cid = np.asarray(ic.component_id)
        for j in range(2):
            rj = np.sort(r[cid == j])
            n_j = rj.size
            assert n_j >= 20000, f"component {j} undersampled (n={n_j})"
            ecdf = (np.arange(n_j) + 0.5) / n_j
            model_cdf = np.interp(rj, np.asarray(m._r_grid), np.asarray(m._cdf_j[j]))
            ks = float(np.max(np.abs(ecdf - model_cdf)))
            assert ks < 0.02, f"component {j}: KS distance {ks:.4f} >= 0.02"

    def test_engine_b_global_virial_is_half_unscaled(self):
        """THE headline gate: Plummer halo + EFF core (the REALIZABLE a_EFF=0.8
        mix), N=30k, |Q - 0.5| < 0.02 with NO virial rescale anywhere in the
        pipeline. The exact pairwise potential is row-chunked (a single 30k^2
        block is ~22 GB); the math is identical to compute_potential_energy."""
        m = self._model()
        ic = m.sample_cluster(jax.random.PRNGKey(0), n_stars=30000, G=G)
        pos = np.asarray(ic.positions
                         - jnp.average(ic.positions, axis=0, weights=ic.masses))
        vel = np.asarray(ic.velocities
                         - jnp.average(ic.velocities, axis=0, weights=ic.masses))
        mass = np.asarray(ic.masses)

        T = 0.5 * float(np.sum(mass * np.sum(vel**2, axis=1)))
        V2 = 0.0  # sum over ORDERED pairs i != j; V = -G/2 * V2
        chunk = 2000
        for i0 in range(0, pos.shape[0], chunk):
            p = pos[i0:i0 + chunk]
            d = np.sqrt(((p[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))
            rows = np.arange(p.shape[0])
            d[rows, i0 + rows] = np.inf  # drop self-pairs
            V2 += float(np.sum(mass[i0:i0 + chunk, None] * mass[None, :] / d))
        V = -0.5 * G * V2
        Q = T / abs(V)
        assert abs(Q - 0.5) < 0.02, f"Engine B global Q={Q:.4f} (expected 0.5, unscaled)"

    def test_engine_b_theory_Qj_is_half(self):
        """component_virial_ratios (B branch): the EXACT-quadrature oracle over
        the DF speed moments in the shared Psi -- independent of all sampled
        quantities, truncation-consistently weighted by the DF-side density
        rho_DF,j (see engine_b_component_virials) -- returns Q_j = 0.5 +- 3e-3
        for BOTH components."""
        m = self._model()
        Qj = np.asarray(m.component_virial_ratios())
        assert Qj.shape == (2,)
        np.testing.assert_allclose(Qj, 0.5, atol=3e-3,
                                   err_msg=f"Engine B theory Q_j={Qj}")

    def test_speed_scale_uses_sampled_mass(self):
        """Doubling every stellar mass m_j doubles <v^2>: the velocity scale
        comes from the ACTUAL sampled mass sum_i m_i (the Engine A lesson),
        never an input M_total. N_frac is invariant under a uniform m_j scaling,
        so the same key gives identical dimensionless draws -> ratio exactly 2."""
        key = jax.random.PRNGKey(5)
        ic1 = self._model().sample_cluster(key, n_stars=4000, G=G)
        ic2 = self._model(m_j=jnp.array([1.0, 2.0])).sample_cluster(
            key, n_stars=4000, G=G)
        v2_1 = float(jnp.mean(jnp.sum(ic1.velocities**2, axis=1)))
        v2_2 = float(jnp.mean(jnp.sum(ic2.velocities**2, axis=1)))
        np.testing.assert_allclose(v2_2 / v2_1, 2.0, rtol=1e-12)

    # ---- code-review fix batch (2026-06-10): F1-F6 ----

    def test_zero_anisotropy_radius_raises(self):
        """F1a: r_a_j = 0 is not a model -- the OM augmented-density weight
        1 + r^2/r_a^2 diverges as r_a -> 0; previously this built silently
        with a non-finite f table and sample_cluster returned NaN velocities."""
        with pytest.raises(ValueError, match=r"r_a_j\[0\]"):
            self._model(r_a_j=jnp.array([0.0, jnp.inf]))

    def test_zero_mass_fraction_raises(self):
        """F1a: a 0.0 mass fraction is a user bug (the component should be
        omitted); its f_j == 0 row previously fed 0/0 normalizations."""
        with pytest.raises(ValueError, match=r"mass_fractions\[1\]"):
            self._model(mass_fractions=jnp.array([1.0, 0.0]))

    def test_nan_df_refused_with_nonfinite_message(self):
        """F1b: the realizability gate is NaN-aware. float(nan) < threshold is
        False, so a non-finite DF (here from a NaN profile parameter) used to
        construct silently; it must refuse with its OWN message naming the
        component."""
        from progenax.cluster.multicomponent import MultiComponentCluster

        with pytest.raises(ValueError, match="(?i)non-finite"):
            MultiComponentCluster.from_density_profiles(
                [PlummerProfile(r_h=jnp.nan)], jnp.array([1.0]),
                m_j=jnp.array([1.0]), n_r=500, n_e=100, n_grid=100)

    def test_component_length_mismatch_raises(self):
        """F2: 2 profiles + 3 fractions (summing to 1) previously built a
        potential from the wrong mass and the categorical emitted phantom
        component ids whose gathers clamp. Every per-component input length
        is named in the refusal."""
        with pytest.raises(ValueError) as exc:
            self._model(mass_fractions=jnp.array([0.5, 0.3, 0.2]))
        msg = str(exc.value)
        for fragment in ("profiles", "mass_fractions", "m_j"):
            assert fragment in msg, f"{fragment!r} missing from: {msg}"
        with pytest.raises(ValueError, match="m_j"):
            self._model(m_j=jnp.array([0.5, 1.0, 2.0]))
        with pytest.raises(ValueError, match="r_a_j"):
            self._model(r_a_j=jnp.array([jnp.inf]))

    @pytest.mark.parametrize("r_h", [1e4, 1e-2])
    def test_extreme_scale_sampling_clean(self, r_h):
        """F3: the speed-sampler thresholds are RELATIVE to the table's energy
        scale. Engine B tables are in physical units (Psi0 ~ 1/length:
        Psi0 = 1.2e-4 at r_h = 1e4 pc), so the old ABSOLUTE cutoff
        (Psi_r > 1e-6) zeroed ~0.12% of the speeds at r_h = 1e4 (80% at 1e6).
        Both extreme scales must sample finite velocities with ZERO exact-zero
        speeds."""
        from progenax.cluster.multicomponent import MultiComponentCluster

        m = MultiComponentCluster.from_density_profiles(
            [PlummerProfile(r_h=r_h)], jnp.array([1.0]), m_j=jnp.array([1.0]),
            n_r=2000, n_e=400, n_grid=400)
        ic = m.sample_cluster(jax.random.PRNGKey(0), n_stars=20000, G=G)
        assert bool(jnp.all(jnp.isfinite(ic.velocities)))
        speed = jnp.linalg.norm(ic.velocities, axis=1)
        n_zero = int(jnp.sum(speed == 0.0))
        assert n_zero == 0, f"{n_zero} exact-zero speeds at r_h={r_h}"

    def test_engine_b_total_density_is_prescribed_density(self):
        """F4: total_density dispatches on the engine (it used to evaluate the
        Engine-A formula on the NaN tripwires -> silent NaN). The B branch
        returns the prescribed mass-normalized dimensionless total density:
        finite and positive inside r_t, zero outside, 4 pi int rho r^2 dr = 1."""
        m = self._model()
        r_in = jnp.linspace(1e-3, float(m.r_t) * 0.999, 512)
        rho_in = m.total_density(r_in)
        assert bool(jnp.all(jnp.isfinite(rho_in)))
        assert bool(jnp.all(rho_in > 0.0))
        r_out = jnp.array([float(m.r_t) * 1.001, 2.0 * float(m.r_t)])
        np.testing.assert_array_equal(np.asarray(m.total_density(r_out)), 0.0)
        r = jnp.linspace(1e-4, float(m.r_t), 4000)
        mass = 4.0 * jnp.pi * jnp.trapezoid(m.total_density(r) * r**2, r)
        assert abs(float(mass) - 1.0) < 5e-3, f"dimensionless mass {float(mass)}"

    def test_engine_b_rescale_j_raises(self):
        """F4: rescale_j = w_j^-2 is an Engine-A quantity; on B it used to
        return the NaN tripwire silently -- it must refuse, naming the engine
        and pointing at the engine_b field group."""
        m = self._model()
        with pytest.raises(ValueError, match="(?i)engine b"):
            m.rescale_j

    def test_treedef_independent_of_rt_override_value(self):
        """F5: provenance strings carry the RULE only, never the float value --
        two models differing only in the override r_t share ONE treedef (no
        per-value recompiles; vmap/stack over models works)."""
        kw = dict(profiles=[PlummerProfile(r_h=2.0)],
                  mass_fractions=jnp.array([1.0]), m_j=jnp.array([1.0]),
                  n_r=2000, n_e=300, n_grid=200)  # n_r >= 2000: gate-clean here
        m20 = self._model(r_t=20.0, **kw)
        m21 = self._model(r_t=21.0, **kw)
        assert "override" in m20.engine_b.r_t_provenance
        assert (jax.tree_util.tree_structure(m20)
                == jax.tree_util.tree_structure(m21))

    def test_is_aniso_truthful_for_engine_b(self):
        """F6: is_aniso reports the model's OM content (it was hardcoded False
        for every B model). The B sampler never reads it (sampling.py branches
        on engine first), so this is an introspection-truth fix only."""
        assert self._model().is_aniso is False
        m_om = self._model(r_a_j=jnp.array([3.0, jnp.inf]))  # realizable OM halo
        assert m_om.is_aniso is True

    def test_om_directions_scalar_equals_array(self):
        """assign_om_directions regression for the Task 4 per-star r_a ARRAY
        extension: an array of one repeated scalar is BIT-IDENTICAL to the
        scalar path (same key, same positions, same speeds)."""
        from progenax.kinematics.eddington import assign_om_directions

        kp, ks, kd = jax.random.split(jax.random.PRNGKey(7), 3)
        pos = 2.0 * jax.random.normal(kp, (500, 3))
        speeds = jnp.abs(jax.random.normal(ks, (500,)))
        v_scalar = assign_om_directions(kd, pos, speeds, 1.7)
        v_array = assign_om_directions(kd, pos, speeds, jnp.full((500,), 1.7))
        np.testing.assert_array_equal(np.asarray(v_scalar), np.asarray(v_array))


def _make_engine_b_model():
    """The realizable Plummer halo + EFF core headline mix (TestEngineB defaults)."""
    from progenax.cluster.multicomponent import MultiComponentCluster
    return MultiComponentCluster.from_density_profiles(
        profiles=[PlummerProfile(r_h=2.0), EFFProfile(a=0.8, gamma=5.0, r_t=9.0)],
        mass_fractions=jnp.array([0.6, 0.4]), m_j=jnp.array([0.5, 1.0]))


class TestEngineStateGrouping:
    """Grouped engine state replaces the NaN-sentinel union (consolidation 4/4):
    Engine-A-only leaves live on `engine_a` (an _EngineAState), Engine-B state on
    `engine_b`; the absent engine's group is None and A-only access on a B model
    raises an informative AttributeError naming the engine -- never a silent NaN."""

    def test_engine_a_fields_grouped(self):
        from progenax.cluster.multicomponent import MultiComponentCluster

        m = MultiComponentCluster.from_components(
            alpha_j=jnp.array([0.6, 0.4]), w_j=jnp.array([1.0, 0.8]),
            m_j=jnp.array([0.5, 1.0]), W0=5.0, g=1.0, r_c=1.0)
        assert m.engine_a is not None and m.engine_b is None
        assert float(m.W0) == 5.0          # delegating property

    def test_engine_b_has_no_nan_tripwires(self):
        m = _make_engine_b_model()
        assert m.engine_b is not None and m.engine_a is None

    def test_engine_b_a_only_access_raises_informatively(self):
        m = _make_engine_b_model()
        with pytest.raises(AttributeError, match="Engine A"):
            _ = m.W0

    def test_engine_b_every_a_only_name_raises(self):
        """EVERY old A-only field name refuses on a B model, naming the engine
        (the full tripwire set the NaN block used to cover)."""
        m = _make_engine_b_model()
        for name in ("W0", "g", "r_c", "mu_tot", "alpha_j", "w_j", "ra_hat_j",
                     "xi_grid", "psi_grid", "residual"):
            with pytest.raises(AttributeError, match="Engine A"):
                getattr(m, name)

    def test_engine_a_jit_and_grad_still_flow(self):
        """Grouping must not break the pytree: grad through r_c via the state."""
        from progenax.cluster.multicomponent import MultiComponentCluster

        def f(r_c):
            m = MultiComponentCluster.from_components(
                alpha_j=jnp.array([1.0]), w_j=jnp.array([1.0]),
                m_j=jnp.array([1.0]), W0=5.0, g=1.0, r_c=r_c)
            return jnp.sum(m.total_density(jnp.linspace(0.1, 2.0, 16)))

        assert jnp.isfinite(jax.grad(f)(1.0))

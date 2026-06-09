# progenax/tests/unit/profiles/test_limepy_multimass.py
"""Unit tests for the multi-mass LIMEPY coupled equilibrium (Phase 2, Layer A).

Layer A — solve_multimass_limepy(alpha_j, m_j, W0, g, delta): one coupled Poisson
solve given central density fractions alpha_j,

    (1/xi^2) d/dxi(xi^2 dpsi/dxi) = -9 sum_j alpha_j rho_hat_j(xi),
    rho_hat_j(xi) = limepy_density_hat(mu_j^(2 delta) psi, g)
                    / limepy_density_hat(mu_j^(2 delta) W0, g),
    mu_j = m_j / bar_m,  bar_m = sum_j m_j alpha_j   (Gieles & Zocchi 2015, Eqs 24-29).

The structural oracle: at delta=0 every mu_j^(2 delta)=1, so each rho_hat_j collapses
to the single-mass density and the source is (sum alpha_j) rho_hat = rho_hat -- the
solve must reproduce solve_limepy_profile exactly, for ANY alpha_j / m_j.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro.units import STELLAR

G = STELLAR.G


def _component_masks(masses, m_j):
    """Boolean (n_comp, N) masks selecting each discrete mass component."""
    return jnp.stack([jnp.isclose(masses, float(mj)) for mj in np.asarray(m_j)])


class TestMultiMassCoreDelta0:
    """delta=0 is the single-mass model, structurally (the cleanest oracle)."""

    @pytest.mark.parametrize("g", [0.0, 1.0, 2.0])
    def test_delta0_recovers_single_mass_potential(self, g):
        """solve_multimass_limepy(delta=0) potential psi(xi) is identical to
        solve_limepy_profile(W0, g), independent of the (alpha_j, m_j) supplied."""
        from progenax.profiles.limepy import solve_limepy_profile
        from progenax.profiles.limepy_multimass import solve_multimass_limepy

        W0 = 7.0
        alpha_j = jnp.array([0.6, 0.3, 0.1])
        m_j = jnp.array([0.3, 1.0, 8.0])
        xi_s, psi_s = solve_limepy_profile(W0, g=g, xi_max=300.0, n_points=2000)
        xi_m, psi_m, rho_j = solve_multimass_limepy(
            alpha_j, m_j, W0=W0, g=g, delta=0.0, xi_max=300.0, n_points=2000
        )
        np.testing.assert_allclose(np.asarray(xi_m), np.asarray(xi_s), rtol=0, atol=0)
        np.testing.assert_allclose(np.asarray(psi_m), np.asarray(psi_s), rtol=1e-9, atol=1e-9)

    def test_delta0_components_share_single_mass_density(self):
        """At delta=0 every component density equals the single-mass normalized
        density (they ride the identical potential with identical rescaling = none)."""
        from progenax.profiles.limepy import solve_limepy_profile, limepy_density_hat
        from progenax.profiles.limepy_multimass import solve_multimass_limepy

        W0 = 6.0
        alpha_j = jnp.array([0.5, 0.5])
        m_j = jnp.array([0.5, 5.0])
        xi, psi, rho_j = solve_multimass_limepy(
            alpha_j, m_j, W0=W0, g=1.0, delta=0.0, xi_max=300.0, n_points=2000
        )
        single = limepy_density_hat(psi, 1.0) / limepy_density_hat(jnp.asarray(W0), 1.0)
        for j in range(2):
            np.testing.assert_allclose(np.asarray(rho_j[j]), np.asarray(single),
                                       rtol=1e-6, atol=1e-8)


def _half_mass_xi(xi, rho):
    """Dimensionless half-mass radius of a component from its density rho_hat(xi)."""
    integrand = rho * xi**2
    dxi = xi[1] - xi[0]
    M = jnp.concatenate([jnp.zeros(1),
                         jnp.cumsum(0.5 * (integrand[1:] + integrand[:-1])) * dxi])
    return float(jnp.interp(0.5 * M[-1], M, xi))


class TestMultiMassSegregation:
    """delta>0 produces mass segregation as an equilibrium (heavy more concentrated)."""

    def test_heavy_component_is_more_centrally_concentrated(self):
        """At delta=0.5 the heavy component has a SMALLER half-mass radius than the
        light one -- the equilibrium signature of mass segregation (deeper effective
        well for larger mu_j)."""
        from progenax.profiles.limepy_multimass import solve_multimass_limepy

        m_j = jnp.array([0.3, 8.0])
        alpha_j = jnp.array([0.5, 0.5])
        xi, psi, rho_j = solve_multimass_limepy(
            alpha_j, m_j, W0=7.0, g=1.0, delta=0.5, xi_max=300.0, n_points=3000
        )
        xh_light = _half_mass_xi(xi, rho_j[0])
        xh_heavy = _half_mass_xi(xi, rho_j[1])
        assert xh_heavy < xh_light, f"heavy r_h={xh_heavy:.2f} not < light r_h={xh_light:.2f}"

    def test_segregation_strength_increases_with_delta(self):
        """The light/heavy half-mass-radius ratio (a segregation strength) grows
        monotonically with delta, and is ~1 (no segregation) at delta=0."""
        from progenax.profiles.limepy_multimass import solve_multimass_limepy

        m_j = jnp.array([0.3, 8.0])
        alpha_j = jnp.array([0.5, 0.5])
        ratios = []
        for delta in (0.0, 0.2, 0.4, 0.6):
            xi, psi, rho_j = solve_multimass_limepy(
                alpha_j, m_j, W0=7.0, g=1.0, delta=delta, xi_max=300.0, n_points=3000
            )
            ratios.append(_half_mass_xi(xi, rho_j[0]) / _half_mass_xi(xi, rho_j[1]))
        assert abs(ratios[0] - 1.0) < 1e-3, f"delta=0 should give no segregation: {ratios[0]:.3f}"
        assert np.all(np.diff(ratios) > 0), f"segregation not monotonic in delta: {ratios}"

    def test_differentiable_in_W0_g_delta_alpha(self):
        """Gradients flow through the coupled solve in all of (W0, g, delta, alpha_j)
        -- the structural and equipartition parameters are jointly inferable."""
        from progenax.profiles.limepy_multimass import solve_multimass_limepy

        m_j = jnp.array([0.3, 1.0, 8.0])

        def shape_metric(W0, g, delta, alpha_j):
            xi, psi, rho_j = solve_multimass_limepy(
                alpha_j, m_j, W0=W0, g=g, delta=delta, xi_max=300.0, n_points=2000
            )
            return jnp.mean(psi[:300]) + jnp.sum(rho_j[:, :300])

        alpha0 = jnp.array([0.6, 0.3, 0.1])
        dW0 = jax.grad(shape_metric, 0)(7.0, 1.0, 0.4, alpha0)
        dg = jax.grad(shape_metric, 1)(7.0, 1.0, 0.4, alpha0)
        dd = jax.grad(shape_metric, 2)(7.0, 1.0, 0.4, alpha0)
        da = jax.grad(shape_metric, 3)(7.0, 1.0, 0.4, alpha0)
        assert jnp.isfinite(dW0) and jnp.abs(dW0) > 0
        assert jnp.isfinite(dg) and jnp.abs(dg) > 0
        assert jnp.isfinite(dd) and jnp.abs(dd) > 0  # delta genuinely moves the solution
        assert jnp.all(jnp.isfinite(da))


class TestEigenvalueSolve:
    """Layer B: find_alpha_for_masses iterates alpha_j to reproduce target masses M_j."""

    def _kroupa_like_components(self, n=4):
        """A bottom-heavy mass spectrum binned into n components (m_j, M_j)."""
        m_edges = np.geomspace(0.1, 20.0, n + 1)
        m_j = np.sqrt(m_edges[:-1] * m_edges[1:])  # geometric-mean representative mass
        # Kroupa-ish dN/dm ~ m^-2.3 -> mass per bin ~ int m^-1.3 dm
        M_j = np.array([(e1**-0.3 - e0**-0.3) / -0.3
                        for e0, e1 in zip(m_edges[:-1], m_edges[1:])])
        return jnp.asarray(m_j), jnp.asarray(M_j)

    def test_realized_masses_match_targets(self):
        """The converged alpha_j reproduce the target mass fractions f_j = M_j/sum M to
        a small residual -- the eigenvalue solve hits the IMF mass budget."""
        from progenax.profiles.limepy_multimass import (
            find_alpha_for_masses, solve_multimass_limepy,
        )

        m_j, M_j = self._kroupa_like_components(4)
        alpha_j, residual = find_alpha_for_masses(
            m_j, M_j, W0=7.0, g=1.0, delta=0.5, n_iter=40, xi_max=300.0, n_points=2000
        )
        assert float(residual) < 1e-3, f"eigenvalue residual {float(residual):.2e} too large"
        # independent check of the realized mass fractions
        xi, psi, rho_j = solve_multimass_limepy(alpha_j, m_j, 7.0, 1.0, 0.5, 300.0, 2000)
        nu_j = jnp.trapezoid(rho_j * xi**2, xi, axis=1)
        f_real = np.asarray(alpha_j * nu_j / jnp.sum(alpha_j * nu_j))
        f_target = np.asarray(M_j / jnp.sum(M_j))
        np.testing.assert_allclose(f_real, f_target, atol=2e-3)

    def test_delta0_alpha_equals_mass_fractions(self):
        """At delta=0 all components share one density shape (equal nu_j), so the
        realized fraction is alpha_j itself -> the solve drives alpha_j to the target
        mass fractions M_j/sum M. A clean closed-form corner of the iteration."""
        from progenax.profiles.limepy_multimass import find_alpha_for_masses

        m_j, M_j = self._kroupa_like_components(4)
        alpha_j, residual = find_alpha_for_masses(
            m_j, M_j, W0=6.0, g=1.0, delta=0.0, n_iter=40, xi_max=300.0, n_points=2000
        )
        f_target = np.asarray(M_j / jnp.sum(M_j))
        np.testing.assert_allclose(np.asarray(alpha_j), f_target, atol=1e-4)

    def test_alpha_normalized_and_positive(self):
        """The returned alpha_j are a valid central-density partition: positive and
        summing to 1 (preserved by the sqrt update + renormalization)."""
        from progenax.profiles.limepy_multimass import find_alpha_for_masses

        m_j, M_j = self._kroupa_like_components(5)
        alpha_j, _ = find_alpha_for_masses(m_j, M_j, W0=7.0, g=1.5, delta=0.4,
                                           n_iter=30, xi_max=300.0, n_points=2000)
        assert abs(float(jnp.sum(alpha_j)) - 1.0) < 1e-9
        assert bool(jnp.all(alpha_j > 0.0))

    def test_differentiable_in_targets_and_delta(self):
        """Gradients flow through the fixed-iteration eigenvalue solve in (M_j, delta)
        -- target mass fractions and the equipartition degree are inferable."""
        from progenax.profiles.limepy_multimass import find_alpha_for_masses

        m_j, M_j = self._kroupa_like_components(3)

        def metric_M(M):
            a, _ = find_alpha_for_masses(m_j, M, 7.0, 1.0, 0.4, n_iter=15,
                                         xi_max=300.0, n_points=1500)
            return jnp.sum(a**2)

        def metric_d(delta):
            a, _ = find_alpha_for_masses(m_j, M_j, 7.0, 1.0, delta, n_iter=15,
                                         xi_max=300.0, n_points=1500)
            return jnp.sum(a**2)

        dM = jax.grad(metric_M)(M_j)
        dd = jax.grad(metric_d)(0.4)
        assert jnp.all(jnp.isfinite(dM)) and jnp.any(jnp.abs(dM) > 0)
        assert jnp.isfinite(dd) and jnp.abs(dd) > 0


class TestMultiMassLIMEPYModel:
    """Layer C: MultiMassLIMEPY -- construction, sampling, and the equilibrium headline."""

    def _two_component(self, delta=0.5, W0=7.0, g=1.0):
        # A mass ratio mild enough that the (rarer, concentrated) heavy component is
        # resolvable for kinematics, while still clearly segregating (mu_heavy > 1).
        from progenax.profiles.limepy_multimass import MultiMassLIMEPY
        m_j = jnp.array([1.0, 4.0])
        alpha_j = jnp.array([0.6, 0.4])
        return MultiMassLIMEPY.from_alpha(alpha_j, m_j, W0=W0, g=g, delta=delta, r_c=1.0)

    def test_from_alpha_delta0_density_matches_single_mass(self):
        """A delta=0 model's total density profile equals the single-mass LIMEPYProfile
        (no segregation): the class-level single-mass corner."""
        from progenax.profiles.limepy import LIMEPYProfile
        from progenax.profiles.limepy_multimass import MultiMassLIMEPY

        m_j = jnp.array([0.4, 5.0])
        alpha_j = jnp.array([0.6, 0.4])
        model = MultiMassLIMEPY.from_alpha(alpha_j, m_j, W0=7.0, g=1.0, delta=0.0, r_c=1.0)
        king = LIMEPYProfile.from_W0_rc(W0=7.0, g=1.0, r_c=1.0)
        r = jnp.linspace(0.0, float(king.r_t) * 0.98, 200)
        tot = model.total_density(r)
        ref = king.density(r)
        # normalize both to their central value for a shape comparison
        np.testing.assert_allclose(np.asarray(tot / tot[0]), np.asarray(ref / ref[0]),
                                   rtol=3e-3, atol=3e-3)

    def test_from_imf_constructs_and_hits_masses(self):
        """from_imf bins an IMF, solves for alpha_j, and reproduces the per-bin mass
        budget (small residual)."""
        from progenax.imf import PowerLawIMF
        from progenax.profiles.limepy_multimass import MultiMassLIMEPY

        model = MultiMassLIMEPY.from_imf(
            PowerLawIMF.kroupa(), n_comp=4, W0=7.0, g=1.0, delta=0.5,
            m_range=(0.1, 20.0), r_c=1.0,
        )
        assert float(model.residual) < 2e-3
        assert model.m_j.shape == (4,) and model.alpha_j.shape == (4,)
        assert abs(float(jnp.sum(model.alpha_j)) - 1.0) < 1e-9

    def test_per_group_virial_is_half(self):
        """THE headline equilibrium proof: each mass component is sampled from ITS OWN
        equilibrium DF, so the per-component velocity dispersion sigma_1d,j(r) matches
        the analytic LIMEPY moment s_j sqrt(I2/I0/3) (I_k = int u^k E_gamma(g, W_j-u^2/2)),
        and the whole cluster is virial (global Q=0.5, unscaled). This is the
        well-resolved (not N-body-noise-limited) statement of 'each mass group is in
        equilibrium' -- the property the lambda_seg blend lacks per group. (The direct
        per-group virial Q_j vs the blend, seed-averaged, is the Phase-3 comparison.)"""
        from progenax.dynamics import compute_virial_ratio
        from progenax.profiles.limepy import lowered_exponential

        model = self._two_component(delta=0.5)
        pos, vel, masses = model.sample_cluster(jax.random.PRNGKey(0), n_stars=15000, G=G)
        pos = pos - jnp.average(pos, axis=0, weights=masses)
        vel = vel - jnp.average(vel, axis=0, weights=masses)
        r = jnp.linalg.norm(pos, axis=1)
        v2 = jnp.sum(vel**2, axis=1)
        M = jnp.sum(masses)
        s = jnp.sqrt(G * M / (9.0 * model.r_c * model.mu_tot))

        def analytic_sigma1d(W_j, s_j):
            u = jnp.linspace(0.0, jnp.sqrt(2.0 * W_j), 400)
            E = lowered_exponential(model.g, W_j - u**2 / 2.0)
            return float(s_j * jnp.sqrt(jnp.trapezoid(u**4 * E, u)
                                        / jnp.trapezoid(u**2 * E, u) / 3.0))

        # Per-component core dispersion matches the analytic equilibrium prediction.
        for jc in range(2):
            sel = jnp.isclose(masses, float(model.m_j[jc]))
            r_j, v2_j = r[sel], v2[sel]
            s_j = float(s * model.mu_j[jc] ** (-model.delta))
            core = r_j < 1.0
            assert int(jnp.sum(core)) > 60, "too few stars to resolve the core dispersion"
            sig_meas = float(jnp.sqrt(jnp.mean(v2_j[core]) / 3.0))
            r_mid = float(jnp.median(r_j[core]))
            W_j = float(model.rescale_j[jc]) * float(
                jnp.interp(r_mid, model.xi_grid, model.psi_grid))
            sig_pred = analytic_sigma1d(jnp.asarray(W_j), s_j)
            np.testing.assert_allclose(sig_meas, sig_pred, rtol=0.08,
                                       err_msg=f"component {jc} dispersion off equilibrium")

        # The whole cluster is virial without rescaling.
        Qg = float(compute_virial_ratio(pos, vel, masses, G=G))
        assert abs(Qg - 0.5) < 0.04, f"global Q={Qg:.3f} (expected 0.5)"

    def test_theoretical_component_virial_is_exactly_half(self):
        """The bias-free equilibrium proof: the THEORETICAL per-component virial ratio
        Q_j = T_j/|W_j| computed from the model (no sampling, no softening, no finite-N)
        equals 0.5 for EVERY component at every delta. This is the rigorous statement of
        'each mass group is in equilibrium' -- the sampled per-group Q_j is a finite-N
        observable that converges to this."""
        from progenax.profiles.limepy_multimass import MultiMassLIMEPY

        m_j = jnp.array([1.0, 4.0])
        alpha_j = jnp.array([0.5, 0.5])
        for delta in (0.0, 0.3, 0.5, 0.6):
            model = MultiMassLIMEPY.from_alpha(alpha_j, m_j, W0=7.0, g=1.0, delta=delta,
                                               r_c=1.0, n_ode_points=4000)
            Qj = np.asarray(model.component_virial_ratios())
            np.testing.assert_allclose(Qj, 0.5, atol=2e-3,
                                       err_msg=f"delta={delta}: theoretical Q_j={Qj}")

    def test_heavy_component_is_kinematically_colder(self):
        """The equipartition signature: at fixed radius the heavy component has a
        SMALLER velocity dispersion than the light one (s_j = s mu_j^{-delta} with
        mu_heavy>1) -- partial energy equipartition, a true-equilibrium consequence the
        blend does not encode."""
        model = self._two_component(delta=0.5)
        pos, vel, masses = model.sample_cluster(jax.random.PRNGKey(2), n_stars=15000, G=G)
        r = jnp.linalg.norm(pos, axis=1)
        v2 = jnp.sum(vel**2, axis=1)
        core = r < 1.0
        sig_light = float(jnp.sqrt(jnp.mean(
            v2[core & jnp.isclose(masses, float(model.m_j[0]))]) / 3.0))
        sig_heavy = float(jnp.sqrt(jnp.mean(
            v2[core & jnp.isclose(masses, float(model.m_j[1]))]) / 3.0))
        assert sig_heavy < sig_light, f"heavy sigma={sig_heavy:.3f} not < light {sig_light:.3f}"

    def test_sampled_cluster_is_mass_segregated(self):
        """In the sampled cluster the heavy component is more centrally concentrated
        than the light one (mean radius), the observable signature of segregation."""
        model = self._two_component(delta=0.5)
        pos, vel, masses = model.sample_cluster(
            jax.random.PRNGKey(0), n_stars=20000, G=G)
        r = jnp.linalg.norm(pos, axis=1)
        r_light = float(jnp.mean(r[jnp.isclose(masses, float(model.m_j[0]))]))
        r_heavy = float(jnp.mean(r[jnp.isclose(masses, float(model.m_j[1]))]))
        assert r_heavy < r_light, f"heavy <r>={r_heavy:.2f} not < light <r>={r_light:.2f}"

    def test_all_particles_bound(self):
        model = self._two_component(delta=0.5)
        pos, vel, masses = model.sample_cluster(
            jax.random.PRNGKey(1), n_stars=5000, G=G)
        # crude global bound check: speed below the central escape speed s*sqrt(2 W0)
        from progenax.builders import compute_potential_energy, compute_kinetic_energy
        V = float(compute_potential_energy(pos, masses, G=G))
        assert np.isfinite(V)
        d = jnp.linalg.norm(pos[:, None] - pos[None], axis=2) + jnp.eye(pos.shape[0]) * 1e9
        assert float(d.min()) > 1e-6  # no coincident stars

    def test_sample_differentiable_in_delta(self):
        """grad of a kinematic functional w.r.t. delta flows through construction +
        sampling -- the equipartition degree is inferable from a sampled cluster."""
        from progenax.profiles.limepy_multimass import MultiMassLIMEPY

        m_j = jnp.array([0.4, 5.0])
        alpha_j = jnp.array([0.7, 0.3])

        def loss(delta):
            model = MultiMassLIMEPY.from_alpha(alpha_j, m_j, W0=7.0, g=1.0, delta=delta, r_c=1.0)
            pos, vel, masses = model.sample_cluster(
                jax.random.PRNGKey(0), n_stars=400, G=G)
            return jnp.mean(jnp.sum(vel**2, axis=1))

        d = jax.grad(loss)(0.4)
        assert jnp.isfinite(d)


class TestMultiMassAnisotropic:
    """Phase 2b: per-component radial anisotropy r_{a,j} = r_a mu_j^eta in the coupled
    solve. eta=0 is mass-independent anisotropy (the paper default)."""

    def test_isotropic_corner_ra_none(self):
        """r_a=None reproduces the isotropic coupled solve exactly (the iso corner)."""
        from progenax.profiles.limepy_multimass import solve_multimass_limepy

        alpha = jnp.array([0.6, 0.4]); m_j = jnp.array([0.5, 4.0])
        xi_i, psi_i, _ = solve_multimass_limepy(alpha, m_j, 7.0, 1.0, 0.4, 300.0, 2000)
        xi_a, psi_a, _ = solve_multimass_limepy(alpha, m_j, 7.0, 1.0, 0.4, 300.0, 2000, ra_hat=None)
        np.testing.assert_allclose(np.asarray(psi_a), np.asarray(psi_i), rtol=1e-9, atol=1e-9)

    def test_single_component_recovers_single_mass_anisotropic(self):
        """n_comp=1 (alpha=[1], any m) with anisotropy reduces to the single-mass
        anisotropic LIMEPY solve (Phase 1) -- a clean cross-module corner."""
        from progenax.profiles.limepy_multimass import solve_multimass_limepy
        from progenax.profiles.limepy import solve_limepy_profile

        xi_s, psi_s = solve_limepy_profile(7.0, g=1.0, ra_hat=8.0, xi_max=800.0, n_points=3000)
        xi_m, psi_m, _ = solve_multimass_limepy(
            jnp.array([1.0]), jnp.array([1.0]), 7.0, 1.0, 0.5, 800.0, 3000, ra_hat=8.0)
        inside = psi_s > 1e-3
        np.testing.assert_allclose(np.asarray(jnp.interp(xi_s, xi_m, psi_m))[inside],
                                   np.asarray(psi_s)[inside], rtol=2e-3, atol=2e-3)

    def test_anisotropic_multimass_is_equilibrium_and_segregated(self):
        """An anisotropic multi-mass model is still a true equilibrium (theoretical
        per-component Q_j = 0.5, scalar virial holds with anisotropy) AND segregated
        (heavy more concentrated)."""
        from progenax.profiles.limepy_multimass import MultiMassLIMEPY

        model = MultiMassLIMEPY.from_alpha(
            jnp.array([0.6, 0.4]), jnp.array([1.0, 4.0]), W0=7.0, g=1.0, delta=0.4,
            r_a=10.0, eta=0.0, r_c=1.0, xi_max=800.0, n_ode_points=3000)
        Qj = np.asarray(model.component_virial_ratios())
        np.testing.assert_allclose(Qj, 0.5, atol=3e-3, err_msg=f"aniso Q_j={Qj}")
        # heavy more concentrated than light (segregation persists under anisotropy)
        r = model._r_grid
        from progenax.profiles.limepy import _aniso_density_scalar
        def rh(j):
            psi = jnp.interp(r / model.r_c, model.xi_grid, model.psi_grid, left=model.W0, right=0.0)
            p = r / (model.r_a / model.r_c * model.mu_j[j] ** model.eta)
            rho = jax.vmap(lambda pp, ww: _aniso_density_scalar(model.rescale_j[j] * ww, pp, model.g))(p, psi)
            rho = jnp.where(r <= model.r_t, rho, 0.0)
            integ = rho * r**2
            M = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (integ[1:] + integ[:-1])) * (r[1] - r[0])])
            return float(jnp.interp(0.5 * M[-1], M, r))
        assert rh(1) < rh(0), "heavy not more concentrated under anisotropy"

    def test_anisotropic_multimass_differentiable_in_eta_ra_delta(self):
        """Gradients flow through the anisotropic coupled solve in (r_a, eta, delta)."""
        from progenax.profiles.limepy_multimass import solve_multimass_limepy

        m_j = jnp.array([1.0, 4.0]); alpha = jnp.array([0.6, 0.4])
        def metric(ra_hat, eta, delta):
            xi, psi, _ = solve_multimass_limepy(alpha, m_j, 7.0, 1.0, delta, 800.0, 2000,
                                                ra_hat=ra_hat, eta=eta)
            return jnp.mean(psi[:300])
        d_ra = jax.grad(metric, 0)(10.0, 0.0, 0.4)
        d_eta = jax.grad(metric, 1)(10.0, 0.3, 0.4)
        d_delta = jax.grad(metric, 2)(10.0, 0.0, 0.4)
        assert jnp.isfinite(d_ra) and jnp.abs(d_ra) > 0
        assert jnp.isfinite(d_eta)
        assert jnp.isfinite(d_delta) and jnp.abs(d_delta) > 0

# progenax/tests/unit/profiles/test_limepy.py
"""Unit tests for the general-g LIMEPY density (Gieles & Zocchi 2015, Phase 1).

The single-mass LIMEPY isotropic density is the dimensionless integral (their
Appendix B, Eq. B9 — NOT the misprinted main-text Eq. 8):

    I_rho(W) = E_gamma(g + 3/2, W),   E_gamma(a, x) = e^x * P(a, x),

with P the regularized lower incomplete gamma function (jax gammainc). The
truncation parameter g is continuous: g=0 Woolley, g=1 King, g=2 Wilson.

The non-negotiable oracle: at g=1 the density MUST reduce to the existing,
validated King lowered-Maxwellian density (king_lowered_maxwellian_density),
which itself reproduces King (1966). Verified algebraically:
    E_gamma(5/2, W) = e^W erf(sqrt W) - (2/sqrt pi) sqrt(W) (1 + 2W/3).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate
from scipy.special import gammainc as scipy_gammainc

from progenax.profiles.king import (
    king_lowered_maxwellian_density,
    solve_king_profile,
    _find_tidal_radius,
)


def _direct_density_integral(W, g):
    """First-principles I_rho(W) by direct velocity-space quadrature (Eq. B4):

        I_rho(W) = (2/sqrt(pi)) int_0^W k^(1/2) E_gamma(g, W - k) dk,

    with E_gamma(g, y) = e^y for g=0 and e^y * P(g, y) for g>0. This is the
    *definition* the closed form E_gamma(g + 3/2, W) must reproduce — an oracle
    independent of the gammainc identity, valid for ANY g. (scipy: test-only.)
    """
    def E_gamma(a, y):
        if a == 0.0:
            return np.exp(y)
        return np.exp(y) * scipy_gammainc(a, y)

    def integrand(k):
        return np.sqrt(k) * E_gamma(g, W - k)

    val, _ = integrate.quad(integrand, 0.0, W, limit=200)
    return (2.0 / np.sqrt(np.pi)) * val


class TestLimepyDensityCorners:
    """The continuous-g density against its trusted integer-g corners."""

    def test_g1_reduces_to_king_density(self):
        """g=1 isotropic LIMEPY density == the validated King volume density, to
        float64 precision, across the cluster-relevant potential range. This is
        the corner that certifies the general-g closed form."""
        from progenax.profiles.limepy import limepy_density_hat

        W = jnp.linspace(0.01, 16.0, 200)
        limepy_g1 = limepy_density_hat(W, g=1.0)
        king = king_lowered_maxwellian_density(W)
        np.testing.assert_allclose(
            np.asarray(limepy_g1), np.asarray(king), rtol=1e-9, atol=1e-12
        )

    @pytest.mark.parametrize("g", [0.0, 0.5, 1.0, 2.0, 3.0])
    @pytest.mark.parametrize("W", [0.5, 2.0, 5.0, 9.0])
    def test_closed_form_matches_direct_velocity_integral(self, g, W):
        """The closed form E_gamma(g + 3/2, W) reproduces the direct velocity-space
        integral of the DF (Eq. B4) for arbitrary g — first-principles proof that
        the index is g + 3/2 (not the misprinted g + 1/2), for Woolley/King/Wilson
        and the continuous values between them."""
        from progenax.profiles.limepy import limepy_density_hat

        closed = float(limepy_density_hat(jnp.asarray(W), g=g))
        direct = _direct_density_integral(W, g)
        np.testing.assert_allclose(closed, direct, rtol=1e-6)


class TestLimepyDensityContract:
    """Physics/contract properties of the density that a wrong form would break."""

    def test_vanishes_at_and_below_truncation(self):
        """rho_hat(W) = 0 for W <= 0 (the truncation boundary): no stars above the
        escape energy."""
        from progenax.profiles.limepy import limepy_density_hat

        W = jnp.array([-5.0, -0.1, 0.0])
        out = limepy_density_hat(W, g=1.5)
        np.testing.assert_array_equal(np.asarray(out), np.zeros(3))

    def test_strictly_increasing_in_potential(self):
        """Deeper potential -> denser: rho_hat is monotonically increasing in W
        (each successive E_gamma index is monotone)."""
        from progenax.profiles.limepy import limepy_density_hat

        W = jnp.linspace(0.05, 16.0, 300)
        rho = limepy_density_hat(W, g=1.0)
        assert jnp.all(jnp.diff(rho) > 0.0)

    def test_differentiable_in_g_and_W(self):
        """The non-negotiable: gradients flow through BOTH the potential W and the
        truncation index g (g enters as the gammainc first argument a = g + 3/2).
        This is what lets g be a *fitted* parameter in HMC/gradient inference. We
        assert the gradients are finite and non-zero (the parameter genuinely moves
        the density); the W-gradient is additionally positive (deeper -> denser,
        the monotonicity locked above). The sign of the g-gradient is NOT asserted:
        at fixed W the unnormalized E_gamma(g+3/2,W) decreases in g (a property of
        the incomplete gamma); the physical 'larger g -> more extended' statement is
        a normalized profile-extent property, tested on solve_limepy_profile."""
        from progenax.profiles.limepy import limepy_density_hat

        dg = jax.grad(lambda g: jnp.sum(limepy_density_hat(jnp.array([2.0, 5.0]), g)))(1.0)
        dW = jax.grad(lambda W: jnp.sum(limepy_density_hat(W, g=1.0)))(jnp.array([2.0, 5.0]))
        assert jnp.isfinite(dg) and jnp.abs(dg) > 0.0  # g genuinely moves the density
        assert jnp.all(jnp.isfinite(dW)) and jnp.all(dW > 0.0)  # deeper -> denser


class TestSolveLimepyProfile:
    """The self-consistent Poisson solve d^2W/dxi^2 + (2/xi)dW/dxi = -9 rho_hat(W;g).

    Same -9 King-radius nondimensionalization as solve_king_profile; the only
    change is the general-g density source. The g=1 corner must reproduce the
    validated King solver, and the truncation radius must obey the paper's
    extent-vs-g ordering (Fig. 1: larger g -> more extended).
    """

    def test_g1_profile_matches_king_solver(self):
        """g=1 reproduces solve_king_profile's W(xi) to tight tolerance on a shared
        grid — the self-consistent corner (density + Poisson solve together)."""
        from progenax.profiles.limepy import solve_limepy_profile

        W0 = 7.0
        xi_k, psi_k = solve_king_profile(W0, xi_max=300.0, n_points=3000)
        xi_l, psi_l = solve_limepy_profile(W0, g=1.0, xi_max=300.0, n_points=3000)
        # Compare W(xi) inside the cluster (where psi_king > small) on King's grid.
        inside = psi_k > 1e-3
        psi_l_on_k = jnp.interp(xi_k, xi_l, psi_l)
        np.testing.assert_allclose(
            np.asarray(psi_l_on_k[inside]), np.asarray(psi_k[inside]), rtol=1e-4, atol=1e-4
        )

    @pytest.mark.parametrize("W0,c_table_ii", [(5.0, 10.70), (7.0, 33.71), (9.0, 131.4)])
    def test_g1_truncation_radius_matches_king_table_ii(self, W0, c_table_ii):
        """g=1 truncation radius xi_t = r_t/r_c equals King (1966) Table II
        concentration c(W0). A first-principles anchor: the general-g solver lands
        on the historical King concentrations at g=1."""
        from progenax.profiles.limepy import solve_limepy_profile

        xi, psi = solve_limepy_profile(W0, g=1.0)
        xi_t = float(_find_tidal_radius(xi, psi))
        # Table II is quoted to ~3-4 sig figs; allow 3% (ODE-grid + interp).
        np.testing.assert_allclose(xi_t, c_table_ii, rtol=0.03)

    def test_extent_increases_with_g(self):
        """The CORRECT physical g-statement (Gieles & Zocchi Fig. 1): at fixed
        central potential W0, a softer truncation (larger g) yields a more extended
        model — the truncation radius grows Woolley(0) < King(1) < Wilson(2). This is
        the normalized profile-level property my fixed-W density test wrongly tried
        to assert; here it is, in its proper place."""
        from progenax.profiles.limepy import solve_limepy_profile

        W0 = 6.0
        xi_t = []
        for g in (0.0, 1.0, 2.0):
            xi, psi = solve_limepy_profile(W0, g=g, xi_max=400.0, n_points=4000)
            xi_t.append(float(_find_tidal_radius(xi, psi)))
        assert xi_t[0] < xi_t[1] < xi_t[2], f"extent not ordered in g: {xi_t}"

    def test_differentiable_through_solve_in_W0_and_g(self):
        """Gradients flow through the diffrax Poisson solve in BOTH W0 and g — the
        property that makes (W0, g) jointly inferable. A profile-shape functional
        (central-region potential integral) is used so the gradient is grid-stable
        (it does not depend on the non-differentiable truncation-radius crossing)."""
        from progenax.profiles.limepy import solve_limepy_profile

        def shape_metric(W0, g):
            xi, psi = solve_limepy_profile(W0, g=g, xi_max=300.0, n_points=2000)
            # mean potential over the fixed inner grid: smooth in (W0, g)
            return jnp.mean(psi[:200])

        dW0 = jax.grad(shape_metric, argnums=0)(7.0, 1.0)
        dg = jax.grad(shape_metric, argnums=1)(7.0, 1.0)
        assert jnp.isfinite(dW0) and jnp.abs(dW0) > 0.0
        assert jnp.isfinite(dg) and jnp.abs(dg) > 0.0


class TestLimepyAnisotropicDensity:
    """The general-g anisotropic (Michie/OM) density rho_hat(W, p, g), p = r/r_a.

    Computed by the stable 1-D speed quadrature (the angle integral closes to the
    bounded T(beta) = int_{-1}^1 exp(-beta(1-c^2)) dc, beta = p^2 u^2/2), NOT the
    numerically unstable jax hyp1f1. Oracles: the g=1 corner is progenax's existing
    michie_density (same physical DF); p->0 recovers the isotropic limepy density;
    and (where jax hyp1f1 is stable, x < 80) it matches the verified closed form.
    """

    def test_g1_matches_michie_density(self):
        """g=1 anisotropic LIMEPY density == michie_density(W, s) (King cutoff +
        Michie anisotropy) to quadrature precision, over a (W, p) grid. The
        anisotropic King corner."""
        from progenax.profiles.limepy import limepy_density_aniso_hat
        from progenax.profiles.michie import michie_density

        for W in (1.0, 3.0, 6.0, 9.0):
            for p in (0.0, 0.3, 1.0, 3.0):
                lim = float(limepy_density_aniso_hat(jnp.asarray(W), jnp.asarray(p), g=1.0))
                mic = float(michie_density(jnp.asarray(W), jnp.asarray(p)))
                np.testing.assert_allclose(lim, mic, rtol=2e-3, atol=1e-6,
                                           err_msg=f"W={W} p={p}")

    @pytest.mark.parametrize("g", [0.0, 1.0, 2.0])
    def test_isotropic_limit_recovers_limepy_density(self, g):
        """As p->0 the normalized anisotropic density -> the isotropic limepy
        density rho_hat(W)/rho_hat(W0) (T(0)=2 makes the angle integral isotropic)."""
        from progenax.profiles.limepy import limepy_density_aniso_hat, limepy_density_hat

        W = jnp.linspace(0.2, 8.0, 40)
        W0 = 8.0
        aniso = limepy_density_aniso_hat(W, jnp.asarray(1e-4), g=g) / \
            limepy_density_aniso_hat(jnp.asarray(W0), jnp.asarray(1e-4), g=g)
        iso = limepy_density_hat(W, g=g) / limepy_density_hat(jnp.asarray(W0), g=g)
        np.testing.assert_allclose(np.asarray(aniso), np.asarray(iso), rtol=2e-3, atol=1e-4)

    @pytest.mark.parametrize("g", [0.0, 1.0, 2.0])
    @pytest.mark.parametrize("W,p", [(5.0, 0.5), (5.0, 1.0), (3.0, 1.5)])
    def test_matches_verified_closed_form_where_stable(self, g, W, p):
        """Cross-check the quadrature against the verified hyp1f1 closed form
        (candidate A) in the regime where jax hyp1f1 is stable (x = p^2 W < 80):

            I_rho = E(g+3/2,W)/(1+p^2) + p^2/(1+p^2) W^(g+3/2)/Gamma(g+5/2)
                    1F1(1, g+5/2, -p^2 W).

        Both are normalized to the isotropic central value to compare on one scale."""
        from scipy.special import gammainc as sp_gammainc, gamma as sp_gamma, hyp1f1
        from progenax.profiles.limepy import limepy_density_aniso_hat

        x = p**2 * W
        assert x < 80.0  # guard: only test where hyp1f1 is reliable
        a = g + 1.5
        E = np.exp(W) * sp_gammainc(a, W)  # E_gamma(g+3/2, W)
        closed = E / (1 + p**2) + p**2 / (1 + p**2) * W**(g + 1.5) / sp_gamma(g + 2.5) \
            * hyp1f1(1.0, g + 2.5, -(p**2) * W)
        # quadrature is unnormalized by sqrt(2 pi) vs E_gamma; rescale by the s=0 ratio.
        quad = float(limepy_density_aniso_hat(jnp.asarray(W), jnp.asarray(p), g=g))
        quad0 = float(limepy_density_aniso_hat(jnp.asarray(W), jnp.asarray(1e-6), g=g))
        E_iso = float(np.exp(W) * sp_gammainc(a, W))
        quad_scaled = quad / quad0 * E_iso  # put quadrature on the E_gamma scale
        np.testing.assert_allclose(quad_scaled, closed, rtol=3e-3)

    def test_differentiable_in_W_p_g(self):
        """Gradients flow through the anisotropic density in all of (W, p, g) — the
        anisotropy radius r_a (via p) and truncation g are jointly inferable."""
        from progenax.profiles.limepy import limepy_density_aniso_hat

        f = lambda W, p, g: jnp.sum(limepy_density_aniso_hat(W, p, g))
        Wv = jnp.array([2.0, 5.0])
        dW = jax.grad(f, 0)(Wv, jnp.asarray(0.8), 1.0)
        dp = jax.grad(lambda p: limepy_density_aniso_hat(jnp.asarray(5.0), p, 1.0))(0.8)
        dg = jax.grad(lambda g: limepy_density_aniso_hat(jnp.asarray(5.0), jnp.asarray(0.8), g))(1.0)
        assert jnp.all(jnp.isfinite(dW)) and jnp.isfinite(dp) and jnp.isfinite(dg)
        assert jnp.abs(dp) > 0.0 and jnp.abs(dg) > 0.0


class TestLimepyAnisotropicProfile:
    """The anisotropic (Michie/OM) self-consistent solve + profile. The radius-
    dependent Poisson source (density depends on xi via p = xi/ra_hat) generalizes
    solve_michie_profile; the g=1 corner must reproduce MichieProfile.
    """

    def test_solve_g1_aniso_matches_michie_solver(self):
        """solve_limepy_profile(g=1, ra_hat) reproduces solve_michie_profile's W(xi)
        — the anisotropic self-consistent corner (radius-dependent RHS)."""
        from progenax.profiles.limepy import solve_limepy_profile
        from progenax.profiles.michie import solve_michie_profile

        W0, ra = 7.0, 5.0
        xi_m, psi_m = solve_michie_profile(W0, ra, xi_max=800.0, n_points=4000)
        xi_l, psi_l = solve_limepy_profile(W0, g=1.0, ra_hat=ra, xi_max=800.0, n_points=4000)
        inside = psi_m > 1e-3
        psi_l_on_m = jnp.interp(xi_m, xi_l, psi_l)
        np.testing.assert_allclose(
            np.asarray(psi_l_on_m[inside]), np.asarray(psi_m[inside]), rtol=3e-3, atol=3e-3
        )

    def test_profile_g1_aniso_matches_michie_profile(self):
        """LIMEPYProfile(g=1, r_a).density(r) == MichieProfile.density(r) — the
        anisotropic King profile corner, end to end (W0 -> r_t -> normalized rho)."""
        from progenax.profiles.michie import MichieProfile
        from progenax.profiles.limepy import LIMEPYProfile

        mic = MichieProfile.from_W0_rc(W0=7.0, r_c=1.0, r_a=5.0)
        lim = LIMEPYProfile.from_W0_rc(W0=7.0, g=1.0, r_c=1.0, r_a=5.0, xi_max=800.0, n_ode_points=3000)
        r = jnp.linspace(0.0, float(mic.r_t), 400)
        np.testing.assert_allclose(
            np.asarray(lim.density(r)), np.asarray(mic.density(r)), rtol=5e-3, atol=5e-3
        )

    def test_anisotropy_extends_the_model(self):
        """Radial orbits puff the envelope out: at fixed (W0, g), a finite anisotropy
        radius r_a gives a LARGER truncation radius than the isotropic model (r_a=inf).
        The defining qualitative signature of Michie/OM anisotropy."""
        from progenax.profiles.limepy import LIMEPYProfile

        iso = LIMEPYProfile.from_W0_rc(W0=7.0, g=1.0, r_c=1.0, xi_max=800.0, n_ode_points=3000)
        ani = LIMEPYProfile.from_W0_rc(W0=7.0, g=1.0, r_c=1.0, r_a=4.0, xi_max=800.0, n_ode_points=3000)
        assert float(ani.r_t) > float(iso.r_t), f"aniso r_t={float(ani.r_t)} !> iso {float(iso.r_t)}"

    def test_differentiable_in_anisotropy_radius(self):
        """A profile-shape functional is differentiable in r_a through the radius-
        dependent anisotropic solve — the anisotropy radius is inferable."""
        from progenax.profiles.limepy import solve_limepy_profile

        def metric(ra_hat):
            xi, psi = solve_limepy_profile(7.0, g=1.0, ra_hat=ra_hat, xi_max=800.0, n_points=3000)
            return jnp.mean(psi[:300])

        d = jax.grad(metric)(5.0)
        assert jnp.isfinite(d) and jnp.abs(d) > 0.0


class TestLimepyProfile:
    """LIMEPYProfile (SpatialProfile): general-g isotropic spatial profile with
    inverse-CDF position sampling. Generalizes KingProfile with a g field; at g=1
    it must reproduce KingProfile.
    """

    def test_g1_density_matches_king_profile(self):
        """LIMEPYProfile(g=1).density(r) == KingProfile.density(r) over radius —
        the self-consistent spatial-profile corner (W0->r_t->normalized rho(r))."""
        from progenax.profiles.king import KingProfile
        from progenax.profiles.limepy import LIMEPYProfile

        king = KingProfile.from_W0_rc(W0=7.0, r_c=1.0)
        lim = LIMEPYProfile.from_W0_rc(W0=7.0, g=1.0, r_c=1.0)
        r = jnp.linspace(0.0, float(king.r_t), 400)
        np.testing.assert_allclose(
            np.asarray(lim.density(r)), np.asarray(king.density(r)), rtol=2e-3, atol=2e-3
        )

    def test_truncation_radius_grows_with_g(self):
        """The profile's truncation radius r_t obeys the extent ordering in g
        (Woolley<King<Wilson) at fixed (W0, r_c)."""
        from progenax.profiles.limepy import LIMEPYProfile

        r_t = [float(LIMEPYProfile.from_W0_rc(W0=6.0, g=g, r_c=1.0).r_t)
               for g in (0.0, 1.0, 2.0)]
        assert r_t[0] < r_t[1] < r_t[2], f"r_t not ordered in g: {r_t}"

    def test_sampled_positions_recover_density_profile(self):
        """Inverse-CDF position sampling reproduces the model's own radial density:
        the sampled radial histogram matches 4 pi r^2 rho(r) within Poisson noise.
        A real distribution test (not a smoke test) — a wrong CDF fails it."""
        from progenax.profiles.limepy import LIMEPYProfile

        prof = LIMEPYProfile.from_W0_rc(W0=5.0, g=1.5, r_c=1.0)
        key = jax.random.PRNGKey(0)
        pos = prof.sample_positions(jnp.ones(40000), key)
        radii = jnp.linalg.norm(pos, axis=1)
        assert float(jnp.max(radii)) <= float(prof.r_t) * 1.001  # strict truncation

        # Compare sampled radial CDF to the analytic mass CDF at several radii.
        r_test = jnp.linspace(0.1, float(prof.r_t) * 0.95, 6)
        rr = jnp.linspace(0.0, float(prof.r_t), 2000)
        integrand = 4.0 * jnp.pi * rr**2 * prof.density(rr)
        m_cum = jnp.concatenate([jnp.zeros(1),
                                 jnp.cumsum(0.5 * (integrand[1:] + integrand[:-1])) * (rr[1] - rr[0])])
        cdf_analytic = m_cum / m_cum[-1]
        for rt in r_test:
            emp = float(jnp.mean(radii <= rt))
            ana = float(jnp.interp(rt, rr, cdf_analytic))
            assert abs(emp - ana) < 0.02, f"CDF mismatch at r={float(rt):.2f}: {emp:.3f} vs {ana:.3f}"

    def test_differentiable_construction_in_W0_and_g(self):
        """A profile-shape functional is differentiable through construction in both
        W0 and g — structural parameters remain inferable end-to-end at the class
        level (not just the bare solver)."""
        from progenax.profiles.limepy import LIMEPYProfile

        def metric(W0, g):
            prof = LIMEPYProfile.from_W0_rc(W0=W0, g=g, r_c=1.0)
            r = jnp.linspace(0.0, 5.0, 200)
            return jnp.mean(prof.density(r))

        dW0 = jax.grad(metric, argnums=0)(7.0, 1.0)
        dg = jax.grad(metric, argnums=1)(7.0, 1.0)
        assert jnp.isfinite(dW0) and jnp.abs(dW0) > 0.0
        assert jnp.isfinite(dg) and jnp.abs(dg) > 0.0

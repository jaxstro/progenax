# tests/unit/profiles/test_density_poisson.py
"""Prescribed-density shared potential + derived domain (Engine B, design c)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progenax import PlummerProfile, EFFProfile, KingProfile


class TestComponentExtent:
    def test_plummer_is_infinite(self):
        from progenax.profiles.density_poisson import component_extent
        assert component_extent(PlummerProfile(r_h=1.0)) is None

    def test_eff_and_king_finite(self):
        from progenax.profiles.density_poisson import component_extent
        assert float(component_extent(EFFProfile(a=1.0, gamma=4.0, r_t=8.0))) == 8.0
        k = KingProfile.from_W0_rc(W0=5.0, r_c=1.0)
        assert float(component_extent(k)) == pytest.approx(float(k.r_t))


class TestDeriveRt:
    def test_max_of_finite_extents(self):
        from progenax.profiles.density_poisson import derive_r_t
        rt, prov = derive_r_t([PlummerProfile(2.0), EFFProfile(a=0.3, gamma=5.0, r_t=8.0)],
                              jnp.array([0.7, 0.3]))
        assert float(rt) == 8.0 and "EFF" in prov

    def test_all_infinite_uses_f_enc_mass_radius(self):
        """Pure-Plummer mix: r_t = radius enclosing f_enc of the SUMMED mass.
        Single Plummer analytic check: M(<r)/M = x^3/(1+x^2)^{3/2} = f_enc."""
        from progenax.profiles.density_poisson import derive_r_t
        p = PlummerProfile(r_h=1.0)
        rt, prov = derive_r_t([p], jnp.array([1.0]), f_enc=0.995)
        c = 0.995 ** (2.0 / 3.0)
        x_exact = float(jnp.sqrt(c / (1.0 - c)))          # ~17.27
        assert float(rt) == pytest.approx(float(p.a) * x_exact, rel=2e-2), prov

    def test_explicit_override_wins(self):
        from progenax.profiles.density_poisson import derive_r_t
        rt, prov = derive_r_t([PlummerProfile(1.0)], jnp.array([1.0]), r_t=30.0)
        assert float(rt) == 30.0 and "override" in prov

    def test_king_conflict_with_override_raises(self):
        from progenax.profiles.density_poisson import derive_r_t
        k = KingProfile.from_W0_rc(W0=5.0, r_c=1.0)
        with pytest.raises(ValueError, match="King"):
            derive_r_t([k], jnp.array([1.0]), r_t=0.5 * float(k.r_t))


class TestSharedPotential:
    def test_single_plummer_matches_analytic(self):
        """Psi from the quadrature pass == GM/sqrt(r^2+a^2) - GM/sqrt(rt^2+a^2)
        (G=1 internal units) to rtol 1e-4 over the interior.

        Amplitude physics: truncating a Plummer at r_t removes the outer shells,
        which shift Phi(r<r_t) by a CONSTANT that cancels in Psi = Phi(r_t)-Phi,
        so the exact Psi keeps the UNTRUNCATED amplitude M_inf = M(<r_t)/
        trunc_frac (= 1.000551 here, a 5.5e-4 offset from M(<r_t) at r_t = 40,
        x_t = 52.2). Using M_inf makes the oracle exact, so rtol 1e-4 genuinely
        probes the quadrature (measured accuracy ~2e-5)."""
        from progenax.profiles.density_poisson import shared_potential
        p = PlummerProfile(r_h=1.0)
        pot = shared_potential([p], jnp.array([1.0]), r_t=jnp.asarray(40.0))
        a = float(p.a)
        # exact untruncated amplitude (see docstring), NOT the truncated mass
        M = float(pot.M_cum_j[0, -1] / pot.trunc_frac_j[0])
        Psi_exact = M / jnp.sqrt(pot.r_grid**2 + a**2) - M / jnp.sqrt(40.0**2 + a**2)
        sel = np.asarray(pot.r_grid) > 0.05
        np.testing.assert_allclose(np.asarray(pot.Psi_grid)[sel],
                                   np.asarray(Psi_exact)[sel], rtol=1e-4)

    def test_two_components_mass_fractions_respected(self):
        """M_j(r_t) proportions == mass_fractions (each component's CDF is
        normalized; the FRACTIONS set the amplitudes)."""
        from progenax.profiles.density_poisson import shared_potential
        pot = shared_potential([PlummerProfile(2.0), PlummerProfile(0.5)],
                               jnp.array([0.7, 0.3]), r_t=jnp.asarray(60.0))
        Mj = np.asarray(pot.M_cum_j[:, -1])
        np.testing.assert_allclose(Mj / Mj.sum(), [0.7, 0.3], atol=5e-3)

    def test_truncated_mass_fraction_diagnostic(self):
        """Plummer truncated at 5a stores M(<rt)/M(inf) = x^3/(1+x^2)^{3/2}."""
        from progenax.profiles.density_poisson import shared_potential
        p = PlummerProfile(r_h=1.0)
        rt = 5.0 * float(p.a)
        pot = shared_potential([p], jnp.array([1.0]), r_t=jnp.asarray(rt))
        x = 5.0
        expect = x**3 / (1 + x**2) ** 1.5
        assert float(pot.trunc_frac_j[0]) == pytest.approx(expect, rel=1e-3)

    def test_mass_fractions_must_sum_to_one(self):
        from progenax.profiles.density_poisson import shared_potential
        with pytest.raises(ValueError, match="mass_fractions"):
            shared_potential([PlummerProfile(1.0), PlummerProfile(2.0)],
                             jnp.array([0.6, 0.6]), r_t=jnp.asarray(20.0))


class TestKingSharedPotential:
    """King dW/dr in the Engine B density path (2c-iii prerequisite fix).

    jnp.gradient of the LINEARLY interpolated psi grid is a staircase; the
    Eddington d^2 rho/dPsi^2 then rings at high Psi (near the center, where
    successive Delta Psi -> 0) and the Abel 1/sqrt(E - Psi) weight focuses the
    ringing into f(E -> Psi0): a single King failed the realizability gate at
    min f/max|f| = -0.68 even though the true King ergodic DF is strictly
    positive. The fix integrates King's own Poisson identity
        dpsi/dxi = -(9/rho_hat_0) xi^-2 int_0^xi rho_hat(psi(s)) s^2 ds
    by cumulative trapezoid of the CLOSED-FORM density (never differentiate
    interpolated data).
    """

    def _single_king(self):
        from progenax.cluster.multicomponent import MultiComponentCluster
        return MultiComponentCluster.from_density_profiles(
            [KingProfile.from_W0_rc(W0=5.0, r_c=1.0)],
            jnp.array([1.0]), m_j=jnp.array([1.0]))

    def test_single_king_passes_realizability_gate(self):
        """A single King component IS an equilibrium: the build must not raise
        and the realizability margin must be at worst grid-level noise."""
        m = self._single_king()
        assert float(m.engine_b.f_min_j[0]) > -1e-5

    def test_king_f_shape_matches_lowered_maxwellian(self):
        """The Engine B King f(E) must BE the lowered Maxwellian e^Ehat - 1.

        Engine A's King DF is f propto e^{E/sigma^2} - 1 with W = psi/sigma^2;
        the tables are G = 1, so sigma^2 = Psi0/W0 and Ehat = E W0/Psi0.
        Pearson correlation > 0.999 pins the SHAPE (amplitude is the
        mass-fraction normalization)."""
        m = self._single_king()
        st = m.engine_b
        E = np.asarray(st.E_grid)
        f = np.asarray(st.f_j_grid[0])
        sigma2 = float(st.Psi_poisson[0]) / 5.0   # sigma^2 = Psi0/W0, W0 = 5
        ref = np.expm1(E / sigma2)
        corr = float(np.corrcoef(f, ref)[0, 1])
        assert corr > 0.999, f"corr(f, e^Ehat - 1) = {corr:.6f}"


class TestKingDrhoDW:
    def test_closed_form_matches_autodiff(self):
        """Pin the closed-form _king_drho_dW (the erf'/boundary cancellation)
        against jax.grad of the King lowered-Maxwellian density itself."""
        from progenax.profiles.density_poisson import _king_drho_dW
        from progenax.profiles.king import king_lowered_maxwellian_density

        W = jnp.linspace(0.01, 9.0, 400)
        ad = jax.vmap(jax.grad(king_lowered_maxwellian_density))(W)
        np.testing.assert_allclose(np.asarray(_king_drho_dW(W)), np.asarray(ad),
                                   rtol=1e-10)

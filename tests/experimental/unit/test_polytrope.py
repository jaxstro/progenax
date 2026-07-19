r"""P3 — polytropic gas envelope (P = K rho^gamma), self-truncating at xi_1.

The polytrope stores the ADIABATIC index ``gamma`` because that is what an equation of
state hands you; the polytropic index ``n = 1/(gamma - 1)`` is derived (ADR-0065).

Useful correspondences used below:

    gamma = 2      <->  n = 1     (theta = sin(xi)/xi, xi_1 = pi -- exact)
    gamma = 5/3    <->  n = 3/2
    gamma = 4/3    <->  n = 3
    gamma = 1.2    <->  n = 5     (infinite extent -- REJECTED)
    gamma -> 1     <->  n -> inf  (isothermal; use BonnorEbertProfile instead)

Only ``gamma > 1.2`` gives a finite radius, so anything at or below it is refused with a
message pointing at the Bonnor-Ebert class rather than silently truncated.
"""

import jax.numpy as jnp
import pytest

from gravoturb.profiles.polytrope import PolytropeProfile


class TestIndexRelation:
    @pytest.mark.parametrize(
        "gamma, n_expected", [(2.0, 1.0), (5.0 / 3.0, 1.5), (4.0 / 3.0, 3.0)]
    )
    def test_n_is_derived_from_gamma(self, gamma, n_expected):
        p = PolytropeProfile(r_h=1.0, gamma=gamma)
        assert float(p.n) == pytest.approx(n_expected, rel=1e-12)


class TestExactTruncation:
    def test_gamma_2_has_xi1_equal_pi(self):
        """n=1 is the exact sin(xi)/xi solution, whose first zero is pi."""
        p = PolytropeProfile(r_h=1.0, gamma=2.0)
        assert float(p.xi_1) == pytest.approx(float(jnp.pi), rel=1e-7)

    def test_gamma_2_density_matches_exact_shape(self):
        """rho ∝ theta^n = sin(xi)/xi for n=1; compare the normalised shape."""
        p = PolytropeProfile(r_h=1.0, gamma=2.0)
        r = jnp.linspace(1e-3, float(p.r_edge) * 0.99, 100)
        xi = r / p.r_0
        exact = jnp.sin(xi) / xi
        got = p.density(r) / p.rho_c
        assert jnp.allclose(got, exact, rtol=1e-6, atol=1e-8)


class TestDensityStructure:
    @pytest.mark.parametrize("gamma", [1.3, 4.0 / 3.0, 5.0 / 3.0, 2.0])
    def test_density_is_positive_inside(self, gamma):
        p = PolytropeProfile(r_h=1.0, gamma=gamma)
        r = jnp.linspace(1e-3, float(p.r_edge) * 0.99, 200)
        assert jnp.all(p.density(r) > 0.0)

    @pytest.mark.parametrize("gamma", [1.3, 5.0 / 3.0, 2.0])
    def test_density_falls_outward(self, gamma):
        p = PolytropeProfile(r_h=1.0, gamma=gamma)
        r = jnp.linspace(1e-3, float(p.r_edge) * 0.99, 200)
        assert jnp.all(jnp.diff(p.density(r)) < 0.0)

    def test_density_vanishes_beyond_edge(self):
        """Self-truncating: theta reaches zero at xi_1, so there is a true edge."""
        p = PolytropeProfile(r_h=1.0, gamma=5.0 / 3.0)
        r = jnp.array([float(p.r_edge) * 1.01, float(p.r_edge) * 5.0])
        assert jnp.all(p.density(r) == 0.0)

    def test_density_goes_to_zero_at_the_edge(self):
        """Unlike Bonnor-Ebert, the polytrope's density vanishes continuously."""
        p = PolytropeProfile(r_h=1.0, gamma=5.0 / 3.0)
        rho_edge = p.density(jnp.array(float(p.r_edge) * 0.9999))
        assert float(rho_edge) < 1e-3 * float(p.rho_c)

    def test_central_density_is_the_maximum(self):
        p = PolytropeProfile(r_h=1.0, gamma=5.0 / 3.0)
        r = jnp.linspace(1e-4, float(p.r_edge) * 0.99, 100)
        assert float(jnp.max(p.density(r))) == pytest.approx(float(p.rho_c), rel=1e-3)


class TestHalfMassInversion:
    @pytest.mark.parametrize("r_h", [0.5, 1.0, 3.7])
    @pytest.mark.parametrize("gamma", [1.3, 5.0 / 3.0, 2.0])
    def test_r_h_round_trip(self, r_h, gamma):
        p = PolytropeProfile(r_h=r_h, gamma=gamma)
        half = p.mass_enclosed(jnp.array(r_h)) / p.total_mass
        assert float(half) == pytest.approx(0.5, abs=1e-6)

    def test_mass_enclosed_is_monotone(self):
        p = PolytropeProfile(r_h=1.0, gamma=5.0 / 3.0)
        r = jnp.linspace(1e-3, float(p.r_edge), 200)
        assert jnp.all(jnp.diff(p.mass_enclosed(r)) > 0.0)

    def test_mass_enclosed_saturates_at_total(self):
        p = PolytropeProfile(r_h=1.0, gamma=5.0 / 3.0)
        m_out = p.mass_enclosed(jnp.array(float(p.r_edge) * 3.0))
        assert float(m_out) == pytest.approx(float(p.total_mass), rel=1e-9)

    def test_r_h_is_inside_the_sphere(self):
        p = PolytropeProfile(r_h=1.0, gamma=5.0 / 3.0)
        assert float(p.r_edge) > 1.0


class TestConcentrationTrend:
    def test_softer_eos_is_more_centrally_concentrated(self):
        """Lower gamma (higher n) concentrates mass; r_edge/r_h must grow as gamma falls.

        This is the physical content of the gamma knob -- a softer equation of state
        supports less, so the cloud is more centrally peaked at fixed half-mass radius.
        """
        ratios = [
            float(PolytropeProfile(r_h=1.0, gamma=g).r_edge)
            for g in (2.0, 5.0 / 3.0, 1.4, 1.3)
        ]
        assert ratios == sorted(ratios), f"r_edge/r_h should rise as gamma falls: {ratios}"


class TestValidation:
    @pytest.mark.parametrize("gamma", [1.2, 1.1, 1.0, 0.8])
    def test_rejects_gamma_at_or_below_infinite_extent(self, gamma):
        """gamma <= 1.2 is n >= 5: infinite radius. Refuse, do not silently truncate."""
        with pytest.raises(ValueError, match="gamma"):
            PolytropeProfile(r_h=1.0, gamma=gamma)

    def test_rejection_message_points_at_bonnor_ebert(self):
        with pytest.raises(ValueError, match="[Bb]onnor"):
            PolytropeProfile(r_h=1.0, gamma=1.0)

    def test_rejects_nonpositive_r_h(self):
        with pytest.raises(ValueError, match="r_h"):
            PolytropeProfile(r_h=0.0, gamma=5.0 / 3.0)


class TestChainDuckType:
    def test_exposes_density_and_r_h(self):
        p = PolytropeProfile(r_h=1.0, gamma=5.0 / 3.0)
        assert hasattr(p, "density")
        assert float(p.r_h) == pytest.approx(1.0)

    def test_density_accepts_a_3d_grid(self):
        p = PolytropeProfile(r_h=1.0, gamma=5.0 / 3.0)
        r = jnp.ones((4, 4, 4)) * 0.5
        assert p.density(r).shape == (4, 4, 4)

r"""AC-BE3 / AC-BE4 — Bonnor-Ebert profile, critical point, and r_h inversion.

The BE critical constants are DERIVED from our own ODE, not imported (ADR-0067 note).
With ``M = 4 pi rho_c r_0^3 m(xi)``, ``r_0 = c_s / sqrt(4 pi G rho_c)`` and
``P_ext = rho_c e^{-psi} c_s^2``, the central density cancels:

    M = (1/sqrt(4 pi)) * c_s^4 / (G^{3/2} P_ext^{1/2}) * [xi^2 psi' e^{-psi/2}]

so the critical sphere is the maximum of the bracket, and the familiar coefficient is
just ``max(bracket)/sqrt(4 pi)``. The literature values (xi_crit ~ 6.451, contrast
~ 14.04, coefficient ~ 1.18) are used only as a CROSS-CHECK on that derivation.

The maximum is a genuine turning point -- ``m_BE`` rises to it and then falls -- which is
exactly why ``mu_BE`` inverts only on the stable branch and ``xi_max`` is the stored
primary (ADR-0066). That non-monotonicity is asserted here rather than assumed.
"""

import jax.numpy as jnp
import pytest

from gravoturb.profiles.bonnor_ebert import BonnorEbertProfile, critical_sphere

# Literature cross-check values (NOT inputs -- see module docstring).
LIT_XI_CRIT = 6.451
LIT_CONTRAST = 14.04
LIT_MASS_COEFF = 1.18


class TestCriticalSphere:
    """AC-BE3 — the critical point, self-derived then cross-checked."""

    def test_xi_crit_matches_literature(self):
        crit = critical_sphere()
        assert float(crit.xi_crit) == pytest.approx(LIT_XI_CRIT, rel=1e-3)

    def test_density_contrast_matches_literature(self):
        crit = critical_sphere()
        assert float(crit.contrast) == pytest.approx(LIT_CONTRAST, rel=1e-3)

    def test_mass_coefficient_matches_literature(self):
        """max(xi^2 psi' e^{-psi/2}) / sqrt(4 pi) -- the 1.18 in M_BE."""
        crit = critical_sphere()
        assert float(crit.mass_coefficient) == pytest.approx(LIT_MASS_COEFF, rel=2e-3)

    def test_be_mass_is_non_monotonic(self):
        """ADR-0066's premise: m_BE rises to a single peak, then falls.

        If this were monotone, mu_BE would be a globally valid parametrization and the
        stored primary should have been mu_BE rather than xi_max.
        """
        crit = critical_sphere()
        assert float(crit.m_be_at(crit.xi_crit)) > float(crit.m_be_at(2.0 * crit.xi_crit))
        assert float(crit.m_be_at(crit.xi_crit)) > float(crit.m_be_at(0.5 * crit.xi_crit))


class TestDensityStructure:
    def test_density_is_positive_inside(self):
        p = BonnorEbertProfile(r_h=1.0, xi_max=6.0)
        r = jnp.linspace(1e-3, float(p.r_edge) * 0.999, 200)
        assert jnp.all(p.density(r) > 0.0)

    def test_density_falls_outward(self):
        p = BonnorEbertProfile(r_h=1.0, xi_max=6.0)
        r = jnp.linspace(1e-3, float(p.r_edge) * 0.999, 200)
        assert jnp.all(jnp.diff(p.density(r)) < 0.0)

    def test_density_vanishes_beyond_edge(self):
        """Pressure-truncated: no material outside r_edge."""
        p = BonnorEbertProfile(r_h=1.0, xi_max=6.0)
        r = jnp.array([float(p.r_edge) * 1.01, float(p.r_edge) * 5.0])
        assert jnp.all(p.density(r) == 0.0)

    def test_central_density_is_the_maximum(self):
        p = BonnorEbertProfile(r_h=1.0, xi_max=6.0)
        r = jnp.linspace(1e-4, float(p.r_edge) * 0.99, 100)
        assert float(jnp.max(p.density(r))) == pytest.approx(float(p.rho_c), rel=1e-3)

    def test_contrast_matches_density_ratio(self):
        """The reported contrast must equal the actual centre-to-edge density ratio."""
        p = BonnorEbertProfile(r_h=1.0, xi_max=6.0)
        rho_edge = p.density(jnp.array(float(p.r_edge) * 0.9999))
        assert float(p.contrast) == pytest.approx(float(p.rho_c / rho_edge), rel=1e-3)


class TestHalfMassInversion:
    """AC-BE4 — build with r_h, measure r_h back."""

    @pytest.mark.parametrize("r_h", [0.5, 1.0, 3.7])
    @pytest.mark.parametrize("xi_max", [3.0, 6.0, 10.0])
    def test_r_h_round_trip(self, r_h, xi_max):
        p = BonnorEbertProfile(r_h=r_h, xi_max=xi_max)
        half = p.mass_enclosed(jnp.array(r_h)) / p.total_mass
        assert float(half) == pytest.approx(0.5, abs=1e-6)

    def test_mass_enclosed_is_monotone(self):
        p = BonnorEbertProfile(r_h=1.0, xi_max=6.0)
        r = jnp.linspace(1e-3, float(p.r_edge), 200)
        assert jnp.all(jnp.diff(p.mass_enclosed(r)) > 0.0)

    def test_mass_enclosed_saturates_at_total(self):
        p = BonnorEbertProfile(r_h=1.0, xi_max=6.0)
        m_out = p.mass_enclosed(jnp.array(float(p.r_edge) * 3.0))
        assert float(m_out) == pytest.approx(float(p.total_mass), rel=1e-9)

    def test_r_h_is_inside_the_sphere(self):
        p = BonnorEbertProfile(r_h=1.0, xi_max=6.0)
        assert float(p.r_edge) > 1.0


class TestCriticalRatioAPI:
    """ADR-0066 — mu_BE derived, and the stable-branch constructor."""

    def test_mu_be_is_one_at_the_critical_radius(self):
        crit = critical_sphere()
        p = BonnorEbertProfile(r_h=1.0, xi_max=float(crit.xi_crit))
        assert float(p.mu_BE) == pytest.approx(1.0, rel=1e-4)

    def test_mu_be_below_one_when_subcritical(self):
        p = BonnorEbertProfile(r_h=1.0, xi_max=3.0)
        assert float(p.mu_BE) < 1.0

    def test_from_critical_ratio_round_trips(self):
        p = BonnorEbertProfile.from_critical_ratio(mu_BE=0.6, r_h=1.0)
        assert float(p.mu_BE) == pytest.approx(0.6, rel=1e-3)

    def test_from_critical_ratio_preserves_r_h(self):
        p = BonnorEbertProfile.from_critical_ratio(mu_BE=0.6, r_h=2.5)
        half = p.mass_enclosed(jnp.array(2.5)) / p.total_mass
        assert float(half) == pytest.approx(0.5, abs=1e-6)

    def test_from_critical_ratio_rejects_supercritical(self):
        """Refuse an ambiguous request rather than silently picking a branch."""
        with pytest.raises(ValueError, match="mu_BE"):
            BonnorEbertProfile.from_critical_ratio(mu_BE=1.5, r_h=1.0)

    def test_from_critical_ratio_rejects_nonpositive(self):
        with pytest.raises(ValueError, match="mu_BE"):
            BonnorEbertProfile.from_critical_ratio(mu_BE=0.0, r_h=1.0)


class TestValidation:
    def test_rejects_nonpositive_xi_max(self):
        with pytest.raises(ValueError, match="xi_max"):
            BonnorEbertProfile(r_h=1.0, xi_max=0.0)

    def test_rejects_nonpositive_r_h(self):
        with pytest.raises(ValueError, match="r_h"):
            BonnorEbertProfile(r_h=0.0, xi_max=6.0)


class TestChainDuckType:
    """AC-BE6 precondition: the gravoturb chain needs exactly .density and .r_h."""

    def test_exposes_density_and_r_h(self):
        p = BonnorEbertProfile(r_h=1.0, xi_max=6.0)
        assert hasattr(p, "density")
        assert float(p.r_h) == pytest.approx(1.0)

    def test_density_accepts_a_3d_grid(self):
        """envelope.py calls profile.density(r) on a 3D radius grid."""
        p = BonnorEbertProfile(r_h=1.0, xi_max=6.0)
        r = jnp.ones((4, 4, 4)) * 0.5
        assert p.density(r).shape == (4, 4, 4)

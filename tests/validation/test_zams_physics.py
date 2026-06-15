"""
Physics validation for the Tout+1996 ZAMS relations (M -> L, R, T_eff, log g + inverse).

The Tout+1996 fits are validated against INDEPENDENT oracles, not against their own
algebra:
  - the four solar ZAMS anchors (L, R, T_eff, log g) computed cell-by-cell from the
    PDF-verified coefficients at M=1, Z=0.02 (zeta=0, so each coefficient is its
    column-`a` value) -- see docs/core-papers/tout1996_zams_coefficients_verified.md;
  - the Stefan-Boltzmann + g = G M / R^2 closure (T_eff and log g are recomputed by
    hand from L and R in cgs and must match the module to round-off -- a genuine
    cross-check that the physics, not just the coefficients, is wired correctly);
  - strict monotonicity of L(M) over the fitted range (homology: more massive ZAMS
    stars are more luminous), which is what makes the luminosity invertible;
  - the inverse Newton round-trip M -> L -> M (the inverse solve must recover the
    forcing mass to rtol 1e-5 over the full 0.1-100 Msun fitted range);
  - the paper's own stated accuracy envelope (L < 7.5% / R < 5% over the fitted box,
    < 3% / < 1.2% at solar), checked at solar against literature ZAMS values.

Reference:
    Tout, Pols, Eggleton & Han (1996), MNRAS 281, 257, Tables 1 & 2.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import progenax  # noqa: F401  (enables float64 at import)
from progenax.stellar import (
    inverse_zams_luminosity,
    zams_effective_temperature,
    zams_luminosity,
    zams_radius,
    zams_surface_gravity,
)

# Solar ZAMS anchors at M=1, Z=0.02, computed from the PDF-verified coefficients
# (docs/core-papers/tout1996_zams_coefficients_verified.md, "Solar ZAMS anchors").
# L and R are exact algebra of the column-`a` coefficients; T_eff and log g follow
# from L,R via Stefan-Boltzmann / g = GM/R^2 with the implementation's cgs constants.
L_SUN_ZAMS = 0.6977165691451518
R_SUN_ZAMS = 0.8882494502975121
TEFF_SUN_ZAMS = 5597.303626190019  # K
LOGG_SUN_ZAMS = 4.540995576621913  # dex (cgs)


class TestSolarAnchors:
    """The four solar ZAMS values reproduce the PDF-verified anchors within the
    paper's stated solar accuracy (L < 3%, R < 1.2%) and to the computed values."""

    def test_luminosity_solar_anchor(self):
        L = float(zams_luminosity(jnp.array(1.0)))
        # within the paper's stated solar L accuracy (< 3%) of the verified anchor
        assert abs(L - L_SUN_ZAMS) / L_SUN_ZAMS < 0.03
        # and exact to the recorded verified value
        assert L == pytest.approx(L_SUN_ZAMS, rel=1e-10)

    def test_radius_solar_anchor(self):
        R = float(zams_radius(jnp.array(1.0)))
        # within the paper's stated solar R accuracy (< 1.2%) of the verified anchor
        assert abs(R - R_SUN_ZAMS) / R_SUN_ZAMS < 0.012
        assert R == pytest.approx(R_SUN_ZAMS, rel=1e-10)

    def test_effective_temperature_solar_anchor(self):
        T = float(zams_effective_temperature(jnp.array(1.0)))
        assert T == pytest.approx(TEFF_SUN_ZAMS, rel=1e-6)
        # ZAMS Sun sits near 5600 K (cooler/smaller than the present-day 5772 K Sun)
        assert 5400.0 < T < 5800.0

    def test_surface_gravity_solar_anchor(self):
        g = float(zams_surface_gravity(jnp.array(1.0)))
        assert g == pytest.approx(LOGG_SUN_ZAMS, rel=1e-6)
        # ZAMS Sun slightly more compact than today (log g_today ~ 4.44)
        assert 4.4 < g < 4.6


class TestStefanBoltzmannClosure:
    """T_eff and log g are not independent fits: they must satisfy
    L = 4 pi R^2 sigma T_eff^4 and g = G M / R^2 exactly. Recompute them by hand
    in cgs from the module's own L(M), R(M) and require round-off agreement -- a
    physics closure independent of the coefficient values."""

    from jaxstro.constants import LSUN_ERG_S, RSUN_CM, SIGMA_SB, G_CGS, MSUN_G

    @pytest.mark.parametrize("M", [0.2, 0.5, 1.0, 5.0, 30.0, 80.0])
    def test_teff_matches_stefan_boltzmann(self, M):
        L_cgs = float(zams_luminosity(jnp.array(M))) * self.LSUN_ERG_S
        R_cgs = float(zams_radius(jnp.array(M))) * self.RSUN_CM
        T_hand = (L_cgs / (4.0 * np.pi * R_cgs**2 * self.SIGMA_SB)) ** 0.25
        T_mod = float(zams_effective_temperature(jnp.array(M)))
        assert T_mod == pytest.approx(T_hand, rel=1e-10)

    @pytest.mark.parametrize("M", [0.2, 0.5, 1.0, 5.0, 30.0, 80.0])
    def test_logg_matches_newton(self, M):
        M_cgs = M * self.MSUN_G
        R_cgs = float(zams_radius(jnp.array(M))) * self.RSUN_CM
        g_hand = np.log10(self.G_CGS * M_cgs / R_cgs**2)
        g_mod = float(zams_surface_gravity(jnp.array(M)))
        assert g_mod == pytest.approx(g_hand, rel=1e-10)


class TestLuminosityMonotonic:
    """L(M) is strictly increasing over the fitted range -- the homology property
    that makes the luminosity invertible (and the inverse Newton well-posed)."""

    def test_strictly_monotonic_over_fitted_range(self):
        M = jnp.logspace(jnp.log10(0.1), jnp.log10(100.0), 400)
        L = zams_luminosity(M)
        assert jnp.all(jnp.diff(L) > 0.0), "L(M) is not strictly monotonic"

    def test_radius_positive_over_fitted_range(self):
        M = jnp.logspace(jnp.log10(0.1), jnp.log10(100.0), 400)
        R = zams_radius(M)
        assert jnp.all(R > 0.0) and jnp.all(jnp.isfinite(R))


class TestInverseRoundTrip:
    """The differentiable Newton invert recovers the forcing mass over the whole
    fitted range (independent oracle: the forward L(M) it inverts)."""

    def test_round_trip_recovers_mass(self):
        M = jnp.logspace(jnp.log10(0.1), jnp.log10(100.0), 50)
        M_rec = inverse_zams_luminosity(zams_luminosity(M))
        assert jnp.allclose(M_rec, M, rtol=1e-5), (
            f"max rel residual {float(jnp.max(jnp.abs(M_rec - M) / M)):.2e}"
        )

    @pytest.mark.parametrize("M", [0.1, 0.5, 1.0, 5.0, 20.0, 100.0])
    def test_round_trip_pointwise(self, M):
        m_rec = float(inverse_zams_luminosity(zams_luminosity(jnp.array(M))))
        assert m_rec == pytest.approx(M, rel=1e-5)


class TestMetallicityDependence:
    """Lower metallicity -> finite, hotter, more luminous ZAMS (lower opacity);
    the rational fits stay finite and positive across the fitted Z box."""

    def test_lower_Z_is_finite_and_more_luminous(self):
        M = jnp.array(1.0)
        L_solar = float(zams_luminosity(M, Z=0.02))
        L_poor = float(zams_luminosity(M, Z=0.001))
        L_vpoor = float(zams_luminosity(M, Z=0.0001))
        assert jnp.isfinite(L_poor) and jnp.isfinite(L_vpoor)
        # metal-poor ZAMS stars are MORE luminous at fixed mass
        assert L_poor > L_solar and L_vpoor > L_solar

    def test_finite_across_Z_box(self):
        for Z in (1e-4, 1e-3, 1e-2, 0.03):
            L = zams_luminosity(jnp.logspace(jnp.log10(0.1), jnp.log10(100.0), 50), Z=Z)
            R = zams_radius(jnp.logspace(jnp.log10(0.1), jnp.log10(100.0), 50), Z=Z)
            assert jnp.all(jnp.isfinite(L)) and jnp.all(L > 0.0)
            assert jnp.all(jnp.isfinite(R)) and jnp.all(R > 0.0)


class TestPublishedAccuracy:
    """Solar L/R reproduce literature ZAMS values within the paper's stated solar
    accuracy (L < 3%, R < 1.2%). The ZAMS Sun is fainter/smaller than today's Sun;
    the standard-solar-model ZAMS values (L0 ~ 0.70 Lsun, R0 ~ 0.89 Rsun; e.g.
    Bahcall et al. 2001) are the independent oracle here."""

    def test_solar_L_within_paper_accuracy(self):
        L = float(zams_luminosity(jnp.array(1.0)))
        # ZAMS solar luminosity ~ 0.70 Lsun (SSM); Tout claims < 3% at solar
        assert abs(L - 0.70) / 0.70 < 0.03

    def test_solar_R_within_paper_accuracy(self):
        R = float(zams_radius(jnp.array(1.0)))
        # ZAMS solar radius ~ 0.89 Rsun (SSM); Tout claims < 1.2% at solar
        assert abs(R - 0.89) / 0.89 < 0.012


class TestDifferentiability:
    """Every forward relation and the Newton invert flow finite gradients (the
    Fisher-information requirement for inference)."""

    def test_dL_dM_finite_positive(self):
        g = jax.grad(lambda m: zams_luminosity(m))(jnp.array(1.0))
        assert jnp.isfinite(g) and float(g) > 0.0

    def test_dR_dM_finite(self):
        g = jax.grad(lambda m: zams_radius(m))(jnp.array(1.0))
        assert jnp.isfinite(g)

    def test_dTeff_dM_finite(self):
        g = jax.grad(lambda m: zams_effective_temperature(m))(jnp.array(1.0))
        assert jnp.isfinite(g)

    def test_dlogg_dM_finite(self):
        g = jax.grad(lambda m: zams_surface_gravity(m))(jnp.array(1.0))
        assert jnp.isfinite(g)

    def test_inverse_differentiable(self):
        g = jax.grad(lambda L: inverse_zams_luminosity(L)[0])(jnp.array([100.0]))
        assert jnp.isfinite(g)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

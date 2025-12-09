"""Tests for paper-calibrated environment-dependent IMF module.

These tests verify:
1. Exact coefficient values from Marks+2012 and Jerabkova+2018
2. Correct threshold behavior (POSITIVE for Marks, NEGATIVE for Jerabkova)
3. Gradient flow for inference applications
4. Validation against published GC values (Marks+2012 Table 1)
"""

import jax
import jax.numpy as jnp
import pytest

from progenax.imf import (
    BirthEnvironment,
    env_to_imf_params,
    alpha3_jerabkova_mecl,
    alpha3_jerabkova_rho,
    alpha3_marks_plane,
    lowmass_slopes_metallicity,
    JERABKOVA_COEFFICIENTS,
    MARKS_COEFFICIENTS,
)
from progenax.imf.differentiable import individual_mass_nll


# =============================================================================
# Test BirthEnvironment
# =============================================================================


class TestBirthEnvironment:
    """Tests for BirthEnvironment data class."""

    def test_from_cluster_mass(self):
        """Can create BirthEnvironment from M_ecl and [Fe/H]."""
        env = BirthEnvironment.from_cluster_mass(M_ecl=1e6, FeH=-1.5)
        assert jnp.isclose(env.log_mecl, 6.0, atol=0.01)
        assert jnp.isclose(env.metallicity, -1.5)

    def test_from_cluster_mass_with_density(self):
        """Can create BirthEnvironment with optional density."""
        env = BirthEnvironment.from_cluster_mass(M_ecl=1e5, FeH=-0.5, log_rho_cl=2.0)
        assert jnp.isclose(env.log_mecl, 5.0, atol=0.01)
        assert jnp.isclose(env.log_rho_cl, 2.0)

    def test_solar_preset(self):
        """Solar preset has correct values."""
        env = BirthEnvironment.solar()
        assert jnp.isclose(env.metallicity, 0.0)
        assert jnp.isclose(env.log_mecl, 3.0)
        assert jnp.isclose(env.sfe, 0.33)  # Default SFE

    def test_massive_gc_preset(self):
        """Massive GC preset has correct values."""
        env = BirthEnvironment.massive_gc(FeH=-1.5)
        assert jnp.isclose(env.metallicity, -1.5)
        assert jnp.isclose(env.log_mecl, 6.0)

    def test_ngc_7078_preset(self):
        """NGC 7078 preset matches Marks+2012 Table 1 values."""
        env = BirthEnvironment.ngc_7078()
        assert jnp.isclose(env.metallicity, -2.16)
        # log_rho_cl = log10(258.13) ≈ 2.41
        assert jnp.isclose(env.log_rho_cl, jnp.log10(258.13), atol=0.01)

    def test_ngc_104_preset(self):
        """NGC 104 preset matches Marks+2012 Table 1 values."""
        env = BirthEnvironment.ngc_104()
        assert jnp.isclose(env.metallicity, -0.76)
        # log_rho_cl = log10(9.54) ≈ 0.98
        assert jnp.isclose(env.log_rho_cl, jnp.log10(9.54), atol=0.01)


# =============================================================================
# Test Jerabkova+2018 Coefficients
# =============================================================================


class TestJerabkovaCoefficients:
    """Tests for Jerabkova+2018 Eq. 9 implementation."""

    def test_coefficient_values(self):
        """Verify exact coefficient values (8π half-mass density convention)."""
        c = JERABKOVA_COEFFICIENTS
        assert c["FeH_coeff"] == -0.14
        assert c["logMecl_coeff"] == 0.6039  # 0.99 × 0.61 = 0.6039 (exact)
        assert c["constant"] == 0.2161       # 8π half-mass density derived
        assert c["x_threshold"] == -0.87     # NEGATIVE threshold
        assert c["alpha3_slope"] == -0.41
        assert c["alpha3_intercept"] == 1.94

    def test_threshold_is_negative(self):
        """CRITICAL: Jerabkova threshold is x >= -0.87 (NEGATIVE)."""
        c = JERABKOVA_COEFFICIENTS
        assert c["x_threshold"] < 0, "Jerabkova threshold must be NEGATIVE"

    def test_massive_cluster_solar(self):
        """Massive cluster at solar metallicity: moderately top-heavy IMF."""
        # M_ecl = 10^6 M_sun (log_mecl_6 = 0), FeH = 0
        # Use coefficients from module for consistency
        c = JERABKOVA_COEFFICIENTS
        alpha3 = alpha3_jerabkova_mecl(jnp.array(0.0), jnp.array(0.0))
        x = c["FeH_coeff"] * 0 + c["logMecl_coeff"] * 0 + c["constant"]  # = 0.2161
        expected = c["alpha3_slope"] * x + c["alpha3_intercept"]  # ~1.85
        assert jnp.isclose(alpha3, expected, atol=0.01)
        assert float(alpha3) < 2.0  # Top-heavy but not extreme

    def test_massive_cluster_metal_poor(self):
        """Massive metal-poor cluster: more top-heavy."""
        # M_ecl = 10^6 M_sun, FeH = -1.5
        c = JERABKOVA_COEFFICIENTS
        alpha3 = alpha3_jerabkova_mecl(jnp.array(0.0), jnp.array(-1.5))
        x = c["FeH_coeff"] * (-1.5) + c["logMecl_coeff"] * 0 + c["constant"]  # ~0.43
        expected = c["alpha3_slope"] * x + c["alpha3_intercept"]  # ~1.76
        assert jnp.isclose(alpha3, expected, atol=0.05)
        assert float(alpha3) < 1.9  # More top-heavy than solar

    def test_small_cluster_canonical(self):
        """Small cluster at solar metallicity: canonical Kroupa."""
        # M_ecl = 10^3 M_sun (log_mecl_6 = -3), FeH = 0
        c = JERABKOVA_COEFFICIENTS
        log_mecl_6 = jnp.array(-3.0)  # 10^3 M_sun cluster
        FeH = jnp.array(0.0)
        x = c["FeH_coeff"] * 0 + c["logMecl_coeff"] * (-3.0) + c["constant"]  # < -0.87
        alpha3 = alpha3_jerabkova_mecl(log_mecl_6, FeH)
        # Below threshold: x < -0.87, so canonical Kroupa
        assert x < c["x_threshold"], f"x={x} should be below threshold"
        assert jnp.isclose(alpha3, c["alpha3_canonical"], atol=0.01)  # CANONICAL KROUPA!

    def test_smooth_transition(self):
        """Smooth mode produces differentiable transition."""
        log_mecl_6 = jnp.array(0.0)
        FeH = jnp.array(0.0)

        alpha3_sharp = alpha3_jerabkova_mecl(log_mecl_6, FeH, smooth=False)
        alpha3_smooth = alpha3_jerabkova_mecl(log_mecl_6, FeH, smooth=True)

        # Should be close but not identical
        assert jnp.isclose(alpha3_sharp, alpha3_smooth, atol=0.1)

    def test_bounds_clipping(self):
        """Output is clipped to [0.5, 2.3]."""
        # Extreme case: very massive, very metal-poor → would give < 0.5
        log_mecl_6 = jnp.array(2.0)  # 10^8 M_sun
        FeH = jnp.array(-2.5)
        alpha3 = alpha3_jerabkova_mecl(log_mecl_6, FeH)
        assert float(alpha3) >= 0.5
        assert float(alpha3) <= 2.3


# =============================================================================
# Test Marks+2012 Coefficients
# =============================================================================


class TestMarksCoefficients:
    """Tests for Marks+2012 Fundamental Plane implementation."""

    def test_coefficient_values(self):
        """Verify exact coefficient values from paper."""
        c = MARKS_COEFFICIENTS
        assert jnp.isclose(c["cos_theta"], -0.139, atol=0.001)
        assert jnp.isclose(c["sin_theta"], 0.990, atol=0.001)
        assert jnp.isclose(c["x_hat_threshold"], 0.87, atol=0.01)  # POSITIVE
        assert jnp.isclose(c["alpha3_slope"], -0.4072, atol=0.001)
        assert jnp.isclose(c["alpha3_intercept"], 1.9383, atol=0.001)

    def test_threshold_is_positive(self):
        """CRITICAL: Marks threshold is x_hat >= +0.87 (POSITIVE)."""
        c = MARKS_COEFFICIENTS
        assert c["x_hat_threshold"] > 0, "Marks threshold must be POSITIVE"

    def test_high_density_metal_poor(self):
        """High density + metal poor: top-heavy IMF."""
        # FeH = -2, log_rho_6 = 1 (10^7 M_sun/pc^3)
        # x_hat = -0.139*(-2) + 0.990*1 = 0.278 + 0.990 = 1.268 > +0.87
        log_rho_6 = jnp.array(1.0)
        FeH = jnp.array(-2.0)
        alpha3 = alpha3_marks_plane(log_rho_6, FeH)
        x_hat = -0.139 * (-2) + 0.990 * 1.0
        expected = -0.4072 * x_hat + 1.9383
        assert jnp.isclose(alpha3, expected, atol=0.02)
        assert float(alpha3) < 2.0  # Top-heavy

    def test_low_density_canonical(self):
        """Low density + solar metallicity: canonical Kroupa."""
        # FeH = 0, log_rho_6 = -2 (10^4 M_sun/pc^3)
        # x_hat = -0.139*0 + 0.990*(-2) = -1.98 < +0.87
        log_rho_6 = jnp.array(-2.0)
        FeH = jnp.array(0.0)
        alpha3 = alpha3_marks_plane(log_rho_6, FeH)
        assert jnp.isclose(alpha3, 2.3, atol=0.01)  # Canonical


# =============================================================================
# Test Low-Mass Slopes (Marks+2012 Eq. 12)
# =============================================================================


class TestLowMassSlopes:
    """Tests for metallicity-dependent low-mass slopes."""

    def test_solar_metallicity(self):
        """Solar metallicity gives canonical slopes."""
        alpha1, alpha2 = lowmass_slopes_metallicity(jnp.array(0.0))
        assert jnp.isclose(alpha1, 1.3, atol=0.01)  # 1.3 + 0.5*0
        assert jnp.isclose(alpha2, 2.3, atol=0.01)  # 2.3 + 0.5*0

    def test_metal_poor(self):
        """Metal-poor gives shallower slopes."""
        # FeH = -2.0: alpha1 = 1.3 + 0.5*(-2) = 0.3
        #             alpha2 = 2.3 + 0.5*(-2) = 1.3
        alpha1, alpha2 = lowmass_slopes_metallicity(jnp.array(-2.0))
        assert jnp.isclose(alpha1, 0.3, atol=0.01)
        assert jnp.isclose(alpha2, 1.3, atol=0.01)

    def test_metal_rich(self):
        """Metal-rich gives steeper slopes."""
        # FeH = +0.5: alpha1 = 1.3 + 0.5*0.5 = 1.55
        #             alpha2 = 2.3 + 0.5*0.5 = 2.55
        alpha1, alpha2 = lowmass_slopes_metallicity(jnp.array(0.5))
        assert jnp.isclose(alpha1, 1.55, atol=0.01)
        assert jnp.isclose(alpha2, 2.55, atol=0.01)

    def test_table4_grid(self):
        """Verify Table 4 values from Marks+2012."""
        expected = {
            -2.0: (0.30, 1.30),
            -1.5: (0.55, 1.55),
            -1.0: (0.80, 1.80),
            0.0: (1.30, 2.30),
            0.5: (1.55, 2.55),
        }

        for FeH, (exp_a1, exp_a2) in expected.items():
            alpha1, alpha2 = lowmass_slopes_metallicity(jnp.array(FeH), clamp_FeH=False)
            assert jnp.isclose(alpha1, exp_a1, atol=0.02), f"alpha1 at FeH={FeH}"
            assert jnp.isclose(alpha2, exp_a2, atol=0.02), f"alpha2 at FeH={FeH}"

    def test_clamping(self):
        """FeH is clamped to [-2.5, +0.5] by default."""
        # FeH = -5 should be clamped to -2.5
        alpha1, alpha2 = lowmass_slopes_metallicity(jnp.array(-5.0), clamp_FeH=True)
        alpha1_clamped, alpha2_clamped = lowmass_slopes_metallicity(
            jnp.array(-2.5), clamp_FeH=True
        )
        assert jnp.isclose(alpha1, alpha1_clamped)
        assert jnp.isclose(alpha2, alpha2_clamped)


# =============================================================================
# Test Unified env_to_imf_params API
# =============================================================================


class TestEnvToIMFParams:
    """Tests for unified env_to_imf_params function."""

    def test_jerabkova_default(self):
        """Default model is jerabkova_generalized (with SFE)."""
        env = BirthEnvironment.from_cluster_mass(M_ecl=1e6, FeH=-1.5)
        params = env_to_imf_params(env)  # default model="jerabkova_generalized"

        # Compute expected alpha3 with SFE=0.33 (default)
        # At sfe=0.33: log10(sfe/0.33) = 0, so same as jerabkova_mecl
        c = JERABKOVA_COEFFICIENTS
        log_mecl_6 = 0.0  # log10(1e6 / 1e6) = 0
        x = c["FeH_coeff"] * (-1.5) + c["logMecl_coeff"] * log_mecl_6 + c["constant"]  # ~0.43
        expected_alpha3 = c["alpha3_slope"] * x + c["alpha3_intercept"]  # ~1.76

        assert jnp.isclose(params.alpha3, expected_alpha3, atol=0.05)

    def test_kroupa_model(self):
        """Kroupa model ignores environment."""
        env = BirthEnvironment.from_cluster_mass(M_ecl=1e6, FeH=-2.0)
        params = env_to_imf_params(env, model="kroupa")

        assert jnp.isclose(params.alpha0, 0.3)
        assert jnp.isclose(params.alpha1, 1.3)
        assert jnp.isclose(params.alpha2, 2.3)
        assert jnp.isclose(params.alpha3, 2.3)

    def test_marks_plane_computes_density(self):
        """marks_plane model computes density from mass+SFE if not provided."""
        # Use very massive cluster with low SFE to push x_hat above threshold
        env = BirthEnvironment.from_cluster_mass(M_ecl=1e7, FeH=-2.0, sfe=0.1)
        # No log_rho_cl set - should compute from M_ecl and SFE
        params = env_to_imf_params(env, model="marks_plane")
        # Should not raise, and should have modified alpha3
        assert params.alpha3 < 2.3  # Top-heavy for very massive, dense cluster

    def test_jerabkova_rho_computes_density(self):
        """jerabkova_rho model computes density from mass+SFE if not provided."""
        env = BirthEnvironment.from_cluster_mass(M_ecl=1e6, FeH=-1.5)
        # No log_rho_cl set - should compute from M_ecl and SFE
        params = env_to_imf_params(env, model="jerabkova_rho")
        # Should not raise, and should have modified alpha3
        assert params.alpha3 < 2.3  # Top-heavy for massive cluster

    def test_invalid_model(self):
        """Invalid model name raises ValueError."""
        env = BirthEnvironment.from_cluster_mass(M_ecl=1e6, FeH=-1.5)

        with pytest.raises(ValueError, match="Unknown model"):
            env_to_imf_params(env, model="invalid_model")

    def test_include_lowmass_variation(self):
        """Low-mass slopes vary with metallicity."""
        env = BirthEnvironment.from_cluster_mass(M_ecl=1e6, FeH=-2.0)

        params_with = env_to_imf_params(env, include_lowmass_variation=True)
        params_without = env_to_imf_params(env, include_lowmass_variation=False)

        # With variation (Marks Eq. 12): alpha1 = 1.3 + 0.5*(-2) = 0.3
        assert jnp.isclose(params_with.alpha1, 0.3, atol=0.05)
        # Without variation: canonical 1.3
        assert jnp.isclose(params_without.alpha1, 1.3)

    def test_smooth_alpha3(self):
        """Smooth transition produces similar but differentiable output."""
        env = BirthEnvironment.from_cluster_mass(M_ecl=1e6, FeH=-1.5)

        params_sharp = env_to_imf_params(env, smooth_alpha3=False)
        params_smooth = env_to_imf_params(env, smooth_alpha3=True)

        assert jnp.isclose(params_sharp.alpha3, params_smooth.alpha3, atol=0.1)


# =============================================================================
# Test Gradient Flow (for Inference)
# =============================================================================


class TestGradientFlow:
    """Tests verifying gradients flow correctly for inference."""

    def test_gradient_env_to_likelihood(self):
        """Gradients flow: env → IMFParams → likelihood."""
        observed_masses = jnp.array([0.5, 1.0, 5.0, 10.0, 50.0])

        def nll(log_mecl, FeH):
            env = BirthEnvironment(metallicity=FeH, log_mecl=log_mecl)
            params = env_to_imf_params(env, smooth_alpha3=True)
            return individual_mass_nll(observed_masses, params)

        grad_fn = jax.grad(nll, argnums=(0, 1))
        g_mecl, g_feh = grad_fn(jnp.array(5.0), jnp.array(-1.0))

        assert jnp.isfinite(g_mecl), "Gradient w.r.t. log_mecl must be finite"
        assert jnp.isfinite(g_feh), "Gradient w.r.t. FeH must be finite"

    def test_gradient_direction(self):
        """Gradient points in physically sensible direction."""
        # More massive stars → higher log_mecl should decrease NLL
        # (because top-heavy IMF has more massive stars)
        massive_stars = jnp.array([20.0, 30.0, 50.0, 80.0, 100.0])

        def nll(log_mecl):
            env = BirthEnvironment(metallicity=jnp.array(-1.0), log_mecl=log_mecl)
            params = env_to_imf_params(env, smooth_alpha3=True)
            return individual_mass_nll(massive_stars, params)

        # At low log_mecl, increasing it should decrease NLL
        grad_fn = jax.grad(nll)
        g = grad_fn(jnp.array(4.0))

        # Negative gradient means increasing log_mecl decreases NLL
        assert float(g) < 0, "Gradient should be negative for massive star sample"

    def test_jit_compatibility(self):
        """env_to_imf_params is JIT-compatible."""
        @jax.jit
        def compute_alpha3(log_mecl, FeH):
            env = BirthEnvironment(metallicity=FeH, log_mecl=log_mecl)
            params = env_to_imf_params(env, smooth_alpha3=True)
            return params.alpha3

        result = compute_alpha3(jnp.array(6.0), jnp.array(-1.5))
        assert jnp.isfinite(result)


# =============================================================================
# Validation Against Published GC Values (Marks+2012 Table 1)
# =============================================================================


class TestGCValidation:
    """Validation against Marks+2012 Table 1 GC birth conditions."""

    @pytest.mark.parametrize(
        "gc_name,FeH,rho_1e6,expected_alpha3",
        [
            # From Marks+2012 Table 1
            # NGC name, [Fe/H], rho_cl [10^6 M_sun/pc^3], expected alpha3
            ("NGC 104", -0.76, 9.54, 1.34),
            ("NGC 6341", -2.28, 66.03, 1.11),
            ("NGC 6752", -1.56, 31.78, 1.27),
            ("NGC 7078", -2.16, 258.13, 0.76),
        ],
    )
    def test_marks_table1_gc(self, gc_name, FeH, rho_1e6, expected_alpha3):
        """Reproduce alpha3 for GCs from Marks+2012 Table 1."""
        log_rho_6 = jnp.log10(jnp.array(rho_1e6))

        computed = alpha3_marks_plane(log_rho_6, jnp.array(FeH))

        # Note: expect some scatter around best-fit line (~0.15)
        assert jnp.isclose(computed, expected_alpha3, atol=0.20), (
            f"{gc_name}: expected {expected_alpha3}, got {float(computed):.2f}"
        )

    def test_ngc_7078_is_most_top_heavy(self):
        """NGC 7078 (M15) should be the most top-heavy GC in the sample."""
        gc_data = [
            ("NGC 104", -0.76, 9.54),
            ("NGC 6341", -2.28, 66.03),
            ("NGC 6752", -1.56, 31.78),
            ("NGC 7078", -2.16, 258.13),
        ]

        alphas = {}
        for name, FeH, rho_1e6 in gc_data:
            log_rho_6 = jnp.log10(jnp.array(rho_1e6))
            alphas[name] = float(alpha3_marks_plane(log_rho_6, jnp.array(FeH)))

        # NGC 7078 should have smallest alpha3 (most top-heavy)
        assert alphas["NGC 7078"] == min(alphas.values()), (
            f"NGC 7078 should be most top-heavy, but got {alphas}"
        )

    def test_density_dependence_stronger_than_metallicity(self):
        """Density (sin_theta=0.99) dominates over metallicity (cos_theta=-0.14)."""
        c = MARKS_COEFFICIENTS
        # sin_theta / |cos_theta| = 0.99 / 0.139 ≈ 7
        # Density is ~7× more important than metallicity
        ratio = c["sin_theta"] / abs(c["cos_theta"])
        assert ratio > 5, f"Density should dominate, ratio = {ratio:.1f}"


# =============================================================================
# Edge Cases and Numerical Stability
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability at edge cases."""

    def test_extreme_metallicity(self):
        """Extreme metallicity values don't cause NaN."""
        for FeH in [-5.0, -3.0, 0.5, 2.0]:
            alpha3 = alpha3_jerabkova_mecl(jnp.array(0.0), jnp.array(FeH))
            assert jnp.isfinite(alpha3), f"NaN at FeH={FeH}"

    def test_extreme_mass(self):
        """Extreme cluster mass values don't cause NaN."""
        for log_mecl_6 in [-5.0, -2.0, 2.0, 5.0]:
            alpha3 = alpha3_jerabkova_mecl(jnp.array(log_mecl_6), jnp.array(0.0))
            assert jnp.isfinite(alpha3), f"NaN at log_mecl_6={log_mecl_6}"

    def test_zero_density(self):
        """Zero density (log_rho_6 → -∞) gives canonical IMF."""
        # log_rho_6 = -10 (very low density)
        alpha3 = alpha3_marks_plane(jnp.array(-10.0), jnp.array(0.0))
        assert jnp.isclose(alpha3, 2.3, atol=0.01), "Low density should give canonical"

    def test_vmap_compatibility(self):
        """Functions are vmap-compatible."""
        log_mecl_6 = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        FeH = jnp.array([-2.0, -1.0, 0.0, -0.5, -1.5])

        alpha3s = jax.vmap(alpha3_jerabkova_mecl)(log_mecl_6, FeH)
        assert alpha3s.shape == (5,)
        assert jnp.all(jnp.isfinite(alpha3s))

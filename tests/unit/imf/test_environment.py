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
from progenax.imf.environment.mapping import (
    alpha3_marks_table3,
    _alpha3_from_x,
    alpha3_jerabkova_generalized,
    x_jerabkova_generalized,
)
from progenax.imf.environment.coefficients import MARKS_TABLE3_COEFFICIENTS


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
        assert jnp.isclose(c["x_hat_threshold"], -0.87, atol=0.01)  # erratum (was +0.87 typo)
        assert jnp.isclose(c["alpha3_slope"], -0.4072, atol=0.001)
        assert jnp.isclose(c["alpha3_intercept"], 1.9383, atol=0.001)

    def test_threshold_is_negative_erratum(self):
        """CRITICAL: Marks threshold is x_hat >= -0.87 (NEGATIVE), per the 2014
        erratum (Marks et al. 2014, MNRAS 442, 3315). The originally printed +0.87
        in Marks+2012 Eq.14/15 was a missing-minus-sign typo; Fig.6 (p.2252) shows
        the canonical knee at x' ~ -0.87, and the authors used -0.87 in their analysis.
        """
        c = MARKS_COEFFICIENTS
        assert c["x_hat_threshold"] < 0, "Marks threshold must be NEGATIVE (erratum)"
        # the line meets canonical 2.3 continuously at the threshold:
        knee = c["alpha3_slope"] * c["x_hat_threshold"] + c["alpha3_intercept"]
        assert jnp.isclose(knee, 2.3, atol=0.02), "threshold must give a continuous knee"

    def test_high_density_metal_poor(self):
        """High density + metal poor: top-heavy IMF."""
        # FeH = -2, log_rho_6 = 1 (10^7 M_sun/pc^3)
        # x_hat = -0.139*(-2) + 0.990*1 = 0.278 + 0.990 = 1.268 >= -0.87 (on the line)
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
        # x_hat = -0.139*0 + 0.990*(-2) = -1.98 < -0.87 (below threshold -> canonical)
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


# =============================================================================
# Test Marks+2012 Table 3 1D relations (alpha3_marks_table3)
# =============================================================================


class TestMarksTable3:
    """Cover alpha3_marks_table3 relations, the '<' branch, and validation."""

    def test_invalid_relation_raises(self):
        """Unknown relation name raises ValueError listing valid options."""
        with pytest.raises(ValueError, match="Unknown relation"):
            alpha3_marks_table3(jnp.array(0.0), relation="not_a_relation")

    @pytest.mark.parametrize("relation", ["mcl", "mecl", "rho", "feh"])
    def test_all_relations_clip_to_bounds(self, relation):
        """Every relation returns a finite alpha3 in [0.5, 2.3]."""
        # Sweep lambda across a wide range
        lam = jnp.linspace(-3.0, 3.0, 25)
        a3 = alpha3_marks_table3(lam, relation=relation)
        assert a3.shape == lam.shape
        assert jnp.all(jnp.isfinite(a3))
        assert jnp.all(a3 >= 0.5 - 1e-9)
        assert jnp.all(a3 <= 2.3 + 1e-9)

    def test_feh_branch_less_than(self):
        """The '<' branch (feh): lambda < lim varies; lambda > lim stays canonical.

        feh coeffs: p=0.66, q=2.63, lim=-0.5, branch='<'.
        So FeH < -0.5 -> 0.66*FeH + 2.63 (top-heavy at low Z);
           FeH > -0.5 -> canonical 2.3.
        """
        coef = MARKS_TABLE3_COEFFICIENTS["feh"]
        assert coef["branch"] == "<", "feh relation must use the '<' branch"
        p, q, lim = coef["p"], coef["q"], coef["lim"]

        # Below lim (metal-poor): varied value
        feh_lo = jnp.array(-1.5)
        a3_lo = alpha3_marks_table3(feh_lo, relation="feh")
        expected_lo = jnp.clip(p * feh_lo + q, 0.5, 2.3)  # = 1.64
        assert jnp.isclose(a3_lo, expected_lo, atol=1e-6)
        assert float(a3_lo) < 2.3, "metal-poor should be top-heavy (below canonical)"

        # Above lim (metal-rich): canonical 2.3
        feh_hi = jnp.array(0.0)
        a3_hi = alpha3_marks_table3(feh_hi, relation="feh")
        assert jnp.isclose(a3_hi, 2.3, atol=1e-6), "above lim -> canonical Kroupa"

    def test_greater_than_branch_rho(self):
        """The '>' branch (rho): lambda > lim varies; lambda < lim canonical."""
        coef = MARKS_TABLE3_COEFFICIENTS["rho"]
        assert coef["branch"] == ">"
        p, q, lim = coef["p"], coef["q"], coef["lim"]

        # Above lim (dense): varied
        lam_hi = jnp.array(2.0)
        a3_hi = alpha3_marks_table3(lam_hi, relation="rho")
        expected_hi = jnp.clip(p * lam_hi + q, 0.5, 2.3)
        assert jnp.isclose(a3_hi, expected_hi, atol=1e-6)

        # Below lim (diffuse): canonical
        lam_lo = jnp.array(-1.0)
        a3_lo = alpha3_marks_table3(lam_lo, relation="rho")
        assert jnp.isclose(a3_lo, 2.3, atol=1e-6)

    def test_smooth_feh_branch_differentiable(self):
        """smooth=True for the '<' (feh) branch is finite and gradient-friendly."""
        def f(feh):
            return alpha3_marks_table3(feh, relation="feh", smooth=True)

        # Gradient near the transition (lim=-0.5) should be finite and non-zero
        g = jax.grad(f)(jnp.array(-0.5))
        assert jnp.isfinite(g)
        # Smooth value at transition midpoint sits between canonical and varied
        val = f(jnp.array(-0.5))
        assert 0.5 <= float(val) <= 2.3


# =============================================================================
# Test _alpha3_from_x smooth/differentiable path
# =============================================================================


class TestAlpha3FromXSmooth:
    """Cover the smooth (tanh) branch of _alpha3_from_x and its clipping."""

    def test_smooth_is_differentiable_and_finite(self):
        """jax.grad through the smooth tanh transition is finite."""
        def f(x):
            return _alpha3_from_x(
                x, threshold=-0.87, slope=-0.41, intercept=1.94, smooth=True
            )

        for x0 in [-2.0, -0.87, 0.0, 1.0]:
            g = jax.grad(f)(jnp.array(x0))
            assert jnp.isfinite(g), f"non-finite gradient at x={x0}"

    def test_smooth_clipped_to_bounds(self):
        """Smooth output is clipped to [0.5, 2.3] even for extreme x."""
        x = jnp.linspace(-10.0, 20.0, 60)
        a3 = _alpha3_from_x(
            x, threshold=-0.87, slope=-0.41, intercept=1.94, smooth=True
        )
        assert jnp.all(a3 >= 0.5 - 1e-9)
        assert jnp.all(a3 <= 2.3 + 1e-9)

    def test_smooth_approaches_canonical_below_threshold(self):
        """Far below threshold, smooth alpha3 -> canonical 2.3."""
        a3 = _alpha3_from_x(
            jnp.array(-5.0), threshold=-0.87, slope=-0.41, intercept=1.94,
            smooth=True, smooth_width=0.2,
        )
        assert jnp.isclose(a3, 2.3, atol=0.05)


# =============================================================================
# Test env_to_imf_params across ALL model branches
# =============================================================================


class TestEnvToIMFParamsAllModels:
    """Cover every model dispatch branch in env_to_imf_params."""

    @pytest.mark.parametrize(
        "model",
        [
            "jerabkova_rho",
            "marks_plane",
            "marks_rho",
            "marks_mcl",
            "marks_mecl",
            "marks_feh",
        ],
    )
    def test_model_branch_returns_valid_params(self, model):
        """Each model returns IMFParams with alpha3 in [0.5, 2.3] and fixed lows."""
        # Massive, dense, metal-poor cluster exercises top-heavy branches
        env = BirthEnvironment.from_cluster_mass(M_ecl=1e7, FeH=-2.0, sfe=0.1)
        params = env_to_imf_params(env, model=model)

        assert jnp.isfinite(params.alpha3), f"{model}: alpha3 not finite"
        assert 0.5 - 1e-9 <= float(params.alpha3) <= 2.3 + 1e-9, (
            f"{model}: alpha3={float(params.alpha3)} out of [0.5, 2.3]"
        )
        # Without lowmass variation, alpha1/alpha2 stay canonical
        assert jnp.isclose(params.alpha0, 0.3)
        assert jnp.isclose(params.alpha1, 1.3)
        assert jnp.isclose(params.alpha2, 2.3)

    def test_marks_feh_uses_metallicity_only(self):
        """marks_feh ignores mass/density and matches the table3 feh relation."""
        env = BirthEnvironment.from_cluster_mass(M_ecl=1e3, FeH=-1.5)
        params = env_to_imf_params(env, model="marks_feh")
        # Should equal alpha3_marks_table3(FeH=-1.5, 'feh')
        expected = alpha3_marks_table3(jnp.array(-1.5), relation="feh")
        assert jnp.isclose(params.alpha3, expected, atol=1e-6)

    def test_marks_mcl_uses_cloud_mass(self):
        """marks_mcl divides M_ecl by SFE (M_cl) before the table3 lookup."""
        # With high mass + low SFE, M_cl is large -> top-heavy via '>' branch
        env = BirthEnvironment.from_cluster_mass(M_ecl=1e7, FeH=-1.0, sfe=0.05)
        params = env_to_imf_params(env, model="marks_mcl")
        M_ecl = 10.0 ** env.log_mecl
        log_mcl_6 = jnp.log10(M_ecl / env.sfe) - 6.0
        expected = alpha3_marks_table3(log_mcl_6, relation="mcl")
        assert jnp.isclose(params.alpha3, expected, atol=1e-5)

    def test_jerabkova_rho_uses_provided_density(self):
        """jerabkova_rho uses env.log_rho_cl directly when provided (else-branch)."""
        env = BirthEnvironment.from_cluster_mass(M_ecl=1e6, FeH=-1.5, log_rho_cl=1.0)
        params = env_to_imf_params(env, model="jerabkova_rho")
        expected = alpha3_jerabkova_rho(jnp.array(1.0), jnp.array(-1.5))
        assert jnp.isclose(params.alpha3, expected, atol=1e-5)

    def test_marks_plane_uses_provided_density(self):
        """marks_plane uses env.log_rho_cl directly when provided (else-branch)."""
        env = BirthEnvironment.from_cluster_mass(M_ecl=1e6, FeH=-2.0, log_rho_cl=1.0)
        params = env_to_imf_params(env, model="marks_plane")
        expected = alpha3_marks_plane(jnp.array(1.0), jnp.array(-2.0))
        assert jnp.isclose(params.alpha3, expected, atol=1e-5)

    def test_include_lowmass_variation_branch(self):
        """include_lowmass_variation=True applies Marks Eq.12 to alpha1, alpha2."""
        env = BirthEnvironment.from_cluster_mass(M_ecl=1e6, FeH=-2.0)
        params = env_to_imf_params(
            env, model="marks_feh", include_lowmass_variation=True
        )
        a1, a2 = lowmass_slopes_metallicity(jnp.array(-2.0), clamp_FeH=True)
        assert jnp.isclose(params.alpha1, a1, atol=1e-6)
        assert jnp.isclose(params.alpha2, a2, atol=1e-6)
        # Metal-poor -> shallower than canonical 1.3
        assert float(params.alpha1) < 1.3

    def test_clamp_domain_false_vs_true_out_of_range_feh(self):
        """clamp_domain=False passes raw FeH; True clamps to [-2.5, 0.5]."""
        # FeH = -4.0 is below the calibrated floor (-2.5)
        env = BirthEnvironment.from_cluster_mass(M_ecl=1e6, FeH=-4.0)

        params_clamped = env_to_imf_params(
            env, model="marks_feh", clamp_domain=True
        )
        params_raw = env_to_imf_params(
            env, model="marks_feh", clamp_domain=False
        )

        # Clamped uses FeH=-2.5; raw uses FeH=-4.0 -> different alpha3
        a3_at_clamp = alpha3_marks_table3(jnp.array(-2.5), relation="feh")
        a3_at_raw = alpha3_marks_table3(jnp.array(-4.0), relation="feh")
        assert jnp.isclose(params_clamped.alpha3, a3_at_clamp, atol=1e-6)
        assert jnp.isclose(params_raw.alpha3, a3_at_raw, atol=1e-6)
        # They genuinely differ (clamping changed the result)
        assert not jnp.isclose(params_clamped.alpha3, params_raw.alpha3, atol=1e-3)


# =============================================================================
# Environment gradient correctness (FD-vs-autodiff) + SFE-extreme robustness
# =============================================================================


def _central_fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2.0 * h)


def _assert_grad_matches_fd(f, x0, h=1e-5, rtol=1e-4, atol=1e-9):
    """Autodiff grad of f at x0 matches the central FD (and is finite/non-zero)."""
    g = jax.grad(f)(x0)
    g_fd = _central_fd(f, x0, h)
    assert jnp.isfinite(g), f"autodiff grad is {g}"
    assert jnp.abs(g) > 1e-6, f"grad effectively zero ({g}); FD says {g_fd}"
    assert jnp.abs(g - g_fd) <= rtol * jnp.abs(g_fd) + atol, (
        f"autodiff {float(g):.6e} vs FD {float(g_fd):.6e} "
        f"(rel {float(jnp.abs(g - g_fd) / (jnp.abs(g_fd) + 1e-12)):.2e})"
    )


class TestEnvGradients:
    """Autodiff env-parameter gradients match central finite differences.

    Evaluated in a top-heavy regime where the varied branch (slope*x+intercept) is
    active and unclipped, so the gradient is non-zero. Uses smooth=True so the
    tanh-relaxed transition is differentiable everywhere.
    """

    def test_jerabkova_generalized_grad_FeH(self):
        _assert_grad_matches_fd(
            lambda fe: alpha3_jerabkova_generalized(fe, 1e6, 0.33, smooth=True), -1.0
        )

    def test_jerabkova_generalized_grad_sfe(self):
        _assert_grad_matches_fd(
            lambda s: alpha3_jerabkova_generalized(-1.0, 1e6, s, smooth=True), 0.33, h=1e-4
        )

    def test_jerabkova_mecl_grad_logmass(self):
        # Mass dependence via the natural log-mass variable (O(0.1) gradient).
        _assert_grad_matches_fd(
            lambda lm: alpha3_jerabkova_mecl(lm, -1.0, smooth=True), 0.0
        )

    def test_marks_plane_grad_logrho(self):
        _assert_grad_matches_fd(
            lambda r: alpha3_marks_plane(r, -1.5, smooth=True), 1.0
        )

    def test_marks_plane_grad_FeH(self):
        _assert_grad_matches_fd(
            lambda fe: alpha3_marks_plane(1.0, fe, smooth=True), -1.5
        )

    def test_x_jerabkova_grad_FeH(self):
        # x is unconditional (no threshold): grad is exactly the FeH coefficient.
        _assert_grad_matches_fd(
            lambda fe: x_jerabkova_generalized(fe, 1e6, 0.33), -1.0
        )


class TestSFEExtreme:
    """alpha3 stays finite and clipped to [0.5, 2.3] at SFE extremes (no NaN)."""

    @pytest.mark.parametrize("sfe", [1e-6, 1e-3, 0.33, 100.0, 1e6])
    def test_alpha3_finite_and_bounded(self, sfe):
        a3 = alpha3_jerabkova_generalized(jnp.array(-1.0), jnp.array(1e6), jnp.array(sfe))
        assert jnp.isfinite(a3), f"alpha3 non-finite at sfe={sfe}: {a3}"
        assert 0.5 - 1e-9 <= float(a3) <= 2.3 + 1e-9, f"alpha3={float(a3)} out of [0.5,2.3] at sfe={sfe}"

    def test_sfe_extremes_saturate(self):
        # sfe -> 0 => most top-heavy (clips to 0.5); sfe -> inf => canonical (2.3)
        a_low = float(alpha3_jerabkova_generalized(jnp.array(-1.0), jnp.array(1e6), jnp.array(1e-6)))
        a_high = float(alpha3_jerabkova_generalized(jnp.array(-1.0), jnp.array(1e6), jnp.array(1e6)))
        assert a_low == pytest.approx(0.5, abs=1e-6), f"sfe->0 should clip to 0.5, got {a_low}"
        assert a_high == pytest.approx(2.3, abs=1e-6), f"sfe->inf should be canonical 2.3, got {a_high}"

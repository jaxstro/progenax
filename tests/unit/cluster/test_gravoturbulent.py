"""Tests for gravoturbulent f_sub derivation (Burkhart 2018).

This module tests the physics-based f_sub derivation chain:

    (Σ, M) → σ_s → s_crit → f_tail → f_sub

Based on the theory document:
    progenax/docs/core-papers/progenax-gravoturbulent-fdf-theory.md
"""

import pytest
from progenax.cluster.fdf_config import (
    GravoturbulentEnv,
    GravoturbulentResult,
    gravoturbulent_summary,
    tail_layer_from_env,
    env_from_preset,
    GRAVOTURBULENT_PRESETS,
)


class TestGravoturbulentSummary:
    """Test the core f_sub derivation chain.

    These tests verify against worked examples in the theory document
    (Sections 11.1-11.3).
    """

    def test_orion_example(self):
        """Verify worked example from theory document (Section 11.1).

        Orion-like GMC:
        - Σ = 150 M☉/pc²
        - M = 12
        - η_survive = 0.6

        Expected values from the theory document:
        - σ_s = 1.78
        - α_vir = 1.33
        - s_crit = 3.84
        - f_tail = 0.107
        - f_sub = 0.064
        """
        env = GravoturbulentEnv(Sigma=150, Mach=12, eta_survive=0.6)
        result = gravoturbulent_summary(env)

        # Check intermediate values match theory document
        assert abs(result.sigma_s - 1.78) < 0.02, f"σ_s={result.sigma_s}, expected 1.78"
        assert abs(result.alpha_vir - 1.33) < 0.02, f"α_vir={result.alpha_vir}, expected 1.33"
        assert abs(result.s_crit - 3.84) < 0.1, f"s_crit={result.s_crit}, expected 3.84"
        assert abs(result.f_tail - 0.107) < 0.02, f"f_tail={result.f_tail}, expected 0.107"
        assert abs(result.f_sub - 0.064) < 0.01, f"f_sub={result.f_sub}, expected 0.064"

    def test_ymc_example(self):
        """Verify worked example from theory document (Section 11.2).

        YMC-forming clump (like W43):
        - Σ = 1500 M☉/pc²
        - M = 25
        - η_survive = 0.85

        Expected values from the theory document:
        - σ_s = 2.15
        - α_vir = 0.133
        - s_crit = 3.00
        - f_tail = 0.372
        - f_sub = 0.316
        """
        env = GravoturbulentEnv(Sigma=1500, Mach=25, eta_survive=0.85)
        result = gravoturbulent_summary(env)

        assert abs(result.sigma_s - 2.15) < 0.02, f"σ_s={result.sigma_s}, expected 2.15"
        assert abs(result.alpha_vir - 0.133) < 0.02, f"α_vir={result.alpha_vir}, expected 0.133"
        # s_crit tolerance is wider due to log sensitivity
        assert abs(result.s_crit - 3.00) < 0.15, f"s_crit={result.s_crit}, expected 3.00"
        assert abs(result.f_tail - 0.372) < 0.03, f"f_tail={result.f_tail}, expected 0.372"
        assert abs(result.f_sub - 0.316) < 0.03, f"f_sub={result.f_sub}, expected 0.316"

    def test_taurus_example(self):
        """Verify worked example from theory document (Section 11.3).

        Diffuse Taurus-like cloud:
        - Σ = 40 M☉/pc²
        - M = 6
        - η_survive = 0.4

        Expected values from the theory document:
        - σ_s = 1.38
        - α_vir = 5.0
        - s_crit = 3.77
        - f_tail = 0.021
        - f_sub = 0.008
        """
        env = GravoturbulentEnv(Sigma=40, Mach=6, eta_survive=0.4)
        result = gravoturbulent_summary(env)

        assert abs(result.sigma_s - 1.38) < 0.02, f"σ_s={result.sigma_s}, expected 1.38"
        assert abs(result.alpha_vir - 5.0) < 0.1, f"α_vir={result.alpha_vir}, expected 5.0"
        assert abs(result.s_crit - 3.77) < 0.1, f"s_crit={result.s_crit}, expected 3.77"
        assert abs(result.f_tail - 0.021) < 0.005, f"f_tail={result.f_tail}, expected 0.021"
        assert abs(result.f_sub - 0.008) < 0.002, f"f_sub={result.f_sub}, expected 0.008"

    def test_result_type(self):
        """Result should be a GravoturbulentResult dataclass."""
        env = GravoturbulentEnv(Sigma=100, Mach=10, eta_survive=0.5)
        result = gravoturbulent_summary(env)

        assert isinstance(result, GravoturbulentResult)
        assert hasattr(result, "sigma_s")
        assert hasattr(result, "alpha_vir")
        assert hasattr(result, "s_crit")
        assert hasattr(result, "u_crit")
        assert hasattr(result, "f_tail")
        assert hasattr(result, "f_sub")


class TestMonotonicity:
    """Test that f_sub behaves monotonically with environment parameters.

    These tests verify the physical behavior of the model:
    - Higher Σ → lower α_vir → more gas in collapsing tail → higher f_sub
    - Higher η_survive → higher f_sub (linear relationship)
    """

    def test_f_sub_increases_with_Sigma(self):
        """Higher Σ → lower α_vir → higher f_tail → higher f_sub.

        This is the key physical prediction: denser clouds produce
        more substructured clusters.
        """
        Sigmas = [50, 100, 300, 1000, 3000]
        f_subs = []

        for Sigma in Sigmas:
            env = GravoturbulentEnv(Sigma=Sigma, Mach=15, eta_survive=0.7)
            result = gravoturbulent_summary(env)
            f_subs.append(result.f_sub)

        # f_sub should strictly increase with Σ
        for i in range(len(f_subs) - 1):
            assert f_subs[i] < f_subs[i + 1], (
                f"f_sub not monotonic with Σ: f_sub({Sigmas[i]})={f_subs[i]:.4f} >= "
                f"f_sub({Sigmas[i+1]})={f_subs[i+1]:.4f}"
            )

    def test_f_sub_increases_with_eta_survive(self):
        """Higher η_survive → higher f_sub (linear scaling).

        f_sub = η_survive × f_tail, so f_tail should be constant
        and f_sub should scale linearly with η_survive.
        """
        env_low = GravoturbulentEnv(Sigma=500, Mach=15, eta_survive=0.3)
        env_high = GravoturbulentEnv(Sigma=500, Mach=15, eta_survive=0.9)

        result_low = gravoturbulent_summary(env_low)
        result_high = gravoturbulent_summary(env_high)

        # f_tail should be identical (same Σ, M)
        assert abs(result_low.f_tail - result_high.f_tail) < 1e-6, (
            f"f_tail differs: {result_low.f_tail} vs {result_high.f_tail}"
        )

        # f_sub should scale with η_survive (0.9 / 0.3 = 3)
        ratio = result_high.f_sub / result_low.f_sub
        assert ratio == pytest.approx(3.0, rel=0.01), (
            f"f_sub ratio={ratio}, expected 3.0 (linear in η_survive)"
        )

    def test_alpha_vir_decreases_with_Sigma(self):
        """Higher Σ → lower α_vir (virial parameter scaling)."""
        env_low_Sigma = GravoturbulentEnv(Sigma=50, Mach=10, eta_survive=0.5)
        env_high_Sigma = GravoturbulentEnv(Sigma=500, Mach=10, eta_survive=0.5)

        result_low = gravoturbulent_summary(env_low_Sigma)
        result_high = gravoturbulent_summary(env_high_Sigma)

        # α_vir = α₀ × (Σ₀/Σ), so 10× higher Σ → 10× lower α_vir
        assert result_high.alpha_vir < result_low.alpha_vir
        ratio = result_low.alpha_vir / result_high.alpha_vir
        assert ratio == pytest.approx(10.0, rel=0.01), f"α_vir ratio={ratio}, expected 10.0"


class TestPresets:
    """Test preset environments from the theory document."""

    def test_all_presets_exist(self):
        """All documented presets should be loadable."""
        expected_presets = ["taurus", "orion", "typical_gmc", "dense_gmc", "ymc_precursor", "starburst"]

        for name in expected_presets:
            env = env_from_preset(name)
            assert isinstance(env, GravoturbulentEnv), f"Preset '{name}' not a GravoturbulentEnv"

    def test_presets_span_f_sub_range(self):
        """Presets should cover a wide dynamic range of f_sub.

        According to the theory document, f_sub should span ~40× from
        Taurus-like to YMC environments.
        """
        f_subs = {}
        for name in GRAVOTURBULENT_PRESETS:
            env = GRAVOTURBULENT_PRESETS[name]
            result = gravoturbulent_summary(env)
            f_subs[name] = result.f_sub

        # Should span at least an order of magnitude
        ratio = max(f_subs.values()) / min(f_subs.values())
        assert ratio > 10, f"f_sub range too narrow: max/min = {ratio:.1f}"

        # Print for reference (not a test assertion)
        # print(f"\nPreset f_sub values:")
        # for name, f in sorted(f_subs.items(), key=lambda x: x[1]):
        #     print(f"  {name}: f_sub = {f:.4f}")

    def test_preset_case_insensitive(self):
        """env_from_preset should be case-insensitive."""
        env1 = env_from_preset("orion")
        env2 = env_from_preset("ORION")
        env3 = env_from_preset("Orion")

        assert env1.Sigma == env2.Sigma == env3.Sigma
        assert env1.Mach == env2.Mach == env3.Mach

    def test_unknown_preset_raises(self):
        """Unknown preset name should raise KeyError with helpful message."""
        with pytest.raises(KeyError) as exc_info:
            env_from_preset("unknown_cloud_type")

        assert "unknown_cloud_type" in str(exc_info.value).lower()


class TestTailLayerFromEnv:
    """Test TailSubstructureLayer creation from GravoturbulentEnv."""

    def test_mode_is_gravoturbulent(self):
        """Layer created from env should have mode='gravoturbulent'."""
        env = GravoturbulentEnv(Sigma=500, Mach=15, eta_survive=0.7)
        tail = tail_layer_from_env(env)

        assert tail.mode == "gravoturbulent"

    def test_env_is_stored(self):
        """Layer should store reference to the source environment."""
        env = GravoturbulentEnv(Sigma=500, Mach=15, eta_survive=0.7)
        tail = tail_layer_from_env(env)

        assert tail.env is env

    def test_result_is_stored(self):
        """Layer should store the full derivation result."""
        env = GravoturbulentEnv(Sigma=500, Mach=15, eta_survive=0.7)
        tail = tail_layer_from_env(env)

        assert tail.result is not None
        assert isinstance(tail.result, GravoturbulentResult)

    def test_f_sub_matches_result(self):
        """Layer f_sub should match the result f_sub."""
        env = GravoturbulentEnv(Sigma=500, Mach=15, eta_survive=0.7)
        tail = tail_layer_from_env(env)

        assert tail.f_sub == tail.result.f_sub

    def test_f_sub_in_valid_range(self):
        """f_sub should always be in [0, 1]."""
        for name in GRAVOTURBULENT_PRESETS:
            env = GRAVOTURBULENT_PRESETS[name]
            tail = tail_layer_from_env(env)

            assert 0 <= tail.f_sub <= 1, f"f_sub={tail.f_sub} out of range for preset '{name}'"


class TestPhysicalConstraints:
    """Test physical constraints and edge cases."""

    def test_f_tail_bounded_by_one(self):
        """f_tail cannot exceed 1 (it's a mass fraction)."""
        # Extreme starburst environment
        env = GravoturbulentEnv(Sigma=10000, Mach=50, eta_survive=1.0)
        result = gravoturbulent_summary(env)

        assert result.f_tail <= 1.0, f"f_tail={result.f_tail} > 1"
        assert result.f_sub <= 1.0, f"f_sub={result.f_sub} > 1"

    def test_f_tail_positive(self):
        """f_tail must be positive."""
        # Very diffuse cloud with high α_vir
        env = GravoturbulentEnv(Sigma=10, Mach=3, eta_survive=0.1)
        result = gravoturbulent_summary(env)

        assert result.f_tail >= 0, f"f_tail={result.f_tail} < 0"
        assert result.f_sub >= 0, f"f_sub={result.f_sub} < 0"

    def test_eta_survive_zero_gives_zero_f_sub(self):
        """If η_survive=0, f_sub should be 0 (no stars survive feedback)."""
        env = GravoturbulentEnv(Sigma=500, Mach=15, eta_survive=0.0)
        result = gravoturbulent_summary(env)

        assert result.f_tail > 0, "f_tail should be positive even with η=0"
        assert result.f_sub == 0.0, f"f_sub={result.f_sub}, expected 0.0 when η=0"

    def test_custom_parameters(self):
        """Test that custom b, phi_x, alpha_0 work correctly."""
        env_default = GravoturbulentEnv(Sigma=200, Mach=12, eta_survive=0.6)
        env_custom = GravoturbulentEnv(
            Sigma=200, Mach=12, eta_survive=0.6,
            b=0.33,  # Solenoidal (narrower PDF)
            phi_x=0.5,  # Larger sonic scale
            alpha_0=3.0,  # Higher reference virial parameter
        )

        result_default = gravoturbulent_summary(env_default)
        result_custom = gravoturbulent_summary(env_custom)

        # With b=0.33 (vs 0.4), σ_s should be smaller (narrower PDF)
        assert result_custom.sigma_s < result_default.sigma_s

        # With alpha_0=3.0 (vs 2.0), α_vir should be larger
        assert result_custom.alpha_vir > result_default.alpha_vir

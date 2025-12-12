"""Tests for gravoturbulent f_sub derivation.

This module tests the physics-based f_sub derivation using both BM19 and PN11:
- BM19 (default): s_t = (α - 0.5) × σ_s²
- PN11 (alternative): s_crit = ln((π²φ_x²/5) × α_vir × M²)

Based on the theory document:
    progenax/docs/core-papers/progenax-gravoturbulent-fdf-theory.md
"""

import pytest
from progenax.cluster.gravoturbulent import (
    GravoturbulentEnv,
    tail_layer_from_env,
    env_from_preset,
    GRAVOTURBULENT_PRESETS,
)
from progenax.gravoturb import bm19_pipeline, pn11_pipeline, BM19Result, PN11Result


class TestPN11Summary:
    """Test the PN11 f_sub derivation chain.

    These tests verify against worked examples in the theory document
    (Sections 11.1-11.3) using the PN11 model.
    """

    def test_orion_example(self):
        """Verify Orion-like GMC example.

        Orion-like GMC:
        - Σ = 150 M☉/pc²
        - M = 12
        - η_survive = 0.6

        With default Σ₀=85 (Heyer & Dame 2015):
        - α_vir = 2.0 × (85/150) = 1.13
        - σ_s ≈ 1.78
        """
        result = pn11_pipeline(mach=12, Sigma=150, eta_survive=0.6)

        assert abs(result.sigma_s - 1.78) < 0.02, f"σ_s={result.sigma_s}, expected 1.78"
        assert abs(result.alpha_vir - 1.13) < 0.05, f"α_vir={result.alpha_vir}, expected 1.13"
        assert result.s_crit > 3.0, f"s_crit={result.s_crit} should be > 3"
        assert 0.05 < result.f_dense < 0.20, f"f_dense={result.f_dense} out of expected range"
        assert 0.03 < result.f_sub < 0.15, f"f_sub={result.f_sub} out of expected range"

    def test_ymc_example(self):
        """Verify YMC-forming clump example.

        YMC-forming clump (like W43):
        - Σ = 1500 M☉/pc²
        - M = 25
        - η_survive = 0.85

        With default Σ₀=85:
        - α_vir = 2.0 × (85/1500) ≈ 0.113
        """
        result = pn11_pipeline(mach=25, Sigma=1500, eta_survive=0.85)

        assert abs(result.sigma_s - 2.15) < 0.02, f"σ_s={result.sigma_s}, expected 2.15"
        assert abs(result.alpha_vir - 0.113) < 0.02, f"α_vir={result.alpha_vir}, expected 0.113"
        assert 2.5 < result.s_crit < 3.5, f"s_crit={result.s_crit} out of expected range"
        assert 0.3 < result.f_dense < 0.5, f"f_dense={result.f_dense} out of expected range"
        assert 0.25 < result.f_sub < 0.45, f"f_sub={result.f_sub} out of expected range"

    def test_taurus_example(self):
        """Verify Taurus-like diffuse cloud example.

        Diffuse Taurus-like cloud:
        - Σ = 40 M☉/pc²
        - M = 6
        - η_survive = 0.4

        With default Σ₀=85:
        - α_vir = 2.0 × (85/40) = 4.25
        """
        result = pn11_pipeline(mach=6, Sigma=40, eta_survive=0.4)

        assert abs(result.sigma_s - 1.38) < 0.02, f"σ_s={result.sigma_s}, expected 1.38"
        assert abs(result.alpha_vir - 4.25) < 0.1, f"α_vir={result.alpha_vir}, expected 4.25"
        assert 3.0 < result.s_crit < 4.5, f"s_crit={result.s_crit} out of expected range"
        assert 0.01 < result.f_dense < 0.05, f"f_dense={result.f_dense} out of expected range"
        assert 0.005 < result.f_sub < 0.02, f"f_sub={result.f_sub} out of expected range"

    def test_result_type(self):
        """Result should be a PN11Result namedtuple."""
        result = pn11_pipeline(mach=10, Sigma=100)

        assert isinstance(result, PN11Result)
        assert hasattr(result, "sigma_s")
        assert hasattr(result, "sigma_s_sq")
        assert hasattr(result, "alpha_vir")
        assert hasattr(result, "s_crit")
        assert hasattr(result, "f_dense")
        assert hasattr(result, "f_sub")


class TestBM19Summary:
    """Test the BM19 f_sub derivation (default model)."""

    def test_result_type(self):
        """Result should be a BM19Result namedtuple."""
        result = bm19_pipeline(mach=10, alpha=2.0)

        assert isinstance(result, BM19Result)
        assert hasattr(result, "sigma_s")
        assert hasattr(result, "s_t")
        assert hasattr(result, "f_dense")
        assert hasattr(result, "f_sub")

    def test_f_sub_positive(self):
        """f_sub should always be positive."""
        result = bm19_pipeline(mach=10, alpha=2.0, eta_survive=0.5)
        assert result.f_sub > 0

    def test_f_sub_bounded(self):
        """f_sub should be bounded in [0, 1]."""
        # High Mach, high α
        result = bm19_pipeline(mach=30, alpha=2.5, eta_survive=1.0)
        assert 0 <= result.f_sub <= 1


class TestMonotonicity:
    """Test that f_sub behaves monotonically with environment parameters.

    These tests verify the physical behavior of the PN11 model:
    - Higher Σ → lower α_vir → more gas in collapsing tail → higher f_sub
    - Higher η_survive → higher f_sub (linear relationship)
    """

    def test_f_sub_increases_with_Sigma(self):
        """Higher Σ → lower α_vir → higher f_dense → higher f_sub."""
        Sigmas = [50, 100, 300, 1000, 3000]
        f_subs = []

        for Sigma in Sigmas:
            result = pn11_pipeline(mach=15, Sigma=Sigma, eta_survive=0.7)
            f_subs.append(result.f_sub)

        # f_sub should strictly increase with Σ
        for i in range(len(f_subs) - 1):
            assert f_subs[i] < f_subs[i + 1], (
                f"f_sub not monotonic with Σ: f_sub({Sigmas[i]})={f_subs[i]:.4f} >= "
                f"f_sub({Sigmas[i+1]})={f_subs[i+1]:.4f}"
            )

    def test_f_sub_increases_with_eta_survive(self):
        """Higher η_survive → higher f_sub (linear scaling)."""
        result_low = pn11_pipeline(mach=15, Sigma=500, eta_survive=0.3)
        result_high = pn11_pipeline(mach=15, Sigma=500, eta_survive=0.9)

        # f_dense should be identical (same Σ, M)
        assert abs(result_low.f_dense - result_high.f_dense) < 1e-6, (
            f"f_dense differs: {result_low.f_dense} vs {result_high.f_dense}"
        )

        # f_sub should scale with η_survive (0.9 / 0.3 = 3)
        ratio = result_high.f_sub / result_low.f_sub
        assert ratio == pytest.approx(3.0, rel=0.01), (
            f"f_sub ratio={ratio}, expected 3.0 (linear in η_survive)"
        )

    def test_alpha_vir_decreases_with_Sigma(self):
        """Higher Σ → lower α_vir (virial parameter scaling)."""
        result_low = pn11_pipeline(mach=10, Sigma=50)
        result_high = pn11_pipeline(mach=10, Sigma=500)

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

        Using PN11 which depends on Σ (surface density), giving wider range.
        BM19 only depends on Mach, so range is narrower.
        """
        f_subs = {}
        for name in GRAVOTURBULENT_PRESETS:
            env = GRAVOTURBULENT_PRESETS[name]
            # Use PN11 which varies with both Σ and M
            result = pn11_pipeline(
                mach=env.Mach,
                Sigma=env.Sigma,
                b=env.b,
                eta_survive=env.eta_survive,
            )
            f_subs[name] = result.f_sub

        # Should span at least a factor of 10
        ratio = max(f_subs.values()) / min(f_subs.values())
        assert ratio > 10, f"f_sub range too narrow: max/min = {ratio:.1f}"

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

    def test_mode_is_bm19_default(self):
        """Layer created from env should have mode='bm19' by default."""
        env = GravoturbulentEnv(Sigma=500, Mach=15, eta_survive=0.7)
        tail = tail_layer_from_env(env)

        assert tail.mode == "bm19"

    def test_mode_pn11_when_requested(self):
        """Layer with mode='pn11' when requested."""
        env = GravoturbulentEnv(Sigma=500, Mach=15, eta_survive=0.7)
        tail = tail_layer_from_env(env, model="pn11")

        assert tail.mode == "pn11"

    def test_env_is_stored(self):
        """Layer should store reference to the source environment."""
        env = GravoturbulentEnv(Sigma=500, Mach=15, eta_survive=0.7)
        tail = tail_layer_from_env(env)

        assert tail.env is env

    def test_result_bm19_stored(self):
        """BM19 mode should store BM19Result."""
        env = GravoturbulentEnv(Sigma=500, Mach=15, eta_survive=0.7)
        tail = tail_layer_from_env(env, model="bm19")

        assert tail.result is not None
        assert isinstance(tail.result, BM19Result)

    def test_result_pn11_stored(self):
        """PN11 mode should store PN11Result."""
        env = GravoturbulentEnv(Sigma=500, Mach=15, eta_survive=0.7)
        tail = tail_layer_from_env(env, model="pn11")

        assert tail.result is not None
        assert isinstance(tail.result, PN11Result)

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

    def test_f_dense_bounded_by_one(self):
        """f_dense cannot exceed 1 (it's a mass fraction)."""
        # Extreme starburst environment
        result = pn11_pipeline(mach=50, Sigma=10000, eta_survive=1.0)

        assert result.f_dense <= 1.0, f"f_dense={result.f_dense} > 1"
        assert result.f_sub <= 1.0, f"f_sub={result.f_sub} > 1"

    def test_f_dense_positive(self):
        """f_dense must be positive."""
        # Very diffuse cloud with high α_vir
        result = pn11_pipeline(mach=3, Sigma=10, eta_survive=0.1)

        assert result.f_dense >= 0, f"f_dense={result.f_dense} < 0"
        assert result.f_sub >= 0, f"f_sub={result.f_sub} < 0"

    def test_eta_survive_zero_gives_zero_f_sub(self):
        """If η_survive=0, f_sub should be 0."""
        result = pn11_pipeline(mach=15, Sigma=500, eta_survive=0.0)

        assert result.f_dense > 0, "f_dense should be positive even with η=0"
        assert result.f_sub == 0.0, f"f_sub={result.f_sub}, expected 0.0 when η=0"

    def test_custom_pn11_parameters(self):
        """Test that custom b, phi_x work correctly in PN11."""
        result_default = pn11_pipeline(mach=12, Sigma=200, eta_survive=0.6)
        result_custom = pn11_pipeline(
            mach=12, Sigma=200, eta_survive=0.6,
            b=0.33,  # Solenoidal (narrower PDF)
            phi_x=0.5,  # Larger sonic scale
        )

        # With b=0.33 (vs 0.4), σ_s should be smaller (narrower PDF)
        assert result_custom.sigma_s < result_default.sigma_s

        # Both should return valid results
        assert 0 < result_default.f_sub < 1
        assert 0 < result_custom.f_sub < 1

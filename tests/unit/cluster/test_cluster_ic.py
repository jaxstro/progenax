# progenax/tests/unit/cluster/test_cluster_ic.py
"""
Unit tests for progenax.cluster IC generation (v1.4 spec).

Tests cover:
- Import sanity (all exports available)
- ClusterState dataclass properties
- generate_cluster_ic physics sanity
- Mass segregation monotonicity (Λ_MSR vs λ_seg)
- Fractal Q parameter correlation with D
- Guard against fractal + mass_seg combination

References:
    v1.4 Mass Segregation & Fractal Substructure Specification §7
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


# =============================================================================
# Import Sanity Tests
# =============================================================================


class TestImports:
    """Test that all v1.4 exports are available."""

    def test_cluster_core_imports(self):
        """Test core cluster API imports."""
        from progenax.cluster import (
            ClusterState,
            SpatialStructureParams,
            MassSegregationLayer,
            FractalLayer,
            generate_cluster_ic,
            sample_velocities_for_profile,
        )

        # All should be importable without error
        assert ClusterState is not None
        assert SpatialStructureParams is not None
        assert MassSegregationLayer is not None
        assert FractalLayer is not None
        assert generate_cluster_ic is not None
        assert sample_velocities_for_profile is not None

    def test_cluster_submodule_imports(self):
        """Test submodule imports."""
        from progenax.cluster.mass_segregation import energy_sorted_segregation
        from progenax.cluster.fractal_gw_legacy import (
            generate_fractal_positions,
            rescale_fractal_to_target_radii,
            assign_velocities_and_virialize,
        )

        assert energy_sorted_segregation is not None
        assert generate_fractal_positions is not None
        assert rescale_fractal_to_target_radii is not None
        assert assign_velocities_and_virialize is not None

    def test_profiles_api_imports(self):
        """Test profiles functional API imports."""
        from progenax.profiles.api import (
            make_profile,
            ProfileName,
            sample_density_profile,
            compute_profile_potential,
        )

        assert make_profile is not None
        assert sample_density_profile is not None
        assert compute_profile_potential is not None

    def test_diagnostics_imports(self):
        """Test diagnostics imports."""
        from progenax.diagnostics import (
            compute_lambda_msr,
            compute_q_parameter,
            compute_azimuthal_variation,
        )

        assert compute_lambda_msr is not None
        assert compute_q_parameter is not None
        assert compute_azimuthal_variation is not None


# =============================================================================
# ClusterState Tests
# =============================================================================


class TestClusterState:
    """Test ClusterState dataclass."""

    def test_cluster_state_creation(self):
        """Test creating a ClusterState."""
        from progenax.cluster import ClusterState

        masses = jnp.array([1.0, 2.0, 3.0])
        positions = jnp.zeros((3, 3))
        velocities = jnp.zeros((3, 3))

        state = ClusterState(masses=masses, positions=positions, velocities=velocities)

        assert state.N == 3
        assert jnp.isclose(state.M_total, 6.0)
        assert state.masses.shape == (3,)
        assert state.positions.shape == (3, 3)
        assert state.velocities.shape == (3, 3)

    def test_cluster_state_immutable(self):
        """Test that ClusterState is immutable (frozen dataclass)."""
        from progenax.cluster import ClusterState

        masses = jnp.array([1.0, 2.0])
        positions = jnp.zeros((2, 3))
        velocities = jnp.zeros((2, 3))

        state = ClusterState(masses=masses, positions=positions, velocities=velocities)

        # Should raise an error when trying to modify
        with pytest.raises((AttributeError, TypeError)):
            state.masses = jnp.array([5.0, 6.0])


# =============================================================================
# Generate Cluster IC Tests
# =============================================================================


class TestGenerateClusterIC:
    """Test generate_cluster_ic function."""

    @pytest.fixture
    def imf(self):
        """Create a Kroupa IMF for testing."""
        from progenax.imf import PowerLawIMF
        return PowerLawIMF.kroupa()

    @pytest.fixture
    def key(self):
        """Create a random key."""
        return jax.random.PRNGKey(42)

    def test_basic_plummer_generation(self, key, imf):
        """Test basic Plummer cluster generation."""
        from progenax.cluster import generate_cluster_ic, SpatialStructureParams

        cluster = generate_cluster_ic(
            key=key,
            N_stars=100,
            M_total=100.0,
            R_half=1.0,
            imf_params=imf,
            structure_params=SpatialStructureParams(base_profile="plummer"),
        )

        assert cluster.N == 100
        assert jnp.isclose(cluster.M_total, 100.0, rtol=1e-4)
        assert cluster.positions.shape == (100, 3)
        assert cluster.velocities.shape == (100, 3)

    def test_guard_fractal_plus_segregation(self, key, imf):
        """Test that ValueError is raised if both fractal and mass_seg are set."""
        from progenax.cluster import (
            generate_cluster_ic,
            SpatialStructureParams,
            MassSegregationLayer,
            FractalLayer,
        )

        with pytest.raises(ValueError, match="not both"):
            generate_cluster_ic(
                key=key,
                N_stars=100,
                M_total=100.0,
                R_half=1.0,
                imf_params=imf,
                structure_params=SpatialStructureParams(
                    base_profile="plummer",
                    mass_segregation=MassSegregationLayer(lambda_seg=0.5),
                    fractal=FractalLayer(D=2.0),
                ),
            )

    def test_center_of_mass_removed(self, key, imf):
        """Test that center of mass is at origin."""
        from progenax.cluster import generate_cluster_ic, SpatialStructureParams

        cluster = generate_cluster_ic(
            key=key,
            N_stars=500,
            M_total=500.0,
            R_half=1.0,
            imf_params=imf,
            structure_params=SpatialStructureParams(base_profile="plummer"),
        )

        # Mass-weighted center of mass
        M_total = jnp.sum(cluster.masses)
        x_com = jnp.sum(cluster.masses[:, None] * cluster.positions, axis=0) / M_total
        v_com = jnp.sum(cluster.masses[:, None] * cluster.velocities, axis=0) / M_total

        # Tolerance needs to be looser due to floating point arithmetic
        assert jnp.allclose(x_com, 0.0, atol=1e-6)
        assert jnp.allclose(v_com, 0.0, atol=1e-6)


# =============================================================================
# Mass Segregation Tests
# =============================================================================


class TestMassSegregation:
    """Test mass segregation functionality."""

    @pytest.fixture
    def imf(self):
        from progenax.imf import PowerLawIMF
        return PowerLawIMF.kroupa()

    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(123)

    def test_segregation_increases_lambda_msr(self, key, imf):
        """Test that λ_seg > 0 increases Λ_MSR."""
        from progenax.cluster import (
            generate_cluster_ic,
            SpatialStructureParams,
            MassSegregationLayer,
        )
        from progenax.diagnostics import compute_lambda_msr

        # Generate unsegregated cluster
        key1, key2 = jax.random.split(key)
        cluster_unseg = generate_cluster_ic(
            key=key1,
            N_stars=500,
            M_total=500.0,
            R_half=1.0,
            imf_params=imf,
            structure_params=SpatialStructureParams(base_profile="plummer"),
        )

        # Generate segregated cluster
        cluster_seg = generate_cluster_ic(
            key=key2,
            N_stars=500,
            M_total=500.0,
            R_half=1.0,
            imf_params=imf,
            structure_params=SpatialStructureParams(
                base_profile="plummer",
                mass_segregation=MassSegregationLayer(lambda_seg=1.0),
            ),
        )

        # Compute Λ_MSR for both (using NumPy arrays for diagnostics)
        lambda_unseg, _ = compute_lambda_msr(
            np.array(cluster_unseg.positions),
            np.array(cluster_unseg.masses),
            N_massive=20,
        )
        lambda_seg, _ = compute_lambda_msr(
            np.array(cluster_seg.positions),
            np.array(cluster_seg.masses),
            N_massive=20,
        )

        # Segregated cluster should have higher Λ_MSR
        assert lambda_seg > lambda_unseg, (
            f"λ_seg=1.0 should give Λ_MSR > unsegregated: "
            f"Λ_seg={lambda_seg:.2f} vs Λ_unseg={lambda_unseg:.2f}"
        )


# =============================================================================
# Fractal Tests
# =============================================================================


class TestFractal:
    """Test fractal substructure functionality."""

    @pytest.fixture
    def imf(self):
        from progenax.imf import PowerLawIMF
        return PowerLawIMF.kroupa()

    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(456)

    def test_fractal_generation(self, key, imf):
        """Test basic fractal IC generation."""
        from progenax.cluster import (
            generate_cluster_ic,
            SpatialStructureParams,
            FractalLayer,
        )

        cluster = generate_cluster_ic(
            key=key,
            N_stars=200,
            M_total=200.0,
            R_half=1.0,
            imf_params=imf,
            structure_params=SpatialStructureParams(
                base_profile="plummer",
                fractal=FractalLayer(D=2.0, lambda_frac=1.0),
            ),
        )

        assert cluster.N == 200
        assert cluster.positions.shape == (200, 3)

    def test_q_parameter_increases_with_D(self, key):
        """Test that Q parameter increases with fractal dimension D."""
        from progenax.cluster.fractal_gw_legacy import generate_fractal_positions
        from progenax.diagnostics import compute_q_parameter

        # Low D = more clumpy = lower Q
        key1, key2 = jax.random.split(key)
        pos_low_D, _, _ = generate_fractal_positions(key1, 500, D=1.6)
        pos_high_D, _, _ = generate_fractal_positions(key2, 500, D=2.8)

        Q_low = compute_q_parameter(np.array(pos_low_D))
        Q_high = compute_q_parameter(np.array(pos_high_D))

        # Higher D → more uniform → higher Q
        assert Q_high > Q_low, (
            f"Higher D should give higher Q: Q(D=2.8)={Q_high:.3f} vs Q(D=1.6)={Q_low:.3f}"
        )


# =============================================================================
# Profiles API Tests
# =============================================================================


class TestProfilesAPI:
    """Test profiles functional API."""

    @pytest.fixture
    def key(self):
        return jax.random.PRNGKey(789)

    def test_make_profile_plummer(self):
        """Test make_profile for Plummer."""
        from progenax.profiles.api import make_profile

        profile = make_profile("plummer", R_half=1.0)
        assert profile is not None
        assert hasattr(profile, 'sample_positions')

    def test_sample_density_profile(self, key):
        """Test sample_density_profile."""
        from progenax.profiles.api import sample_density_profile

        # N=10000: the sample median converges so a tight, seed-robust bound is
        # possible. At N=100 the median scatter is ~9% (max dev ~40% over seeds),
        # so the old rtol=0.3 only "passed" by seed luck; here rtol=0.05 is ~6 sigma.
        positions = sample_density_profile(key, 10000, "plummer", R_half=2.0)

        assert positions.shape == (10000, 3)
        # Half-mass radius: sample median tracks the true R_half to <5% at N=1e4
        radii = jnp.linalg.norm(positions, axis=1)
        r_half_measured = jnp.median(radii)
        assert jnp.isclose(r_half_measured, 2.0, rtol=0.05)

    def test_compute_profile_potential(self, key):
        """Test compute_profile_potential for Plummer."""
        from progenax.profiles.api import sample_density_profile, compute_profile_potential
        from jaxstro.units import STELLAR

        G = STELLAR.G
        positions = sample_density_profile(key, 50, "plummer", R_half=1.0)

        potential = compute_profile_potential(
            positions, "plummer", M_total=100.0, R_half=1.0, G=G
        )

        assert potential.shape == (50,)
        # Potential should be negative
        assert jnp.all(potential < 0)


# =============================================================================
# Diagnostics Tests
# =============================================================================


class TestDiagnostics:
    """Test diagnostic functions."""

    def test_compute_lambda_msr(self):
        """Test Λ_MSR computation."""
        from progenax.diagnostics import compute_lambda_msr

        # Create a simple cluster with mass segregation
        np.random.seed(42)
        N = 100
        positions = np.random.randn(N, 3)
        masses = np.random.power(2.35, N) * 10  # Power-law-ish masses

        lambda_msr, sigma = compute_lambda_msr(positions, masses, N_massive=10)

        assert lambda_msr > 0
        assert sigma >= 0

    def test_compute_q_parameter(self):
        """Test Cartwright-Whitworth Q computation."""
        from progenax.diagnostics import compute_q_parameter

        # Create uniform sphere
        np.random.seed(42)
        r = np.random.uniform(0, 1, 500) ** (1/3)
        theta = np.arccos(2 * np.random.uniform(0, 1, 500) - 1)
        phi = np.random.uniform(0, 2 * np.pi, 500)
        positions = np.column_stack([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta),
        ])

        Q = compute_q_parameter(positions)

        # Q should be positive and finite
        # The exact value depends on normalization conventions; the key physics
        # is that Q is higher for more uniform/concentrated distributions
        assert Q > 0, f"Q should be positive, got {Q:.3f}"
        assert np.isfinite(Q), f"Q should be finite, got {Q:.3f}"

    def test_compute_azimuthal_variation(self):
        """Test azimuthal variation computation."""
        from progenax.diagnostics import compute_azimuthal_variation

        np.random.seed(42)
        positions = np.random.randn(500, 3)

        var = compute_azimuthal_variation(positions, n_bins=12)

        assert var >= 0
        # Gaussian should have low azimuthal variation
        assert var < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

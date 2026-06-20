"""Baseline Q parameter tests for different spatial profiles.

These tests verify that our Q parameter computation gives expected values
for standard distributions, anchoring our implementation to CW04.

Key baselines:
    - Uniform sphere: Q ≈ 0.79 (CW04 '3D0')
    - Plummer sphere: Q >> 1 (centrally concentrated, outside CW04 calibration range)
    - FDF + uniform ≈ bare uniform (FDF doesn't break baseline)

References:
    Cartwright & Whitworth (2004) MNRAS 348, 589 - Table 1
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progenax.diagnostics.substructure import compute_q_parameter
from progenax.profiles.uniform import UniformSphereProfile


class TestUniformSphereBaseline:
    """Test Q for uniform sphere matches CW04 '3D0'."""

    def test_q_matches_cw04_range(self):
        """Uniform sphere Q should be in CW04's 0.79 ± 0.04 range.

        CW04 Table 1 gives Q ≈ 0.79 for '3D0' (uniform sphere).
        We accept slightly wider range (0.75-0.90) due to:
            - Different convex hull algorithms
            - Finite N effects
            - Monte Carlo variance
        """
        profile = UniformSphereProfile(R=1.0)

        Q_values = []
        for seed in range(10):
            key = jax.random.PRNGKey(seed)
            masses = jnp.ones(300)
            positions = profile.sample_positions(masses, key)
            Q = compute_q_parameter(np.asarray(positions))
            Q_values.append(Q)

        Q_mean = np.mean(Q_values)
        Q_std = np.std(Q_values)

        # CW04: Q = 0.79 ± 0.04, we accept 0.75-0.90
        assert 0.75 < Q_mean < 0.90, (
            f"Expected Q ≈ 0.79 for uniform sphere (CW04 '3D0'), "
            f"got {Q_mean:.3f} ± {Q_std:.3f}"
        )

    def test_q_converges_with_n(self):
        """Q should be approximately N-independent for N > 100.

        CW04 notes that Q is roughly independent of N for N > 100.
        """
        profile = UniformSphereProfile(R=1.0)
        key = jax.random.PRNGKey(42)

        Q_by_N = {}
        for N in [100, 300, 500, 1000]:
            subkey = jax.random.fold_in(key, N)
            masses = jnp.ones(N)
            positions = profile.sample_positions(masses, subkey)
            Q_by_N[N] = compute_q_parameter(np.asarray(positions))

        Q_values = list(Q_by_N.values())
        Q_range = max(Q_values) - min(Q_values)
        Q_mean = np.mean(Q_values)

        # Q should vary by less than 25% across different N
        assert Q_range / Q_mean < 0.25, (
            f"Q should be N-independent, but varies by {Q_range / Q_mean * 100:.1f}%: "
            f"{Q_by_N}"
        )


class TestPlummerBaseline:
    """Test Q for Plummer profile (centrally concentrated)."""

    @pytest.fixture
    def plummer_positions(self):
        """Generate Plummer sphere positions for testing."""
        # Use simple inverse CDF sampling for Plummer
        # M(<r)/M = r³/(r² + a²)^(3/2) where a = r_h/sqrt(2^(2/3) - 1)
        rng = np.random.default_rng(42)
        N = 300

        # Plummer scale radius from half-mass radius
        r_h = 1.0
        a = r_h / np.sqrt(2 ** (2 / 3) - 1)

        # Sample radii via inverse CDF
        u = rng.uniform(0, 1, N)
        # Inverse CDF: r = a * u^(1/3) / sqrt(1 - u^(2/3))
        u_clipped = np.clip(u, 1e-10, 1 - 1e-10)
        radii = a * u_clipped ** (1 / 3) / np.sqrt(1 - u_clipped ** (2 / 3))

        # Sample angles
        cos_theta = rng.uniform(-1, 1, N)
        sin_theta = np.sqrt(1 - cos_theta**2)
        phi = rng.uniform(0, 2 * np.pi, N)

        # Cartesian
        x = radii * sin_theta * np.cos(phi)
        y = radii * sin_theta * np.sin(phi)
        z = radii * cos_theta

        return np.column_stack([x, y, z])

    def test_plummer_q_gt_uniform(self, plummer_positions):
        """Plummer Q should be greater than uniform sphere Q.

        Plummer is centrally concentrated, which leads to:
            - Larger mean pairwise separation s̄ (some stars at large r)
            - Shorter MST edges m̄ (concentration creates short connections)
            → Higher Q = m̄ / s̄

        Note: Plummer Q >> 1 is expected and NOT a bug. CW04's Q calibration
        assumes fractal/uniform distributions, not concentrated profiles.
        """
        # Get uniform baseline
        profile = UniformSphereProfile(R=1.0)
        key = jax.random.PRNGKey(42)
        masses = jnp.ones(300)
        uniform_positions = profile.sample_positions(masses, key)
        Q_uniform = compute_q_parameter(np.asarray(uniform_positions))

        # Get Plummer Q
        Q_plummer = compute_q_parameter(plummer_positions)

        assert Q_plummer > Q_uniform, (
            f"Plummer Q ({Q_plummer:.2f}) should be > uniform Q ({Q_uniform:.2f})"
        )

    def test_plummer_q_outside_cw04_range(self, plummer_positions):
        """Plummer Q should be significantly above CW04's calibrated range.

        CW04 calibrates Q ∈ [0.5, 1.0] for fractal/uniform distributions.
        Plummer's central concentration gives Q >> 1.

        This is expected behavior, not a bug.
        """
        Q_plummer = compute_q_parameter(plummer_positions)

        # Plummer (centrally concentrated) gives Q > 1, well above the CW04
        # fractal/uniform band (~0.5-0.95). Under the corrected A=πR² convention
        # Plummer Q ≈ 1.2-1.7; assert > 1.1 (still clearly concentrated).
        assert Q_plummer > 1.1, (
            f"Plummer Q ({Q_plummer:.2f}) should be > 1 due to concentration. "
            "If Q < 1.1, something may be wrong with sampling."
        )


# NOTE: TestEnvToFDFLayerRanges / TestEnvToFDFLayerPhysics were removed in P5 — they
# tested cluster.fdf_config.env_to_fdf_layer (deleted). The underlying physics (σ_ln_ρ,
# χ, b from environment) is covered below via BirthEnvironment + the clean-room subsystem.


class TestBirthEnvironmentTurbulence:
    """Test BirthEnvironment turbulence methods."""

    def test_turbulent_mach_typical_values(self):
        """Mach number should be in physical range for typical clusters.

        Using Larson velocity-size relation: σ_v = σ_v0 × (R_cloud)^α
        where R_cloud is derived from Marks+2012 cloud density:
        - Small OC (10³ M☉): M ~ 2.8
        - Large OC (10⁴ M☉): M ~ 3.2
        - YMC (10⁵ M☉): M ~ 3.8
        - GC (10⁶ M☉): M ~ 4.4

        These are MUCH more realistic than old virial-based estimates (M ~ 20-400)
        which incorrectly used stellar r_h instead of cloud radius.

        Note: Marks+2012 derived cloud densities are higher than typical GMCs
        (~10⁴-10⁶ vs ~10²-10³ M☉/pc³), giving smaller cloud radii and lower
        Mach numbers. Users can override with explicit log_rho_cl for typical
        GMC densities, which would give M ~ 7-15.
        """
        from progenax.imf.environment import BirthEnvironment

        # Expected ranges with Marks+2012 cloud densities
        test_cases = [
            (1e3, 2, 5),  # Small OC: M ~ 2.8
            (1e4, 2.5, 5),  # Large OC: M ~ 3.2
            (1e5, 3, 5.5),  # YMC: M ~ 3.8
            (1e6, 3.5, 6),  # GC: M ~ 4.4
        ]

        for M_ecl, M_min, M_max in test_cases:
            env = BirthEnvironment.from_cluster_mass(M_ecl=M_ecl)
            mach = float(env.turbulent_mach())

            assert M_min < mach < M_max, (
                f"M_ecl={M_ecl:.0e}: Mach = {mach:.1f} outside expected [{M_min}, {M_max}]"
            )

    def test_sigma_ln_rho_typical_values(self):
        """σ_ln_ρ should be in physical range from Federrath+2010.

        σ_ln_ρ = sqrt(ln(1 + b²M²)) with environment-dependent b and
        Larson-derived Mach numbers. The b parameter is derived from cloud
        density via b_from_environment():
        - Low-density clouds: b ~ 0.33 (more solenoidal)
        - High-density cores: b ~ 0.7 (more compressive)

        Expected ranges (with environment-dependent b):
        - Small OC (10³ M☉): σ_ln_ρ ~ 1.0-1.3
        - Large OC (10⁴ M☉): σ_ln_ρ ~ 1.1-1.5
        - YMC (10⁵ M☉): σ_ln_ρ ~ 1.2-1.6
        - GC (10⁶ M☉): σ_ln_ρ ~ 1.3-1.8

        These are slightly higher than fixed b=0.4 values because b increases
        with cloud density for massive clusters.
        """
        from progenax.imf.environment import BirthEnvironment

        # Expected ranges with environment-dependent b
        # Higher-mass clusters have denser clouds → higher b → higher σ_ln_ρ
        test_cases = [
            (1e3, 0.8, 1.5),  # Small OC
            (1e4, 0.9, 1.6),  # Large OC
            (1e5, 1.0, 1.7),  # YMC
            (1e6, 1.1, 1.9),  # GC
        ]

        for M_ecl, sigma_min, sigma_max in test_cases:
            env = BirthEnvironment.from_cluster_mass(M_ecl=M_ecl)
            sigma = float(env.sigma_ln_rho())

            assert sigma_min < sigma < sigma_max, (
                f"M_ecl={M_ecl:.0e}: σ_ln_ρ = {sigma:.2f} outside expected [{sigma_min}, {sigma_max}]"
            )

    def test_spectral_slope_supersonic(self):
        """Density spectrum flattens (β < Kolmogorov) and decreases with mass (Kim&Ryu 2005).

        Corrected 2026-06: β is the DENSITY power-spectrum slope, which *flattens* with
        Mach (Kim & Ryu 2005), NOT the velocity Burgers slope ~4 (the prior bug). These
        clouds are mildly supersonic (M~3-4) → β~3.0-3.3, shallower than the Kolmogorov
        ceiling 11/3 and monotonically decreasing as the cluster (hence Mach) grows.
        """
        from progenax.imf.environment import BirthEnvironment

        prev = None
        for M_ecl in [1e3, 1e4, 1e5, 1e6]:
            env = BirthEnvironment.from_cluster_mass(M_ecl=M_ecl)
            beta = float(env.spectral_slope())

            assert 2.0 <= beta < 11.0 / 3.0, (
                f"M_ecl={M_ecl:.0e}: β = {beta:.2f} should be in [2, 11/3) (density spectrum)"
            )
            if prev is not None:
                assert beta < prev, (
                    f"β should decrease with cluster mass (Mach); got {beta:.3f} >= {prev:.3f}"
                )
            prev = beta


class TestBFromEnvironment:
    """Test environment-dependent turbulence driving parameter b."""

    def test_b_low_density_solenoidal(self):
        """Low-density clouds should have solenoidal driving (b ~ 0.33)."""
        from progenax.cluster.turbulence import b_from_environment

        # Low density: 100 M☉/pc³ (log = 2)
        b_low = float(b_from_environment(jnp.array(2.0)))

        # Should be close to solenoidal limit (0.33)
        assert 0.30 < b_low < 0.40, (
            f"Low density (10² M☉/pc³) should give b ≈ 0.33, got {b_low:.2f}"
        )

    def test_b_high_density_compressive(self):
        """High-density cores should have compressive driving (b ~ 0.7)."""
        from progenax.cluster.turbulence import b_from_environment

        # High density: 10⁶ M☉/pc³ (log = 6)
        b_high = float(b_from_environment(jnp.array(6.0)))

        # Should be close to compressive limit (0.7)
        assert 0.60 < b_high < 0.75, (
            f"High density (10⁶ M☉/pc³) should give b ≈ 0.7, got {b_high:.2f}"
        )

    def test_b_increases_with_density(self):
        """b should increase monotonically with cloud density."""
        from progenax.cluster.turbulence import b_from_environment

        log_rhos = [2.0, 3.0, 4.0, 5.0, 6.0]
        b_values = [float(b_from_environment(jnp.array(lr))) for lr in log_rhos]

        # Check monotonic increase
        for i in range(len(b_values) - 1):
            assert b_values[i] < b_values[i + 1], (
                f"b should increase with density: {b_values}"
            )


# TestEnvToFDFLayerPhysics removed in P5 (tested the deleted cluster.fdf_config.env_to_fdf_layer).
# The b-from-density physics is covered by TestBFromEnvironment above (core turbulence).

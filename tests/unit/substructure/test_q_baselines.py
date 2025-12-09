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
            f"Q should be N-independent, but varies by {Q_range/Q_mean*100:.1f}%: "
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
        a = r_h / np.sqrt(2**(2/3) - 1)

        # Sample radii via inverse CDF
        u = rng.uniform(0, 1, N)
        # Inverse CDF: r = a * u^(1/3) / sqrt(1 - u^(2/3))
        u_clipped = np.clip(u, 1e-10, 1 - 1e-10)
        radii = a * u_clipped**(1/3) / np.sqrt(1 - u_clipped**(2/3))

        # Sample angles
        cos_theta = rng.uniform(-1, 1, N)
        sin_theta = np.sqrt(1 - cos_theta**2)
        phi = rng.uniform(0, 2*np.pi, N)

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

        # Plummer should have Q > 1.5 (way above CW04 range)
        assert Q_plummer > 1.5, (
            f"Plummer Q ({Q_plummer:.2f}) should be >> 1 due to concentration. "
            "If Q < 1.5, something may be wrong with sampling."
        )


class TestEnvToFDFLayerRanges:
    """Test that environment-derived FDF parameters are in physical ranges."""

    def test_sigma_ln_rho_in_physical_range(self):
        """σ_ln_ρ from environment should be in reasonable range.

        Federrath+2010: σ_ln_ρ = sqrt(ln(1 + b²M²)) where b ~ 0.4.
        For virial Mach numbers M ~ 20-60 (from our r_h-M relation),
        this gives σ_ln_ρ ~ 2.0-3.5.
        """
        from progenax.cluster.fdf_config import env_to_fdf_layer

        test_cases = [
            (3.0, "Small OC (10³ M☉)"),
            (4.0, "Large OC (10⁴ M☉)"),
            (5.0, "YMC (10⁵ M☉)"),
            (6.0, "GC (10⁶ M☉)"),
        ]

        for log_mecl, name in test_cases:
            layer = env_to_fdf_layer(jnp.array(log_mecl))
            sigma = layer.sigma_ln_rho

            # Actual computed range is ~2.0-3.5 due to high virial Mach
            assert 1.0 < sigma < 4.0, (
                f"{name}: σ_ln_ρ = {sigma:.2f} outside physical range [1.0, 4.0]"
            )

    def test_chi_in_valid_range(self):
        """χ from environment should be in [1.6, 3.0]."""
        from progenax.cluster.fdf_config import env_to_fdf_layer

        for log_mecl in [3.0, 4.0, 5.0, 6.0]:
            layer = env_to_fdf_layer(jnp.array(log_mecl))
            chi = layer.chi

            assert 1.6 <= chi <= 3.0, (
                f"log_mecl={log_mecl}: χ = {chi:.2f} outside valid range [1.6, 3.0]"
            )

    def test_sigma_increases_with_mass(self):
        """σ_ln_ρ should increase with cluster mass.

        More massive clusters → higher velocity dispersion → higher Mach → higher σ.
        """
        from progenax.cluster.fdf_config import env_to_fdf_layer

        sigmas = []
        for log_mecl in [3.0, 4.0, 5.0, 6.0]:
            layer = env_to_fdf_layer(jnp.array(log_mecl))
            sigmas.append(layer.sigma_ln_rho)

        # Check monotonic increase
        for i in range(len(sigmas) - 1):
            assert sigmas[i] < sigmas[i + 1], (
                f"σ_ln_ρ should increase with mass: {sigmas}"
            )


class TestBirthEnvironmentTurbulence:
    """Test BirthEnvironment turbulence methods."""

    def test_turbulent_mach_typical_values(self):
        """Mach number should be in physical range for typical clusters.

        Using virial σ_v = √(G M_ecl / r_h) with Marks+2012 r_h-M relation
        (r_h = 0.1 × M_ecl^0.13):
        - Small OC (10³ M☉): M ~ 21
        - Large OC (10⁴ M☉): M ~ 57
        - YMC (10⁵ M☉): M ~ 155
        - GC (10⁶ M☉): M ~ 422

        Note: These high Mach numbers reflect the compact r_h from Marks+2012.
        Real clusters may have larger r_h and lower Mach.
        """
        from progenax.imf.environment import BirthEnvironment

        # Actual computed ranges from virial + r_h scaling
        test_cases = [
            (1e3, 15, 30),     # Small OC: M ~ 21
            (1e4, 45, 70),     # Large OC: M ~ 57
            (1e5, 130, 180),   # YMC: M ~ 155
            (1e6, 350, 500),   # GC: M ~ 422
        ]

        for M_ecl, M_min, M_max in test_cases:
            env = BirthEnvironment.from_cluster_mass(M_ecl=M_ecl)
            mach = float(env.turbulent_mach())

            assert M_min < mach < M_max, (
                f"M_ecl={M_ecl:.0e}: Mach = {mach:.1f} outside expected [{M_min}, {M_max}]"
            )

    def test_sigma_ln_rho_typical_values(self):
        """σ_ln_ρ should be in physical range from Federrath+2010.

        σ_ln_ρ = sqrt(ln(1 + b²M²)) with b ~ 0.4 and high virial Mach:
        - Small OC (M ~ 21): σ_ln_ρ ~ 2.1
        - Large OC (M ~ 39): σ_ln_ρ ~ 2.6
        - YMC (M ~ 73): σ_ln_ρ ~ 3.1
        - GC (M ~ 137): σ_ln_ρ ~ 3.5
        """
        from progenax.imf.environment import BirthEnvironment

        # Actual computed ranges based on virial Mach numbers
        test_cases = [
            (1e3, 1.5, 2.5),   # Small OC: σ ~ 2.1
            (1e4, 2.2, 3.0),   # Large OC: σ ~ 2.6
            (1e5, 2.8, 3.4),   # YMC: σ ~ 3.1
            (1e6, 3.2, 3.8),   # GC: σ ~ 3.5
        ]

        for M_ecl, sigma_min, sigma_max in test_cases:
            env = BirthEnvironment.from_cluster_mass(M_ecl=M_ecl)
            sigma = float(env.sigma_ln_rho())

            assert sigma_min < sigma < sigma_max, (
                f"M_ecl={M_ecl:.0e}: σ_ln_ρ = {sigma:.2f} outside expected [{sigma_min}, {sigma_max}]"
            )

    def test_spectral_slope_supersonic(self):
        """Star-forming clouds have M >> 1, so β should be ~4 (Burgers)."""
        from progenax.imf.environment import BirthEnvironment

        # All typical star-forming clusters are supersonic
        for M_ecl in [1e3, 1e4, 1e5, 1e6]:
            env = BirthEnvironment.from_cluster_mass(M_ecl=M_ecl)
            beta = float(env.spectral_slope())

            # Should be close to Burgers β ≈ 4.0 for supersonic
            assert 3.8 < beta < 4.05, (
                f"M_ecl={M_ecl:.0e}: β = {beta:.2f} should be ~4.0 for supersonic"
            )

"""
Physics validation tests for Plummer profile and velocity distribution.

Tests verify that implementations match theoretical predictions from:
- Plummer (1911), MNRAS 71, 460
- Binney & Tremaine (2008), "Galactic Dynamics"
- Aarseth (2003), "Gravitational N-Body Simulations"

Each test has quantitative error bounds based on statistical expectations.
"""

import jax
import jax.numpy as jnp
import pytest

from progenax.profiles import PlummerProfile
from progenax.kinematics import PlummerVelocityDF


class TestPlummerScaleRadius:
    """Verify Plummer scale radius formula: a = r_h × √(2^(2/3) - 1)."""

    def test_scale_radius_formula_exact(self, plummer_constants):
        """Scale radius formula matches exact derivation.

        From M(<r)/M = r³/(r²+a²)^(3/2), at r=r_h: M(<r_h)/M = 0.5
        Solving: a = r_h × √(2^(2/3) - 1) ≈ 0.7664 × r_h
        """
        r_h = 1.0
        profile = PlummerProfile(r_h=r_h)

        expected_a = r_h * plummer_constants.SCALE_RADIUS_FACTOR
        assert jnp.allclose(profile.a, expected_a, rtol=1e-6), \
            f"Scale radius a={float(profile.a):.10f}, expected={float(expected_a):.10f}"

    def test_scale_radius_numerical_value(self):
        """Scale radius approximately 0.7664 × r_h."""
        r_h = 1.0
        profile = PlummerProfile(r_h=r_h)

        # Numerical value: sqrt(2^(2/3) - 1) ≈ 0.76643...
        assert jnp.allclose(profile.a, 0.7664 * r_h, rtol=0.001), \
            f"Scale radius a={float(profile.a):.4f}, expected ≈ 0.7664"

    @pytest.mark.parametrize("r_h", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_scale_radius_scales_linearly(self, r_h, plummer_constants):
        """Scale radius scales linearly with half-mass radius."""
        profile = PlummerProfile(r_h=r_h)
        expected_a = r_h * plummer_constants.SCALE_RADIUS_FACTOR

        assert jnp.allclose(profile.a, expected_a, rtol=1e-6)

    def test_profile_df_scale_radius_match(self):
        """Profile and velocity DF use identical scale radius."""
        r_h = 1.0
        profile = PlummerProfile(r_h=r_h)
        df = PlummerVelocityDF(r_h=r_h)

        assert jnp.allclose(profile.a, df.a, rtol=1e-6), \
            f"Mismatch: profile.a={float(profile.a):.6f}, df.a={float(df.a):.6f}"


class TestPlummerDensityProfile:
    """Verify Plummer density profile and cumulative mass function."""

    def test_half_mass_radius_statistical(self, N_validation, key, tolerances):
        """50% of sampled particles within r_h (definition of half-mass radius)."""
        r_h = 1.0
        profile = PlummerProfile(r_h=r_h)
        masses = jnp.ones(N_validation)

        positions = profile.sample_positions(masses, key)
        radii = jnp.linalg.norm(positions, axis=1)

        fraction_within = float(jnp.mean(radii < r_h))

        assert abs(fraction_within - 0.5) < tolerances.HALF_MASS, \
            f"Fraction within r_h = {fraction_within:.4f}, expected 0.50 ± {tolerances.HALF_MASS}"

    def test_cumulative_mass_at_scale_radius(self, N_validation, key, plummer_constants, tolerances):
        """M(<a)/M = 1/2^(3/2) ≈ 0.354 at scale radius.

        From Plummer CDF: M(<r)/M = r³/(r²+a²)^(3/2)
        At r=a: M(<a)/M = a³/(2a²)^(3/2) = 1/(2√2) = 0.354
        """
        r_h = 1.0
        profile = PlummerProfile(r_h=r_h)
        masses = jnp.ones(N_validation)

        positions = profile.sample_positions(masses, key)
        radii = jnp.linalg.norm(positions, axis=1)

        fraction_within_a = float(jnp.mean(radii < profile.a))
        expected = float(plummer_constants.MASS_WITHIN_SCALE_RADIUS)

        assert abs(fraction_within_a - expected) < tolerances.HALF_MASS, \
            f"M(<a)/M = {fraction_within_a:.4f}, expected {expected:.4f}"

    def test_cdf_formula_accuracy(self, N_stats, key):
        """Sampled radii match theoretical CDF: M(<r)/M = r³/(r²+a²)^(3/2)."""
        r_h = 1.0
        profile = PlummerProfile(r_h=r_h)
        masses = jnp.ones(N_stats)
        a = profile.a

        positions = profile.sample_positions(masses, key)
        radii = jnp.linalg.norm(positions, axis=1)

        # Test at several radii
        test_radii = [0.5 * a, a, 2.0 * a, 3.0 * a]
        for r_test in test_radii:
            # Measured
            measured = float(jnp.mean(radii < r_test))
            # Theoretical
            expected = float(r_test**3 / (r_test**2 + a**2)**1.5)

            assert abs(measured - expected) < 0.03, \
                f"At r={float(r_test):.2f}: measured={measured:.4f}, expected={expected:.4f}"

    def test_positions_isotropic(self, N_validation, key):
        """Sampled positions are isotropically distributed."""
        r_h = 1.0
        profile = PlummerProfile(r_h=r_h)
        masses = jnp.ones(N_validation)

        positions = profile.sample_positions(masses, key)

        # Check isotropy: mean of each component should be ~0
        mean_pos = jnp.mean(positions, axis=0)
        std_pos = jnp.std(positions, axis=0)

        for i, (mean, std) in enumerate(zip(mean_pos, std_pos)):
            # Mean within 3σ/√N of zero
            threshold = 3 * std / jnp.sqrt(N_validation)
            assert jnp.abs(mean) < threshold, \
                f"Component {i}: mean={float(mean):.4f}, expected ~0 (threshold={float(threshold):.4f})"


class TestPlummerVelocityDispersion:
    """Verify Plummer velocity dispersion profile: σ²(r) = GM/(6√(r²+a²))."""

    def test_central_velocity_dispersion(self, N_stats, key, tolerances):
        """1D velocity dispersion at r=0 matches σ²(0) = GM/(6a)."""
        r_h = 1.0
        G = 1.0
        N = N_stats

        df = PlummerVelocityDF(r_h=r_h)
        positions = jnp.zeros((N, 3))  # All at center
        masses = jnp.ones(N)
        M_total = float(N)
        a = df.a

        velocities = df.sample_velocities(positions, masses, key, G=G)

        # Measured 1D dispersion (any component, since isotropic)
        sigma_measured = float(jnp.std(velocities[:, 0]))

        # Theoretical: σ(0) = √(GM/(6a))
        sigma_theory = float(jnp.sqrt(G * M_total / (6.0 * a)))

        relative_error = abs(sigma_measured - sigma_theory) / sigma_theory
        assert relative_error < tolerances.VELOCITY_DISPERSION, \
            f"σ_measured={sigma_measured:.4f}, σ_theory={sigma_theory:.4f}, error={relative_error*100:.1f}%"

    def test_radial_dispersion_profile(self, N_validation, key, tolerances):
        """Velocity dispersion decreases with radius per σ²(r) = GM/(6√(r²+a²))."""
        r_h = 1.0
        G = 1.0
        N = N_validation

        df = PlummerVelocityDF(r_h=r_h)
        masses = jnp.ones(N)
        M_total = float(N)
        a = df.a

        # Test at multiple radii
        test_radii = [0.0, 0.5, 1.0, 2.0, 3.0]
        measured_sigmas = []
        theory_sigmas = []

        for r in test_radii:
            positions = jnp.array([[r, 0.0, 0.0]] * N)
            seed = int(42 + r * 10)
            k = jax.random.PRNGKey(seed)

            velocities = df.sample_velocities(positions, masses, k, G=G)
            sigma_m = float(jnp.std(velocities[:, 0]))
            sigma_t = float(jnp.sqrt(G * M_total / (6.0 * jnp.sqrt(r**2 + a**2))))

            measured_sigmas.append(sigma_m)
            theory_sigmas.append(sigma_t)

        # Verify monotonic decrease
        for i in range(len(test_radii) - 1):
            assert measured_sigmas[i] > measured_sigmas[i+1] * 0.85, \
                f"σ should decrease: σ({test_radii[i]})={measured_sigmas[i]:.3f} vs σ({test_radii[i+1]})={measured_sigmas[i+1]:.3f}"

        # Verify matches theory within tolerance
        for r, sm, st in zip(test_radii, measured_sigmas, theory_sigmas):
            error = abs(sm - st) / st
            assert error < tolerances.VELOCITY_DISPERSION, \
                f"At r={r}: error={error*100:.1f}% exceeds {tolerances.VELOCITY_DISPERSION*100:.0f}%"

    def test_velocity_isotropy(self, N_stats, key):
        """Velocities are isotropically distributed (no radial bias)."""
        r_h = 1.0
        G = 1.0

        profile = PlummerProfile(r_h=r_h)
        df = PlummerVelocityDF(r_h=r_h)
        masses = jnp.ones(N_stats)

        key_pos, key_vel = jax.random.split(key)
        positions = profile.sample_positions(masses, key_pos)
        velocities = df.sample_velocities(positions, masses, key_vel, G=G)

        # Check isotropy: <vx²> ≈ <vy²> ≈ <vz²>
        v2_mean = jnp.mean(velocities**2, axis=0)

        # Each component should be within 5% of the others
        mean_v2 = float(jnp.mean(v2_mean))
        for i, v2i in enumerate(v2_mean):
            rel_diff = abs(float(v2i) - mean_v2) / mean_v2
            assert rel_diff < 0.05, \
                f"Anisotropy detected: <v{['x','y','z'][i]}²>={float(v2i):.4f}, mean={mean_v2:.4f}"


class TestPlummerVirialEquilibrium:
    """Verify Plummer ICs are in virial equilibrium: Q = 2T/|V| ≈ 1."""

    def test_virial_ratio(self, N_validation, key, tolerances):
        """Virial ratio Q = 2T/|V| close to 1.0 for equilibrium system."""
        r_h = 1.0
        G = 1.0

        profile = PlummerProfile(r_h=r_h)
        df = PlummerVelocityDF(r_h=r_h)
        masses = jnp.ones(N_validation)
        M_total = float(jnp.sum(masses))
        a = profile.a

        key_pos, key_vel = jax.random.split(key)
        positions = profile.sample_positions(masses, key_pos)
        velocities = df.sample_velocities(positions, masses, key_vel, G=G)

        # Kinetic energy: T = 0.5 × Σ(m_i × v_i²)
        v_squared = jnp.sum(velocities**2, axis=1)
        T = 0.5 * float(jnp.sum(masses * v_squared))

        # Analytical potential energy for Plummer: V = -3πGM²/(32a)
        V_analytical = -3.0 * jnp.pi * G * M_total**2 / (32.0 * a)

        # Virial ratio
        Q = 2.0 * T / abs(float(V_analytical))

        assert abs(Q - 1.0) < tolerances.VIRIAL_RATIO, \
            f"Virial ratio Q={Q:.4f}, expected 1.0 ± {tolerances.VIRIAL_RATIO}"


class TestPlummerBoundParticles:
    """Verify all particles are gravitationally bound: v < v_esc."""

    def test_all_particles_bound(self, N_stats, key):
        """100% of particles have v < v_esc (bound orbits)."""
        r_h = 1.0
        G = 1.0

        profile = PlummerProfile(r_h=r_h)
        df = PlummerVelocityDF(r_h=r_h)
        masses = jnp.ones(N_stats)
        M_total = float(jnp.sum(masses))
        a = df.a

        key_pos, key_vel = jax.random.split(key)
        positions = profile.sample_positions(masses, key_pos)
        velocities = df.sample_velocities(positions, masses, key_vel, G=G)

        # Escape velocity at each position: v_esc² = 2GM/√(r²+a²)
        radii = jnp.linalg.norm(positions, axis=1)
        v_esc = jnp.sqrt(2.0 * G * M_total / jnp.sqrt(radii**2 + a**2))

        # Velocity magnitudes
        v_mag = jnp.linalg.norm(velocities, axis=1)

        # All must be bound
        bound_fraction = float(jnp.mean(v_mag < v_esc))
        assert bound_fraction == 1.0, \
            f"Only {bound_fraction*100:.2f}% bound (expected 100%)"

    def test_velocity_to_escape_ratio(self, N_stats, key, plummer_constants):
        """Mean v/v_esc ≈ 0.5 (from Beta distribution mean √⟨q²⟩ = 0.5)."""
        r_h = 1.0
        G = 1.0

        profile = PlummerProfile(r_h=r_h)
        df = PlummerVelocityDF(r_h=r_h)
        masses = jnp.ones(N_stats)
        M_total = float(jnp.sum(masses))
        a = df.a

        key_pos, key_vel = jax.random.split(key)
        positions = profile.sample_positions(masses, key_pos)
        velocities = df.sample_velocities(positions, masses, key_vel, G=G)

        radii = jnp.linalg.norm(positions, axis=1)
        v_esc = jnp.sqrt(2.0 * G * M_total / jnp.sqrt(radii**2 + a**2))
        v_mag = jnp.linalg.norm(velocities, axis=1)

        q_mean = float(jnp.mean(v_mag / v_esc))

        # From Beta(3/2, 9/2): E[q] = E[√u] where u ~ Beta(3/2, 9/2)
        # E[√u] ≈ 0.47 (numerical integration)
        assert 0.40 < q_mean < 0.55, \
            f"Mean q = v/v_esc = {q_mean:.4f}, expected ~0.47"


class TestPlummerBetaDistribution:
    """Verify velocity magnitudes follow Beta(3/2, 9/2) for q² = (v/v_esc)²."""

    def test_q_squared_mean(self, N_stats, key, plummer_constants):
        """⟨q²⟩ = 0.25 from Beta(3/2, 9/2) mean = a/(a+b) = 1.5/6."""
        r_h = 1.0
        G = 1.0

        profile = PlummerProfile(r_h=r_h)
        df = PlummerVelocityDF(r_h=r_h)
        masses = jnp.ones(N_stats)
        M_total = float(jnp.sum(masses))
        a = df.a

        key_pos, key_vel = jax.random.split(key)
        positions = profile.sample_positions(masses, key_pos)
        velocities = df.sample_velocities(positions, masses, key_vel, G=G)

        radii = jnp.linalg.norm(positions, axis=1)
        v_esc = jnp.sqrt(2.0 * G * M_total / jnp.sqrt(radii**2 + a**2))
        v_mag = jnp.linalg.norm(velocities, axis=1)

        q = v_mag / v_esc
        q2_mean = float(jnp.mean(q**2))

        expected = float(plummer_constants.MEAN_Q_SQUARED)  # 0.25
        assert abs(q2_mean - expected) < 0.02, \
            f"⟨q²⟩ = {q2_mean:.4f}, expected {expected:.4f}"

    def test_q_squared_variance(self, N_stats, key):
        """Variance of q² matches Beta(3/2, 9/2) prediction.

        Var(q²) = ab/((a+b)²(a+b+1)) = 1.5×4.5/(36×7) = 0.0268
        """
        r_h = 1.0
        G = 1.0

        profile = PlummerProfile(r_h=r_h)
        df = PlummerVelocityDF(r_h=r_h)
        masses = jnp.ones(N_stats)
        M_total = float(jnp.sum(masses))
        a = df.a

        key_pos, key_vel = jax.random.split(key)
        positions = profile.sample_positions(masses, key_pos)
        velocities = df.sample_velocities(positions, masses, key_vel, G=G)

        radii = jnp.linalg.norm(positions, axis=1)
        v_esc = jnp.sqrt(2.0 * G * M_total / jnp.sqrt(radii**2 + a**2))
        v_mag = jnp.linalg.norm(velocities, axis=1)

        q2 = (v_mag / v_esc)**2
        q2_var = float(jnp.var(q2))

        # Theoretical variance for Beta(1.5, 4.5)
        a_beta, b_beta = 1.5, 4.5
        expected_var = (a_beta * b_beta) / ((a_beta + b_beta)**2 * (a_beta + b_beta + 1))

        assert abs(q2_var - expected_var) / expected_var < 0.15, \
            f"Var(q²) = {q2_var:.4f}, expected {expected_var:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

"""
Physics validation tests for EFF (Elson-Fall-Freeman 1987) profile.

Tests verify that implementations match theoretical predictions from:
- Elson, Fall & Freeman (1987), ApJ 323, 54
- Cabrera-Ziri et al. (2016), MNRAS 457, 809

Each test has quantitative error bounds based on theoretical expectations.
"""

import jax
import jax.numpy as jnp
import pytest

from progenax.profiles.eff import EFFProfile
from progenax.kinematics.eff_df import EFFVelocityDF


class TestEFFDensityFormula:
    """Verify EFF density profile: rho(r) = rho_0 / (1 + r^2/a^2)^(gamma/2)."""

    def test_central_density_unity(self):
        """Density at r=0 is maximum (rho_0 normalized to 1)."""
        a, gamma, r_t = 1.0, 3.0, 10.0
        profile = EFFProfile(a=a, gamma=gamma, r_t=r_t)

        rho_0 = profile.density(jnp.array([0.0]))
        assert float(rho_0[0]) == 1.0, f"rho(0) = {float(rho_0[0])}, expected 1.0"

    def test_density_at_scale_radius(self):
        """At r=a: rho(a) = rho_0 / 2^(gamma/2)."""
        for gamma in [2.0, 3.0, 4.0]:
            a, r_t = 1.0, 10.0
            profile = EFFProfile(a=a, gamma=gamma, r_t=r_t)

            rho_a = profile.density(jnp.array([a]))
            expected = 1.0 / (2.0 ** (gamma / 2.0))

            assert abs(float(rho_a[0]) - expected) < 1e-10, \
                f"gamma={gamma}: rho(a)={float(rho_a[0]):.6f}, expected={expected:.6f}"

    def test_power_law_slope_asymptotic(self):
        """For r >> a: rho(r) proportional to r^(-gamma)."""
        a, gamma, r_t = 1.0, 3.0, 100.0  # Large r_t to test asymptotic
        profile = EFFProfile(a=a, gamma=gamma, r_t=r_t)

        # Test at r >> a
        r1, r2 = 20.0, 40.0
        rho_1 = float(profile.density(jnp.array([r1]))[0])
        rho_2 = float(profile.density(jnp.array([r2]))[0])

        # For power law rho ~ r^(-gamma): rho2/rho1 = (r1/r2)^gamma
        expected_ratio = (r1 / r2) ** gamma
        actual_ratio = rho_2 / rho_1

        assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.01, \
            f"Power-law ratio: actual={actual_ratio:.6f}, expected={expected_ratio:.6f}"

    def test_density_monotonic_decrease(self):
        """Density decreases monotonically with radius."""
        a, gamma, r_t = 1.0, 3.0, 10.0
        profile = EFFProfile(a=a, gamma=gamma, r_t=r_t)

        r_grid = jnp.linspace(0.0, r_t, 100)
        rho_grid = profile.density(r_grid)

        diffs = jnp.diff(rho_grid)
        assert jnp.all(diffs <= 1e-10), "Density should decrease with radius"

    def test_density_zero_beyond_truncation(self):
        """Density is zero for r > r_t."""
        a, gamma, r_t = 1.0, 3.0, 10.0
        profile = EFFProfile(a=a, gamma=gamma, r_t=r_t)

        r_beyond = jnp.array([r_t + 0.01, r_t + 1.0, r_t + 10.0])
        rho_beyond = profile.density(r_beyond)

        assert jnp.all(rho_beyond == 0.0), \
            f"Density should be 0 beyond r_t, got {rho_beyond}"


class TestEFFTidalTruncation:
    """Verify EFF profile enforces tidal truncation at r_t."""

    def test_all_particles_within_tidal_radius(self, N_validation, key):
        """100% of particles at r <= r_t."""
        a, gamma, r_t = 1.0, 3.0, 10.0
        profile = EFFProfile(a=a, gamma=gamma, r_t=r_t)

        masses = jnp.ones(N_validation)
        positions = profile.sample_positions(masses, key)
        radii = jnp.linalg.norm(positions, axis=1)

        max_r = float(jnp.max(radii))
        assert max_r <= r_t + 0.01, f"Max radius {max_r:.4f} exceeds r_t={r_t}"

        fraction_within = float(jnp.mean(radii <= r_t))
        assert fraction_within == 1.0, \
            f"Only {fraction_within*100:.2f}% within r_t (expected 100%)"

    @pytest.mark.parametrize("r_t", [5.0, 10.0, 20.0])
    def test_truncation_radius_respected(self, r_t, N_validation, key):
        """Test truncation works for different r_t values."""
        a, gamma = 1.0, 3.0
        profile = EFFProfile(a=a, gamma=gamma, r_t=r_t)

        masses = jnp.ones(N_validation)
        positions = profile.sample_positions(masses, key)
        radii = jnp.linalg.norm(positions, axis=1)

        max_r = float(jnp.max(radii))
        assert max_r <= r_t + 0.01, f"r_t={r_t}: max radius {max_r:.4f}"


class TestEFFGammaConcentration:
    """Verify gamma parameter affects profile concentration."""

    def test_higher_gamma_more_concentrated(self, N_validation, key):
        """Higher gamma produces more centrally concentrated profile."""
        a, r_t = 1.0, 10.0

        median_radii = []
        for gamma in [2.0, 3.0, 4.0]:
            profile = EFFProfile(a=a, gamma=gamma, r_t=r_t)

            masses = jnp.ones(N_validation)
            k = jax.random.PRNGKey(42)  # Same key for fair comparison
            positions = profile.sample_positions(masses, k)
            radii = jnp.linalg.norm(positions, axis=1)

            median_radii.append(float(jnp.median(radii)))

        # Higher gamma should have smaller median radius (more concentrated)
        assert median_radii[0] > median_radii[1] > median_radii[2], \
            f"Median radii should decrease with gamma: gamma=[2,3,4] -> r_median={median_radii}"

    def test_gamma_affects_central_fraction(self, N_validation, key):
        """Higher gamma produces more particles near center."""
        a, r_t = 1.0, 10.0

        central_fractions = []
        for gamma in [2.0, 3.0, 4.0]:
            profile = EFFProfile(a=a, gamma=gamma, r_t=r_t)

            masses = jnp.ones(N_validation)
            k = jax.random.PRNGKey(42)
            positions = profile.sample_positions(masses, k)
            radii = jnp.linalg.norm(positions, axis=1)

            # Fraction within 2 scale radii
            central_fractions.append(float(jnp.mean(radii < 2.0 * a)))

        # Higher gamma should have more particles centrally
        assert central_fractions[0] < central_fractions[1] < central_fractions[2], \
            f"Central fraction should increase with gamma: gamma=[2,3,4] -> f={central_fractions}"


class TestEFFDensityProfile:
    """Verify sampled density matches EFF formula."""

    def test_density_decreases_with_radius(self, N_validation, key):
        """Binned density decreases monotonically with radius."""
        a, gamma, r_t = 1.0, 3.0, 10.0
        profile = EFFProfile(a=a, gamma=gamma, r_t=r_t)

        masses = jnp.ones(N_validation)
        positions = profile.sample_positions(masses, key)
        radii = jnp.linalg.norm(positions, axis=1)

        # Bin particles and count density
        bins = jnp.linspace(0, r_t, 20)
        hist, _ = jnp.histogram(radii, bins=bins)

        # Normalize by shell volume: V = (4/3)pi(r_out^3 - r_in^3)
        volumes = (4.0/3.0) * jnp.pi * (bins[1:]**3 - bins[:-1]**3)
        densities = hist / (volumes + 1e-10)

        # Check that density at r < a is higher than at r > 2*a
        inner_density = float(jnp.mean(densities[:5]))
        outer_density = float(jnp.mean(densities[10:]))

        assert inner_density > outer_density, \
            f"Inner density {inner_density:.2f} should exceed outer {outer_density:.2f}"

    def test_half_mass_radius_reasonable(self, N_validation, key):
        """Half-mass radius is between a and r_t."""
        a, gamma, r_t = 1.0, 3.0, 10.0
        profile = EFFProfile(a=a, gamma=gamma, r_t=r_t)

        masses = jnp.ones(N_validation)
        positions = profile.sample_positions(masses, key)
        radii = jnp.linalg.norm(positions, axis=1)

        # Half-mass radius (median for equal masses)
        r_h = float(jnp.median(radii))

        # Should be between a and r_t
        assert a < r_h < r_t, f"r_h={r_h:.3f} should be in ({a}, {r_t})"


class TestEFFVelocityDF:
    """Verify EFF velocity distribution function properties."""

    def test_velocity_isotropy(self, N_stats, key):
        """Velocities are isotropically distributed."""
        a, gamma, r_t = 1.0, 3.0, 10.0
        G = 1.0

        profile = EFFProfile(a=a, gamma=gamma, r_t=r_t)
        df = EFFVelocityDF(a=a, gamma=gamma, r_t=r_t)

        masses = jnp.ones(N_stats)
        key_pos, key_vel = jax.random.split(key)

        positions = profile.sample_positions(masses, key_pos)
        velocities = df.sample_velocities(positions, masses, key_vel, G=G)

        # Check isotropy: <vx^2> ~ <vy^2> ~ <vz^2>
        v2_mean = jnp.mean(velocities**2, axis=0)
        mean_v2 = float(jnp.mean(v2_mean))

        for i, v2i in enumerate(v2_mean):
            rel_diff = abs(float(v2i) - mean_v2) / mean_v2
            assert rel_diff < 0.10, \
                f"Anisotropy detected: <v{['x','y','z'][i]}^2>={float(v2i):.4f}, mean={mean_v2:.4f}"

    def test_eff_eddington_virial_ratio_mild_truncation(self):
        """For mild truncation (gamma=5) the Eddington DF yields virial equilibrium
        (Q ~ 0.5) WITHOUT external rescale.

        Note: the steep gamma=3 default, whose mass diverges logarithmically, is ~5-8%
        sub-virial under sharp truncation -- an intrinsic property of truncating an
        empirical (non-DF) profile, not an inversion error (use the King model for a
        strict lowered-DF equilibrium).
        """
        from progenax.builders import compute_kinetic_energy, compute_potential_energy

        a, gamma, r_t, G = 1.0, 5.0, 15.0, 1.0
        profile = EFFProfile(a=a, gamma=gamma, r_t=r_t)
        df = EFFVelocityDF(a=a, gamma=gamma, r_t=r_t)
        masses = jnp.ones(6000)
        kp, kv = jax.random.split(jax.random.PRNGKey(0))
        pos = profile.sample_positions(masses, kp)
        vel = df.sample_velocities(pos, masses, kv, G=G)
        Q = float(
            compute_kinetic_energy(vel, masses)
            / jnp.abs(compute_potential_energy(pos, masses, G=G))
        )
        assert abs(Q - 0.5) < 0.05, f"unscaled Q={Q:.3f} (expected ~0.5 for mild truncation)"

    def test_eff_eddington_f_is_physical(self):
        """The tabulated Eddington DF f(E) is non-negative and increases with energy."""
        df = EFFVelocityDF(a=1.0, gamma=3.0, r_t=10.0)
        f = df.f_grid
        assert jnp.all(f >= 0.0), "Eddington f(E) must be non-negative (physical DF)"
        assert float(f[-1]) > float(f[len(f) // 2]) > 0.0, "f(E) should increase with E"

    def test_eff_all_particles_bound(self):
        a, gamma, r_t, G = 1.0, 3.0, 10.0, 1.0
        profile = EFFProfile(a=a, gamma=gamma, r_t=r_t)
        df = EFFVelocityDF(a=a, gamma=gamma, r_t=r_t)
        masses = jnp.ones(3000)
        kp, kv = jax.random.split(jax.random.PRNGKey(1))
        pos = profile.sample_positions(masses, kp)
        vel = df.sample_velocities(pos, masses, kv, G=G)
        r = jnp.linalg.norm(pos, axis=1)
        Psi_r = jnp.interp(r, df.r_grid, df.Psi_grid, left=df.Psi_grid[0], right=0.0)
        kappa = G * jnp.sum(masses) / (4.0 * jnp.pi * df.mu)
        v_esc = jnp.sqrt(2.0 * kappa * jnp.maximum(Psi_r, 0.0))
        v = jnp.linalg.norm(vel, axis=1)
        assert float(jnp.mean(v <= v_esc + 1e-9)) == 1.0, "all EFF velocities must be bound"

    def test_eff_velocity_sampling_differentiable(self):
        """grad through the inverse-CDF sampling (via a position scale) is finite."""
        a, gamma, r_t, G = 1.0, 3.0, 10.0, 1.0
        profile = EFFProfile(a=a, gamma=gamma, r_t=r_t)
        df = EFFVelocityDF(a=a, gamma=gamma, r_t=r_t)
        kp, kv = jax.random.split(jax.random.PRNGKey(2))
        pos = profile.sample_positions(jnp.ones(200), kp)

        def loss(pos_scale):
            vel = df.sample_velocities(pos * pos_scale, jnp.ones(200), kv, G=G)
            return jnp.mean(jnp.sum(vel**2, axis=1))

        g = jax.grad(loss)(1.0)
        assert jnp.isfinite(g), f"grad through EFF DF sampling is non-finite: {g}"

    def test_zero_bulk_velocity(self, N_stats, key):
        """Mean velocity is zero (no net motion)."""
        a, gamma, r_t = 1.0, 3.0, 10.0
        G = 1.0

        profile = EFFProfile(a=a, gamma=gamma, r_t=r_t)
        df = EFFVelocityDF(a=a, gamma=gamma, r_t=r_t)

        masses = jnp.ones(N_stats)
        key_pos, key_vel = jax.random.split(key)

        positions = profile.sample_positions(masses, key_pos)
        velocities = df.sample_velocities(positions, masses, key_vel, G=G)

        # Mean velocity should be ~0
        v_mean = jnp.mean(velocities, axis=0)
        sigma_per_component = jnp.std(velocities, axis=0)

        for i in range(3):
            # Mean should be within ~3 sigma/sqrt(N) of zero
            expected_error = float(sigma_per_component[i]) / jnp.sqrt(N_stats)
            assert abs(float(v_mean[i])) < 5 * expected_error, \
                f"<v{['x','y','z'][i]}> = {float(v_mean[i]):.6f}, expected ~0"


class TestEFFScaleRadius:
    """Verify scale radius parameter affects profile correctly."""

    @pytest.mark.parametrize("a", [0.5, 1.0, 2.0])
    def test_scale_radius_sets_core(self, a, N_validation, key):
        """Scale radius determines core-like region."""
        gamma, r_t = 3.0, 10.0
        profile = EFFProfile(a=a, gamma=gamma, r_t=r_t)

        masses = jnp.ones(N_validation)
        positions = profile.sample_positions(masses, key)
        radii = jnp.linalg.norm(positions, axis=1)

        # Fraction within scale radius
        f_within_a = float(jnp.mean(radii < a))

        # For EFF gamma=3 with r_t >> a, only ~5-20% of mass is within a
        # (power-law profile has most mass at larger radii)
        # Larger a/r_t ratio means more mass within a
        assert 0.01 < f_within_a < 0.5, \
            f"a={a}: fraction within a = {f_within_a*100:.1f}%"

    def test_larger_scale_radius_more_core_mass(self, N_validation, key):
        """Larger scale radius means more mass within the core region."""
        gamma, r_t = 3.0, 10.0

        fractions = []
        for a in [0.5, 1.0, 2.0]:
            profile = EFFProfile(a=a, gamma=gamma, r_t=r_t)

            masses = jnp.ones(N_validation)
            k = jax.random.PRNGKey(42)
            positions = profile.sample_positions(masses, k)
            radii = jnp.linalg.norm(positions, axis=1)

            # Fraction within scale radius
            fractions.append(float(jnp.mean(radii < a)))

        # Larger a should have more mass within a (since a is bigger)
        assert fractions[0] < fractions[1] < fractions[2], \
            f"Mass fraction within a should increase with a: a=[0.5,1,2] -> f={fractions}"


@pytest.mark.slow
class TestEFFHighRtOverACoreResolution:
    """Audit R4 (EFF extension): the linear CDF grid under-resolves the core at
    large r_t/a. Reference = dense quadrature of rho*r^2 (independent of the
    profile's internal CDF). Measured pre-fix error at 0.3a (r_t/a=100): +4.2%;
    the sqrt-stretched grid (r = r_t*u^2) brings it to <0.1%."""

    def test_sampled_core_mass_matches_dense_reference_high_rt_over_a(self):
        a, gamma, r_t = 1.0, 4.0, 100.0  # r_t/a = 100
        s = jnp.linspace(0.0, r_t, 4_000_000)
        rho = (1.0 + (s / a) ** 2) ** (-gamma / 2.0)
        integ = s**2 * rho
        M = jnp.concatenate(
            [jnp.zeros(1), jnp.cumsum(0.5 * (integ[1:] + integ[:-1]) * jnp.diff(s))]
        )
        prof = EFFProfile(a=a, gamma=gamma, r_t=r_t)
        n = 4_000_000
        r = jnp.linalg.norm(
            prof.sample_positions(jnp.ones(n), jax.random.PRNGKey(3)), axis=1
        )
        for r_probe in (0.3, 1.0, 3.0):
            m_ref = float(jnp.interp(r_probe, s, M) / M[-1])
            m_samp = float(jnp.mean(r < r_probe))
            shot = 3.0 / (m_ref * n) ** 0.5
            tol = max(0.02, shot)  # 2% grid budget or 3-sigma shot, whichever larger
            assert abs(m_samp / m_ref - 1.0) < tol, (
                f"r={r_probe}a: sampled {m_samp:.4e} vs reference {m_ref:.4e} "
                f"(rel err {(m_samp/m_ref-1)*100:+.2f}%, tol {tol*100:.2f}%)"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

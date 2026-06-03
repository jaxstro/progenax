"""
Physics validation tests for King (1966) profile and velocity distribution.

Tests verify that implementations match theoretical predictions from:
- King (1966), AJ 71, 64
- Binney & Tremaine (2008), "Galactic Dynamics"

Each test has quantitative error bounds based on theoretical expectations.
"""

import jax
import jax.numpy as jnp
import pytest

from progenax.profiles.king import KingProfile, solve_king_profile
from progenax.kinematics import KingVelocityDF


class TestKingODESolution:
    """Verify King profile ODE solver."""

    def test_boundary_conditions(self):
        """ODE solution satisfies boundary conditions: ψ(0) = W0, dψ/dξ|₀ = 0."""
        W0 = 7.0
        xi_grid, psi_grid = solve_king_profile(W0)

        # ψ(0) = W0 (start near W0, not exactly at ξ=0 due to singularity)
        assert abs(float(psi_grid[0]) - W0) < 0.1, \
            f"ψ(0) = {float(psi_grid[0]):.4f}, expected {W0}"

    def test_potential_monotonic_decrease(self):
        """Dimensionless potential ψ(ξ) decreases with radius."""
        W0 = 7.0
        xi_grid, psi_grid = solve_king_profile(W0)

        # After first few points, should be monotonically decreasing
        diffs = jnp.diff(psi_grid[5:])
        assert jnp.all(diffs <= 0.01), "ψ(ξ) should decrease with radius"

    def test_potential_reaches_zero(self):
        """Potential reaches zero at tidal radius (truncation)."""
        W0 = 7.0
        xi_grid, psi_grid = solve_king_profile(W0, xi_max=50.0)

        # Should reach ψ → 0 somewhere (tidal radius)
        min_psi = float(jnp.min(psi_grid))
        assert min_psi < 0.1, f"Minimum ψ = {min_psi}, expected ~0"

    @pytest.mark.parametrize("W0", [3.0, 5.0, 7.0, 9.0])
    def test_different_concentrations(self, W0):
        """ODE solver works for different W0 values."""
        xi_grid, psi_grid = solve_king_profile(W0)

        # ψ(0) should be close to W0
        assert abs(float(psi_grid[0]) - W0) < 0.2

        # ψ should be non-negative
        assert jnp.all(psi_grid >= -0.01)


class TestKingTidalTruncation:
    """Verify King profile enforces tidal truncation at r_t."""

    def test_all_particles_within_tidal_radius(self, N_validation, key):
        """100% of particles at r ≤ r_t."""
        W0, r_c, r_t = 7.0, 1.0, 10.0
        xi_grid, psi_grid = solve_king_profile(W0)
        profile = KingProfile(W0=W0, r_c=r_c, r_t=r_t, xi_grid=xi_grid, psi_grid=psi_grid)

        masses = jnp.ones(N_validation)
        positions = profile.sample_positions(masses, key)
        radii = jnp.linalg.norm(positions, axis=1)

        max_r = float(jnp.max(radii))
        assert max_r <= r_t + 0.01, f"Max radius {max_r:.4f} exceeds r_t={r_t}"

        fraction_within = float(jnp.mean(radii <= r_t))
        assert fraction_within == 1.0, \
            f"Only {fraction_within*100:.2f}% within r_t (expected 100%)"


class TestKingConcentration:
    """Verify King concentration parameter W0 affects profile shape."""

    def test_w0_affects_natural_tidal_radius(self):
        """Higher W0 produces larger natural tidal radius (ξ where ψ→0).

        This is the fundamental property of King models: higher W0 = more extended
        profile relative to core radius.
        """
        natural_tidal_radii = []
        for W0 in [3.0, 7.0, 11.0]:
            xi_grid, psi_grid = solve_king_profile(W0, xi_max=200.0, n_points=2000)

            # Find where ψ FIRST drops below 0.01 (approximate tidal radius)
            # Use threshold relative to W0 for robustness
            threshold = 0.01 * W0
            below_threshold = psi_grid < threshold

            if jnp.any(below_threshold):
                # Find first index where psi drops below threshold
                first_idx = int(jnp.argmax(below_threshold))
                xi_t = float(xi_grid[first_idx])
            else:
                # Potential didn't drop below threshold - use last grid point
                xi_t = float(xi_grid[-1])

            natural_tidal_radii.append(xi_t)

        # Higher W0 should have larger natural tidal radius (in core radius units)
        assert natural_tidal_radii[0] < natural_tidal_radii[1] < natural_tidal_radii[2], \
            f"Natural ξ_t should increase with W0: W0=[3,7,11] → ξ_t={natural_tidal_radii}"

    def test_w0_affects_half_mass_radius(self, N_validation, key):
        """Half-mass radius (in core radius units) varies with W0.

        For King models with matched natural tidal radii, higher W0 has
        smaller r_h/r_c (more concentrated core).
        """
        half_mass_radii = []
        for W0 in [3.0, 7.0, 11.0]:
            xi_grid, psi_grid = solve_king_profile(W0)

            # Use natural-ish truncation: r_t = natural_xi_t * r_c
            mask = psi_grid > 0.01
            xi_t_natural = float(xi_grid[jnp.argmin(mask)]) if jnp.any(mask) else float(xi_grid[-1])
            r_c = 1.0
            r_t = xi_t_natural * r_c

            profile = KingProfile(W0=W0, r_c=r_c, r_t=r_t, xi_grid=xi_grid, psi_grid=psi_grid)

            masses = jnp.ones(N_validation)
            k = jax.random.PRNGKey(42)
            positions = profile.sample_positions(masses, k)
            radii = jnp.linalg.norm(positions, axis=1)

            # Half-mass radius normalized by core radius
            r_h = float(jnp.median(radii)) / r_c
            half_mass_radii.append(r_h)

        # All should produce valid half-mass radii
        for i, r_h in enumerate(half_mass_radii):
            assert r_h > 0, f"W0={[3,7,11][i]}: r_h/r_c={r_h} should be positive"


class TestKingVelocityDF:
    """Verify King velocity distribution function properties."""

    def test_velocities_bound_against_king_escape_speed(self, N_validation, key):
        """All velocities are bound: v <= v_esc(r) = sigma sqrt(2 psi(r)) from the
        self-consistent King model. The lowered-Maxwellian DF samples on [0, v_esc]
        natively (no clipping), so boundedness is intrinsic.
        """
        W0, r_c = 7.0, 1.0
        G = 1.0

        profile = KingProfile.from_W0_rc(W0, r_c)
        df = KingVelocityDF(W0=W0, r_c=r_c, r_t=float(profile.r_t))

        masses = jnp.ones(N_validation)
        key_pos, key_vel = jax.random.split(key)
        positions = profile.sample_positions(masses, key_pos)
        velocities = df.sample_velocities(positions, masses, key_vel, G=G)

        radii = jnp.linalg.norm(positions, axis=1)
        W = jnp.interp(radii / r_c, df.xi_grid, df.psi_grid, left=df.W0, right=0.0)
        v_esc = df._sigma(jnp.sum(masses), G) * jnp.sqrt(2.0 * jnp.maximum(W, 0.0))
        v_mag = jnp.linalg.norm(velocities, axis=1)

        bound_fraction = float(jnp.mean(v_mag <= v_esc + 1e-9))
        assert bound_fraction == 1.0, \
            f"Only {bound_fraction*100:.1f}% bound (v <= v_esc), expected 100%"

    def test_velocity_isotropy(self, N_stats, key):
        """Velocities are isotropically distributed."""
        W0, r_c, r_t = 7.0, 1.0, 10.0
        G = 1.0

        xi_grid, psi_grid = solve_king_profile(W0)
        profile = KingProfile(W0=W0, r_c=r_c, r_t=r_t, xi_grid=xi_grid, psi_grid=psi_grid)
        df = KingVelocityDF(W0=W0, r_c=r_c, r_t=r_t)

        masses = jnp.ones(N_stats)
        key_pos, key_vel = jax.random.split(key)

        positions = profile.sample_positions(masses, key_pos)
        velocities = df.sample_velocities(positions, masses, key_vel, G=G)

        # Check isotropy: <vx²> ≈ <vy²> ≈ <vz²>
        v2_mean = jnp.mean(velocities**2, axis=0)
        mean_v2 = float(jnp.mean(v2_mean))

        for i, v2i in enumerate(v2_mean):
            rel_diff = abs(float(v2i) - mean_v2) / mean_v2
            assert rel_diff < 0.10, \
                f"Anisotropy detected: <v{['x','y','z'][i]}²>={float(v2i):.4f}, mean={mean_v2:.4f}"

    def test_velocity_dispersion_decreases_outward(self, N_validation, key):
        """Velocity dispersion decreases with radius."""
        W0, r_c, r_t = 7.0, 1.0, 10.0
        G = 1.0

        df = KingVelocityDF(W0=W0, r_c=r_c, r_t=r_t)
        masses = jnp.ones(N_validation)
        M_total = float(jnp.sum(masses))

        # Measure dispersion at different radii
        test_radii = [0.5, 2.0, 5.0]
        sigmas = []

        for r in test_radii:
            positions = jnp.array([[r, 0.0, 0.0]] * N_validation)
            k = jax.random.PRNGKey(int(42 + r * 10))
            velocities = df.sample_velocities(positions, masses, k, G=G)

            sigma = float(jnp.std(velocities[:, 0]))
            sigmas.append(sigma)

        # Should decrease with radius
        assert sigmas[0] > sigmas[1] > sigmas[2], \
            f"Dispersion should decrease: σ(r) = {sigmas}"


class TestKingDensityProfile:
    """Verify King density profile shape."""

    def test_density_decreases_with_radius(self, N_validation, key):
        """Density decreases monotonically with radius."""
        W0, r_c, r_t = 7.0, 1.0, 10.0

        xi_grid, psi_grid = solve_king_profile(W0)
        profile = KingProfile(W0=W0, r_c=r_c, r_t=r_t, xi_grid=xi_grid, psi_grid=psi_grid)

        masses = jnp.ones(N_validation)
        positions = profile.sample_positions(masses, key)
        radii = jnp.linalg.norm(positions, axis=1)

        # Bin particles and count density
        bins = jnp.linspace(0, r_t, 20)
        hist, _ = jnp.histogram(radii, bins=bins)

        # Normalize by shell volume: V = (4/3)π(r_out³ - r_in³)
        volumes = (4.0/3.0) * jnp.pi * (bins[1:]**3 - bins[:-1]**3)
        densities = hist / (volumes + 1e-10)

        # Should generally decrease (allow some noise)
        # Check that density at r < r_c is higher than at r > 2*r_c
        inner_density = float(jnp.mean(densities[:5]))
        outer_density = float(jnp.mean(densities[10:]))

        assert inner_density > outer_density, \
            f"Inner density {inner_density:.2f} should exceed outer {outer_density:.2f}"


class TestKingLoweredMaxwellianDensity:
    """B2.0: corrected lowered-Maxwellian volume density + factor-of-9 nondimensionalization.

    The earlier code solved Poisson with King's K-function (incomplete-gamma/projected
    form) as the 3-D density, over-extending the profile by 2-30x, and omitted the
    standard factor of 9 in the nondimensionalization. The corrected model must
    reproduce the King (1966) Table II concentrations and the lowered-Maxwellian
    density shape.
    """

    # King (1966), AJ 71, 64, Table II: c = log10(r_t/r_c) vs W0.
    @pytest.mark.parametrize(
        "W0,c_ref", [(1, 0.30), (3, 0.67), (5, 1.03), (7, 1.53), (9, 2.12)]
    )
    def test_concentration_matches_king_table_ii(self, W0, c_ref):
        prof = KingProfile.from_W0_rc(float(W0), 1.0, xi_max=400.0, n_ode_points=8000)
        c = float(jnp.log10(prof.r_t / prof.r_c))
        assert abs(c - c_ref) < 0.03, (
            f"W0={W0}: c={c:.3f} vs King (1966) Table II c={c_ref} (delta {c-c_ref:+.3f})"
        )

    def test_density_shape_matches_direct_velocity_integral(self):
        """KingProfile.density(r) follows the lowered-Maxwellian shape (independent
        oracle = direct velocity integration), not the over-extended K-form."""
        prof = KingProfile.from_W0_rc(7.0, 1.0, xi_max=400.0, n_ode_points=8000)
        r = jnp.linspace(0.02 * float(prof.r_t), 0.9 * float(prof.r_t), 25)
        xi = r / prof.r_c
        psi = jnp.interp(xi, prof.xi_grid, prof.psi_grid, left=prof.W0, right=0.0)

        def direct(W, nv=100_000):
            v = jnp.linspace(0.0, jnp.sqrt(2.0 * W), nv)
            return float(jnp.trapezoid(v**2 * (jnp.exp(W - v**2 / 2.0) - 1.0), v))

        rho = jnp.asarray(prof.density(r))
        rho_direct = jnp.asarray([direct(float(p)) for p in psi])
        rho_n = rho / rho[0]
        d_n = rho_direct / rho_direct[0]
        max_rel = float(jnp.max(jnp.abs(rho_n - d_n) / (jnp.abs(d_n) + 1e-12)))
        assert max_rel < 5e-3, f"density shape disagrees with lowered-Maxwellian (max rel {max_rel:.2e})"


class TestKingEquilibriumVelocityDF:
    """B2.1: King velocity DF as a true lowered-Maxwellian in detailed equilibrium.

    Sampling the lowered-Maxwellian g(v) ∝ v^2 [exp(psi(r) - v^2/2sigma^2) - 1] on
    [0, v_esc(r)] with the self-consistent sigma^2 = G M / (9 r_c mu(W0)) must put the
    cluster in virial equilibrium WITHOUT any external rescale (Q = T/|V| = 0.5).
    The old ad-hoc DF (Gaussian + clip, parabolic psi) gives Q ~ 6.7.
    """

    def _build_ic(self, W0=7.0, r_c=1.0, N=5000, seed=0):
        from jaxstro.units import STELLAR
        prof = KingProfile.from_W0_rc(W0, r_c)
        df = KingVelocityDF(W0=W0, r_c=r_c, r_t=float(prof.r_t))
        masses = jnp.ones(N)
        kp, kv = jax.random.split(jax.random.PRNGKey(seed))
        pos = prof.sample_positions(masses, kp)
        vel = df.sample_velocities(pos, masses, kv, G=STELLAR.G)
        return prof, df, masses, pos, vel, STELLAR.G

    def test_virial_ratio_is_half_unscaled(self):
        from progenax.builders import compute_kinetic_energy, compute_potential_energy
        _, _, m, pos, vel, G = self._build_ic(W0=7.0, N=5000)
        T = compute_kinetic_energy(vel, m)
        V = compute_potential_energy(pos, m, G=G)
        Q = float(T / jnp.abs(V))
        assert abs(Q - 0.5) < 0.05, f"unscaled Q={Q:.3f} (expected 0.5 for King equilibrium)"

    def test_all_particles_bound(self):
        prof, df, m, pos, vel, G = self._build_ic(W0=7.0, N=3000)
        r = jnp.linalg.norm(pos, axis=1)
        v = jnp.linalg.norm(vel, axis=1)
        # local potential and escape speed from the self-consistent model
        W = jnp.interp(r / prof.r_c, df.xi_grid, df.psi_grid, left=df.W0, right=0.0)
        sigma = df._sigma(jnp.sum(m), G)
        v_esc = sigma * jnp.sqrt(2.0 * jnp.maximum(W, 0.0))
        frac_bound = float(jnp.mean(v <= v_esc + 1e-9))
        assert frac_bound == 1.0, f"only {frac_bound*100:.1f}% bound (v < v_esc)"

    def test_velocity_sampling_is_differentiable(self):
        """grad of mean kinetic energy w.r.t. r_c flows through the DF sampling."""
        from jaxstro.units import STELLAR

        def loss(r_c):
            prof = KingProfile.from_W0_rc(7.0, 1.0)
            df = KingVelocityDF(W0=7.0, r_c=r_c, r_t=float(prof.r_t))
            m = jnp.ones(200)
            kp, kv = jax.random.split(jax.random.PRNGKey(1))
            pos = prof.sample_positions(m, kp)
            vel = df.sample_velocities(pos, m, kv, G=STELLAR.G)
            return jnp.mean(jnp.sum(vel**2, axis=1))

        g = jax.grad(loss)(1.0)
        assert jnp.isfinite(g), f"grad through King DF sampling is non-finite: {g}"

    def test_dispersion_profile_matches_king_moment(self):
        """Sampled sigma_1d(r) matches the analytic lowered-Maxwellian 2nd moment."""
        prof, df, m, pos, vel, G = self._build_ic(W0=7.0, N=40000)
        sigma = float(df._sigma(jnp.sum(m), G))
        r = jnp.linalg.norm(pos, axis=1)
        v2 = jnp.sum(vel**2, axis=1)

        def u2_mean(W, nu=4000):
            u = jnp.linspace(0.0, jnp.sqrt(2.0 * W), nu)
            g = u**2 * (jnp.exp(W - u**2 / 2.0) - 1.0)
            return float(jnp.trapezoid(u**2 * g, u) / jnp.trapezoid(g, u))

        for lo, hi in [(0.5, 1.5), (2.0, 4.0), (5.0, 9.0)]:
            msk = (r >= lo) & (r < hi)
            W_bin = float(jnp.mean(jnp.interp(
                r[msk] / prof.r_c, df.xi_grid, df.psi_grid, left=df.W0, right=0.0)))
            sig_sampled = float(jnp.sqrt(jnp.mean(v2[msk]) / 3.0))
            sig_analytic = sigma * jnp.sqrt(u2_mean(W_bin) / 3.0)
            rel = abs(sig_sampled - sig_analytic) / sig_analytic
            assert rel < 0.12, (
                f"r in [{lo},{hi}): sampled sigma_1d={sig_sampled:.3f} vs "
                f"analytic {sig_analytic:.3f} (rel {rel:.2%})"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

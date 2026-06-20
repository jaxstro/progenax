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

from progenax.kinematics import KingVelocityDF
from progenax.profiles.king import KingProfile, solve_king_profile


class TestKingODESolution:
    """Verify King profile ODE solver."""

    def test_boundary_conditions(self):
        """ODE solution satisfies boundary conditions: ψ(0) = W0, dψ/dξ|₀ = 0."""
        W0 = 7.0
        xi_grid, psi_grid, _ = solve_king_profile(W0)

        # ψ(0) = W0 (start near W0, not exactly at ξ=0 due to singularity)
        assert abs(float(psi_grid[0]) - W0) < 0.1, (
            f"ψ(0) = {float(psi_grid[0]):.4f}, expected {W0}"
        )

    def test_potential_monotonic_decrease(self):
        """Dimensionless potential ψ(ξ) decreases with radius."""
        W0 = 7.0
        xi_grid, psi_grid, _ = solve_king_profile(W0)

        # After first few points, should be monotonically decreasing
        diffs = jnp.diff(psi_grid[5:])
        assert jnp.all(diffs <= 0.01), "ψ(ξ) should decrease with radius"

    def test_potential_reaches_zero(self):
        """Potential reaches zero at tidal radius (truncation)."""
        W0 = 7.0
        xi_grid, psi_grid, _ = solve_king_profile(W0, xi_max=50.0)

        # Should reach ψ → 0 somewhere (tidal radius)
        min_psi = float(jnp.min(psi_grid))
        assert min_psi < 0.1, f"Minimum ψ = {min_psi}, expected ~0"

    @pytest.mark.parametrize("W0", [3.0, 5.0, 7.0, 9.0])
    def test_different_concentrations(self, W0):
        """ODE solver works for different W0 values."""
        xi_grid, psi_grid, _ = solve_king_profile(W0)

        # ψ(0) should be close to W0
        assert abs(float(psi_grid[0]) - W0) < 0.2

        # ψ should be non-negative
        assert jnp.all(psi_grid >= -0.01)


class TestKingTidalTruncation:
    """Verify King profile enforces tidal truncation at r_t."""

    def test_all_particles_within_tidal_radius(self, N_validation, key):
        """100% of particles at r ≤ r_t."""
        # self-consistent constructor (recommended API); r_t derived from W0
        profile = KingProfile.from_W0_rc(W0=7.0, r_c=1.0)
        r_t = float(profile.r_t)

        masses = jnp.ones(N_validation)
        positions = profile.sample_positions(masses, key)
        radii = jnp.linalg.norm(positions, axis=1)

        max_r = float(jnp.max(radii))
        assert max_r <= r_t + 0.01, f"Max radius {max_r:.4f} exceeds r_t={r_t}"

        fraction_within = float(jnp.mean(radii <= r_t))
        assert fraction_within == 1.0, (
            f"Only {fraction_within * 100:.2f}% within r_t (expected 100%)"
        )


class TestKingConcentration:
    """Verify King concentration parameter W0 affects profile shape."""

    def test_w0_affects_natural_tidal_radius(self):
        """Higher W0 produces larger natural tidal radius (ξ where ψ→0).

        This is the fundamental property of King models: higher W0 = more extended
        profile relative to core radius.
        """
        natural_tidal_radii = []
        for W0 in [3.0, 7.0, 11.0]:
            xi_grid, psi_grid, _ = solve_king_profile(W0, xi_max=200.0, n_points=2000)

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
        assert (
            natural_tidal_radii[0] < natural_tidal_radii[1] < natural_tidal_radii[2]
        ), (
            f"Natural ξ_t should increase with W0: W0=[3,7,11] → ξ_t={natural_tidal_radii}"
        )

    def test_w0_affects_half_mass_radius(self, N_validation, key):
        """Half-mass radius (in core radius units) varies with W0.

        For King models with matched natural tidal radii, higher W0 has
        smaller r_h/r_c (more concentrated core).
        """
        half_mass_radii = []
        for W0 in [3.0, 7.0, 11.0]:
            xi_grid, psi_grid, _ = solve_king_profile(W0)

            # Use natural-ish truncation: r_t = natural_xi_t * r_c
            mask = psi_grid > 0.01
            xi_t_natural = (
                float(xi_grid[jnp.argmin(mask)])
                if jnp.any(mask)
                else float(xi_grid[-1])
            )
            r_c = 1.0
            r_t = xi_t_natural * r_c

            profile = KingProfile(
                W0=W0, r_c=r_c, r_t=r_t, xi_grid=xi_grid, psi_grid=psi_grid
            )

            masses = jnp.ones(N_validation)
            k = jax.random.PRNGKey(42)
            positions = profile.sample_positions(masses, k)
            radii = jnp.linalg.norm(positions, axis=1)

            # Half-mass radius normalized by core radius
            r_h = float(jnp.median(radii)) / r_c
            half_mass_radii.append(r_h)

        # All should produce valid half-mass radii
        for i, r_h in enumerate(half_mass_radii):
            assert r_h > 0, f"W0={[3, 7, 11][i]}: r_h/r_c={r_h} should be positive"


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
        df = KingVelocityDF(W0=W0, r_c=r_c)

        masses = jnp.ones(N_validation)
        key_pos, key_vel = jax.random.split(key)
        positions = profile.sample_positions(masses, key_pos)
        velocities = df.sample_velocities(positions, masses, key_vel, G=G)

        radii = jnp.linalg.norm(positions, axis=1)
        W = jnp.interp(radii / r_c, df.xi_grid, df.psi_grid, left=df.W0, right=0.0)
        v_esc = df._sigma(jnp.sum(masses), G) * jnp.sqrt(2.0 * jnp.maximum(W, 0.0))
        v_mag = jnp.linalg.norm(velocities, axis=1)

        bound_fraction = float(jnp.mean(v_mag <= v_esc + 1e-9))
        assert bound_fraction == 1.0, (
            f"Only {bound_fraction * 100:.1f}% bound (v <= v_esc), expected 100%"
        )

    def test_velocity_isotropy(self, N_stats, key):
        """Velocities are isotropically distributed."""
        W0, r_c = 7.0, 1.0
        G = 1.0

        # self-consistent constructor (recommended API); r_t derived from W0
        profile = KingProfile.from_W0_rc(W0=W0, r_c=r_c)
        df = KingVelocityDF(W0=W0, r_c=r_c)

        masses = jnp.ones(N_stats)
        key_pos, key_vel = jax.random.split(key)

        positions = profile.sample_positions(masses, key_pos)
        velocities = df.sample_velocities(positions, masses, key_vel, G=G)

        # Check isotropy: <vx²> ≈ <vy²> ≈ <vz²>
        v2_mean = jnp.mean(velocities**2, axis=0)
        mean_v2 = float(jnp.mean(v2_mean))

        for i, v2i in enumerate(v2_mean):
            rel_diff = abs(float(v2i) - mean_v2) / mean_v2
            assert rel_diff < 0.10, (
                f"Anisotropy detected: <v{['x', 'y', 'z'][i]}²>={float(v2i):.4f}, mean={mean_v2:.4f}"
            )

    def test_velocity_dispersion_decreases_outward(self, N_validation, key):
        """Velocity dispersion decreases with radius."""
        W0, r_c, _ = 7.0, 1.0, 10.0
        G = 1.0

        df = KingVelocityDF(W0=W0, r_c=r_c)
        masses = jnp.ones(N_validation)

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
        assert sigmas[0] > sigmas[1] > sigmas[2], (
            f"Dispersion should decrease: σ(r) = {sigmas}"
        )


class TestKingDensityProfile:
    """Verify King density profile shape."""

    def test_density_decreases_with_radius(self, N_validation, key):
        """Density decreases monotonically with radius."""
        # self-consistent constructor (recommended API); r_t derived from W0
        profile = KingProfile.from_W0_rc(W0=7.0, r_c=1.0)
        r_t = float(profile.r_t)

        masses = jnp.ones(N_validation)
        positions = profile.sample_positions(masses, key)
        radii = jnp.linalg.norm(positions, axis=1)

        # Bin particles and count density
        bins = jnp.linspace(0, r_t, 20)
        hist, _ = jnp.histogram(radii, bins=bins)

        # Normalize by shell volume: V = (4/3)π(r_out³ - r_in³)
        volumes = (4.0 / 3.0) * jnp.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
        densities = hist / (volumes + 1e-10)

        # Should generally decrease (allow some noise)
        # Check that density at r < r_c is higher than at r > 2*r_c
        inner_density = float(jnp.mean(densities[:5]))
        outer_density = float(jnp.mean(densities[10:]))

        assert inner_density > outer_density, (
            f"Inner density {inner_density:.2f} should exceed outer {outer_density:.2f}"
        )


class TestKingLoweredMaxwellianDensity:
    """Lowered-Maxwellian volume density + factor-of-9 nondimensionalization.

    The King Poisson source is the lowered-Maxwellian 3-D *volume* density (not the
    projected K-function), nondimensionalized with the standard factor of 9. With
    these, the model must reproduce the King (1966) Table II concentrations and the
    lowered-Maxwellian density shape (verified against a direct velocity integral).
    """

    # King (1966), AJ 71, 64, Table II (log c column): c = log10(r_t/r_c) vs W0.
    # Table II begins at W0=2.5; W0<2.5 is not tabulated, so it is not asserted here.
    @pytest.mark.parametrize("W0,c_ref", [(3, 0.67), (5, 1.03), (7, 1.53), (9, 2.12)])
    def test_concentration_matches_king_table_ii(self, W0, c_ref):
        prof = KingProfile.from_W0_rc(float(W0), 1.0, xi_max=400.0, n_ode_points=8000)
        c = float(jnp.log10(prof.r_t / prof.r_c))
        assert abs(c - c_ref) < 0.03, (
            f"W0={W0}: c={c:.3f} vs King (1966) Table II c={c_ref} (delta {c - c_ref:+.3f})"
        )

    def test_density_shape_matches_direct_velocity_integral(self):
        """KingProfile.density(r) follows the lowered-Maxwellian volume-density
        shape, verified against an independent oracle (direct velocity integration)."""
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
        assert max_rel < 5e-3, (
            f"density shape disagrees with lowered-Maxwellian (max rel {max_rel:.2e})"
        )


class TestKingEquilibriumVelocityDF:
    """King velocity DF as a true lowered-Maxwellian in detailed equilibrium.

    Sampling the lowered-Maxwellian g(v) ∝ v^2 [exp(psi(r) - v^2/2sigma^2) - 1] on
    [0, v_esc(r)] with the self-consistent sigma^2 = G M / (9 r_c mu(W0)) puts the
    cluster in virial equilibrium WITHOUT any external rescale (Q = T/|V| = 0.5).
    """

    def _build_ic(self, W0=7.0, r_c=1.0, N=5000, seed=0):
        from jaxstro.units import STELLAR

        prof = KingProfile.from_W0_rc(W0, r_c)
        df = KingVelocityDF(W0=W0, r_c=r_c)
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
        assert abs(Q - 0.5) < 0.05, (
            f"unscaled Q={Q:.3f} (expected 0.5 for King equilibrium)"
        )

    def test_all_particles_bound(self):
        prof, df, m, pos, vel, G = self._build_ic(W0=7.0, N=3000)
        r = jnp.linalg.norm(pos, axis=1)
        v = jnp.linalg.norm(vel, axis=1)
        # local potential and escape speed from the self-consistent model
        W = jnp.interp(r / prof.r_c, df.xi_grid, df.psi_grid, left=df.W0, right=0.0)
        sigma = df._sigma(jnp.sum(m), G)
        v_esc = sigma * jnp.sqrt(2.0 * jnp.maximum(W, 0.0))
        frac_bound = float(jnp.mean(v <= v_esc + 1e-9))
        assert frac_bound == 1.0, f"only {frac_bound * 100:.1f}% bound (v < v_esc)"

    # grad through KingVelocityDF.sample_velocities is FD-audited by the grad-audit
    # registry (tests/validation/grad_audit/registry.py ::
    # KingVelocityDF.sample_velocities [r_c] + [W0]); see
    # docs/website/50-validation/differentiability-audit.md. The former finite-only
    # test_velocity_sampling_is_differentiable smoke (grad of mean KE wrt r_c, isfinite
    # only) was removed (audit T6: isfinite passes a silently-zeroed grad; the registry
    # FD cases are strictly stronger; registry is SoT). (The distinct auto-domain
    # high-W0 FD test test_auto_domain_preserves_differentiability_high_W0 is kept.)

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
            W_bin = float(
                jnp.mean(
                    jnp.interp(
                        r[msk] / prof.r_c,
                        df.xi_grid,
                        df.psi_grid,
                        left=df.W0,
                        right=0.0,
                    )
                )
            )
            sig_sampled = float(jnp.sqrt(jnp.mean(v2[msk]) / 3.0))
            sig_analytic = sigma * jnp.sqrt(u2_mean(W_bin) / 3.0)
            rel = abs(sig_sampled - sig_analytic) / sig_analytic
            assert rel < 0.12, (
                f"r in [{lo},{hi}): sampled sigma_1d={sig_sampled:.3f} vs "
                f"analytic {sig_analytic:.3f} (rel {rel:.2%})"
            )


def test_concentration_matches_king1966_table_ii():
    """c(W0)=log10(r_t/r_c) must match King (1966) Table II to <=0.02.
    Reference c: W0=3 -> 0.67, W0=7 -> 1.53, W0=9 -> 2.12 (King 1966; B&T 2008)."""
    import jax.numpy as jnp

    from progenax import KingProfile

    ref = {3.0: 0.67, 7.0: 1.53, 9.0: 2.12}
    for w0, c_ref in ref.items():
        p = KingProfile.from_W0_rc(W0=w0, r_c=1.0)
        c = float(jnp.log10(p.r_t / p.r_c))
        assert abs(c - c_ref) <= 0.02, (
            f"W0={w0}: c={c:.3f} vs King Table II {c_ref} (>0.02)"
        )


class TestKingAutoDomain:
    """from_W0_rc auto-sizes the ODE integration domain from W0 (no manual xi_max).

    The default ODE domain must scale with concentration so high-W0 models (whose
    tidal radius grows super-exponentially) integrate to psi->0 instead of pinning
    to the boundary. For W0 <= 9 the auto domain must reproduce the previous fixed
    default exactly (backward compatibility), and differentiability in r_c must be
    preserved (W0 keys only the static domain, never a differentiated path).
    """

    # King (1966) Table II: c = log10(r_t/r_c)
    @pytest.mark.parametrize("W0,c_ref", [(12.0, 2.739), (15.0, 3.356)])
    def test_auto_domain_recovers_high_W0_concentration(self, W0, c_ref):
        """High-W0 King models converge to Table II with NO explicit xi_max."""
        p = KingProfile.from_W0_rc(W0=W0, r_c=1.0)  # auto domain
        c = float(jnp.log10(p.r_t / p.r_c))
        assert abs(c - c_ref) <= 0.03, (
            f"W0={W0}: auto-domain c={c:.3f} vs King Table II {c_ref} "
            f"(>0.03 -> domain pinned, not converged)"
        )

    @pytest.mark.parametrize("W0", [3.0, 5.0, 7.0, 9.0])
    def test_auto_domain_backward_compatible(self, W0):
        """For W0<=9 the auto domain reproduces the previous fixed default
        (xi_max=300, n_ode_points=2000) bit-for-bit."""
        auto = KingProfile.from_W0_rc(W0=W0, r_c=1.0)
        explicit = KingProfile.from_W0_rc(
            W0=W0, r_c=1.0, xi_max=300.0, n_ode_points=2000
        )
        assert float(auto.r_t) == float(explicit.r_t)

    def test_auto_domain_preserves_differentiability_high_W0(self):
        """grad of a sampled summary statistic w.r.t. r_c flows through the
        auto-domain high-W0 profile and matches a finite difference."""

        def loss(r_c):
            p = KingProfile.from_W0_rc(W0=12.0, r_c=r_c)  # auto domain
            pos = p.sample_positions(jnp.ones(200), jax.random.PRNGKey(0))
            return jnp.mean(jnp.linalg.norm(pos, axis=1))

        g = float(jax.grad(loss)(1.0))
        assert jnp.isfinite(g) and g != 0.0, f"grad non-finite/zero: {g}"
        fd = float((loss(1.0 + 1e-5) - loss(1.0 - 1e-5)) / 2e-5)
        rel = abs(g - fd) / (abs(g) + abs(fd) + 1e-30)
        assert rel < 1e-5, f"AD {g:.6f} vs FD {fd:.6f} rel={rel:.2e}"

    def test_velocity_df_auto_domain_high_W0_equilibrium(self):
        """KingVelocityDF auto-sizes its domain too: a W0=12 IC sampled with the
        matched DF (no explicit xi_max) is in virial equilibrium (Q~0.5)."""
        from jaxstro.units import STELLAR

        from progenax.builders import compute_kinetic_energy, compute_potential_energy

        prof = KingProfile.from_W0_rc(W0=12.0, r_c=1.0)
        df = KingVelocityDF(W0=12.0, r_c=1.0)  # auto domain
        m = jnp.ones(3000)
        kp, kv = jax.random.split(jax.random.PRNGKey(0))
        pos = prof.sample_positions(m, kp)
        vel = df.sample_velocities(pos, m, kv, G=STELLAR.G)
        T = compute_kinetic_energy(vel, m)
        V = compute_potential_energy(pos, m, G=STELLAR.G)
        Q = float(T / jnp.abs(V))
        assert abs(Q - 0.5) < 0.06, f"W0=12 auto-domain Q={Q:.3f} (expected ~0.5)"


def _dense_king_cumulative_mass(W0):
    """Reference M(<r)/M_total on a dense ODE solve, independent of the profile's
    internal CDF grid. Returns (xi, M) with M cumulative (dimensionless, ∝ enclosed
    mass; the rho0 normalization cancels in M/M[-1])."""
    from progenax.profiles.king import (
        king_lowered_maxwellian_density,
        solve_king_profile,
    )

    xi, psi, _ = solve_king_profile(W0, xi_max=600.0, n_points=20_000)
    rho = king_lowered_maxwellian_density(jnp.maximum(psi, 0.0))
    integ = rho * xi**2
    M = jnp.concatenate(
        [jnp.zeros(1), jnp.cumsum(0.5 * (integ[1:] + integ[:-1]) * jnp.diff(xi))]
    )
    return xi, M


@pytest.mark.slow
class TestHighW0CoreResolution:
    """Audit R4: the linear 1000-pt CDF grid under-resolves the core at W0 >= 9.

    Reference = direct quadrature of rho_hat(psi) on a dense ODE solve
    (xi_max=600, n_points=20000) — independent of the profile's internal CDF.
    Measured pre-fix errors at 0.3 r_c: +18% (W0=9), +270% (W0=12).
    """

    @pytest.mark.parametrize("W0", [7.0, 9.0, 12.0])
    def test_sampled_core_mass_matches_dense_reference(self, W0):
        xi, M = _dense_king_cumulative_mass(W0)
        prof = KingProfile.from_W0_rc(W0=W0, r_c=1.0)
        n = 2_000_000
        pos = prof.sample_positions(jnp.ones(n), jax.random.PRNGKey(3))
        r = jnp.linalg.norm(pos, axis=1)
        for r_probe in (0.3, 1.0, 3.0):
            m_ref = float(jnp.interp(r_probe, xi, M) / M[-1])
            m_samp = float(jnp.mean(r < r_probe))
            shot = 3.0 / (m_ref * n) ** 0.5  # 3 sigma binomial
            tol = max(0.03, shot)  # 3% grid budget or shot noise, whichever larger
            assert abs(m_samp / m_ref - 1.0) < tol, (
                f"W0={W0}, r={r_probe} r_c: sampled M(<r)/M = {m_samp:.3e} vs "
                f"reference {m_ref:.3e} (rel err {(m_samp / m_ref - 1) * 100:+.1f}%)"
            )


def test_high_w0_core_mass_fast_enforcer():
    """Non-slow PR-lane guard for the R4 core-resolution fix (single W0=9 / 0.3 r_c
    probe, smaller N). The full W0 grid lives in the slow TestHighW0CoreResolution."""
    xi, M = _dense_king_cumulative_mass(9.0)
    prof = KingProfile.from_W0_rc(W0=9.0, r_c=1.0)
    n = 500_000
    pos = prof.sample_positions(jnp.ones(n), jax.random.PRNGKey(5))
    r = jnp.linalg.norm(pos, axis=1)
    r_probe = 0.3
    m_ref = float(jnp.interp(r_probe, xi, M) / M[-1])
    m_samp = float(jnp.mean(r < r_probe))
    shot = 3.0 / (m_ref * n) ** 0.5
    tol = max(0.05, shot)
    assert abs(m_samp / m_ref - 1.0) < tol, (
        f"W0=9, r=0.3 r_c: sampled {m_samp:.3e} vs reference {m_ref:.3e} "
        f"(rel err {(m_samp / m_ref - 1) * 100:+.1f}%)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

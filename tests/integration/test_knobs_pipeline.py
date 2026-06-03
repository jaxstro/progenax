"""Integration tests for IC knobs pipeline."""

import jax
import jax.numpy as jnp
import pytest

# --- QUARANTINED (Batch 0, audit M4) -----------------------------------------
# This integration test is refactor-orphaned: it exercises a pre-refactor
# "knobs pipeline" API that no longer exists. The skip is placed ABOVE the
# progenax imports so the dead `from progenax.profiles.mass_segregation import
# apply_mass_segregation_baumgardt` (and the removed progenax.substructure.fractal
# / apply_fractal_overlay_radial) never execute and abort suite collection.
# Resolution (rewrite to the current energy_sorted_segregation API vs delete) is
# tracked for Batch 4 per-item approval.
# See: docs/notes/2026-06-02-knobs-pipeline-stale-test.md
pytest.skip(
    "Refactor-orphaned knobs-pipeline test: references the pre-refactor API "
    "(progenax.profiles.mass_segregation.apply_mass_segregation_baumgardt, since "
    "redesigned into progenax.cluster.mass_segregation.energy_sorted_segregation "
    "with an incompatible signature; progenax.substructure.fractal and "
    "apply_fractal_overlay_radial removed). Quarantined to restore suite "
    "collection. Tracked in docs/notes/2026-06-02-knobs-pipeline-stale-test.md "
    "(Batch 4, per-item approval).",
    allow_module_level=True,
)

from progenax.profiles import PlummerProfile
from progenax.kinematics import PlummerVelocityDF
from progenax.imf import PowerLawIMF
from progenax.builders import virial_scale, to_com_frame, compute_kinetic_energy, compute_potential_energy
from progenax.kinematics.anisotropy import apply_osipkov_merritt
from progenax.kinematics.rotation import apply_solid_body_rotation
from progenax.profiles.mass_segregation import apply_mass_segregation_baumgardt
from progenax.tidal import apply_tidal_truncation
from progenax.binaries.population import (
    LogNormalPeriod,
    ThermalEccentricity,
    sample_isotropic_orientations,
)


class TestKnobsPipeline:
    """End-to-end tests for applying multiple knobs."""

    def test_full_pipeline(self):
        """Full IC generation with all knobs applied."""
        N = 500
        G = 0.00450  # Stellar units
        r_h = 2.0

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 10)

        # 1. Sample masses
        imf = PowerLawIMF.kroupa()
        masses = imf.sample(keys[0], N)

        # 2. Sample positions
        profile = PlummerProfile(r_h=r_h)
        positions = profile.sample_positions(masses, keys[1])

        # 3. Sample velocities
        velocity_df = PlummerVelocityDF(r_h=r_h)
        velocities = velocity_df.sample_velocities(positions, masses, keys[2], G=G)

        # 4. Apply mass segregation (Baumgardt energy-ranked orbit assignment)
        positions, velocities = apply_mass_segregation_baumgardt(
            positions, velocities, masses, s=0.3, key=keys[3], G=G
        )

        # 5. Apply Osipkov-Merritt anisotropy
        velocities = apply_osipkov_merritt(velocities, positions, keys[4], r_a=r_h)

        # 6. Apply rotation
        velocities = apply_solid_body_rotation(
            velocities, positions, omega=0.05, axis=jnp.array([0., 0., 1.])
        )

        # 7. Apply tidal truncation
        r_t = 5 * r_h
        positions, velocities, masses, mask = apply_tidal_truncation(
            positions, velocities, masses, r_t
        )

        # 8. Center on COM
        positions, velocities = to_com_frame(positions, velocities, masses)

        # 9. Virial scale to Q = 0.5 (equilibrium)
        velocities = virial_scale(positions, velocities, masses, Q_target=0.5, G=G)

        # Verify final state
        assert positions.shape[0] > 0  # Some particles kept
        assert velocities.shape == positions.shape
        assert masses.shape[0] == positions.shape[0]

        # Check virial ratio: Q = T/|V| = 0.5 for equilibrium
        T = compute_kinetic_energy(velocities, masses)
        V = compute_potential_energy(positions, masses, G=G)
        Q = T / jnp.abs(V)  # Q = T/|V| (NOT 2T/|V|)
        assert jnp.abs(Q - 0.5) < 0.01  # Should be virial equilibrium

    def test_binary_parameters_sampling(self):
        """Test sampling binary orbital parameters."""
        N = 100
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)

        # Sample periods
        P_dist = LogNormalPeriod(mu_log_P=4.0, sigma_log_P=2.0)
        periods = P_dist.sample(keys[0], N)
        assert periods.shape == (N,)
        assert jnp.all(periods > 0)

        # Sample eccentricities
        e_dist = ThermalEccentricity()
        eccentricities = e_dist.sample(keys[1], N)
        assert eccentricities.shape == (N,)
        assert jnp.all(eccentricities >= 0)
        assert jnp.all(eccentricities < 1)

        # Sample orientations
        inc, Omega, omega, M_anom = sample_isotropic_orientations(keys[2], N)
        assert inc.shape == (N,)
        assert jnp.all(inc >= 0) and jnp.all(inc <= jnp.pi)

    def test_jit_compatibility(self):
        """All knobs work under JIT compilation."""
        G = 0.00450  # Stellar units

        @jax.jit
        def apply_all_knobs(positions, velocities, masses, key):
            keys = jax.random.split(key, 3)

            # Mass segregation (Baumgardt energy-ranked orbit assignment)
            positions, velocities = apply_mass_segregation_baumgardt(
                positions, velocities, masses, s=0.3, key=keys[0], G=G
            )

            # Anisotropy
            velocities = apply_osipkov_merritt(velocities, positions, keys[1], r_a=1.0)

            # Rotation
            velocities = apply_solid_body_rotation(
                velocities, positions, omega=0.1, axis=jnp.array([0., 0., 1.])
            )

            return positions, velocities

        key = jax.random.PRNGKey(42)
        positions = jax.random.normal(key, (100, 3))
        velocities = jax.random.normal(jax.random.PRNGKey(0), (100, 3))
        masses = jnp.ones(100)

        pos_out, vel_out = apply_all_knobs(positions, velocities, masses, jax.random.PRNGKey(1))

        assert pos_out.shape == (100, 3)
        assert vel_out.shape == (100, 3)

    def test_gradient_flow_through_knobs(self):
        """Gradients flow through all differentiable knobs."""
        G = 0.00450  # Stellar units

        def loss_fn(s_param, r_a, omega):
            key = jax.random.PRNGKey(42)
            positions = jax.random.normal(key, (50, 3))
            velocities = jax.random.normal(jax.random.PRNGKey(0), (50, 3))
            masses = jnp.ones(50)

            # Apply knobs (Baumgardt segregation with s parameter)
            positions, velocities = apply_mass_segregation_baumgardt(
                positions, velocities, masses, s=s_param, key=jax.random.PRNGKey(1), G=G
            )
            velocities = apply_osipkov_merritt(velocities, positions, jax.random.PRNGKey(2), r_a=r_a)
            velocities = apply_solid_body_rotation(
                velocities, positions, omega=omega, axis=jnp.array([0., 0., 1.])
            )

            # Simple loss: total kinetic energy
            return 0.5 * jnp.sum(masses[:, None] * velocities**2)

        # Compute gradients
        grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
        grads = grad_fn(0.3, 1.0, 0.1)

        # All gradients should be finite
        for g in grads:
            assert jnp.isfinite(g)

    def test_knobs_preserve_particle_count(self):
        """Non-truncation knobs preserve particle count."""
        N = 200
        G = 0.00450  # Stellar units
        key = jax.random.PRNGKey(42)
        positions = jax.random.normal(key, (N, 3))
        velocities = jax.random.normal(jax.random.PRNGKey(0), (N, 3))
        masses = jnp.ones(N)

        # Mass segregation preserves count (Baumgardt returns positions AND velocities)
        pos_seg, vel_seg = apply_mass_segregation_baumgardt(
            positions, velocities, masses, s=0.5, key=jax.random.PRNGKey(1), G=G
        )
        assert pos_seg.shape[0] == N
        assert vel_seg.shape[0] == N

        # Anisotropy preserves count
        vel_aniso = apply_osipkov_merritt(velocities, positions, jax.random.PRNGKey(2), r_a=1.0)
        assert vel_aniso.shape[0] == N

        # Rotation preserves count
        vel_rot = apply_solid_body_rotation(velocities, positions, omega=0.1, axis=jnp.array([0., 0., 1.]))
        assert vel_rot.shape[0] == N


class TestAdvancedKnobsPipeline:
    """Integration tests for advanced IC knobs (Tasks 1-6)."""

    def test_baumgardt_segregation_pipeline(self):
        """Full workflow with Baumgardt energy-ranked orbit assignment."""
        from progenax.profiles.mass_segregation import apply_mass_segregation_baumgardt

        N = 200
        G = 0.00450
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 5)

        # 1. Sample masses
        imf = PowerLawIMF.kroupa()
        masses = imf.sample(keys[0], N)

        # 2. Sample positions & velocities
        profile = PlummerProfile(r_h=1.0)
        positions = profile.sample_positions(masses, keys[1])

        velocity_df = PlummerVelocityDF(r_h=1.0)
        velocities = velocity_df.sample_velocities(positions, masses, keys[2], G=G)

        # 3. Apply Baumgardt-style mass segregation
        positions_seg, velocities_seg = apply_mass_segregation_baumgardt(
            positions, velocities, masses, s=0.8, key=keys[3], G=G
        )

        # 4. Verify outputs
        assert positions_seg.shape == (N, 3)
        assert velocities_seg.shape == (N, 3)

        # 5. Check mass segregation: most massive stars should be more centrally located
        radii = jnp.linalg.norm(positions_seg, axis=1)
        mass_percentile_90 = jnp.percentile(masses, 90)
        massive_mask = masses >= mass_percentile_90

        mean_r_massive = jnp.mean(radii[massive_mask])
        mean_r_all = jnp.mean(radii)

        # With s=0.8 (strong segregation), massive stars should be more central
        assert mean_r_massive < mean_r_all

    def test_fractal_overlay_pipeline(self):
        """Smooth positions → fractal overlay → verify structure."""
        from progenax.substructure.fractal import (
            generate_fractal_positions,
            apply_fractal_overlay_radial,
            apply_fractal_overlay_blend,
        )

        N = 300
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 5)

        # 1. Generate smooth Plummer positions
        imf = PowerLawIMF.kroupa()
        masses = imf.sample(keys[0], N)
        profile = PlummerProfile(r_h=1.0)
        positions_smooth = profile.sample_positions(masses, keys[1])

        # 2. Apply radial-preserving fractal overlay (McLuster-style)
        positions_fractal_radial = apply_fractal_overlay_radial(
            positions_smooth, keys[2], d_fractal=2.0
        )

        assert positions_fractal_radial.shape == (N, 3)

        # Verify radial profile preserved
        radii_smooth = jnp.sort(jnp.linalg.norm(positions_smooth, axis=1))
        radii_fractal = jnp.sort(jnp.linalg.norm(positions_fractal_radial, axis=1))

        # Radii should match exactly (sorted)
        assert jnp.allclose(radii_smooth, radii_fractal, rtol=1e-5)

        # 3. Apply blend overlay (experimental)
        positions_fractal_blend = apply_fractal_overlay_blend(
            positions_smooth, keys[3], d_fractal=2.0, lambda_frac=0.5
        )

        assert positions_fractal_blend.shape == (N, 3)

        # 4. Test pure fractal generation
        positions_pure_fractal = generate_fractal_positions(N, keys[4], d_fractal=2.0)
        assert positions_pure_fractal.shape == (N, 3)

        # Should be in unit sphere
        radii_pure = jnp.linalg.norm(positions_pure_fractal, axis=1)
        assert jnp.all(radii_pure <= 1.0)

    def test_radial_binary_fraction_pipeline(self):
        """Sample binaries with radially varying f_b(r)."""
        from progenax.binaries.population import RadialBinaryFraction

        N = 500
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        # 1. Generate cluster positions
        imf = PowerLawIMF.kroupa()
        masses = imf.sample(keys[0], N)
        profile = PlummerProfile(r_h=1.0)
        positions = profile.sample_positions(masses, keys[1])

        # 2. Compute radii
        radii = jnp.linalg.norm(positions, axis=1)

        # 3. Core-enhanced binary fraction
        rbf = RadialBinaryFraction(fb0=0.5, A=0.5, alpha=1.0, r_scale=1.0)

        # Compute f_b(r)
        fb_r = rbf.compute(radii)
        assert fb_r.shape == (N,)
        assert jnp.all(fb_r >= 0.0) and jnp.all(fb_r <= 1.0)

        # Sample binary membership
        is_binary = rbf.sample_membership(radii, keys[2])
        assert is_binary.shape == (N,)
        assert is_binary.dtype == jnp.bool_

        # 4. Verify core enhancement: inner stars have higher binary fraction
        inner_mask = radii < 0.5
        outer_mask = radii > 2.0

        if jnp.sum(inner_mask) > 0 and jnp.sum(outer_mask) > 0:
            fb_inner = jnp.mean(fb_r[inner_mask])
            fb_outer = jnp.mean(fb_r[outer_mask])
            assert fb_inner > fb_outer  # Core-enhanced

    def test_mass_dependent_binaries_pipeline(self):
        """Route by mass to different distributions (Sana vs solar-type)."""
        from progenax.binaries.population import (
            LogNormalPeriod,
            SanaOBPeriod,
            ThermalEccentricity,
            MoeEccentricity,
            MassDependentBinaryConfig,
            sample_mass_dependent_orbits,
        )

        N = 400
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)

        # 1. Sample masses (mix of low and high mass)
        imf = PowerLawIMF.kroupa()
        masses = imf.sample(keys[0], N)

        # 2. Configure mass-dependent binary prescriptions
        config = MassDependentBinaryConfig(
            m_break=8.0,
            low_mass_period=LogNormalPeriod(mu_log_P=4.8, sigma_log_P=2.3),
            high_mass_period=SanaOBPeriod(),
            low_mass_eccentricity=ThermalEccentricity(),
            high_mass_eccentricity=MoeEccentricity(P_circ=10.0, P_thermal=1000.0),
        )

        # 3. Sample orbital parameters
        periods, eccentricities = sample_mass_dependent_orbits(masses, config, keys[1])

        assert periods.shape == (N,)
        assert eccentricities.shape == (N,)
        assert jnp.all(periods > 0)
        assert jnp.all(eccentricities >= 0) and jnp.all(eccentricities < 1)

        # 4. Verify mass-dependent routing
        low_mass_mask = masses < config.m_break
        high_mass_mask = masses >= config.m_break

        if jnp.sum(low_mass_mask) > 0 and jnp.sum(high_mass_mask) > 0:
            # High-mass stars should have shorter periods on average (Sana+2012)
            mean_P_low = jnp.mean(periods[low_mass_mask])
            mean_P_high = jnp.mean(periods[high_mass_mask])

            # Sana OB periods are typically shorter than solar-type
            # (log P in [0.3, 3.5] vs solar-type log P ~ 4.8)
            assert mean_P_high < mean_P_low

    def test_two_component_pipeline(self):
        """Generate cluster with two populations (extended + concentrated)."""
        from progenax.populations import TwoComponentConfig, generate_two_component_cluster

        N = 600
        G = 0.00450
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)

        # 1. Sample masses
        imf = PowerLawIMF.kroupa()
        masses = imf.sample(keys[0], N)

        # 2. Configure two-component cluster
        # Population A: Extended halo
        profile_A = PlummerProfile(r_h=2.0)
        df_A = PlummerVelocityDF(r_h=2.0)

        # Population B: Concentrated core
        profile_B = PlummerProfile(r_h=0.5)
        df_B = PlummerVelocityDF(r_h=0.5)

        config = TwoComponentConfig(
            f_A=0.3,  # 30% in extended halo
            profile_A=profile_A,
            profile_B=profile_B,
            velocity_df_A=df_A,
            velocity_df_B=df_B,
        )

        # 3. Generate cluster (random population assignment)
        positions, velocities, pop_id = generate_two_component_cluster(
            masses, config, keys[1], G=G
        )

        assert positions.shape == (N, 3)
        assert velocities.shape == (N, 3)
        assert pop_id.shape == (N,)

        # 4. Verify population fractions
        n_pop_A = jnp.sum(pop_id == 0)
        n_pop_B = jnp.sum(pop_id == 1)

        assert n_pop_A + n_pop_B == N
        # Should be approximately f_A fraction in pop A
        assert jnp.abs(n_pop_A / N - config.f_A) < 0.1  # Within 10% (stochastic)

        # 5. Verify population A is more extended than B
        radii = jnp.linalg.norm(positions, axis=1)
        radii_A = radii[pop_id == 0]
        radii_B = radii[pop_id == 1]

        if len(radii_A) > 0 and len(radii_B) > 0:
            mean_r_A = jnp.mean(radii_A)
            mean_r_B = jnp.mean(radii_B)
            assert mean_r_A > mean_r_B  # Pop A is extended

    def test_all_advanced_knobs_jit(self):
        """Verify all new knobs work under JIT together."""
        from progenax.profiles.mass_segregation import apply_mass_segregation_baumgardt
        from progenax.substructure.fractal import apply_fractal_overlay_radial
        from progenax.binaries.population import (
            RadialBinaryFraction,
            MassDependentBinaryConfig,
            LogNormalPeriod,
            SanaOBPeriod,
            ThermalEccentricity,
            MoeEccentricity,
            sample_mass_dependent_orbits,
        )
        from progenax.populations import TwoComponentConfig, generate_two_component_cluster

        @jax.jit
        def apply_all_advanced_knobs(masses, key):
            """Apply all advanced knobs in one JIT-compiled function."""
            G = 0.00450
            keys = jax.random.split(key, 10)

            # 1. Two-component cluster
            profile_A = PlummerProfile(r_h=2.0)
            profile_B = PlummerProfile(r_h=0.5)
            df_A = PlummerVelocityDF(r_h=2.0)
            df_B = PlummerVelocityDF(r_h=0.5)

            config_2comp = TwoComponentConfig(
                f_A=0.3, profile_A=profile_A, profile_B=profile_B,
                velocity_df_A=df_A, velocity_df_B=df_B
            )

            positions, velocities, pop_id = generate_two_component_cluster(
                masses, config_2comp, keys[0], G=G
            )

            # 2. Apply Baumgardt mass segregation
            positions, velocities = apply_mass_segregation_baumgardt(
                positions, velocities, masses, s=0.5, key=keys[1], G=G
            )

            # 3. Apply fractal overlay
            positions = apply_fractal_overlay_radial(positions, keys[2], d_fractal=2.3)

            # 4. Radial binary fraction
            radii = jnp.linalg.norm(positions, axis=1)
            rbf = RadialBinaryFraction(fb0=0.5, A=0.3, alpha=1.0, r_scale=1.0)
            is_binary = rbf.sample_membership(radii, keys[3])

            # 5. Mass-dependent binary orbital parameters
            config_binaries = MassDependentBinaryConfig(
                m_break=8.0,
                low_mass_period=LogNormalPeriod(),
                high_mass_period=SanaOBPeriod(),
                low_mass_eccentricity=ThermalEccentricity(),
                high_mass_eccentricity=MoeEccentricity(),
            )
            periods, eccentricities = sample_mass_dependent_orbits(masses, config_binaries, keys[4])

            # Return final state
            return positions, velocities, pop_id, is_binary, periods, eccentricities

        # Run JIT-compiled function
        N = 100
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)

        imf = PowerLawIMF.kroupa()
        masses = imf.sample(keys[0], N)

        positions, velocities, pop_id, is_binary, periods, ecc = apply_all_advanced_knobs(
            masses, keys[1]
        )

        # Verify all outputs
        assert positions.shape == (N, 3)
        assert velocities.shape == (N, 3)
        assert pop_id.shape == (N,)
        assert is_binary.shape == (N,)
        assert periods.shape == (N,)
        assert ecc.shape == (N,)

        # All outputs should be finite
        assert jnp.all(jnp.isfinite(positions))
        assert jnp.all(jnp.isfinite(velocities))
        assert jnp.all(jnp.isfinite(periods))
        assert jnp.all(jnp.isfinite(ecc))

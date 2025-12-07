"""Integration tests for IC knobs pipeline."""

import jax
import jax.numpy as jnp
import pytest

from progenax.profiles import PlummerProfile
from progenax.kinematics import PlummerVelocityDF
from progenax.imf import PowerLawIMF
from progenax.builders import virial_scale, to_com_frame, compute_kinetic_energy, compute_potential_energy
from progenax.kinematics.anisotropy import apply_osipkov_merritt
from progenax.kinematics.rotation import apply_solid_body_rotation
from progenax.profiles.mass_segregation import apply_mass_segregation
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

        # 4. Apply mass segregation
        m_ref = jnp.median(masses)
        positions = apply_mass_segregation(positions, masses, eta=0.3, m_ref=m_ref)

        # 5. Apply Osipkov-Merritt anisotropy
        velocities = apply_osipkov_merritt(velocities, positions, keys[3], r_a=r_h)

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

        # 9. Virial scale
        velocities = virial_scale(positions, velocities, masses, Q_target=1.0, G=G)

        # Verify final state
        assert positions.shape[0] > 0  # Some particles kept
        assert velocities.shape == positions.shape
        assert masses.shape[0] == positions.shape[0]

        # Check virial ratio
        T = compute_kinetic_energy(velocities, masses)
        V = compute_potential_energy(positions, masses, G=G)
        Q = 2 * T / jnp.abs(V)
        assert jnp.abs(Q - 1.0) < 0.01  # Should be virial equilibrium

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
        @jax.jit
        def apply_all_knobs(positions, velocities, masses, key):
            keys = jax.random.split(key, 3)

            # Mass segregation
            positions = apply_mass_segregation(positions, masses, eta=0.3, m_ref=1.0)

            # Anisotropy
            velocities = apply_osipkov_merritt(velocities, positions, keys[0], r_a=1.0)

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
        def loss_fn(eta, r_a, omega):
            key = jax.random.PRNGKey(42)
            positions = jax.random.normal(key, (50, 3))
            velocities = jax.random.normal(jax.random.PRNGKey(0), (50, 3))
            masses = jnp.ones(50)

            # Apply knobs
            positions = apply_mass_segregation(positions, masses, eta=eta, m_ref=1.0)
            velocities = apply_osipkov_merritt(velocities, positions, jax.random.PRNGKey(1), r_a=r_a)
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
        key = jax.random.PRNGKey(42)
        positions = jax.random.normal(key, (N, 3))
        velocities = jax.random.normal(jax.random.PRNGKey(0), (N, 3))
        masses = jnp.ones(N)

        # Mass segregation preserves count
        pos_seg = apply_mass_segregation(positions, masses, eta=0.5, m_ref=1.0)
        assert pos_seg.shape[0] == N

        # Anisotropy preserves count
        vel_aniso = apply_osipkov_merritt(velocities, positions, jax.random.PRNGKey(1), r_a=1.0)
        assert vel_aniso.shape[0] == N

        # Rotation preserves count
        vel_rot = apply_solid_body_rotation(velocities, positions, omega=0.1, axis=jnp.array([0., 0., 1.]))
        assert vel_rot.shape[0] == N

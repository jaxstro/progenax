"""End-to-end integration tests for progenax."""

import jax
import jax.numpy as jnp
import pytest


class TestIMFToICPipeline:
    """Test IMF sampling → IC generation pipeline."""

    def test_kroupa_to_plummer_ic(self):
        """Generate Plummer IC with Kroupa IMF masses."""
        from progenax.imf import PowerLawIMF
        from progenax.profiles import PlummerProfile
        from progenax.kinematics import PlummerVelocityDF
        from progenax.builders import build_spatial_ic

        # Sample masses from Kroupa IMF
        imf = PowerLawIMF.kroupa()
        key = jax.random.PRNGKey(42)
        key_imf, key_ic = jax.random.split(key)
        masses = imf.sample(key_imf, 100)

        # Create Plummer IC
        profile = PlummerProfile(r_h=1.0)
        velocity_df = PlummerVelocityDF(r_h=1.0)
        G = 1.0

        result = build_spatial_ic(
            profile=profile,
            masses=masses,
            velocity_df=velocity_df,
            Q=1.0,
            key=key_ic,
            G=G,
        )

        assert result.positions.shape == (100, 3)
        assert result.velocities.shape == (100, 3)
        assert jnp.all(result.masses > 0)

    def test_chabrier_to_king_ic(self):
        """Generate King IC with Chabrier IMF masses."""
        from progenax.imf import ChabrierIMF
        from progenax.profiles import KingProfile, solve_king_profile
        from progenax.kinematics import KingVelocityDF
        from progenax.builders import build_spatial_ic

        # Sample masses from Chabrier IMF
        imf = ChabrierIMF()
        key = jax.random.PRNGKey(42)
        key_imf, key_ic = jax.random.split(key)
        masses = imf.sample(key_imf, 50)

        # Create King IC
        W0 = 6.0
        xi_grid, psi_grid = solve_king_profile(W0=W0)
        profile = KingProfile(W0=W0, r_c=1.0, r_t=10.0, xi_grid=xi_grid, psi_grid=psi_grid)
        velocity_df = KingVelocityDF(W0=W0, r_c=profile.r_c, r_t=profile.r_t)
        G = 1.0

        result = build_spatial_ic(
            profile=profile,
            masses=masses,
            velocity_df=velocity_df,
            Q=1.0,
            key=key_ic,
            G=G,
        )

        assert result.positions.shape == (50, 3)


class TestAnalyticalValidation:
    """Test analytical ICs have correct properties."""

    def test_two_body_energy_conservation(self):
        """Two-body system should have correct total energy."""
        from progenax.analytical import two_body_kepler, two_body_energy
        from progenax.builders import compute_kinetic_energy, compute_potential_energy

        G = 1.0
        M1, M2, a = 1.0, 0.5, 2.0

        ic = two_body_kepler(M1=M1, M2=M2, a=a, e=0.0, G=G)

        T = compute_kinetic_energy(ic.velocities, ic.masses)
        V = compute_potential_energy(ic.positions, ic.masses, G=G)
        E_computed = T + V
        E_analytical = two_body_energy(M1, M2, a, G)

        # Energy should match within 1%
        assert jnp.abs(E_computed - E_analytical) / jnp.abs(E_analytical) < 0.01

    def test_figure_eight_symmetry(self):
        """Figure-8 should have 3-fold symmetry."""
        from progenax.analytical import three_body_figure_eight

        G = 1.0
        ic = three_body_figure_eight(mass=1.0, scale=1.0, G=G)

        # All three masses should be equal
        assert jnp.allclose(ic.masses, 1.0)

        # Distances from origin should be approximately equal (3-fold symmetry)
        r = jnp.linalg.norm(ic.positions, axis=1)
        assert jnp.std(r) / jnp.mean(r) < 0.1  # <10% variation


class TestBinaryICGeneration:
    """Test binary star IC generation."""

    def test_binary_from_elements(self):
        """Create binary from orbital elements."""
        from progenax.binaries import KeplerElements

        G = 1.0
        elements = KeplerElements(a=1.0, e=0.3, i=0.0, Omega=0.0, omega=0.0, M0=0.0)
        r1, v1, r2, v2 = elements.to_binary_state(m1=1.0, m2=0.5, G=G)

        # COM at origin
        m1, m2 = 1.0, 0.5
        com = (m1 * r1 + m2 * r2) / (m1 + m2)
        assert jnp.allclose(com, 0.0, atol=1e-10)

        # Total momentum zero
        p_total = m1 * v1 + m2 * v2
        assert jnp.allclose(p_total, 0.0, atol=1e-10)

    def test_batch_binary_generation(self):
        """Batch generate N binaries."""
        from progenax.binaries import batch_elements_to_resolved

        G = 1.0
        N = 20
        m1 = jnp.ones(N)
        m2 = jnp.ones(N) * 0.5
        logP = jnp.ones(N) * 2.0
        e = jnp.linspace(0.0, 0.5, N)
        zeros = jnp.zeros(N)

        r1, v1, r2, v2 = batch_elements_to_resolved(
            m1, m2, logP, e, zeros, zeros, zeros, zeros,
            G=G, day_in_time_units=1.0
        )

        assert r1.shape == (N, 3)
        assert r2.shape == (N, 3)


class TestDifferentiability:
    """Test that key operations are differentiable."""

    def test_imf_ppf_gradient(self):
        """IMF PPF should be differentiable."""
        from progenax.imf import PowerLawIMF

        imf = PowerLawIMF.kroupa()

        def loss(u):
            return jnp.sum(imf.ppf(u))

        grad_fn = jax.grad(loss)
        u = jnp.array([0.3, 0.5, 0.7])
        grads = grad_fn(u)

        assert jnp.all(jnp.isfinite(grads))
        assert jnp.all(grads > 0)  # dm/du > 0 (monotonic)

    def test_spatial_profile_gradient(self):
        """Spatial profile should be differentiable w.r.t. r_h."""
        from progenax.profiles import PlummerProfile

        def total_radius(r_h):
            profile = PlummerProfile(r_h=r_h)
            key = jax.random.PRNGKey(42)
            masses = jnp.ones(10)
            positions = profile.sample_positions(masses, key)
            return jnp.sum(jnp.linalg.norm(positions, axis=1))

        grad_fn = jax.grad(total_radius)
        grad = grad_fn(1.0)

        assert jnp.isfinite(grad)


class TestProtocolCompliance:
    """Test that classes implement protocols correctly."""

    def test_plummer_implements_spatial_profile(self):
        from progenax.protocols import SpatialProfile
        from progenax.profiles import PlummerProfile

        profile = PlummerProfile(r_h=1.0)
        assert isinstance(profile, SpatialProfile)

    def test_plummer_df_implements_velocity_df(self):
        from progenax.protocols import VelocityDF
        from progenax.kinematics import PlummerVelocityDF

        df = PlummerVelocityDF(r_h=1.0)
        assert isinstance(df, VelocityDF)

    def test_kroupa_implements_imf_protocol(self):
        from progenax.protocols import IMFProtocol
        from progenax.imf import PowerLawIMF

        imf = PowerLawIMF.kroupa()
        # Check required attributes
        assert hasattr(imf, 'm_min')
        assert hasattr(imf, 'm_max')
        assert hasattr(imf, 'logpdf')
        assert hasattr(imf, 'cdf')
        assert hasattr(imf, 'ppf')
        assert hasattr(imf, 'sample')

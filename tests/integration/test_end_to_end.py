"""End-to-end integration tests for progenax."""

import jax
import jax.numpy as jnp


class TestIMFToICPipeline:
    """Test IMF sampling → IC generation pipeline."""

    def test_kroupa_to_plummer_ic(self):
        """Generate Plummer IC with Kroupa IMF masses."""
        from progenax.builders import build_spatial_ic
        from progenax.imf import PowerLawIMF
        from progenax.kinematics import PlummerVelocityDF
        from progenax.profiles import PlummerProfile

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
            Q=0.5,  # Q = T/|V|, 0.5 for equilibrium
            key=key_ic,
            G=G,
        )

        assert result.positions.shape == (100, 3)
        assert result.velocities.shape == (100, 3)
        assert jnp.all(result.masses > 0)

    def test_chabrier_to_king_ic(self):
        """Generate King IC with Chabrier IMF masses."""
        from progenax.builders import build_spatial_ic
        from progenax.imf import ChabrierIMF
        from progenax.kinematics import KingVelocityDF
        from progenax.profiles import KingProfile

        # Sample masses from Chabrier IMF
        imf = ChabrierIMF()
        key = jax.random.PRNGKey(42)
        key_imf, key_ic = jax.random.split(key)
        masses = imf.sample(key_imf, 50)

        # Create King IC
        W0 = 6.0
        profile = KingProfile.from_W0_rc(W0=W0, r_c=1.0)  # self-consistent r_t
        velocity_df = KingVelocityDF(W0=W0, r_c=profile.r_c)
        G = 1.0

        result = build_spatial_ic(
            profile=profile,
            masses=masses,
            velocity_df=velocity_df,
            Q=0.5,  # Q = T/|V|, 0.5 for equilibrium
            key=key_ic,
            G=G,
        )

        assert result.positions.shape == (50, 3)


class TestAnalyticalValidation:
    """Test analytical ICs have correct properties."""

    def test_two_body_energy_conservation(self):
        """Two-body system should have correct total energy."""
        from progenax.analytical import two_body_energy, two_body_kepler
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
        """The canonical Chenciner–Montgomery figure-eight is a COLLINEAR, point-symmetric
        configuration at t=0 — NOT 3-fold spatially symmetric (that was a prior bug, which
        gave |L|=1.6 and an orbit that did not close). Bodies 1,2 sit at ±(x1,y1); body 3
        sits at the origin; total angular momentum is zero."""
        from progenax.analytical import three_body_figure_eight

        G = 1.0
        ic = three_body_figure_eight(mass=1.0, scale=1.0, G=G)

        # All three masses equal
        assert jnp.allclose(ic.masses, 1.0)

        # Point symmetry r1 = -r2, and body 3 at the origin (the collinear CMS config)
        assert jnp.allclose(ic.positions[0], -ic.positions[1], atol=1e-12)
        assert jnp.allclose(ic.positions[2], 0.0, atol=1e-12)

        # Zero total angular momentum (the defining property of the figure-eight)
        L = jnp.sum(ic.masses[:, None] * jnp.cross(ic.positions, ic.velocities), axis=0)
        assert jnp.linalg.norm(L) < 1e-10


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
            m1, m2, logP, e, zeros, zeros, zeros, zeros, G=G, day_in_time_units=1.0
        )

        assert r1.shape == (N, 3)
        assert r2.shape == (N, 3)


# Differentiability of the public IC operations is owned by the grad-audit registry
# (tests/validation/grad_audit/registry.py); see
# docs/website/50-validation/differentiability-audit.md. The former finite-only
# TestDifferentiability smoke tests were removed (audit T6: isfinite passes a
# silently-zeroed grad; the registry FD cases are strictly stronger; registry is SoT):
#   - test_spatial_profile_gradient -> PlummerProfile.sample_positions [r_h] (FD-covered).
#   - test_imf_ppf_gradient (grad wrt the uniform draw u, a non-param du-monotonicity
#     smoke) is redundant -- the parameter channels of the IMF PPFs are FD-audited by
#     the registry's PowerLawIMF/ChabrierIMF/Maschberger .ppf cases.


class TestProtocolCompliance:
    """Test that classes implement protocols correctly."""

    def test_plummer_implements_spatial_profile(self):
        from progenax.profiles import PlummerProfile
        from progenax.protocols import SpatialProfile

        profile = PlummerProfile(r_h=1.0)
        assert isinstance(profile, SpatialProfile)

    def test_plummer_df_implements_velocity_df(self):
        from progenax.kinematics import PlummerVelocityDF
        from progenax.protocols import VelocityDF

        df = PlummerVelocityDF(r_h=1.0)
        assert isinstance(df, VelocityDF)

    def test_kroupa_implements_imf_protocol(self):
        from progenax.imf import PowerLawIMF

        imf = PowerLawIMF.kroupa()
        # Check required attributes
        assert hasattr(imf, "m_min")
        assert hasattr(imf, "m_max")
        assert hasattr(imf, "logpdf")
        assert hasattr(imf, "cdf")
        assert hasattr(imf, "ppf")
        assert hasattr(imf, "sample")

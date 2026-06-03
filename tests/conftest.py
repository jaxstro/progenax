"""
Shared fixtures and configuration for progenax tests.

Provides:
- Standard particle counts (N) for different test tiers
- Tolerance thresholds for physics validation
- Common profile/DF factory functions
- JAX configuration
"""

import pytest
import jax
import jax.numpy as jnp


# =============================================================================
# JAX Configuration
# =============================================================================

def pytest_configure(config):
    """Enable float64 for all tests and configure warnings."""
    jax.config.update("jax_enable_x64", True)

    # Suppress Equinox warning about field(init=False) - this is intentional
    # design in EnvironmentIMF where environment→IMF conversion is not meant
    # to be differentiated through
    config.addinivalue_line(
        "filterwarnings",
        "ignore:Using.*field.*init=False.*:UserWarning"
    )


# =============================================================================
# Standard Particle Counts
# =============================================================================

@pytest.fixture
def N_unit():
    """Small N for fast unit tests."""
    return 100


@pytest.fixture
def N_integration():
    """Medium N for integration tests."""
    return 1000


@pytest.fixture
def N_validation():
    """Large N for physics validation (statistical accuracy)."""
    return 5000


@pytest.fixture
def N_stats():
    """Very large N for precise statistical tests."""
    return 10000


# =============================================================================
# Physics Tolerance Thresholds
# =============================================================================

class PhysicsTolerances:
    """Standard tolerances for physics validation tests.

    Based on literature values and statistical expectations.
    """
    # Exact formulas (machine precision)
    EXACT = 1e-10

    # High precision (numerical methods)
    HIGH = 1e-6

    # Standard physics tests (statistical + numerical)
    STANDARD = 0.05  # 5%

    # Relaxed (large statistical fluctuations)
    RELAXED = 0.10  # 10%

    # Very relaxed (qualitative tests)
    QUALITATIVE = 0.20  # 20%

    # Specific physics targets
    # Q = T/|V| for a sampled Plummer (analytic V) at N=5000: |Q-0.5| ~ 0.002,
    # 40-seed std 0.005 (max 0.011) -> 0.05 is an ~11-sigma, discriminating bound.
    # (Was 0.20, which would have accepted a 40%-wrong kinetic energy.)
    VIRIAL_RATIO = 0.05      # Q = T/|V| within 5% of 0.5 (regime-anchored, N=5000)
    HALF_MASS = 0.03         # Half-mass radius: 3% tolerance
    VELOCITY_DISPERSION = 0.10  # σ(r) profile: 10% tolerance
    CDF_MONOTONIC = 1e-10    # CDF must be strictly increasing
    BOUND_FRACTION = 1.0     # 100% of particles must be bound


@pytest.fixture
def tolerances():
    """Standard physics tolerances."""
    return PhysicsTolerances()


# =============================================================================
# Random Keys
# =============================================================================

@pytest.fixture
def key():
    """Standard JAX random key for reproducibility."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def key_factory():
    """Factory for generating multiple keys."""
    def _make_key(seed=42):
        return jax.random.PRNGKey(seed)
    return _make_key


# =============================================================================
# Profile Factories
# =============================================================================

@pytest.fixture
def plummer_profile():
    """Standard Plummer profile for testing."""
    from progenax.profiles import PlummerProfile
    return PlummerProfile(r_h=1.0)


@pytest.fixture
def king_profile():
    """Standard King profile for testing (W0=7, moderate concentration)."""
    from progenax.profiles import KingProfile
    return KingProfile(W0=7.0, r_c=1.0, r_t=10.0)


@pytest.fixture
def eff_profile():
    """Standard EFF profile for testing."""
    from progenax.profiles import EFFProfile
    return EFFProfile(a=1.0, gamma=3.0, r_t=10.0)


# =============================================================================
# Velocity DF Factories
# =============================================================================

@pytest.fixture
def plummer_df():
    """Standard Plummer velocity DF for testing."""
    from progenax.kinematics import PlummerVelocityDF
    return PlummerVelocityDF(r_h=1.0)


@pytest.fixture
def king_df():
    """Standard King velocity DF for testing."""
    from progenax.kinematics import KingVelocityDF
    return KingVelocityDF(W0=7.0, r_c=1.0, r_t=10.0)


@pytest.fixture
def eff_df():
    """Standard EFF velocity DF for testing."""
    from progenax.kinematics import EFFVelocityDF
    return EFFVelocityDF(a=1.0, gamma=3.0, r_t=10.0)


# =============================================================================
# IMF Factories
# =============================================================================

@pytest.fixture
def kroupa_imf():
    """Standard Kroupa IMF for testing."""
    from progenax.imf import KroupaIMF
    return KroupaIMF()


@pytest.fixture
def chabrier_imf():
    """Standard Chabrier IMF for testing."""
    from progenax.imf import ChabrierIMF
    return ChabrierIMF()


# =============================================================================
# Standard Test Data
# =============================================================================

@pytest.fixture
def equal_masses(N_validation):
    """Equal masses array (1.0 M_sun each)."""
    return jnp.ones(N_validation)


@pytest.fixture
def unit_G():
    """Standard gravitational constant for testing."""
    return 1.0


# =============================================================================
# Physics Constants (for validation)
# =============================================================================

class PlummerConstants:
    """Plummer model constants from literature.

    References:
        Plummer (1911), MNRAS 71, 460
        Binney & Tremaine (2008), "Galactic Dynamics"
    """
    # Scale radius to half-mass radius ratio
    # a = r_h * sqrt(2^(2/3) - 1) ≈ 0.7664 * r_h
    SCALE_RADIUS_FACTOR = jnp.sqrt(2**(2/3) - 1)

    # Mass fraction within scale radius
    # M(<a)/M = 1/2^(3/2) ≈ 0.354
    MASS_WITHIN_SCALE_RADIUS = 1.0 / (2**1.5)

    # Beta distribution parameters for velocity sampling
    # q² ~ Beta(3/2, 9/2) where q = v/v_esc
    BETA_A = 1.5
    BETA_B = 4.5

    # Mean of q² = a/(a+b) = 1.5/6 = 0.25
    MEAN_Q_SQUARED = 0.25

    # Analytical potential energy: V = -3π G M² / (32a)
    PE_COEFFICIENT = 3.0 * jnp.pi / 32.0


@pytest.fixture
def plummer_constants():
    """Plummer model constants."""
    return PlummerConstants()


class KingConstants:
    """King model constants from literature.

    References:
        King (1966), AJ 71, 64
        Binney & Tremaine (2008), "Galactic Dynamics"
    """
    # Reference K-function values (computed from implementation)
    # K(W) = erf(√W) - (2/√π)√W exp(-W)
    K_REF_W3 = 0.8884  # K(3.0)
    K_REF_W5 = 0.9814  # K(5.0)
    K_REF_W7 = 0.9971  # K(7.0)

    # W0 ranges for different concentration
    W0_LOW = 3.0    # Low concentration
    W0_MED = 7.0    # Medium concentration
    W0_HIGH = 12.0  # High concentration


@pytest.fixture
def king_constants():
    """King model constants."""
    return KingConstants()


class IMFConstants:
    """IMF constants from literature.

    References:
        Salpeter (1955), ApJ 121, 161
        Kroupa (2001), MNRAS 322, 231
        Chabrier (2003), PASP 115, 763
    """
    # Salpeter slope
    SALPETER_ALPHA = 2.35

    # Kroupa multi-slope parameters
    KROUPA_ALPHAS = (0.3, 1.3, 2.3)  # Below 0.08, 0.08-0.5, above 0.5 M_sun
    KROUPA_BREAKS = (0.08, 0.5)  # M_sun

    # Chabrier log-normal parameters
    CHABRIER_MC = 0.08  # M_sun (characteristic mass)
    CHABRIER_SIGMA = 0.69  # Log-normal width
    CHABRIER_ALPHA_HIGH = 2.35  # Power law slope above 1 M_sun


@pytest.fixture
def imf_constants():
    """IMF constants."""
    return IMFConstants()

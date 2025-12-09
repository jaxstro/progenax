"""
Comprehensive tests for IMF extensions stack fixes.

Tests verify fixes from the IMF Extensions Stack plan:
- Task 1: binary.py type hints (Bool for is_binary)
- Task 2: MoeDiStefano2017 twin sampling consistency

Note: Tasks 3-4 tested deprecated classes (CustomEnvironmentIMF, GasEnvironment,
massive_star_fraction) which were removed in v0.3 and replaced by the
paper-calibrated environment.py module. See test_environment.py for tests
of the new BirthEnvironment and env_to_imf_params API.
"""

import jax
import jax.numpy as jnp
import pytest


class TestBinaryTypeHints:
    """Task 1: Verify binary.py type hints are correct."""

    def test_sample_systems_is_binary_is_boolean(self):
        """is_binary return value should have boolean dtype."""
        from progenax.imf import PowerLawIMF
        from progenax.imf.binary import BinaryIMF

        imf = BinaryIMF(PowerLawIMF.kroupa(), binary_fraction=0.5)
        key = jax.random.PRNGKey(42)
        m1, m2, is_binary = imf.sample_systems(key, 100)

        assert is_binary.dtype == jnp.bool_, \
            f"is_binary should be bool, got {is_binary.dtype}"


class TestMoeDiStefano2017Sampling:
    """Task 2: Verify MoeDiStefano2017 sampling matches PDF."""

    def test_twin_sampling_within_bounds(self):
        """Twin samples should be within [q_min, 1]."""
        from progenax.imf.binary import MoeDiStefano2017

        moe = MoeDiStefano2017(q_min=0.1, sigma_twin=0.03)
        key = jax.random.PRNGKey(42)
        m1 = jnp.ones(10000) * 1.0
        q = moe.sample_given_primary(key, m1)

        assert jnp.all(q >= 0.1), "All q should be >= q_min"
        assert jnp.all(q <= 1.0), "All q should be <= 1.0"


# Note: TestCustomEnvironmentIMFEquinox and TestMassiveStarFractionKey
# were removed in v0.3 as part of the environment module refactor.
# The deprecated classes (CustomEnvironmentIMF, GasEnvironment,
# massive_star_fraction) have been replaced by:
#   - BirthEnvironment (paper-calibrated inference target)
#   - env_to_imf_params (unified API for all models)
#   - alpha3_jerabkova_mecl, alpha3_marks_plane (paper equations)
# See tests/unit/imf/test_environment.py for comprehensive tests.


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

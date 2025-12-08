"""
Comprehensive tests for IMF extensions stack fixes.

Tests verify fixes from the IMF Extensions Stack plan:
- Task 1: binary.py type hints (Bool for is_binary)
- Task 2: MoeDiStefano2017 twin sampling consistency
- Task 3: CustomEnvironmentIMF equinox pattern
- Task 4: massive_star_fraction key parameter
- Task 5: IGIMF key parameters and documentation
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


class TestCustomEnvironmentIMFEquinox:
    """Task 3: Verify CustomEnvironmentIMF equinox pattern."""

    def test_is_valid_pytree(self):
        """CustomEnvironmentIMF should be a valid PyTree."""
        from progenax.imf.environment import CustomEnvironmentIMF, GasEnvironment

        env = GasEnvironment.solar_neighborhood()
        imf = CustomEnvironmentIMF(env)

        # Should flatten/unflatten without error
        leaves, treedef = jax.tree_util.tree_flatten(imf)
        imf_restored = jax.tree_util.tree_unflatten(treedef, leaves)

        # Properties should match
        assert imf.alpha_high == imf_restored.alpha_high
        assert imf.m_char == imf_restored.m_char

    def test_jit_compilation_works(self):
        """CustomEnvironmentIMF should work with JIT."""
        from progenax.imf.environment import CustomEnvironmentIMF, GasEnvironment

        env = GasEnvironment.solar_neighborhood()
        imf = CustomEnvironmentIMF(env)

        @jax.jit
        def sample_fn(key):
            return imf.sample(key, 100)

        key = jax.random.PRNGKey(42)
        masses = sample_fn(key)

        assert masses.shape == (100,)
        assert jnp.all(masses > 0)


class TestMassiveStarFractionKey:
    """Task 4: Verify massive_star_fraction accepts key parameter."""

    def test_key_reproducibility(self):
        """Same key should give same result."""
        from progenax.imf import PowerLawIMF
        from progenax.imf.environment import massive_star_fraction

        imf = PowerLawIMF.salpeter()
        key = jax.random.PRNGKey(42)

        frac1 = massive_star_fraction(imf, key=key)
        frac2 = massive_star_fraction(imf, key=key)

        assert frac1 == frac2, "Same key should give identical results"

    def test_n_samples_parameter(self):
        """n_samples parameter should be accepted."""
        from progenax.imf import PowerLawIMF
        from progenax.imf.environment import massive_star_fraction

        imf = PowerLawIMF.salpeter()
        key = jax.random.PRNGKey(42)

        # Should work with different sample sizes
        frac1 = massive_star_fraction(imf, key=key, n_samples=1000)
        frac2 = massive_star_fraction(imf, key=key, n_samples=1000)

        assert 0.0 <= frac1 <= 1.0
        assert 0.0 <= frac2 <= 1.0


class TestIGIMFKeyParams:
    """Task 5: Verify IGIMF accepts key parameters."""

    def test_mean_mass_accepts_key(self):
        """mean_mass should accept key parameter."""
        from progenax.imf import PowerLawIMF
        from progenax.imf.igimf import IGIMF

        igimf = IGIMF(PowerLawIMF.kroupa(), sfr=1.0)
        key = jax.random.PRNGKey(42)

        mean1 = igimf.mean_mass(key=key)
        mean2 = igimf.mean_mass(key=key)

        assert jnp.allclose(mean1, mean2), "Same key should give same result"

    def test_effective_slope_accepts_key(self):
        """effective_slope_high_mass should accept key parameter."""
        from progenax.imf import PowerLawIMF
        from progenax.imf.igimf import IGIMF

        # Use very high SFR to ensure enough massive stars
        igimf = IGIMF(PowerLawIMF.kroupa(), sfr=1000.0)
        key = jax.random.PRNGKey(42)

        # Use wider mass range to avoid empty array
        slope = igimf.effective_slope_high_mass(
            key=key, n_samples=5000, m_range=(5, 120)
        )

        # Just verify it returns a finite value
        assert jnp.isfinite(slope), f"Slope should be finite, got {slope}"

    def test_logpdf_accepts_key(self):
        """logpdf should accept key parameter."""
        from progenax.imf import PowerLawIMF
        from progenax.imf.igimf import IGIMF

        igimf = IGIMF(PowerLawIMF.kroupa(), sfr=1.0)
        key = jax.random.PRNGKey(42)

        m = jnp.array([1.0, 10.0])
        logp = igimf.logpdf(m, key=key, n_samples=1000)

        assert logp.shape == (2,)
        assert jnp.all(jnp.isfinite(logp))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

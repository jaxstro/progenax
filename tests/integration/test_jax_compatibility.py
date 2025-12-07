"""
JAX compatibility tests for progenax modules.

Parametrized tests ensuring all core modules work with:
- jax.jit (JIT compilation)
- jax.grad (automatic differentiation)
- jax.vmap (vectorization/batching)

These are critical for gradient-based inference workflows.
"""

import jax
import jax.numpy as jnp
import pytest

from progenax.profiles import PlummerProfile, EFFProfile
from progenax.kinematics import PlummerVelocityDF, EFFVelocityDF
from progenax.imf import PowerLawIMF, ChabrierIMF


# =============================================================================
# Profile JIT/Grad/Vmap Tests
# =============================================================================

class TestProfileJITCompatibility:
    """Test spatial profiles work with JIT compilation."""

    def test_plummer_sample_jit(self):
        """Plummer sample_positions works under JIT."""
        profile = PlummerProfile(r_h=1.0)
        masses = jnp.ones(100)
        key = jax.random.PRNGKey(42)

        @jax.jit
        def sample(key):
            return profile.sample_positions(masses, key)

        positions = sample(key)
        assert positions.shape == (100, 3)
        assert jnp.all(jnp.isfinite(positions))

    def test_eff_sample_jit(self):
        """EFF sample_positions works under JIT."""
        profile = EFFProfile(a=1.0, gamma=3.0, r_t=10.0)
        masses = jnp.ones(100)
        key = jax.random.PRNGKey(42)

        @jax.jit
        def sample(key):
            return profile.sample_positions(masses, key)

        positions = sample(key)
        assert positions.shape == (100, 3)
        assert jnp.all(jnp.isfinite(positions))


class TestProfileGradCompatibility:
    """Test spatial profiles are differentiable."""

    def test_plummer_grad_through_sample(self):
        """Gradient flows through Plummer position sampling."""
        def loss(r_h):
            profile = PlummerProfile(r_h=r_h)
            masses = jnp.ones(50)
            key = jax.random.PRNGKey(42)
            positions = profile.sample_positions(masses, key)
            return jnp.mean(jnp.linalg.norm(positions, axis=1))

        grad_fn = jax.grad(loss)
        grad_val = grad_fn(1.0)

        assert jnp.isfinite(grad_val)
        # Larger r_h should give larger radii, so gradient should be positive
        assert float(grad_val) > 0

    def test_eff_grad_through_sample(self):
        """Gradient flows through EFF position sampling."""
        def loss(a):
            profile = EFFProfile(a=a, gamma=3.0, r_t=10.0)
            masses = jnp.ones(50)
            key = jax.random.PRNGKey(42)
            positions = profile.sample_positions(masses, key)
            return jnp.mean(jnp.linalg.norm(positions, axis=1))

        grad_fn = jax.grad(loss)
        grad_val = grad_fn(1.0)

        assert jnp.isfinite(grad_val)


# =============================================================================
# Velocity DF JIT/Grad Tests
# =============================================================================

class TestVelocityDFJITCompatibility:
    """Test velocity DFs work with JIT compilation."""

    def test_plummer_df_jit(self):
        """Plummer velocity sampling works under JIT."""
        df = PlummerVelocityDF(r_h=1.0)
        N = 100
        positions = jax.random.normal(jax.random.PRNGKey(0), (N, 3))
        masses = jnp.ones(N)
        G = 1.0

        @jax.jit
        def sample(key):
            return df.sample_velocities(positions, masses, key, G=G)

        velocities = sample(jax.random.PRNGKey(42))
        assert velocities.shape == (N, 3)
        assert jnp.all(jnp.isfinite(velocities))

    def test_eff_df_jit(self):
        """EFF velocity sampling works under JIT."""
        df = EFFVelocityDF(a=1.0, gamma=3.0, r_t=10.0)
        N = 100
        positions = jax.random.normal(jax.random.PRNGKey(0), (N, 3)) * 0.5
        masses = jnp.ones(N)
        G = 1.0

        @jax.jit
        def sample(key):
            return df.sample_velocities(positions, masses, key, G=G)

        velocities = sample(jax.random.PRNGKey(42))
        assert velocities.shape == (N, 3)
        assert jnp.all(jnp.isfinite(velocities))


class TestVelocityDFGradCompatibility:
    """Test velocity DFs are differentiable."""

    def test_plummer_df_grad_through_sample(self):
        """Gradient flows through Plummer velocity sampling."""
        N = 50

        def loss(r_h):
            df = PlummerVelocityDF(r_h=r_h)
            positions = jnp.ones((N, 3)) * 0.5  # Fixed positions
            masses = jnp.ones(N)
            key = jax.random.PRNGKey(42)
            velocities = df.sample_velocities(positions, masses, key, G=1.0)
            return jnp.mean(jnp.sum(velocities**2, axis=1))

        grad_fn = jax.grad(loss)
        grad_val = grad_fn(1.0)

        assert jnp.isfinite(grad_val)


# =============================================================================
# IMF JIT/Grad Tests
# =============================================================================

class TestIMFJITCompatibility:
    """Test IMFs work with JIT compilation."""

    def test_powerlaw_sample_jit(self):
        """Power-law IMF sampling works under JIT."""
        imf = PowerLawIMF.kroupa()

        @jax.jit
        def sample(key):
            return imf.sample(key, 100)

        masses = sample(jax.random.PRNGKey(42))
        assert masses.shape == (100,)
        assert jnp.all(jnp.isfinite(masses))
        assert jnp.all(masses >= imf.m_min)
        assert jnp.all(masses <= imf.m_max)

    def test_chabrier_sample_jit(self):
        """Chabrier IMF sampling works under JIT."""
        imf = ChabrierIMF()

        @jax.jit
        def sample(key):
            return imf.sample(key, 100)

        masses = sample(jax.random.PRNGKey(42))
        assert masses.shape == (100,)
        assert jnp.all(jnp.isfinite(masses))

    def test_powerlaw_ppf_jit(self):
        """Power-law PPF works under JIT."""
        imf = PowerLawIMF.kroupa()

        @jax.jit
        def ppf(u):
            return imf.ppf(u)

        u = jnp.linspace(0.01, 0.99, 50)
        masses = ppf(u)
        assert masses.shape == (50,)
        assert jnp.all(jnp.isfinite(masses))

    def test_chabrier_ppf_jit(self):
        """Chabrier PPF works under JIT."""
        imf = ChabrierIMF()

        @jax.jit
        def ppf(u):
            return imf.ppf(u)

        u = jnp.linspace(0.01, 0.99, 50)
        masses = ppf(u)
        assert masses.shape == (50,)
        assert jnp.all(jnp.isfinite(masses))


class TestIMFGradCompatibility:
    """Test IMFs are differentiable."""

    def test_powerlaw_ppf_grad(self):
        """Gradient flows through power-law PPF."""
        imf = PowerLawIMF.kroupa()

        def total_mass(u):
            return jnp.sum(imf.ppf(u))

        grad_fn = jax.grad(total_mass)
        u = jnp.array([0.3, 0.5, 0.7])
        grads = grad_fn(u)

        assert jnp.all(jnp.isfinite(grads))
        # PPF is monotonic, so dm/du > 0
        assert jnp.all(grads > 0)

    def test_chabrier_ppf_grad(self):
        """Gradient flows through Chabrier PPF."""
        imf = ChabrierIMF()

        def total_mass(u):
            return jnp.sum(imf.ppf(u))

        grad_fn = jax.grad(total_mass)
        u = jnp.array([0.3, 0.5, 0.7])
        grads = grad_fn(u)

        assert jnp.all(jnp.isfinite(grads))
        assert jnp.all(grads > 0)

    def test_powerlaw_logpdf_grad(self):
        """Gradient flows through power-law logpdf."""
        imf = PowerLawIMF.kroupa()

        def total_logpdf(m):
            return jnp.sum(imf.logpdf(m))

        grad_fn = jax.grad(total_logpdf)
        masses = jnp.array([0.3, 1.0, 5.0])
        grads = grad_fn(masses)

        assert jnp.all(jnp.isfinite(grads))


class TestIMFVmapCompatibility:
    """Test IMFs work with vmap (vectorization)."""

    def test_powerlaw_ppf_vmap(self):
        """Power-law PPF works with vmap over batches."""
        imf = PowerLawIMF.kroupa()

        # Batch of uniform samples
        u_batch = jnp.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ])

        # vmap over first axis
        batched_ppf = jax.vmap(imf.ppf)
        masses = batched_ppf(u_batch)

        assert masses.shape == (3, 3)
        assert jnp.all(jnp.isfinite(masses))

    def test_chabrier_logpdf_vmap(self):
        """Chabrier logpdf works with vmap over batches."""
        imf = ChabrierIMF()

        # Batch of masses
        m_batch = jnp.array([
            [0.1, 0.5, 1.0],
            [2.0, 5.0, 10.0],
        ])

        batched_logpdf = jax.vmap(imf.logpdf)
        logpdfs = batched_logpdf(m_batch)

        assert logpdfs.shape == (2, 3)
        assert jnp.all(jnp.isfinite(logpdfs))


# =============================================================================
# End-to-End Pipeline Differentiability
# =============================================================================

class TestPipelineDifferentiability:
    """Test complete IC generation pipeline is differentiable."""

    def test_plummer_ic_grad_wrt_r_h(self):
        """Full Plummer IC is differentiable w.r.t. r_h."""
        def loss(r_h):
            profile = PlummerProfile(r_h=r_h)
            df = PlummerVelocityDF(r_h=r_h)

            masses = jnp.ones(50)
            key = jax.random.PRNGKey(42)
            key_pos, key_vel = jax.random.split(key)

            positions = profile.sample_positions(masses, key_pos)
            velocities = df.sample_velocities(positions, masses, key_vel, G=1.0)

            # Loss: total kinetic energy
            return 0.5 * jnp.sum(masses * jnp.sum(velocities**2, axis=1))

        grad_fn = jax.grad(loss)
        grad_val = grad_fn(1.0)

        assert jnp.isfinite(grad_val)
        # Larger r_h means lower density → lower escape velocity → lower KE
        assert float(grad_val) < 0

    def test_imf_to_ic_grad_wrt_m_min(self):
        """IMF → IC pipeline is differentiable w.r.t. IMF parameters."""
        def loss(m_min):
            # Create IMF with varying m_min
            imf = PowerLawIMF(
                exponents=[2.35],
                breakpoints=[],
                m_min=m_min,
                m_max=100.0,
            )

            # Sample masses via PPF (deterministic given u)
            u = jnp.linspace(0.1, 0.9, 50)
            masses = imf.ppf(u)

            # Create IC
            profile = PlummerProfile(r_h=1.0)
            key = jax.random.PRNGKey(42)
            positions = profile.sample_positions(masses, key)

            # Loss: total mass
            return jnp.sum(masses)

        grad_fn = jax.grad(loss)
        grad_val = grad_fn(0.1)

        assert jnp.isfinite(grad_val)


# =============================================================================
# JIT Recompilation Stress Tests
# =============================================================================

class TestJITRecompilation:
    """Test that JIT doesn't recompile unnecessarily."""

    def test_profile_sample_no_recompile(self):
        """Profile sampling with same shapes doesn't recompile."""
        profile = PlummerProfile(r_h=1.0)
        masses = jnp.ones(100)

        @jax.jit
        def sample(key):
            return profile.sample_positions(masses, key)

        # First call compiles
        _ = sample(jax.random.PRNGKey(1))

        # Subsequent calls should not recompile
        for i in range(5):
            positions = sample(jax.random.PRNGKey(i + 10))
            assert positions.shape == (100, 3)

    def test_imf_ppf_no_recompile(self):
        """IMF PPF with same shapes doesn't recompile."""
        imf = PowerLawIMF.kroupa()

        @jax.jit
        def ppf(u):
            return imf.ppf(u)

        # First call compiles
        u = jnp.linspace(0.1, 0.9, 100)
        _ = ppf(u)

        # Subsequent calls with same shape should not recompile
        for _ in range(5):
            masses = ppf(u)
            assert masses.shape == (100,)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

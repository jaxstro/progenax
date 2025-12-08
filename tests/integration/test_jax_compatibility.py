"""
JAX compatibility tests for progenax modules.

Consolidated tests ensuring core modules work with JIT, grad, and vmap.
One test per category per module type.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxstro.units import STELLAR
from progenax.profiles import PlummerProfile, EFFProfile
from progenax.kinematics import PlummerVelocityDF
from progenax.imf import PowerLawIMF, ChabrierIMF

G = STELLAR.G  # ≈ 0.00450 [pc³ Msun⁻¹ Myr⁻²]


class TestProfileJAXCompatibility:
    """Test spatial profiles work with JAX transformations."""

    def test_plummer_jit(self):
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

    def test_plummer_grad(self):
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


class TestVelocityDFJAXCompatibility:
    """Test velocity DFs work with JAX transformations."""

    def test_plummer_df_jit(self):
        """Plummer velocity sampling works under JIT."""
        df = PlummerVelocityDF(r_h=1.0)
        N = 100
        positions = jax.random.normal(jax.random.PRNGKey(0), (N, 3))
        masses = jnp.ones(N)

        @jax.jit
        def sample(key):
            return df.sample_velocities(positions, masses, key, G=G)

        velocities = sample(jax.random.PRNGKey(42))
        assert velocities.shape == (N, 3)
        assert jnp.all(jnp.isfinite(velocities))

    def test_plummer_df_grad(self):
        """Gradient flows through Plummer velocity sampling."""
        N = 50

        def loss(r_h):
            df = PlummerVelocityDF(r_h=r_h)
            positions = jnp.ones((N, 3)) * 0.5  # Fixed positions
            masses = jnp.ones(N)
            key = jax.random.PRNGKey(42)
            velocities = df.sample_velocities(positions, masses, key, G=G)
            return jnp.mean(jnp.sum(velocities**2, axis=1))

        grad_fn = jax.grad(loss)
        grad_val = grad_fn(1.0)

        assert jnp.isfinite(grad_val)


class TestIMFJAXCompatibility:
    """Test IMFs work with JAX transformations."""

    def test_powerlaw_jit(self):
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

    def test_powerlaw_grad(self):
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

    def test_powerlaw_vmap(self):
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
            velocities = df.sample_velocities(positions, masses, key_vel, G=G)

            # Loss: total kinetic energy
            return 0.5 * jnp.sum(masses * jnp.sum(velocities**2, axis=1))

        grad_fn = jax.grad(loss)
        grad_val = grad_fn(1.0)

        assert jnp.isfinite(grad_val)
        # Larger r_h means lower density → lower escape velocity → lower KE
        assert float(grad_val) < 0

"""MichieVelocityDF: the 2-D (v_r, v_t) sampler for the Michie-King anisotropic model.

Headline checks: beta(r) ~ 0 at the centre and increases outward (radial anisotropy);
virial equilibrium Q ~ 0.5 unscaled; large r_a -> isotropic; JIT.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxstro.units import STELLAR
from progenax.profiles.michie import MichieProfile

G = STELLAR.G


def _shell(r, N, seed):
    dirs = jax.random.normal(jax.random.PRNGKey(seed), (N, 3))
    dirs = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)
    return r * dirs


def _beta(v, pos):
    r_hat = pos / jnp.linalg.norm(pos, axis=1, keepdims=True)
    v_r = jnp.sum(v * r_hat, axis=1)
    v_t2 = jnp.sum(v**2, axis=1) - v_r**2
    return 1.0 - jnp.mean(v_t2) / (2.0 * jnp.mean(v_r**2))


class TestMichieVelocityDF:
    def test_beta_isotropic_center_radial_outward(self):
        from progenax.kinematics.michie_df import MichieVelocityDF

        df = MichieVelocityDF(W0=7.0, r_c=1.0, r_a=8.0)
        N = 40000
        betas = []
        for r, seed in [(1.0, 0), (8.0, 1), (25.0, 2)]:
            pos = _shell(r, N, seed)
            v = df.sample_velocities(pos, jnp.ones(N), jax.random.PRNGKey(seed + 10), G=G)
            betas.append(float(_beta(v, pos)))
        assert abs(betas[0]) < 0.06, f"beta(center)={betas[0]:.3f} should be ~0"
        assert betas[0] < betas[1] < betas[2], f"beta must increase outward: {betas}"
        assert betas[2] > 0.3, f"outer beta={betas[2]:.3f} should be clearly radial"

    def test_large_r_a_isotropic(self):
        from progenax.kinematics.michie_df import MichieVelocityDF

        df = MichieVelocityDF(W0=7.0, r_c=1.0, r_a=1e4)
        N = 40000
        pos = _shell(10.0, N, seed=3)
        v = df.sample_velocities(pos, jnp.ones(N), jax.random.PRNGKey(13), G=G)
        assert abs(float(_beta(v, pos))) < 0.05, "large r_a must be ~isotropic"

    def test_virial_equilibrium(self):
        from progenax.kinematics.michie_df import MichieVelocityDF
        from progenax.dynamics.virial import compute_virial_ratio

        N = 4000
        masses = jnp.ones(N)
        prof = MichieProfile.from_W0_rc(W0=7.0, r_c=1.0, r_a=8.0)
        pos = prof.sample_positions(masses, jax.random.PRNGKey(0))
        df = MichieVelocityDF(W0=7.0, r_c=1.0, r_a=8.0)
        v = df.sample_velocities(pos, masses, jax.random.PRNGKey(1), G=G)
        Q = float(compute_virial_ratio(pos, v, masses, G=G))
        assert abs(Q - 0.5) < 0.08, f"Michie-King Q={Q:.3f} should be ~0.5 unscaled"

    def test_jit_compatible(self):
        from progenax.kinematics.michie_df import MichieVelocityDF

        df = MichieVelocityDF(W0=7.0, r_c=1.0, r_a=8.0)
        N = 128
        pos = _shell(5.0, N, seed=4)
        v = jax.jit(lambda p, m, k: df.sample_velocities(p, m, k, G=G))(
            pos, jnp.ones(N), jax.random.PRNGKey(5)
        )
        assert v.shape == (128, 3) and jnp.all(jnp.isfinite(v))

    def test_grad_wrt_W0_matches_fd(self):
        """jax.grad flows through the anisotropic ODE solve + 2-D sampler (W0)."""
        from progenax.kinematics.michie_df import MichieVelocityDF

        pos = jnp.array([[2.0, 0, 0], [5.0, 0, 0], [10.0, 0, 0], [20.0, 0, 0]])
        masses = jnp.ones(4)
        key = jax.random.PRNGKey(0)

        def loss(W0):
            df = MichieVelocityDF(W0=W0, r_c=1.0, r_a=8.0)
            v = df.sample_velocities(pos, masses, key, G=G)
            return jnp.mean(jnp.sum(v**2, axis=1))

        g = jax.grad(loss)(7.0)
        g_fd = (loss(7.0 + 1e-3) - loss(7.0 - 1e-3)) / 2e-3
        assert jnp.isfinite(g), "grad through the Michie ODE+sampler must be finite"
        assert jnp.abs(g - g_fd) <= 5e-2 * jnp.abs(g_fd) + 1e-9, (
            f"grad d<|v|^2>/dW0={float(g)} vs FD {float(g_fd)}"
        )

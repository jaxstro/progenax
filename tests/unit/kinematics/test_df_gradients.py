"""FD-vs-autodiff gradient checks and JIT-compatibility for the velocity DFs.

The kinematics DFs are public differentiable entry points (loss flows through the
sampled velocities w.r.t. the DF shape parameters), but nothing pinned that the
gradients actually flow and are correct. These tests close that gap:

    - finite-difference vs jax.grad on mean(|v|^2) w.r.t. each DF parameter
    - jax.jit(sample_velocities) runs and returns finite velocities

Positions and the RNG key are held FIXED so the loss is a smooth deterministic
function of the parameter (the Beta / inverse-CDF draws don't move), making the
central finite difference a faithful reference.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxstro.units import STELLAR
from progenax.kinematics.plummer_df import PlummerVelocityDF
from progenax.kinematics.king_df import KingVelocityDF
from progenax.kinematics.eff_df import EFFVelocityDF

G = STELLAR.G


def _fd(f, x, eps):
    """Central finite difference of scalar->scalar f at x."""
    return (f(x + eps) - f(x - eps)) / (2.0 * eps)


def _fixed_positions(N, r_scale, seed):
    """Deterministic isotropic positions inside ~r_scale (param-independent)."""
    k = jax.random.PRNGKey(seed)
    u = jax.random.uniform(k, (N,), minval=0.05, maxval=0.95)
    r = r_scale * u
    dirs = jax.random.normal(jax.random.PRNGKey(seed + 1), (N, 3))
    dirs = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)
    return r[:, None] * dirs


class TestPlummerDFGradients:
    def test_grad_wrt_r_h_matches_fd(self):
        pos = _fixed_positions(300, r_scale=1.5, seed=0)
        masses = jnp.ones(300)
        key = jax.random.PRNGKey(7)

        def loss(r_h):
            v = PlummerVelocityDF(r_h=r_h).sample_velocities(pos, masses, key, G=G)
            return jnp.mean(jnp.sum(v**2, axis=1))

        g = jax.grad(loss)(1.0)
        g_fd = _fd(loss, 1.0, 1e-4)
        assert jnp.isfinite(g)
        assert jnp.abs(g - g_fd) <= 1e-3 * jnp.abs(g_fd) + 1e-6, (
            f"Plummer grad d<|v|^2>/dr_h={float(g)} vs FD {float(g_fd)}"
        )

    def test_jit_compatible(self):
        pos = _fixed_positions(128, r_scale=1.0, seed=3)
        masses = jnp.ones(128)
        df = PlummerVelocityDF(r_h=1.0)
        v = jax.jit(lambda p, m, k: df.sample_velocities(p, m, k, G=G))(
            pos, masses, jax.random.PRNGKey(1)
        )
        assert v.shape == (128, 3) and jnp.all(jnp.isfinite(v))


class TestKingDFGradients:
    @pytest.mark.parametrize("param,base,eps,rtol", [
        ("r_c", 1.0, 1e-4, 2e-3),
        ("W0", 7.0, 1e-3, 5e-2),  # W0 flows through the King ODE solve
    ])
    def test_grad_matches_fd(self, param, base, eps, rtol):
        pos = _fixed_positions(300, r_scale=3.0, seed=10)
        masses = jnp.ones(300)
        key = jax.random.PRNGKey(11)

        def loss(x):
            kw = {"W0": 7.0, "r_c": 1.0}
            kw[param] = x
            v = KingVelocityDF(**kw).sample_velocities(pos, masses, key, G=G)
            return jnp.mean(jnp.sum(v**2, axis=1))

        g = jax.grad(loss)(base)
        g_fd = _fd(loss, base, eps)
        assert jnp.isfinite(g)
        assert jnp.abs(g - g_fd) <= rtol * jnp.abs(g_fd) + 1e-6, (
            f"King grad d<|v|^2>/d{param}={float(g)} vs FD {float(g_fd)}"
        )

    def test_jit_compatible(self):
        pos = _fixed_positions(128, r_scale=3.0, seed=12)
        masses = jnp.ones(128)
        df = KingVelocityDF(W0=7.0, r_c=1.0)
        v = jax.jit(lambda p, m, k: df.sample_velocities(p, m, k, G=G))(
            pos, masses, jax.random.PRNGKey(2)
        )
        assert v.shape == (128, 3) and jnp.all(jnp.isfinite(v))


class TestEFFDFGradients:
    @pytest.mark.parametrize("param,base,eps,rtol", [
        ("a", 1.0, 1e-4, 5e-2),
        ("gamma", 3.0, 1e-3, 5e-2),  # gamma flows through the Eddington table
    ])
    def test_grad_matches_fd(self, param, base, eps, rtol):
        pos = _fixed_positions(300, r_scale=4.0, seed=20)
        masses = jnp.ones(300)
        key = jax.random.PRNGKey(21)

        def loss(x):
            kw = {"a": 1.0, "gamma": 3.0, "r_t": 10.0}
            kw[param] = x
            v = EFFVelocityDF(**kw).sample_velocities(pos, masses, key, G=G)
            return jnp.mean(jnp.sum(v**2, axis=1))

        g = jax.grad(loss)(base)
        g_fd = _fd(loss, base, eps)
        assert jnp.isfinite(g), f"EFF grad w.r.t. {param} not finite"
        assert jnp.abs(g - g_fd) <= rtol * jnp.abs(g_fd) + 1e-5, (
            f"EFF grad d<|v|^2>/d{param}={float(g)} vs FD {float(g_fd)}"
        )

    def test_jit_compatible(self):
        pos = _fixed_positions(128, r_scale=4.0, seed=22)
        masses = jnp.ones(128)
        df = EFFVelocityDF(a=1.0, gamma=3.0, r_t=10.0)
        v = jax.jit(lambda p, m, k: df.sample_velocities(p, m, k, G=G))(
            pos, masses, jax.random.PRNGKey(4)
        )
        assert v.shape == (128, 3) and jnp.all(jnp.isfinite(v))

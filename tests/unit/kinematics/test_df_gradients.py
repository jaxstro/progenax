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

from jaxstro.units import STELLAR
from progenax.kinematics.plummer_df import PlummerVelocityDF
from progenax.kinematics.king_df import KingVelocityDF
from progenax.kinematics.eff_df import EFFVelocityDF

G = STELLAR.G


def _fixed_positions(N, r_scale, seed):
    """Deterministic isotropic positions inside ~r_scale (param-independent)."""
    k = jax.random.PRNGKey(seed)
    u = jax.random.uniform(k, (N,), minval=0.05, maxval=0.95)
    r = r_scale * u
    dirs = jax.random.normal(jax.random.PRNGKey(seed + 1), (N, 3))
    dirs = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)
    return r[:, None] * dirs


class TestPlummerDFGradients:
    # AD-vs-FD for PlummerVelocityDF.sample_velocities(r_h) is owned by the grad-audit
    # registry (tests/validation/grad_audit/registry.py :: PlummerVelocityDF.sample_velocities);
    # see docs/website/50-validation/differentiability-audit.md. (audit T6 consolidation; registry is SoT)
    def test_jit_compatible(self):
        pos = _fixed_positions(128, r_scale=1.0, seed=3)
        masses = jnp.ones(128)
        df = PlummerVelocityDF(r_h=1.0)
        v = jax.jit(lambda p, m, k: df.sample_velocities(p, m, k, G=G))(
            pos, masses, jax.random.PRNGKey(1)
        )
        assert v.shape == (128, 3) and jnp.all(jnp.isfinite(v))


class TestKingDFGradients:
    # AD-vs-FD for KingVelocityDF.sample_velocities(W0) and (r_c) is owned by the grad-audit
    # registry (tests/validation/grad_audit/registry.py :: KingVelocityDF.sample_velocities,
    # both the W0 and r_c channels); see docs/website/50-validation/differentiability-audit.md.
    # (audit T6 consolidation; registry is SoT)
    def test_jit_compatible(self):
        pos = _fixed_positions(128, r_scale=3.0, seed=12)
        masses = jnp.ones(128)
        df = KingVelocityDF(W0=7.0, r_c=1.0)
        v = jax.jit(lambda p, m, k: df.sample_velocities(p, m, k, G=G))(
            pos, masses, jax.random.PRNGKey(2)
        )
        assert v.shape == (128, 3) and jnp.all(jnp.isfinite(v))


class TestEFFDFGradients:
    # AD-vs-FD for EFFVelocityDF.sample_velocities(gamma) and (a) is owned by the grad-audit
    # registry (tests/validation/grad_audit/registry.py :: EFFVelocityDF.sample_velocities,
    # both the gamma and a channels); see docs/website/50-validation/differentiability-audit.md.
    # (audit T6 consolidation; registry is SoT)
    def test_jit_compatible(self):
        pos = _fixed_positions(128, r_scale=4.0, seed=22)
        masses = jnp.ones(128)
        df = EFFVelocityDF(a=1.0, gamma=3.0, r_t=10.0)
        v = jax.jit(lambda p, m, k: df.sample_velocities(p, m, k, G=G))(
            pos, masses, jax.random.PRNGKey(4)
        )
        assert v.shape == (128, 3) and jnp.all(jnp.isfinite(v))

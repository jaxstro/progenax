"""Demo-harness unit tests for scripts/_demo_binaries.py (B12).

Demo-harness tier — NOT released-core. These exercise the reusable pieces of the
binary-inflated dynamical-mass demo: the isotropic LOS projection, the
sigma-independent flux-weighted blend kernel K_orb (Moe P-q-e + ZAMS L), and the
differentiable binned single+binary mixture model.
"""
import numpy as np
import jax.numpy as jnp
import progenax  # noqa: F401  -- enables float64 at import

from scripts._demo_binaries import project_los_velocity


def test_los_projection_is_isotropic_mean_zero():
    """Dotting a fixed 3-velocity onto random isotropic LOS directions gives a
    mean-zero projection with std = |v|/sqrt(3) (variance shared over 3 axes)."""
    rng = np.random.default_rng(0)
    v = jnp.array([10.0, 0.0, 0.0])  # 10 km/s along x
    dirs = rng.normal(size=(20000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    los = np.array([float(project_los_velocity(v, jnp.asarray(d))) for d in dirs])
    assert abs(los.mean()) < 0.3                        # isotropic -> mean ~0
    assert np.isclose(los.std(), 10.0 / np.sqrt(3), rtol=0.05)  # variance/3 per axis

"""Demo-harness unit tests for scripts/_demo_binaries.py (B12).

Demo-harness tier — NOT released-core. These exercise the reusable pieces of the
binary-inflated dynamical-mass demo: the isotropic LOS projection, the
sigma-independent flux-weighted blend kernel K_orb (Moe P-q-e + ZAMS L), and the
differentiable binned single+binary mixture model.
"""
import numpy as np
import jax.numpy as jnp
import progenax  # noqa: F401  -- enables float64 at import

from scripts._demo_binaries import (
    project_los_velocity,
    build_korb_kernel,
    _kernel_std,
)


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


def test_korb_high_q_self_cancels():
    """Equal-mass (q=1) binaries: L1=L2, v1=-v2 -> flux-weighted Delta cancels to
    ~0, so K_orb is far narrower than for a low-q (primary-reflex) pool."""
    v_grid, k = build_korb_kernel(n_pool=20000, q_fixed=1.0, Z=1e-3, seed=1)
    spread_hi = _kernel_std(v_grid, k)
    v_grid2, k2 = build_korb_kernel(n_pool=20000, q_fixed=0.2, Z=1e-3, seed=1)
    spread_lo = _kernel_std(v_grid2, k2)
    assert spread_hi < 0.3 * spread_lo   # high-q cancels, low-q (primary) does not


def test_korb_normalized_and_zero_mean():
    """K_orb integrates to 1 (proper density) and is ~zero-mean (random phase +
    isotropic orientation give a symmetric blend-velocity distribution)."""
    v_grid, k = build_korb_kernel(n_pool=50000, Z=1e-3, seed=2)
    dv = v_grid[1] - v_grid[0]
    assert np.isclose(np.sum(k) * dv, 1.0, atol=1e-3)   # normalized density
    assert abs(np.sum(k * v_grid) * dv) < 0.5           # ~zero mean

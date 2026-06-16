"""Stage-1 OED demo unit tests (Task 1: predicted observable + per-star Fisher blocks)."""
import sys
import pathlib

import jax
import jax.numpy as jnp
import progenax  # noqa: F401  -- enables float64
from jaxstro.units import STELLAR

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
import _demo_oed as oed  # noqa: E402


def test_predict_sigma_shape_and_units():
    th = oed.theta_truth()                      # (3,) = (r_a, M, r_h)
    sig = oed.predict_sigma(th, oed.R_BINS, STELLAR.G)   # (3, K) channels x bins
    assert sig.shape == (3, oed.R_BINS.shape[0])
    assert jnp.all(sig > 0)
    # isotropic-ish check: at small R, los ~ pm_r ~ pm_t within 30%
    inner = sig[:, 0]
    assert jnp.max(inner) / jnp.min(inner) < 1.5


def test_per_star_blocks_shape_and_symmetry():
    th = oed.theta_truth()
    Mb, sig = oed.per_star_blocks(th, oed.R_BINS, oed.EPS, STELLAR.G)
    K = oed.R_BINS.shape[0]
    assert Mb.shape == (3, K, 3, 3)             # channel, bin, P, P
    # each block is symmetric PSD rank-1: M = 2 J J^T / denom
    assert jnp.allclose(Mb, jnp.swapaxes(Mb, -1, -2), atol=1e-12)
    # diagonal entries non-negative
    assert jnp.all(jnp.diagonal(Mb, axis1=-2, axis2=-1) >= -1e-12)

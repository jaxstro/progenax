"""Unit tests for the 2-D Limber angular band-power predictor (A-new1, Piece 2).

``angular_bandpowers_2d_limber`` walks the validated forward chain
    rho_g(beta) -> d_n (exact BM19 density Hermite) -> xi_rho (Mehler, NOT expm1)
    -> Limber slab projection -> 2-D FFT (Wiener-Khinchin, derived norm = 1) -> |k|-binned.
``add_poisson_shot`` adds the validated count-observable shot model on top.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.experimental


def test_bandpowers_shape():
    """Returns band-powers of length len(k_edges)-1 (and matching centers, n_modes)."""
    from gravoturb.inference.covariance import angular_bandpowers_2d_limber

    shape = (32, 32, 32)
    k_edges = jnp.linspace(1.0, 12.0, 7)  # 6 bins
    kc, bp, nm = angular_bandpowers_2d_limber(
        shape, 3.0, 8.0, 0.4, 2.5, depth=32.0, k_edges=k_edges, n_max=8, n_quad=64
    )
    assert bp.shape == (len(k_edges) - 1,)
    assert kc.shape == (len(k_edges) - 1,)
    assert nm.shape == (len(k_edges) - 1,)


def test_bandpowers_differentiable_in_beta_and_mach():
    """jax.grad of sum(bandpowers) wrt beta and wrt mach are finite and nonzero."""
    from gravoturb.inference.covariance import angular_bandpowers_2d_limber

    shape = (32, 32, 32)
    k_edges = jnp.linspace(1.0, 12.0, 7)

    def total_beta(beta):
        return jnp.sum(
            angular_bandpowers_2d_limber(
                shape, beta, 8.0, 0.4, 2.5, 32.0, k_edges, 8, 64
            )[1]
        )

    def total_mach(mach):
        return jnp.sum(
            angular_bandpowers_2d_limber(
                shape, 3.0, mach, 0.4, 2.5, 32.0, k_edges, 8, 64
            )[1]
        )

    g_beta = float(jax.grad(total_beta)(3.0))
    g_mach = float(jax.grad(total_mach)(8.0))
    assert np.isfinite(g_beta) and g_beta != 0.0
    assert np.isfinite(g_mach) and g_mach != 0.0


def test_bandpower_slope_steepens_with_beta():
    """Higher beta -> steeper predicted band-power slope (more large-scale power)."""
    from gravoturb.inference.covariance import angular_bandpowers_2d_limber

    shape = (48, 48, 48)
    k_edges = jnp.linspace(1.0, 18.0, 10)

    def slope(beta):
        kc, bp, _ = angular_bandpowers_2d_limber(
            shape, beta, 8.0, 0.4, 2.5, 48.0, k_edges, 14, 128
        )
        kc, bp = np.asarray(kc), np.asarray(bp)
        m = (bp > 0) & np.isfinite(bp)
        # negative power-law exponent; steeper = more negative
        return np.polyfit(np.log(kc[m]), np.log(bp[m]), 1)[0]

    s25 = slope(2.5)
    s35 = slope(3.5)
    assert s35 < s25  # higher beta -> steeper (more negative) slope


def test_slab_full_depth_reduces_to_full_projection():
    """At depth = n_los the slab projection equals the periodic full projection."""
    from gravoturb.inference.covariance import (
        _angular_bandpowers_from_xi_rho_full,
        angular_bandpowers_2d_limber,
    )

    shape = (32, 32, 32)
    k_edges = jnp.linspace(1.0, 12.0, 7)
    _, bp_slab, _ = angular_bandpowers_2d_limber(
        shape,
        3.0,
        8.0,
        0.4,
        2.5,
        depth=float(shape[2]),
        k_edges=k_edges,
        n_max=8,
        n_quad=64,
    )
    _, bp_full, _ = _angular_bandpowers_from_xi_rho_full(
        shape, 3.0, 8.0, 0.4, 2.5, k_edges=k_edges, n_max=8, n_quad=64
    )
    assert np.allclose(np.asarray(bp_slab), np.asarray(bp_full), rtol=1e-10, atol=1e-12)


def test_add_poisson_shot_limits():
    """Shot model: high-k (clustering->0) approaches n_bar_sky; low-k dominated by clustering."""
    from gravoturb.inference.covariance import add_poisson_shot

    n_bar_sky = 50.0
    depth = 96.0
    # synthetic clustering band-powers: large at low k, ~0 at high k
    clustering = jnp.array([1.0e3, 1.0e1, 1.0e-1, 1.0e-6])
    total = add_poisson_shot(clustering, n_bar_sky, depth)
    total = np.asarray(total)
    scale = (n_bar_sky / depth) ** 2
    # high-k bin: clustering negligible -> total ~ shot = n_bar_sky
    assert total[-1] == pytest.approx(n_bar_sky, rel=1e-4)
    # low-k bin: clustering dominates over shot
    assert scale * 1.0e3 > n_bar_sky
    assert total[0] == pytest.approx(scale * 1.0e3 + n_bar_sky, rel=1e-10)

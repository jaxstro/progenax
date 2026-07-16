"""Unit tests for the analytic log+ projected-band-power predictor (Phase-2 step 1).

``predict_logp_bandpowers`` is the forward model mu(beta) = A_s(beta, mach) * T_fixed(k), where A_s is
the analytic projected LOG-density band-powers (Limber of xi_s via the log-density Hermite c_n) and T
is a calibrate-once, beta-independent per-bin transfer. Phase-0 (validation/_d03,_d05) established
that this keeps the beta-response purely analytic (no emulated slope) with a beta-stable (~5%) transfer
-- the property rank-G lacks.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.experimental

SHAPE = (32, 32, 32)
K_EDGES = jnp.linspace(1.0, 12.0, 7)  # 6 bins
NB = len(K_EDGES) - 1


def _slope(k_edges, P):
    k = 0.5 * (np.asarray(k_edges)[:-1] + np.asarray(k_edges)[1:])
    P = np.asarray(P)
    return np.polyfit(np.log(k), np.log(P), 1)[0]


def test_uses_logdensity_chain_steeper_than_density():
    """A_s (log-density) band-powers are steeper than the density predictor at the same beta.

    Distinguishes predict_logp_bandpowers (log-density c_n chain) from angular_bandpowers_2d_limber
    (density d_n chain): Phase-0 D01 measured A_s slope ~ -3.2 vs A_rho ~ -2.4 at beta=3, i.e. the
    log-density observable is markedly steeper.
    """
    from gravoturb.inference.covariance import angular_bandpowers_2d_limber
    from gravoturb.inference.projected_logp import predict_logp_bandpowers

    T = jnp.ones(NB)
    mu = predict_logp_bandpowers(
        SHAPE, 3.0, 8.0, 0.4, 2.5, 32.0, K_EDGES, T, n_max=8, n_quad=64
    )
    _kc, P_rho, _nm = angular_bandpowers_2d_limber(
        SHAPE, 3.0, 8.0, 0.4, 2.5, 32.0, K_EDGES, 8, 64
    )
    assert _slope(K_EDGES, mu) < _slope(K_EDGES, P_rho) - 0.3


def test_transfer_scales_per_bin():
    """mu(T) == mu(ones) * T elementwise (the transfer is a per-bin multiplicative factor)."""
    from gravoturb.inference.projected_logp import predict_logp_bandpowers

    T = jnp.linspace(0.5, 2.0, NB)
    base = predict_logp_bandpowers(
        SHAPE, 3.0, 8.0, 0.4, 2.5, 32.0, K_EDGES, jnp.ones(NB), 8, 64
    )
    scaled = predict_logp_bandpowers(SHAPE, 3.0, 8.0, 0.4, 2.5, 32.0, K_EDGES, T, 8, 64)
    np.testing.assert_allclose(
        np.asarray(scaled), np.asarray(base) * np.asarray(T), rtol=1e-10
    )


def test_differentiable_in_beta():
    """jax.grad of sum(mu) wrt beta is finite and nonzero (analytic beta-response)."""
    from gravoturb.inference.projected_logp import predict_logp_bandpowers

    T = jnp.ones(NB)

    def total(beta):
        return jnp.sum(
            predict_logp_bandpowers(SHAPE, beta, 8.0, 0.4, 2.5, 32.0, K_EDGES, T, 8, 64)
        )

    g = float(jax.grad(total)(3.0))
    assert np.isfinite(g) and g != 0.0


def test_slope_steepens_with_beta():
    """Higher beta -> steeper (more negative) predicted log-density band-power slope."""
    from gravoturb.inference.projected_logp import predict_logp_bandpowers

    T = jnp.ones(NB)
    lo = predict_logp_bandpowers(SHAPE, 2.5, 8.0, 0.4, 2.5, 32.0, K_EDGES, T, 8, 64)
    hi = predict_logp_bandpowers(SHAPE, 3.5, 8.0, 0.4, 2.5, 32.0, K_EDGES, T, 8, 64)
    assert _slope(K_EDGES, hi) < _slope(K_EDGES, lo)


def test_analytic_emulator_matches_direct_model():
    """Interpolating the SMOOTH analytic A_s(beta) table reproduces the direct model to <0.5%.

    Unlike a noisy-simulation emulator (whose interpolation corrupts the beta-slope -> the v2h flaw),
    A_s is deterministic and smooth, so a table + linear interp preserves the beta-response. Tested at
    a random OFF-node beta against the direct analytic chain.
    """
    from gravoturb.inference.projected_logp import (
        interp_logp_bandpowers,
        precompute_a_s_table,
        predict_logp_bandpowers,
    )

    beta_nodes = jnp.linspace(2.0, 11.0 / 3.0, 96)
    table = precompute_a_s_table(
        SHAPE, 8.0, 0.4, 2.5, 32.0, K_EDGES, beta_nodes, n_max=8, n_quad=64
    )
    assert np.asarray(table).shape == (96, NB)
    T = jnp.linspace(0.8, 1.2, NB)
    beta_off = 2.917  # off-node
    mu_emu = interp_logp_bandpowers(beta_off, beta_nodes, table, T)
    mu_dir = predict_logp_bandpowers(
        SHAPE, beta_off, 8.0, 0.4, 2.5, 32.0, K_EDGES, T, 8, 64
    )
    np.testing.assert_allclose(np.asarray(mu_emu), np.asarray(mu_dir), rtol=5e-3)


def test_analytic_emulator_differentiable():
    """jax.grad through the interpolated emulator wrt beta is finite and nonzero."""
    from gravoturb.inference.projected_logp import (
        interp_logp_bandpowers,
        precompute_a_s_table,
    )

    beta_nodes = jnp.linspace(2.0, 11.0 / 3.0, 96)
    table = precompute_a_s_table(
        SHAPE, 8.0, 0.4, 2.5, 32.0, K_EDGES, beta_nodes, n_max=8, n_quad=64
    )
    T = jnp.ones(NB)
    g = float(
        jax.grad(lambda b: jnp.sum(interp_logp_bandpowers(b, beta_nodes, table, T)))(
            2.917
        )
    )
    assert np.isfinite(g) and g != 0.0


def _logplus_limit_reference(
    shape, beta, mach, b, alpha, depth, k_edges, n_max, n_quad
):
    """Deterministic high-N limit: the Mehler 2-pt of log_+(Sigma/L) (NO Poisson noise).

    As n_bar -> inf the Poisson-smoothed map m(Sigma) -> log_+(Sigma/L) (piecewise: ln above the mean,
    linear below), so the shot model's clustering term must converge to the band-powers of this
    deterministic log_+ of the lognormal projected density -- NOT pure ln (that is the ln-vs-log_+
    difference the calibrated transfer absorbs)."""
    from gravoturb.inference.covariance import (
        _angular_bandpowers_from_xi_rho_2d,
        _xi_rho_grid,
    )
    from jaxstro.numerics.quadrature import hermite_coefficients
    from gravoturb.theory.log_correlations import gaussianized_xi
    from gravoturb.theory.projection import limber_project_slab

    xi_rho = _xi_rho_grid(shape, beta, mach, b, alpha, n_max, n_quad)
    xi_Sigma = limber_project_slab(xi_rho, depth, los_axis=2)
    L = float(depth)
    s2 = jnp.log1p(xi_Sigma[0, 0] / L**2)
    s = jnp.sqrt(s2)
    rho_g = jnp.log1p(xi_Sigma / L**2) / s2

    def lp_map(g):
        x = jnp.exp(s * g - 0.5 * s2)  # Sigma/L, mean 1
        return jnp.where(x > 1.0, jnp.log(jnp.where(x > 0, x, 1.0)), x - 1.0)

    a = hermite_coefficients(lp_map, n_max, n_quad)
    return np.asarray(
        _angular_bandpowers_from_xi_rho_2d(gaussianized_xi(rho_g, a), k_edges)[1]
    )


def test_shot_model_differentiable_in_beta():
    """jax.grad of the shot-transfer forward model wrt beta is finite and nonzero."""
    from gravoturb.inference.projected_logp import predict_logp_bandpowers_shot

    def total(beta):
        return jnp.sum(
            predict_logp_bandpowers_shot(
                SHAPE,
                beta,
                8.0,
                0.4,
                2.5,
                32.0,
                K_EDGES,
                n_bar_3d=0.4,
                n_max=8,
                n_quad=64,
                n_count_max=400,
            )
        )

    g = float(jax.grad(total)(3.0))
    assert np.isfinite(g) and g != 0.0


def test_shot_floor_positive_and_decreases_with_nbar():
    """W_shot (white Poisson floor) is positive and SMALLER at higher mean count."""
    from gravoturb.inference.projected_logp import logp_shot_components

    _Pc_lo, W_lo = logp_shot_components(
        SHAPE,
        3.0,
        8.0,
        0.4,
        2.5,
        32.0,
        K_EDGES,
        n_bar_3d=0.3,
        n_max=8,
        n_quad=64,
        n_count_max=400,
    )
    _Pc_hi, W_hi = logp_shot_components(
        SHAPE,
        3.0,
        8.0,
        0.4,
        2.5,
        32.0,
        K_EDGES,
        n_bar_3d=3.0,
        n_max=8,
        n_quad=64,
        n_count_max=2000,
    )
    assert float(W_lo) > 0.0 and float(W_hi) > 0.0
    assert float(W_hi) < float(W_lo)


def test_shot_clustering_reduces_to_log_limit_at_high_nbar():
    """At high mean count the clustering term -> the lognormal-copula log predictor (A_logSig)."""
    from gravoturb.inference.projected_logp import logp_shot_components

    # high n_bar but modest absolute count (low n_bar_3d on a deep box) so the Poisson sum is fully
    # resolved by n_count_max (the shot model is cheap only at low counts -- exactly the regime we need
    # it for). n_bar_sky = n_bar_3d * depth = 8 * 32 = 256 -> log limit well-approached.
    sh, dep, nmax, nq = (32, 32, 32), 32.0, 8, 64
    ke = jnp.linspace(2.0, 12.0, 6)
    P_clust, _W = logp_shot_components(
        sh,
        3.0,
        8.0,
        0.4,
        2.5,
        dep,
        ke,
        n_bar_3d=8.0,
        n_max=nmax,
        n_quad=nq,
        n_count_max=6000,
    )
    ref = _logplus_limit_reference(sh, 3.0, 8.0, 0.4, 2.5, dep, ke, nmax, nq)
    np.testing.assert_allclose(np.asarray(P_clust), ref, rtol=0.08)


def test_loglike_stationary_and_peaked_at_truth():
    """Noiseless data = mu(beta_true): the Gaussian log-like is peaked at beta_true with grad ~ 0."""
    from gravoturb.inference.projected_logp import (
        logp_loglike,
        predict_logp_bandpowers,
    )

    T = jnp.ones(NB)
    prec = jnp.eye(NB)
    beta_true = 3.0
    data = predict_logp_bandpowers(
        SHAPE, beta_true, 8.0, 0.4, 2.5, 32.0, K_EDGES, T, 8, 64
    )

    def ll(beta):
        return logp_loglike(
            data, beta, 8.0, 0.4, 2.5, 32.0, SHAPE, K_EDGES, T, prec, 8, 64
        )

    assert float(ll(beta_true)) > float(ll(beta_true + 0.3))
    assert float(ll(beta_true)) > float(ll(beta_true - 0.3))
    g = float(jax.grad(ll)(beta_true))
    assert np.isfinite(g) and abs(g) < 1e-4  # stationary at the truth (noiseless)


def test_loglike_differentiable_off_truth():
    """jax.grad of the log-like wrt beta is finite away from the truth (gradient-based inference)."""
    from gravoturb.inference.projected_logp import (
        logp_loglike,
        predict_logp_bandpowers,
    )

    T = jnp.ones(NB)
    prec = jnp.eye(NB)
    data = predict_logp_bandpowers(SHAPE, 3.0, 8.0, 0.4, 2.5, 32.0, K_EDGES, T, 8, 64)
    g = float(
        jax.grad(
            lambda b: logp_loglike(
                data, b, 8.0, 0.4, 2.5, 32.0, SHAPE, K_EDGES, T, prec, 8, 64
            )
        )(2.5)
    )
    assert np.isfinite(g) and g != 0.0


def test_calibrate_transfer_makes_model_match_fiducial_mean():
    """T = mean(observable rows) / A_s_fid, so A_s_fid * T reproduces the observable mean exactly.

    This is the defining contract: at the fiducial, mu(theta_fid) = A_s(theta_fid) * T == E[data|fid].
    The beta-RESPONSE is unaffected (T is a constant) -- step-1 tests cover that.
    """
    from gravoturb.inference.projected_logp import calibrate_transfer

    rng = np.random.default_rng(0)
    rows = rng.uniform(1.0, 5.0, size=(16, NB))  # synthetic observable band-power rows
    a_s_fid = np.linspace(2.0, 0.2, NB)  # synthetic analytic A_s at fiducial
    T = calibrate_transfer(jnp.asarray(rows), jnp.asarray(a_s_fid))
    assert np.asarray(T).shape == (NB,)
    np.testing.assert_allclose(
        np.asarray(a_s_fid) * np.asarray(T), rows.mean(axis=0), rtol=1e-10
    )

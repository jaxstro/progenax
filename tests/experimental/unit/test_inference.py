"""Phase 5 (Milestone 1): differentiable inference -- covariance, likelihood, Fisher.

Data vector d(theta) = [log-density power-spectrum band-powers P_s(k_i), CIC variance
sigma^2_N(R)]. Band-powers (Anna 2026-06-05) give the exact diagonal Gaussian covariance
2 P_s^2 / N_modes; validated against the realization mock covariance (Hartlap-corrected).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.experimental


def test_power_spectrum_bandpowers_positive_and_differentiable():
    """P_s(k_i) band-powers (radially-binned FFT of the xi_s grid) are positive and
    differentiable in (mach,b,alpha,beta)."""
    from gravoturb_fdf.inference.covariance import power_spectrum_bandpowers

    shape, k_edges = (32, 32, 32), jnp.linspace(1.0, 13.0, 7)
    kc, P, nmodes = power_spectrum_bandpowers(shape, 3.0, 5.0, 0.4, 2.5, k_edges, n_max=12)
    assert P.shape == kc.shape and kc.shape[0] == 6
    assert jnp.all(P > 0) and jnp.all(nmodes > 0)

    for j, name in enumerate(["beta", "mach", "b", "alpha"]):
        args = [3.0, 5.0, 0.4, 2.5]
        g = jax.grad(lambda v, j=j: jnp.sum(power_spectrum_bandpowers(
            shape, *[v if i == j else args[i] for i in range(4)], k_edges, n_max=12)[1]))(args[j])
        assert np.isfinite(float(g))


def test_bandpowers_match_mock():
    """Analytic band-powers == ensemble-mean measured periodogram band-powers of the
    smooth-copula log-density field (forward fidelity of P_s(k), the Fourier dual of AC11)."""
    from gravoturb_fdf.inference.covariance import power_spectrum_bandpowers, measured_bandpowers
    from gravoturb_fdf.field.field import gaussian_random_field
    from gravoturb_fdf.validation.measure import smooth_copula_field

    shape, beta, mach, b, alpha = (32, 32, 32), 3.0, 5.0, 0.4, 2.5
    k_edges = jnp.linspace(1.0, 13.0, 7)
    _, P_an, _ = power_spectrum_bandpowers(shape, beta, mach, b, alpha, k_edges, n_max=14)

    key = jax.random.PRNGKey(0)
    acc = np.zeros(len(k_edges) - 1)
    n_real = 24
    for i in range(n_real):
        s = np.asarray(smooth_copula_field(
            gaussian_random_field(shape, beta, jax.random.fold_in(key, i)), mach, b, alpha))
        acc += measured_bandpowers(s, shape, k_edges)
    P_mock = acc / n_real
    rel = np.abs(np.asarray(P_an) - P_mock) / np.abs(P_mock)
    assert np.median(rel) < 0.05


def test_mock_covariance_psd_and_hartlap():
    """mock_covariance is symmetric positive-definite and recovers a known covariance;
    the Hartlap inverse-debias factor is in (0,1) and -> 1 as n_real -> inf."""
    from gravoturb_fdf.inference.covariance import (
        mock_covariance, hartlap_factor, mock_precision)

    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4))
    true_cov = A @ A.T + 0.1 * np.eye(4)
    rows = rng.multivariate_normal(np.zeros(4), true_cov, size=600)
    C = mock_covariance(rows)
    assert np.allclose(C, C.T)
    assert np.all(np.linalg.eigvalsh(C) > 0)
    assert np.allclose(C, true_cov, rtol=0.0, atol=0.25 * np.abs(true_cov).max())

    assert 0.0 < hartlap_factor(600, 4) < 1.0
    assert hartlap_factor(1_000_000, 4) == pytest.approx(1.0, abs=1e-4)
    prec = mock_precision(rows)  # Hartlap-corrected C^{-1}
    assert np.allclose(prec, prec.T)
    assert np.all(np.linalg.eigvalsh(prec) > 0)


def test_gaussian_bandpower_covariance_underestimates_mock():
    """Documents the Phase-5 finding (regression guard): the diagnostic Gaussian band-power
    covariance 2P^2/N UNDERESTIMATES the true mock band-power variance for the non-Gaussian
    log-density field -- which is WHY the Fisher uses the mock covariance (Anna 2026-06-05)."""
    from gravoturb_fdf.inference.covariance import (
        power_spectrum_bandpowers, gaussian_bandpower_covariance, measured_bandpowers,
        mock_covariance)
    from gravoturb_fdf.field.field import gaussian_random_field
    from gravoturb_fdf.validation.measure import smooth_copula_field

    shape, beta, mach, b, alpha = (32, 32, 32), 3.0, 5.0, 0.4, 2.5
    k_edges = jnp.linspace(2.0, 14.0, 7)
    _, P, nmodes = power_spectrum_bandpowers(shape, beta, mach, b, alpha, k_edges, n_max=14)
    gauss_diag = np.diag(np.asarray(gaussian_bandpower_covariance(P, nmodes)))

    key = jax.random.PRNGKey(0)
    rows = [measured_bandpowers(np.asarray(smooth_copula_field(
        gaussian_random_field(shape, beta, jax.random.fold_in(key, i)), mach, b, alpha)),
        shape, k_edges) for i in range(80)]
    mock_diag = np.diag(mock_covariance(rows))
    ratio = mock_diag / gauss_diag
    assert np.all(ratio > 1.3)          # Gaussian C is an underestimate everywhere
    assert ratio[-1] > 2.0 * ratio[0]   # excess grows toward small scales (high k)


# --- Task 5.2: Gaussian likelihood on the data vector --------------------------

_CFG = dict(shape=(24, 24, 24), k_edges=jnp.linspace(2.0, 11.0, 5), cell_sizes=(4,),
            n_bar=30.0, n_max=12)
_THETA = jnp.array([5.0, 0.4, 2.5, 3.0])  # (mach, b, alpha, beta)


def test_data_vector_shape_and_differentiable():
    """d(theta) = [P_s(k_i) band-powers, sigma^2_N(c)] is finite, correctly shaped, and
    differentiable in theta=(mach,b,alpha,beta)."""
    from gravoturb_fdf.inference.likelihood import data_vector

    d = data_vector(_THETA, **_CFG)
    assert d.shape == (4 + 1,)  # 4 band-power bins + 1 CIC variance
    assert jnp.all(jnp.isfinite(d))
    g = np.asarray(jax.grad(lambda th: jnp.sum(data_vector(th, **_CFG)))(_THETA))
    assert np.all(np.isfinite(g)) and np.any(np.abs(g) > 0)


def test_gaussian_loglike_max_at_truth_and_differentiable():
    """gaussian_loglike on NOISELESS data (= d(theta_true)) peaks at theta_true (zero residual)
    and is a finite, differentiable scalar."""
    from gravoturb_fdf.inference.likelihood import data_vector, gaussian_loglike

    data = data_vector(_THETA, **_CFG)
    precision = jnp.diag(1.0 / data**2)  # any PD precision -> max at truth
    ll0 = float(gaussian_loglike(data, _THETA, precision, **_CFG))
    assert ll0 == pytest.approx(0.0, abs=1e-8)  # zero residual at truth

    for j, dth in enumerate([0.5, 0.05, 0.3, 0.3]):
        pert = _THETA.at[j].add(dth)
        assert float(gaussian_loglike(data, pert, precision, **_CFG)) < ll0 - 1e-6

    g = np.asarray(jax.grad(lambda th: gaussian_loglike(data, th, precision, **_CFG))(_THETA))
    assert np.all(np.isfinite(g))

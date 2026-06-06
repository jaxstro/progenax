"""Unit tests for gravoturb_fdf.theory.projection (Phase 2).

The analytic normalized Gaussian correlation rho_g(r; beta) for P(k) = k^{-beta} on the
simulator's grid -- the beta-carrier of the predicted 2-point. Validated against the
realization oracle (gaussian_correlation_measured) and required differentiable in beta.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.experimental


def test_rho_g_grid_normalized_at_zero_lag():
    """rho_g(0) = 1 by construction (xi / xi[0])."""
    from gravoturb_fdf.theory.projection import gaussian_correlation_grid

    rho = gaussian_correlation_grid((24, 24, 24), 3.0)
    assert float(rho[0, 0, 0]) == pytest.approx(1.0, abs=1e-12)
    assert float(jnp.max(jnp.abs(rho))) == pytest.approx(1.0, abs=1e-9)  # peak at 0


def test_rho_g_grid_matches_measured_oracle():
    """Analytic rho_g(r;beta) == ensemble-mean measured rho_g from gaussian_random_field."""
    from gravoturb_fdf.theory.projection import gaussian_correlation_grid
    from gravoturb_fdf.field.field import gaussian_random_field
    from gravoturb_fdf.validation.measure import (
        gaussian_correlation_measured, radial_average)

    # The .real full-grid construction matches the analytic rho_g = IFFT[k^-beta]/(.)[0]
    # in expectation (proof: <autocov(Re h)> = (1/2) IFFT(|amp|^2)), so the residual is
    # measurement noise -> tightens with n_real (20.9%/4.3%/3.8% max at n_real 8/16/32).
    shape, beta, n_bins, n_real = (40, 40, 40), 3.0, 12, 16
    rho_grid = np.asarray(gaussian_correlation_grid(shape, beta))
    _, rho_a = radial_average(rho_grid, n_bins=n_bins)

    key = jax.random.PRNGKey(0)
    accs = []
    for i in range(n_real):
        g = gaussian_random_field(shape, beta, jax.random.fold_in(key, i))
        _, rho_m = gaussian_correlation_measured(np.asarray(g), n_bins=n_bins)
        accs.append(rho_m)
    rho_m = np.mean(accs, axis=0)

    mask = rho_a > 0.1  # signal-dominated bins (small-rho bins are noise/signal-limited)
    rel = np.abs(rho_a[mask] - rho_m[mask]) / np.abs(rho_a[mask])
    assert np.median(rel) < 0.05
    assert rel.max() < 0.10


def test_grf_realizes_power_law_spectrum():
    """gaussian_random_field must realize P(k) ~ k^{-beta}: the azimuthally-averaged
    ratio P_meas(k)/k^{-beta} is flat (to <10%) across k -- no high-k excess from
    un-enforced Hermitian symmetry."""
    from gravoturb_fdf.field.field import gaussian_random_field
    from gravoturb_fdf.theory.projection import _kmag_grid

    shape, beta, n_real = (40, 40, 40), 3.0, 8
    kmag = np.asarray(_kmag_grid(shape))
    ideal = np.where(kmag > 0, np.where(kmag > 0, kmag, 1.0) ** (-beta), 0.0)
    key = jax.random.PRNGKey(0)
    power = np.zeros(shape)
    for i in range(n_real):
        g = np.asarray(gaussian_random_field(shape, beta, jax.random.fold_in(key, i)))
        g = g - g.mean()
        power += np.abs(np.fft.fftn(g)) ** 2 / g.size
    power /= n_real

    kf, pf, idf = kmag.ravel(), power.ravel(), ideal.ravel()
    edges = np.linspace(1.0, 18.0, 9)
    ratios = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (kf >= lo) & (kf < hi)
        if m.any():
            ratios.append(pf[m].mean() / idf[m].mean())
    ratios = np.array(ratios)
    assert (ratios.max() - ratios.min()) / ratios.mean() < 0.10


def test_rho_g_grid_differentiable_in_beta():
    """d rho_g(r;beta)/d beta finite and nonzero at a representative separation."""
    from gravoturb_fdf.theory.projection import gaussian_correlation_grid

    def rho_at(beta):
        return gaussian_correlation_grid((24, 24, 24), beta)[3, 0, 0]

    grad = float(jax.grad(rho_at)(3.0))
    assert np.isfinite(grad) and abs(grad) > 0.0


# --- Task 2.2: smoothing at scale R (window, sigma_g^2(R)) ---------------------


def test_window_functions_normalized_at_zero():
    """W(0) = 1 for both top-hat and Gaussian windows (grad-safe at k=0)."""
    from gravoturb_fdf.theory.projection import gaussian_window, top_hat_window

    assert float(top_hat_window(jnp.array(0.0))) == pytest.approx(1.0, abs=1e-8)
    assert float(gaussian_window(jnp.array(0.0))) == pytest.approx(1.0, abs=1e-12)


def test_smoothed_variance_fraction_limits_and_monotone():
    """sigma_g^2(R)/sigma_g^2(0): ->1 as R->0, decreasing in R."""
    from gravoturb_fdf.theory.projection import smoothed_variance_fraction

    shape, beta = (32, 32, 32), 3.0
    assert float(smoothed_variance_fraction(shape, beta, 1e-3)) == pytest.approx(1.0, abs=0.02)
    fs = np.array([float(smoothed_variance_fraction(shape, beta, R))
                   for R in (0.5, 1.0, 2.0, 4.0, 8.0)])
    assert np.all(np.diff(fs) < 0.0)


def test_smoothed_variance_matches_oracle():
    """Analytic sigma_g^2(R)/sigma_g^2(0) == the ensemble-summed-periodogram ratio
    sum_real sum_k |g_k|^2 W^2 / sum_real sum_k |g_k|^2 (an unbiased estimator of
    sum P W^2 / sum P; avoids the ratio-of-sums bias from few low-k modes that a
    per-realization var(g_R)/var(g) average suffers)."""
    from gravoturb_fdf.theory.projection import (
        _kmag_grid, smoothed_variance_fraction, top_hat_window)
    from gravoturb_fdf.field.field import gaussian_random_field

    shape, beta, R = (40, 40, 40), 3.0, 2.0
    pred = float(smoothed_variance_fraction(shape, beta, R))
    W2 = np.asarray(top_hat_window(jnp.asarray(np.asarray(_kmag_grid(shape)) * R))) ** 2
    key = jax.random.PRNGKey(0)
    x_tot = y_tot = 0.0
    for i in range(16):
        g = np.asarray(gaussian_random_field(shape, beta, jax.random.fold_in(key, i)))
        pk = np.abs(np.fft.fftn(g)) ** 2
        x_tot += float(np.sum(pk * W2))
        y_tot += float(np.sum(pk))
    assert abs(x_tot / y_tot - pred) / pred < 0.05


def test_smoothed_variance_differentiable():
    """sigma_g^2(R) differentiable in beta and R (d/dR < 0)."""
    from gravoturb_fdf.theory.projection import smoothed_variance_fraction

    gb = float(jax.grad(lambda beta: smoothed_variance_fraction((24, 24, 24), beta, 2.0))(3.0))
    gR = float(jax.grad(lambda R: smoothed_variance_fraction((24, 24, 24), 3.0, R))(2.0))
    assert np.isfinite(gb) and np.isfinite(gR) and gR < 0.0


# --- Task 2.3: Limber projection 3D -> 2D --------------------------------------


def test_limber_project_grid_exact_identity():
    """Discrete Limber is exact: the 2D autocovariance of the LOS-projected field equals
    N_los * sum over the LOS axis of the 3D autocovariance (periodic identity)."""
    from gravoturb_fdf.theory.projection import limber_project_grid
    from gravoturb_fdf.validation.measure import autocovariance_3d

    rng = np.random.default_rng(0)
    f = rng.normal(size=(20, 20, 16))
    sigma_col = f.sum(axis=2)  # project along LOS (z)
    xi_col = autocovariance_3d(sigma_col)
    proj = np.asarray(limber_project_grid(jnp.asarray(autocovariance_3d(f)), los_axis=2))
    assert np.allclose(proj, xi_col, rtol=1e-9, atol=1e-12)


def test_limber_project_radial_matches_gaussian_closed_form():
    """w(r_perp) = int xi(sqrt(r_perp^2+l^2)) dl; for xi=exp(-r^2/2sigma^2) the closed
    form is sigma*sqrt(2pi)*exp(-r_perp^2/2sigma^2)."""
    from gravoturb_fdf.theory.projection import limber_project_radial

    sigma = 2.0
    xi = lambda r: jnp.exp(-0.5 * (r / sigma) ** 2)
    r_perp = jnp.array([0.0, 1.0, 2.0, 3.0])
    w = limber_project_radial(xi, r_perp, half_depth=8.0 * sigma, n_nodes=2001)
    expected = sigma * np.sqrt(2 * np.pi) * np.exp(-0.5 * (np.asarray(r_perp) / sigma) ** 2)
    assert np.allclose(np.asarray(w), expected, rtol=1e-4)


def test_limber_project_radial_differentiable():
    """w(r_perp) differentiable in a parameter of xi (finite nonzero grad)."""
    from gravoturb_fdf.theory.projection import limber_project_radial

    def w0(sigma):
        xi = lambda r: jnp.exp(-0.5 * (r / sigma) ** 2)
        return limber_project_radial(xi, jnp.array([1.0]), half_depth=20.0, n_nodes=1001)[0]

    grad = float(jax.grad(w0)(2.0))
    assert np.isfinite(grad) and abs(grad) > 0.0

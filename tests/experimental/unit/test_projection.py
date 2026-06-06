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

"""Unit tests for gravoturb_fdf.theory.cic (Phase 3: counts-in-cells).

CIC counts trace the LINEAR (mean-1) density rho_tilde = rho/<rho> (the simulator places
stars proportional to rho -- a Cox / doubly-stochastic Poisson process). So the CIC
clustering term xi_bar(R) = Var(rho_tilde_cell) is the cell-averaged LINEAR-density 2-point,
NOT the log-density xi_s. The cell scale R regularizes the alpha<=2 fat tail (cell-averaging
IS the smoothing). Route A (Anna 2026-06-05): the moment uses the exact marginal-induced
linear-rho Gaussianization series; the count distribution P(N) (Task 3.3) uses Route B.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.experimental


def test_cic_variance_formula():
    """sigma^2_N = N_bar + N_bar^2 xi_bar: Poisson floor (xi_bar=0 -> var=mean) plus the
    clustering over-dispersion term."""
    from gravoturb_fdf.theory.cic import cic_variance

    assert float(cic_variance(10.0, 0.0)) == pytest.approx(10.0)   # pure Poisson
    assert float(cic_variance(10.0, 0.1)) == pytest.approx(20.0)   # 10 + 100*0.1
    assert float(cic_variance(4.0, 0.25)) == pytest.approx(8.0)    # 4 + 16*0.25


def test_cell_averaged_xi_rho_vs_field_oracle():
    """Route A: the analytic cell-averaged LINEAR-density 2-point xi_bar_rho(R) ==
    ensemble-mean Var of the realized rho_tilde smoothed at scale R (top-hat). Isolates
    the linear-rho Gaussianization series from Poisson shot noise. alpha=2.5 keeps
    <rho^2> finite so the series converges cleanly (alpha<=2 stress + convergence -> AC13)."""
    from gravoturb_fdf.theory.cic import cell_averaged_xi_rho
    from gravoturb_fdf.theory.projection import top_hat_window
    from gravoturb_fdf.field.field import gaussian_random_field
    from gravoturb_fdf.validation.measure import smooth_copula_field, smoothed_linear_variance

    shape, beta, R = (48, 48, 48), 3.0, 2.5
    mach, b, alpha, n_real = 5.0, 0.4, 2.5, 16
    pred = float(cell_averaged_xi_rho(shape, beta, R, mach, b, alpha, n_max=20))

    key = jax.random.PRNGKey(0)
    acc = 0.0
    for i in range(n_real):
        g = gaussian_random_field(shape, beta, jax.random.fold_in(key, i))
        rho = np.exp(np.asarray(smooth_copula_field(g, mach, b, alpha)))
        acc += smoothed_linear_variance(rho, R, top_hat_window)
    oracle = acc / n_real

    assert pred > 0.0
    assert abs(pred - oracle) / oracle < 0.10


def test_cic_moments_differentiable():
    """jax.grad of sigma^2_N(R) wrt (mach, b, alpha, beta) is finite and nonzero -- the
    heavy linear tail must not produce NaN gradients (cf. the Task 1.2 clip)."""
    from gravoturb_fdf.theory.cic import cell_averaged_xi_rho, cic_variance

    shape, R, n_bar = (24, 24, 24), 2.0, 5.0

    def sigma2(params):
        mach, b, alpha, beta = params
        xi_bar = cell_averaged_xi_rho(shape, beta, R, mach, b, alpha, n_max=12)
        return cic_variance(n_bar, xi_bar)

    grad = np.asarray(jax.grad(sigma2)(jnp.array([5.0, 0.4, 2.5, 3.0])))
    assert np.all(np.isfinite(grad))
    assert np.any(np.abs(grad) > 0.0)

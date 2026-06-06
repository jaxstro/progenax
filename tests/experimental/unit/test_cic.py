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


# --- Task 3.2: smoothed density PDF p_R (Route B: reduced-variance BM19) -------


def test_smoothed_log_variance_limits():
    """sigma_s^2(R) (exact smoothed log-density variance, the log-map analog of Route A):
    -> sigma_s_squared(mach,b) as R->0 (cell = point, full variance), strictly decreasing
    in R (more smoothing), and differentiable in (mach,b,alpha,beta)."""
    from gravoturb_fdf.theory.cic import smoothed_log_variance
    from gravoturb_fdf.theory.bm19 import sigma_s_squared

    shape, beta = (32, 32, 32), 3.0
    mach, b, alpha = 5.0, 0.4, 2.5
    full = float(sigma_s_squared(mach, b))
    at0 = float(smoothed_log_variance(shape, beta, 1e-3, mach, b, alpha, n_max=12))
    assert at0 == pytest.approx(full, rel=0.02)  # R->0 recovers the full log-variance

    sigs = np.array([float(smoothed_log_variance(shape, beta, R, mach, b, alpha, n_max=12))
                     for R in (0.5, 1.0, 2.0, 4.0, 8.0)])
    assert np.all(np.diff(sigs) < 0.0)  # decreasing in R

    grad = float(jax.grad(lambda be: smoothed_log_variance(shape, be, 2.0, mach, b, alpha,
                                                           n_max=12))(beta))
    assert np.isfinite(grad) and abs(grad) > 0.0


def test_smoothed_pdf_normalized_and_R0_limit():
    """Route B p_R(s) = reduced-variance BM19 (effective Mach from sigma_s^2(R)): integrates
    to 1, and as R->0 (cell = point) recovers the full unsmoothed BM19 volume PDF."""
    from gravoturb_fdf.theory.cic import smoothed_pdf
    from gravoturb_fdf.theory.pdf import bm19_volume_pdf

    shape, beta = (32, 32, 32), 3.0
    mach, b, alpha = 5.0, 0.4, 2.5
    s = jnp.linspace(-15.0, 40.0, 8000)

    p = smoothed_pdf(s, shape, beta, 2.0, mach, b, alpha, n_max=12)
    assert float(jnp.trapezoid(p, s)) == pytest.approx(1.0, rel=2e-3)

    p0 = smoothed_pdf(s, shape, beta, 1e-3, mach, b, alpha, n_max=12)
    full = bm19_volume_pdf(s, mach, b, alpha)
    assert float(jnp.max(jnp.abs(p0 - full))) < 1e-2 * float(jnp.max(full))


def test_smoothed_pdf_log_variance_matches_in_body_limit():
    """Large alpha (negligible tail): the log-variance of p_R equals sigma_s^2(R), so the
    reduced-variance construction is faithful (Var of s under p_R == the smoothed log-var)."""
    from gravoturb_fdf.theory.cic import smoothed_log_variance, smoothed_pdf

    shape, beta = (32, 32, 32), 3.0
    mach, b, alpha, R = 5.0, 0.4, 8.0, 2.0  # alpha=8 -> tail negligible, ~ pure lognormal
    s = jnp.linspace(-15.0, 15.0, 8000)
    p = smoothed_pdf(s, shape, beta, R, mach, b, alpha, n_max=12)
    mean = float(jnp.trapezoid(s * p, s))
    var = float(jnp.trapezoid((s - mean) ** 2 * p, s))
    sig2R = float(smoothed_log_variance(shape, beta, R, mach, b, alpha, n_max=12))
    assert var == pytest.approx(sig2R, rel=0.05)


# --- Task 3.3: compound-Poisson count distribution P(N) -----------------------


def test_pN_normalizes_and_mean_matches():
    """P(N) = int Poisson(N | N_bar e^s/mu) p_R(s) ds: sum_N P(N) = 1 and sum_N N P(N) = N_bar
    (the compound-Poisson identities). N range wide enough to capture the over-dispersed tail."""
    from gravoturb_fdf.theory.cic import count_distribution

    N = jnp.arange(0, 250)
    n_bar = 8.0
    pN = count_distribution(N, n_bar, (24, 24, 24), 3.0, 2.0, 5.0, 0.4, 3.0, n_max=12)
    assert float(jnp.sum(pN)) == pytest.approx(1.0, rel=2e-3)
    assert float(jnp.sum(N * pN)) == pytest.approx(n_bar, rel=2e-2)


def test_pN_overdispersed():
    """Clustering makes counts over-dispersed relative to pure Poisson: Var(N) > N_bar.
    Heavy regime (R=2, alpha=3): the density fluctuations broaden the count distribution."""
    from gravoturb_fdf.theory.cic import count_distribution

    N = jnp.arange(0, 400)
    n_bar = 10.0
    pN = count_distribution(N, n_bar, (24, 24, 24), 3.0, 2.0, 6.0, 0.5, 3.0, n_max=12)
    mean = float(jnp.sum(N * pN))
    var = float(jnp.sum(N**2 * pN) - mean**2)
    assert var > 1.3 * n_bar  # clearly over-dispersed (Poisson would give var == mean)


def test_pN_compound_poisson_moment_identity():
    """Var(N) == N_bar + N_bar^2 Var_{p_R}(rho_tilde), the exact compound-Poisson identity.
    Validated in a light regime (R=4, alpha=4) where the lambda^2 tail is captured by the N
    range (in heavy regimes the identity still holds but needs a far larger N_max to sum the
    rare high-density tail -- see the AC13 convergence note)."""
    from gravoturb_fdf.theory.cic import count_distribution, smoothed_pdf

    N = jnp.arange(0, 300)
    n_bar = 8.0
    shape, beta, R, mach, b, alpha = (24, 24, 24), 3.0, 4.0, 5.0, 0.4, 4.0
    pN = count_distribution(N, n_bar, shape, beta, R, mach, b, alpha, n_max=12)
    mean = float(jnp.sum(N * pN))
    var = float(jnp.sum(N**2 * pN) - mean**2)

    s = jnp.linspace(-15.0, 30.0, 1024)  # count_distribution's own grid (shared -> exact)
    p = smoothed_pdf(s, shape, beta, R, mach, b, alpha, n_max=12)
    p = p / jnp.trapezoid(p, s)
    mu = float(jnp.trapezoid(jnp.exp(s) * p, s))
    var_rho = float(jnp.trapezoid((jnp.exp(s) / mu - 1.0) ** 2 * p, s))
    assert var == pytest.approx(n_bar + n_bar**2 * var_rho, rel=0.02)


def test_sample_cic_counts_clean_poisson():
    """sample_cic_counts is a true inhomogeneous-Poisson CIC: count_cell ~ Poisson(n_bar*rho_cell).
    Mean ~ n_bar and Var(N) ~ N_bar + N_bar^2 Var(rho_cell) (the clean Cox relation, no fine-cell
    pile-up artifact) -- so it matches the count_distribution model and is resolution-independent."""
    from gravoturb_fdf.field.sampling import sample_cic_counts
    from gravoturb_fdf.field.field import gaussian_random_field
    from gravoturb_fdf.validation.measure import smooth_copula_field

    shape, c, n_bar = (24, 24, 24), 4, 30.0
    M = shape[0] // c
    g = gaussian_random_field(shape, 3.0, jax.random.PRNGKey(0))
    s = jnp.asarray(smooth_copula_field(g, 5.0, 0.4, 2.5))
    counts = sample_cic_counts(s, n_bar, c, jax.random.PRNGKey(1))
    assert counts.shape == (M, M, M)
    assert float(counts.mean()) == pytest.approx(n_bar, rel=0.12)

    rho_cell = (jnp.exp(s) / jnp.mean(jnp.exp(s))).reshape(
        M, c, M, c, M, c).mean(axis=(1, 3, 5))
    expected_var = n_bar + n_bar**2 * float(jnp.var(rho_cell))
    assert float(jnp.var(counts)) == pytest.approx(expected_var, rel=0.25)


def test_pN_differentiable():
    """jax.grad of a functional of P(N) wrt (mach,b,alpha,beta) is finite and nonzero."""
    from gravoturb_fdf.theory.cic import count_distribution

    N = jnp.arange(0, 200)

    def tail_mass(params):
        mach, b, alpha, beta = params
        pN = count_distribution(N, 10.0, (24, 24, 24), beta, 2.0, mach, b, alpha, n_max=10)
        return jnp.sum(jnp.where(N > 25, pN, 0.0))  # P(N>25): the over-dense cells

    grad = np.asarray(jax.grad(tail_mass)(jnp.array([6.0, 0.5, 3.0, 3.0])))
    assert np.all(np.isfinite(grad))
    assert np.any(np.abs(grad) > 0.0)

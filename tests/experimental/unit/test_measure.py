"""Unit tests for gravoturb.diagnostics.measure (the oracle-measurement side).

These utilities measure 2-point statistics from realization fields and build the
theory-consistent ``smooth_copula_field`` (s = log_density_icdf(Phi(g_hat)) - shift on an
EXACTLY unit-variance g_hat) used as the clean oracle for the Gaussianization series
(AC11). numpy/scipy are permitted here (validation path, non-differentiable).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.experimental


def test_autocovariance_3d_recovers_cosine_mode():
    """xi(r) of a single cosine mode = 0.5 cos(2 pi k0 dx / N) (Wiener-Khinchin)."""
    from gravoturb.diagnostics.measure import autocovariance_3d

    n, k0 = 16, 2
    i = np.arange(n)
    f = np.cos(2 * np.pi * k0 * i / n)[:, None, None] * np.ones((n, n, n))
    xi = autocovariance_3d(f)
    dx = np.arange(n)
    expected = 0.5 * np.cos(2 * np.pi * k0 * dx / n)
    assert np.allclose(xi[:, 0, 0], expected, atol=1e-10)


def test_autocovariance_3d_zero_lag_is_variance():
    """xi(0) = Var(field)."""
    from gravoturb.diagnostics.measure import autocovariance_3d

    rng = np.random.default_rng(0)
    f = rng.normal(size=(24, 24, 24))
    xi = autocovariance_3d(f)
    assert xi[0, 0, 0] == pytest.approx(float(np.var(f)), rel=1e-10)


def test_measured_2pt_white_noise_decorrelates():
    """White noise: rho(r>0) ~ 0 (within sampling noise); variance recovered."""
    from gravoturb.diagnostics.measure import gaussian_correlation_measured

    rng = np.random.default_rng(1)
    g = rng.normal(size=(32, 32, 32))
    r, rho = gaussian_correlation_measured(g, n_bins=12)
    assert np.all(r > 0.0)  # zero-lag excluded
    assert np.max(np.abs(rho)) < 0.05  # no spurious correlation


def test_smooth_copula_field_marginal_lognormal_limit():
    """smooth_copula_field marginal: large alpha => Var(s) ~ sigma_s^2."""
    from gravoturb.diagnostics.measure import smooth_copula_field
    from gravoturb.theory.density_pdf import sigma_s_squared

    mach, b, alpha = 5.0, 0.4, 6.0
    sig2 = float(sigma_s_squared(mach, b))
    key = jax.random.PRNGKey(3)
    g = jax.random.normal(key, (48, 48, 48))
    s = smooth_copula_field(g, mach, b, alpha)
    assert float(np.var(s)) == pytest.approx(sig2, rel=0.03)


def test_smooth_copula_field_mean_density_unity():
    """smooth_copula_field marginal: <e^s> ~ 1 (alpha=3, finite 2nd moment)."""
    from gravoturb.diagnostics.measure import smooth_copula_field

    key = jax.random.PRNGKey(4)
    g = jax.random.normal(key, (48, 48, 48))
    s = smooth_copula_field(g, 5.0, 0.4, 3.0)
    assert float(np.mean(np.exp(s))) == pytest.approx(1.0, rel=3e-2)


# --- Task 4: measure_exceedances (mock -> POT exceedance histogram for the alpha block) -------


def test_measure_exceedances_counts_edges_and_alpha_recovery():
    """measure_exceedances reduces a gas log-density field to (counts, edges, s_max, n_tail) above
    s_thr. Mechanics: edges span [s_thr, s_max] closed at the realized max; counts sum to n_tail =
    #(s>s_thr). Physics: on a pure-exponential tail above s_thr, maximizing tail_exceedance_loglike
    over alpha recovers alpha_true to within ~3 sigma (sigma = alpha/sqrt(N_tail))."""
    from gravoturb.diagnostics.measure import measure_exceedances
    from gravoturb.inference.likelihood import tail_exceedance_loglike

    rng = np.random.default_rng(0)
    alpha_true, s_thr, n_tail_draw = 2.5, 1.0, 40000
    body = rng.uniform(-5.0, 0.99, size=30000)  # all below s_thr
    tail = s_thr + rng.exponential(1.0 / alpha_true, size=n_tail_draw)
    s_field = np.concatenate([body, tail])

    counts, edges, s_max, n_tail = measure_exceedances(s_field, s_thr, n_bins=30)
    # --- mechanics ---
    assert n_tail == int((s_field > s_thr).sum()) == n_tail_draw
    assert edges[0] == pytest.approx(s_thr) and edges[-1] == pytest.approx(s_max)
    assert s_max == pytest.approx(float(s_field.max()))
    assert int(round(float(counts.sum()))) == n_tail  # every exceedance binned

    # --- physics: 1-D MLE over alpha recovers the tail slope ---
    alphas = np.linspace(1.5, 4.0, 501)
    theta = lambda a: jnp.array([5.0, 0.4, a, 3.0])
    lls = np.array(
        [
            float(
                tail_exceedance_loglike(
                    jnp.asarray(counts), jnp.asarray(edges), theta(a), s_thr, s_max
                )
            )
            for a in alphas
        ]
    )
    alpha_hat = alphas[int(np.argmax(lls))]
    sigma = alpha_true / np.sqrt(n_tail)
    assert abs(alpha_hat - alpha_true) < 3.0 * sigma


# --- Task 3: measure_log_count_variance (data-side log_plus variance, Neyrinck+2011 Eq 2) -----


def test_measure_log_count_variance_matches_log_plus():
    from gravoturb.diagnostics.measure import measure_log_count_variance

    rng = np.random.default_rng(0)
    n_bar = 5.0
    counts = rng.poisson(n_bar, size=(16, 16, 16))
    v = measure_log_count_variance(counts, n_bar)
    # reference: same Neyrinck Eq 2 transform, numpy
    d = counts / n_bar - 1.0
    A = np.where(d > 0.0, np.log1p(np.where(d > 0.0, d, 0.0)), d)
    assert abs(v - float(np.var(A))) < 1e-12
    assert v >= 0.0


# --- Task 6: estimate_log_count_variance_var (fixed fiducial var_v for the count block) -------


def test_log_count_variance_estimator_var_positive():
    """The fiducial estimator variance of measure_log_count_variance over an n_real mock ensemble
    (used as the fixed var_v in log_count_variance_loglike) is a finite, strictly positive number."""
    from gravoturb.diagnostics.measure import estimate_log_count_variance_var

    vv = estimate_log_count_variance_var(
        mach=8.0,
        b=0.4,
        alpha=2.5,
        beta=3.0,
        shape=(24, 24, 24),
        cell_size=4,
        n_bar=5.0,
        n_real=8,
        key=jax.random.PRNGKey(0),
    )
    assert vv > 0.0 and vv == vv


# --- Task A2: 2D measurement helpers (project_counts_los, angular band-powers) -----------------


def test_project_counts_los_sums_slices():
    import numpy as np
    from gravoturb.diagnostics.measure import project_counts_los

    c = np.ones((8, 8, 8))
    assert np.allclose(project_counts_los(c, depth=8), 8.0)
    assert np.allclose(project_counts_los(c, depth=3), 3.0)


def test_measure_angular_bandpowers_2d_shape_and_positive():
    import numpy as np
    from gravoturb.diagnostics.measure import measure_angular_bandpowers_2d

    rng = np.random.default_rng(0)
    bp = measure_angular_bandpowers_2d(
        rng.normal(size=(32, 32)), np.linspace(1.0, 8.0, 4)
    )
    assert bp.shape == (3,) and np.all(bp >= 0.0)


def test_measure_log_count_variance_is_shape_agnostic_2d():
    """measure_log_count_variance is shape-agnostic: it runs on a 2D (projected) count map and
    returns a finite, non-negative float. The 2D inference path relies on this same statistic."""
    import numpy as np
    from gravoturb.diagnostics.measure import measure_log_count_variance

    rng = np.random.default_rng(0)
    n_bar = 5.0
    counts2d = rng.poisson(n_bar, size=(16, 16))
    v = measure_log_count_variance(counts2d, n_bar)
    assert isinstance(v, float) and np.isfinite(v) and v >= 0.0


def test_detrended_log_count_variance_reduces_and_detrends():
    """Phase 5 / A1: measure_log_count_variance_detrended(counts, n_bar_cells)
    (i) reduces EXACTLY to the homogeneous statistic for uniform n_bar_cells;
    (ii) on a PURE-envelope Poisson catalog (no turbulence) removes the envelope
    inflation (statistic ~ shot level, matching a homogeneous pure-Poisson reference),
    where the homogeneous statistic is visibly inflated."""
    import jax
    import jax.numpy as jnp
    from gravoturb.diagnostics.measure import (
        measure_log_count_variance,
        measure_log_count_variance_detrended,
    )
    from gravoturb.realization.envelope import apply_spherical_envelope
    from gravoturb.realization.placement import sample_cic_counts

    from progenax import PlummerProfile

    shape, box, cell, n_stars = (32,) * 3, 4.0, 4, 20000.0
    nb = n_stars / (shape[0] // cell) ** 3

    # (i) exact reduction
    rng = np.random.default_rng(0)
    cnt = rng.poisson(nb, size=(8, 8, 8)).astype(float)
    hom = measure_log_count_variance(cnt, nb)
    det = measure_log_count_variance_detrended(cnt, np.full((8, 8, 8), nb))
    assert det == pytest.approx(hom, rel=1e-12)

    # (ii) the physically meaningful transfer statement: the TURBULENCE EXCESS
    # (statistic minus its matched pure-shot baseline, both masked+detrended) on an
    # ENVELOPED catalog matches the same excess measured on the periodic box —
    # the envelope window is fully absorbed by (known-intensity detrend + declared
    # effective-volume mask n_bar_i >= n_min).
    from gravoturb.diagnostics.measure import envelope_cell_intensity
    from gravoturb.realization.pipeline import build_turbulent_field

    prof = PlummerProfile(r_h=0.5)
    n_cells = envelope_cell_intensity(prof, box, shape, cell, nb)
    excess_env, excess_per = [], []
    for sd in range(6):
        fld = build_turbulent_field(8.0, 0.5, 1.8, 3.0, shape, jax.random.PRNGKey(sd))
        s_env = apply_spherical_envelope(fld.s, prof, box)
        s_env = s_env - jnp.log(jnp.mean(jnp.exp(s_env)))
        s_env0 = apply_spherical_envelope(jnp.zeros(shape), prof, box)
        s_env0 = s_env0 - jnp.log(jnp.mean(jnp.exp(s_env0)))
        k = jax.random.PRNGKey(sd)
        cnt_te = np.asarray(sample_cic_counts(s_env, nb, cell, jax.random.fold_in(k, 1)))
        cnt_0e = np.asarray(sample_cic_counts(s_env0, nb, cell, jax.random.fold_in(k, 2)))
        v_te = measure_log_count_variance_detrended(cnt_te, n_cells)
        v_0e = measure_log_count_variance_detrended(cnt_0e, n_cells)
        excess_env.append(v_te - v_0e)
        cnt_tp = np.asarray(sample_cic_counts(fld.s, nb, cell, jax.random.fold_in(k, 3)))
        cnt_0p = np.random.default_rng(100 + sd).poisson(nb, size=(8, 8, 8)).astype(float)
        v_tp = measure_log_count_variance(cnt_tp, nb)
        v_0p = measure_log_count_variance(cnt_0p, nb)
        excess_per.append(v_tp - v_0p)
    d = np.mean(excess_env) - np.mean(excess_per)
    scatter = max(np.std(excess_env), np.std(excess_per)) / np.sqrt(6)
    assert np.mean(excess_per) > 0.2                     # turbulence signal present
    assert abs(d) < 4.0 * scatter + 0.05                 # transfer holds within scatter

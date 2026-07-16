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
    from gravoturb.inference.covariance import power_spectrum_bandpowers

    shape, k_edges = (32, 32, 32), jnp.linspace(1.0, 13.0, 7)
    kc, P, nmodes = power_spectrum_bandpowers(
        shape, 3.0, 5.0, 0.4, 2.5, k_edges, n_max=12
    )
    assert P.shape == kc.shape and kc.shape[0] == 6
    assert jnp.all(P > 0) and jnp.all(nmodes > 0)

    for j, name in enumerate(["beta", "mach", "b", "alpha"]):
        args = [3.0, 5.0, 0.4, 2.5]
        g = jax.grad(
            lambda v, j=j: jnp.sum(
                power_spectrum_bandpowers(
                    shape,
                    *[v if i == j else args[i] for i in range(4)],
                    k_edges,
                    n_max=12,
                )[1]
            )
        )(args[j])
        assert np.isfinite(float(g))


def test_bandpowers_match_mock():
    """Analytic band-powers == ensemble-mean measured periodogram band-powers of the
    smooth-copula log-density field (forward fidelity of P_s(k), the Fourier dual of AC11)."""
    from gravoturb.diagnostics.measure import smooth_copula_field
    from gravoturb.inference.covariance import (
        measured_bandpowers,
        power_spectrum_bandpowers,
    )
    from gravoturb.realization.gaussian_field import gaussian_random_field

    shape, beta, mach, b, alpha = (32, 32, 32), 3.0, 5.0, 0.4, 2.5
    k_edges = jnp.linspace(1.0, 13.0, 7)
    _, P_an, _ = power_spectrum_bandpowers(
        shape, beta, mach, b, alpha, k_edges, n_max=14
    )

    key = jax.random.PRNGKey(0)
    acc = np.zeros(len(k_edges) - 1)
    n_real = 24
    for i in range(n_real):
        s = np.asarray(
            smooth_copula_field(
                gaussian_random_field(shape, beta, jax.random.fold_in(key, i)),
                mach,
                b,
                alpha,
            )
        )
        acc += measured_bandpowers(s, shape, k_edges)
    P_mock = acc / n_real
    rel = np.abs(np.asarray(P_an) - P_mock) / np.abs(P_mock)
    assert np.median(rel) < 0.05


def test_mock_covariance_psd_and_hartlap():
    """mock_covariance is symmetric positive-definite and recovers a known covariance;
    the Hartlap inverse-debias factor is in (0,1) and -> 1 as n_real -> inf."""
    from gravoturb.inference.covariance import (
        hartlap_factor,
        mock_covariance,
        mock_precision,
    )

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
    from gravoturb.diagnostics.measure import smooth_copula_field
    from gravoturb.inference.covariance import (
        gaussian_bandpower_covariance,
        measured_bandpowers,
        mock_covariance,
        power_spectrum_bandpowers,
    )
    from gravoturb.realization.gaussian_field import gaussian_random_field

    shape, beta, mach, b, alpha = (32, 32, 32), 3.0, 5.0, 0.4, 2.5
    k_edges = jnp.linspace(2.0, 14.0, 7)
    _, P, nmodes = power_spectrum_bandpowers(
        shape, beta, mach, b, alpha, k_edges, n_max=14
    )
    gauss_diag = np.diag(np.asarray(gaussian_bandpower_covariance(P, nmodes)))

    key = jax.random.PRNGKey(0)
    rows = [
        measured_bandpowers(
            np.asarray(
                smooth_copula_field(
                    gaussian_random_field(shape, beta, jax.random.fold_in(key, i)),
                    mach,
                    b,
                    alpha,
                )
            ),
            shape,
            k_edges,
        )
        for i in range(80)
    ]
    mock_diag = np.diag(mock_covariance(rows))
    ratio = mock_diag / gauss_diag
    assert np.all(ratio > 1.3)  # Gaussian C is an underestimate everywhere
    assert ratio[-1] > 2.0 * ratio[0]  # excess grows toward small scales (high k)


# --- Task 5.2: Gaussian likelihood on the data vector --------------------------

_CFG = dict(
    shape=(24, 24, 24),
    k_edges=jnp.linspace(2.0, 11.0, 5),
    cell_sizes=(4,),
    n_bar=30.0,
    n_max=12,
)
_THETA = jnp.array([5.0, 0.4, 2.5, 3.0])  # (mach, b, alpha, beta)


def test_data_vector_shape_and_differentiable():
    """d(theta) = [P_s(k_i) band-powers, sigma^2_N(c)] is finite, correctly shaped, and
    differentiable in theta=(mach,b,alpha,beta)."""
    from gravoturb.inference.likelihood import data_vector

    d = data_vector(_THETA, **_CFG)
    assert d.shape == (4 + 1,)  # 4 band-power bins + 1 CIC variance
    assert jnp.all(jnp.isfinite(d))
    g = np.asarray(jax.grad(lambda th: jnp.sum(data_vector(th, **_CFG)))(_THETA))
    assert np.all(np.isfinite(g)) and np.any(np.abs(g) > 0)


def test_gaussian_loglike_max_at_truth_and_differentiable():
    """gaussian_loglike on NOISELESS data (= d(theta_true)) peaks at theta_true (zero residual)
    and is a finite, differentiable scalar."""
    from gravoturb.inference.likelihood import data_vector, gaussian_loglike

    data = data_vector(_THETA, **_CFG)
    precision = jnp.diag(1.0 / data**2)  # any PD precision -> max at truth
    ll0 = float(gaussian_loglike(data, _THETA, precision, **_CFG))
    assert ll0 == pytest.approx(0.0, abs=1e-8)  # zero residual at truth

    for j, dth in enumerate([0.5, 0.05, 0.3, 0.3]):
        pert = _THETA.at[j].add(dth)
        assert float(gaussian_loglike(data, pert, precision, **_CFG)) < ll0 - 1e-6

    g = np.asarray(
        jax.grad(lambda th: gaussian_loglike(data, th, precision, **_CFG))(_THETA)
    )
    assert np.all(np.isfinite(g))


# --- Task 5.3: Fisher matrix + forecast ---------------------------------------


def test_fisher_3param_pd_and_errors_finite():
    """With b fixed, the free params (mach, alpha, beta) are identifiable: F = J^T Cinv J is
    symmetric positive-definite and the marginal errors sigma(theta_i) are finite and positive."""
    from gravoturb.inference.fisher import fisher_matrix, marginal_errors
    from gravoturb.inference.likelihood import data_vector

    data = data_vector(_THETA, **_CFG)
    prec = jnp.diag(1.0 / data**2)
    F = fisher_matrix(_THETA, prec, free=(0, 2, 3), **_CFG)
    assert F.shape == (3, 3)
    assert jnp.allclose(F, F.T)
    assert jnp.all(jnp.linalg.eigvalsh(F) > 0)  # identifiable
    sig = marginal_errors(F)
    assert jnp.all(jnp.isfinite(sig)) and jnp.all(sig > 0)


def test_fisher_full_4param_singular_mach_b_degeneracy():
    """The predicted statistics depend on (mach,b) ONLY via sigma_s^2 = ln(1+(b*mach)^2), so the
    full 4-param Fisher is rank-3 singular -- value the data can't break the mach-b degeneracy
    (need fixed b). A genuine model property, asserted here as a guard."""
    from gravoturb.inference.fisher import fisher_matrix
    from gravoturb.inference.likelihood import data_vector

    data = data_vector(_THETA, **_CFG)
    prec = jnp.diag(1.0 / data**2)
    F = fisher_matrix(_THETA, prec, free=(0, 1, 2, 3), **_CFG)
    ev = jnp.linalg.eigvalsh(F)
    assert float(ev[0] / ev[-1]) < 1e-8  # one ~zero eigenvalue = the mach-b degeneracy


def test_fisher_errors_shrink_as_sqrt_volume():
    """sigma(theta_i) ~ 1/sqrt(V_survey): scaling the precision by n_boxes (independent survey
    volumes add Fisher information) shrinks the errors by 1/sqrt(n_boxes)."""
    from gravoturb.inference.fisher import fisher_matrix, marginal_errors
    from gravoturb.inference.likelihood import data_vector

    data = data_vector(_THETA, **_CFG)
    prec = jnp.diag(1.0 / data**2)
    s1 = marginal_errors(fisher_matrix(_THETA, prec, free=(0, 2, 3), **_CFG))
    s4 = marginal_errors(fisher_matrix(_THETA, 4.0 * prec, free=(0, 2, 3), **_CFG))
    assert jnp.allclose(s4, s1 / 2.0, rtol=1e-5)


# --- Task 6.1: compound-Poisson count (1-pt) log-likelihood --------------------

_CCFG = dict(shape=(24, 24, 24), cell_size=4, n_bar=30.0, n_max=12)


def test_count_loglike_max_at_truth_and_differentiable():
    """1-pt count log-likelihood sum_N hist[N] log P(N|theta). On the noiseless expected
    histogram hist = n_cells * P(N|theta_true) it is maximal at theta_true (Gibbs' inequality:
    cross-entropy is minimised when the model PDF matches), and differentiable in theta."""
    from gravoturb.inference.likelihood import count_loglike
    from gravoturb.theory.counts_in_cells import count_distribution
    from gravoturb.theory.projection import box_window_sq_grid

    N = jnp.arange(0, 250)
    w2 = box_window_sq_grid(_CCFG["shape"], _CCFG["cell_size"])
    pN_true = count_distribution(
        N,
        _CCFG["n_bar"],
        _CCFG["shape"],
        3.0,
        float(_CCFG["cell_size"]),
        5.0,
        0.4,
        2.5,
        n_max=_CCFG["n_max"],
        w2=w2,
    )
    n_cells = (_CCFG["shape"][0] // _CCFG["cell_size"]) ** 3
    hist = np.asarray(pN_true) * n_cells  # noiseless expected histogram

    ll0 = float(count_loglike(hist, _THETA, **_CCFG))
    for j, dth in enumerate([0.5, 0.05, 0.3, 0.3]):
        assert float(count_loglike(hist, _THETA.at[j].add(dth), **_CCFG)) < ll0 - 1e-6

    g = np.asarray(jax.grad(lambda th: count_loglike(hist, th, **_CCFG))(_THETA))
    assert np.all(np.isfinite(g))


def test_count_loglike_constrains_alpha_strongly():
    """The COUNT distribution's high-N tail pins alpha (the PDF-tail slope) -- the curvature of
    the count log-likelihood in alpha is sharp, the M1-vs-M2 improvement that rescues alpha."""
    from gravoturb.inference.likelihood import count_loglike
    from gravoturb.theory.counts_in_cells import count_distribution
    from gravoturb.theory.projection import box_window_sq_grid

    N = jnp.arange(0, 250)
    w2 = box_window_sq_grid(_CCFG["shape"], _CCFG["cell_size"])
    pN_true = count_distribution(
        N,
        _CCFG["n_bar"],
        _CCFG["shape"],
        3.0,
        4.0,
        5.0,
        0.4,
        2.5,
        n_max=_CCFG["n_max"],
        w2=w2,
    )
    n_cells = (_CCFG["shape"][0] // _CCFG["cell_size"]) ** 3
    hist = np.asarray(pN_true) * n_cells

    # second derivative in alpha = Fisher curvature; must be clearly negative (a real constraint)
    d2 = float(
        jax.grad(jax.grad(lambda a: count_loglike(hist, _THETA.at[2].set(a), **_CCFG)))(
            2.5
        )
    )
    assert d2 < -1.0  # sharply peaked in alpha


# --- Task 5: tail-robust log-count-variance Gaussian block --------------------


def test_log_count_variance_loglike_peaks_at_truth():
    """The Gaussian sigma_s^2 -> mach block: on noiseless data (measured = prediction at the
    injected theta) the loglike peaks at the injected mach (ll(8) > ll(5) and ll(8) > ll(12)),
    and is differentiable in mach (finite grad)."""
    from gravoturb.inference.likelihood import log_count_variance_loglike
    from gravoturb.theory.counts_in_cells import predict_log_count_variance
    from gravoturb.theory.projection import box_window_sq_grid

    shape, c, n_bar = (24, 24, 24), 4, 5.0
    w2 = box_window_sq_grid(shape, c)
    # noiseless "measured" = prediction at truth
    meas = float(
        predict_log_count_variance(n_bar, shape, 3.0, float(c), 8.0, 0.4, 2.5, w2=w2)
    )
    ll = lambda m: float(
        log_count_variance_loglike(
            meas, jnp.array([m, 0.4, 2.5, 3.0]), shape, c, n_bar, var_v=1e-3
        )
    )
    assert ll(8.0) > ll(5.0) and ll(8.0) > ll(12.0)  # peaks at the injected mach
    g = float(
        jax.grad(
            lambda m: log_count_variance_loglike(
                meas, jnp.array([m, 0.4, 2.5, 3.0]), shape, c, n_bar, var_v=1e-3
            )
        )(8.0)
    )
    assert g == g


# --- Task 6.2: blackjax NUTS driver -------------------------------------------


def test_run_nuts_recovers_gaussian():
    """The thin blackjax NUTS wrapper recovers a known 2-D Gaussian target (mean + std) --
    validates window-adaptation + sampling independent of our likelihood."""
    from gravoturb.inference.hmc import run_nuts

    mu = jnp.array([1.0, -2.0])
    logdensity = lambda x: -0.5 * jnp.sum((x - mu) ** 2)
    samples = run_nuts(
        logdensity, jnp.zeros(2), jax.random.PRNGKey(0), n_warmup=300, n_samples=1500
    )
    assert samples.shape == (1500, 2)
    assert np.allclose(np.asarray(samples.mean(0)), np.asarray(mu), atol=0.15)
    assert np.allclose(np.asarray(samples.std(0)), 1.0, atol=0.2)


def test_run_nuts_diagnostic_shapes_and_recovers_gaussian():
    import blackjax  # noqa
    from gravoturb.inference.hmc import run_nuts_diagnostic

    # 2-D unit Gaussian target
    logdensity = lambda x: -0.5 * jnp.sum(x**2)
    out = run_nuts_diagnostic(
        logdensity,
        jnp.zeros(2),
        jax.random.PRNGKey(0),
        n_warmup=300,
        n_samples=400,
        n_chains=4,
    )
    assert out["positions"].shape == (4, 400, 2)  # (chains, samples, dim)
    assert out["divergences"].shape == (4, 400)  # bool per step
    assert out["tree_depth"].shape == (4, 400)
    assert out["energy"].shape == (4, 400)
    m = out["positions"].reshape(-1, 2).mean(0)
    s = out["positions"].reshape(-1, 2).std(0)
    assert jnp.all(jnp.abs(m) < 0.15) and jnp.all(jnp.abs(s - 1.0) < 0.15)
    assert out["divergences"].mean() < 0.02  # healthy target -> few divergences


def test_density_pdf_loglike_constrains_alpha():
    """1-pt density-PDF block: sum_bins hist[s] log p_BM19(s;M,b,alpha) -- the faithful,
    alpha-sensitive observable (a gas-density tracer; stars don't carry alpha). Maximal at
    theta_true on the noiseless expected histogram, differentiable, and SHARPLY peaked in
    alpha (d2/dalpha2 << 0) -- this is the block that recovers the PDF-tail slope alpha."""
    from gravoturb.inference.likelihood import density_pdf_loglike
    from gravoturb.theory.density_cdf import log_density_pdf

    s = jnp.linspace(-8.0, 25.0, 400)
    theta = jnp.array([5.0, 0.4, 2.5, 3.0])
    p_true = log_density_pdf(s, 5.0, 0.4, 2.5)
    hist = (
        np.asarray(p_true / jnp.trapezoid(p_true, s)) * 1e4
    )  # noiseless expected histogram

    ll0 = float(density_pdf_loglike(hist, s, theta))
    for j, dth in [(0, 0.5), (2, 0.3)]:  # mach, alpha
        assert float(density_pdf_loglike(hist, s, theta.at[j].add(dth))) < ll0 - 1e-6
    g = np.asarray(jax.grad(lambda th: density_pdf_loglike(hist, s, th))(theta))
    assert np.all(np.isfinite(g))
    d2 = float(
        jax.grad(jax.grad(lambda a: density_pdf_loglike(hist, s, theta.at[2].set(a))))(
            2.5
        )
    )
    assert d2 < -10.0  # sharply constrains alpha (the PDF-tail slope)


def test_bounded_transforms_roundtrip_and_jacobian():
    """The bounded->unconstrained reparametrization (mach>0, alpha>1, beta>0) round-trips and
    its log-Jacobian is finite (needed so HMC samples in unconstrained space)."""
    from gravoturb.inference.hmc import (
        log_jacobian,
        to_constrained,
        to_unconstrained,
    )

    theta_c = jnp.array([5.0, 2.5, 3.0])  # (mach, alpha, beta), all in-bounds
    z = to_unconstrained(theta_c)
    assert np.allclose(np.asarray(to_constrained(z)), np.asarray(theta_c), atol=1e-10)
    assert np.isfinite(float(log_jacobian(z)))


# --- Task 1 (Phase 6 finish): POT truncated-exponential tail-exceedance log-likelihood ---
#
# alpha (the BM19 density-PDF tail slope) is recovered from the gas-density exceedances above a
# fixed threshold s_thr: above s_t the BM19 tail C e^{-alpha s} is memoryless, so the exceedances
# x = s - s_thr in [0, L], L = s_max - s_thr, are TRUNCATED-EXPONENTIAL rate alpha. This decouples
# alpha from sigma_s^2(mach,b) exactly (the lognormal norm cancels) and is the principled fix for
# the finite-field tail truncation that biased the old full-PDF fit high (~2.8 vs truth 2.5).


def _trunc_exp_binprob(edges, s_thr, s_max, alpha):
    """Reference truncated-exponential per-bin probability (numpy), edges in s-space."""
    x = np.asarray(edges) - s_thr
    L = s_max - s_thr
    cdf = (1.0 - np.exp(-alpha * x)) / (1.0 - np.exp(-alpha * L))
    return np.diff(cdf)


def test_tail_exceedance_loglike_max_at_truth():
    """On the noiseless expected exceedance histogram at alpha_true, the POT truncated-exponential
    likelihood is maximal at alpha_true (Gibbs' inequality), strictly above alpha_true +/- 0.3.
    s_thr, s_max are data-derived constants; only theta[2]=alpha is fit."""
    from gravoturb.inference.likelihood import tail_exceedance_loglike

    s_thr, s_max, alpha_true, n_tail = 0.0, 3.0, 2.5, 2000.0
    edges = jnp.linspace(s_thr, s_max, 21)
    counts = jnp.asarray(_trunc_exp_binprob(edges, s_thr, s_max, alpha_true) * n_tail)
    theta = lambda a: jnp.array([5.0, 0.4, a, 3.0])

    ll0 = float(tail_exceedance_loglike(counts, edges, theta(alpha_true), s_thr, s_max))
    for da in (+0.3, -0.3):
        llp = float(
            tail_exceedance_loglike(counts, edges, theta(alpha_true + da), s_thr, s_max)
        )
        assert llp < ll0 - 1e-6


def test_tail_exceedance_loglike_constrains_alpha_sharply():
    """d2(ll)/dalpha2 << 0 at truth: the exceedance slope sharply pins alpha (= -N_tail*I(alpha))."""
    from gravoturb.inference.likelihood import tail_exceedance_loglike

    s_thr, s_max, alpha_true, n_tail = 0.0, 3.0, 2.5, 2000.0
    edges = jnp.linspace(s_thr, s_max, 21)
    counts = jnp.asarray(_trunc_exp_binprob(edges, s_thr, s_max, alpha_true) * n_tail)

    d2 = float(
        jax.grad(
            jax.grad(
                lambda a: tail_exceedance_loglike(
                    counts, edges, jnp.array([5.0, 0.4, a, 3.0]), s_thr, s_max
                )
            )
        )(alpha_true)
    )
    assert d2 < -10.0


def test_tail_exceedance_loglike_grad_finite_incl_small_alphaL():
    """grad in alpha is finite for alpha in [1.1, 6] AND in the alpha*L -> 0 limit (L = 1e-3) --
    guards the stable -expm1 normalizer against 0/0."""
    from gravoturb.inference.likelihood import tail_exceedance_loglike

    s_thr = 0.0
    for s_max in (3.0, 1e-3):
        edges = jnp.linspace(s_thr, s_max, 11)
        counts = jnp.ones(10) * 50.0  # every bin populated
        for a in (1.1, 2.5, 6.0):
            g = float(
                jax.grad(
                    lambda av: tail_exceedance_loglike(
                        counts, edges, jnp.array([5.0, 0.4, av, 3.0]), s_thr, s_max
                    )
                )(a)
            )
            assert np.isfinite(g)


def test_tail_exceedance_loglike_shift_invariance():
    """Shift-immunity: adding a constant c to (s-edges, s_thr, s_max) together leaves ll unchanged
    -- the POT alpha-block needs no mean-1 shift, only s_thr/s_max co-measured with the counts."""
    from gravoturb.inference.likelihood import tail_exceedance_loglike

    s_thr, s_max = 0.7, 3.4
    edges = jnp.linspace(s_thr, s_max, 16)
    counts = jnp.asarray(_trunc_exp_binprob(edges, s_thr, s_max, 2.5) * 1500.0)
    theta = jnp.array([5.0, 0.4, 2.3, 3.0])
    ll = float(tail_exceedance_loglike(counts, edges, theta, s_thr, s_max))

    c = 10.0
    ll_shift = float(
        tail_exceedance_loglike(counts, edges + c, theta, s_thr + c, s_max + c)
    )
    assert ll_shift == pytest.approx(ll, abs=1e-9)


# --- Task 3: truncation-corrected Fisher forecast for alpha (the sigma(alpha) vs N_tail law) ---


def test_alpha_fisher_info_untruncated_limit():
    """Per-exceedance Fisher info of the truncated exponential -> 1/alpha^2 as L -> inf (the EVT/
    Hill asymptote sigma(alpha)=alpha/sqrt(N))."""
    from gravoturb.inference.fisher import alpha_fisher_info

    for a in (1.5, 2.5, 4.0):
        assert float(alpha_fisher_info(a, 1e3)) == pytest.approx(1.0 / a**2, rel=1e-6)


def test_sigma_alpha_truncation_corrected_value_and_asymptote():
    """sigma(alpha)*sqrt(N) = 1/sqrt(I): at (alpha=2.5, L=3) the truncation correction gives ~2.54
    (above the naive alpha/sqrt(N)=2.5); as L -> inf it relaxes to the alpha/sqrt(N) asymptote."""
    from gravoturb.inference.fisher import sigma_alpha

    assert float(sigma_alpha(2.5, 3.0, 1.0)) == pytest.approx(2.5399, abs=1e-3)
    assert float(sigma_alpha(2.5, 1e3, 1.0)) == pytest.approx(2.5, rel=1e-5)
    # 1/sqrt(N_tail) scaling
    assert float(sigma_alpha(2.5, 3.0, 100.0)) == pytest.approx(
        float(sigma_alpha(2.5, 3.0, 1.0)) / 10.0, rel=1e-6
    )


def test_alpha_fisher_info_monotone_in_L():
    """Information grows with the tail's dynamic range L (truncation only loses info)."""
    from gravoturb.inference.fisher import alpha_fisher_info

    Is = [float(alpha_fisher_info(2.5, L)) for L in (0.5, 1.0, 2.0, 3.0, 5.0)]
    assert all(Is[i] < Is[i + 1] for i in range(len(Is) - 1))
    assert Is[-1] < 1.0 / 2.5**2  # always below the untruncated ceiling

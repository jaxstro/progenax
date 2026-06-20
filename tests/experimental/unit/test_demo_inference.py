"""Unit tests for the scripts-local physics-direct inference helper layer.

This is the first *tested* scripts helper, so the sys.path hack to reach
``scripts/_demo_inference.py`` lives here (the module itself stays a plain
sibling of ``scripts/_plotstyle.py``, not a packaged API).

float64 is enabled explicitly (the helper deliberately does not depend on
progenax, which would auto-enable it).
"""

import jax

jax.config.update("jax_enable_x64", True)

import pathlib
import sys

import jax.numpy as jnp
import numpy as np

# tests/experimental/unit/test_demo_inference.py
#   -> parents[0]=unit, [1]=experimental, [2]=tests, [3]=repo root.
# Lives in the experimental tier: the helper pulls optax/blackjax (the
# [experimental] extra), so it must NOT gate the released-core unit shard,
# which syncs --extra dev only (audit R1 follow-up: red CI had hidden this).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import _demo_inference as di  # noqa: E402


def _uniform_radius_positions(key, n, r_max=5.0):
    """Positions whose radii are uniform in [0, r_max] (so radial bins fill),
    with isotropic random directions."""
    k_r, k_dir = jax.random.split(key)
    r = jax.random.uniform(k_r, (n,), minval=0.0, maxval=r_max)
    directions = jax.random.normal(k_dir, (n, 3))
    directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)
    return r[:, None] * directions


class TestBinnedSigma1d:
    def test_isotropic_gaussian_recovers_sigma(self):
        key = jax.random.PRNGKey(0)
        n_per = 40_000
        sigmas = jnp.array([1.0, 2.5, 4.0])
        n_groups = 3
        r_edges = jnp.linspace(0.0, 5.0, 6)  # 5 bins

        pos_list, vel_list, gid_list = [], [], []
        for j in range(n_groups):
            kp, kv, key = jax.random.split(key, 3)
            pos_list.append(_uniform_radius_positions(kp, n_per))
            vel_list.append(jax.random.normal(kv, (n_per, 3)) * sigmas[j])
            gid_list.append(jnp.full((n_per,), j, dtype=jnp.int32))

        pos = jnp.concatenate(pos_list)
        vel = jnp.concatenate(vel_list)
        gid = jnp.concatenate(gid_list)

        sig_hat, se, weight, n = di.binned_sigma1d(
            pos, vel, gid, n_groups, r_edges, n_min=30
        )

        assert sig_hat.shape == (n_groups, 5)
        # Every populated bin must recover sigma_j within 3*SE.
        for j in range(n_groups):
            for k in range(5):
                if weight[j, k] > 0:
                    dev = jnp.abs(sig_hat[j, k] - sigmas[j])
                    assert dev <= 3.0 * se[j, k], (j, k, float(dev), float(se[j, k]))

    def test_se_scaling(self):
        key = jax.random.PRNGKey(1)
        n = 20_000
        pos = _uniform_radius_positions(key, n)
        vel = jax.random.normal(jax.random.PRNGKey(2), (n, 3)) * 2.0
        gid = jnp.zeros((n,), dtype=jnp.int32)
        r_edges = jnp.linspace(0.0, 5.0, 4)

        sig_hat, se, weight, n_bin = di.binned_sigma1d(
            pos, vel, gid, 1, r_edges, n_min=30
        )
        # SE of the 3-component pooled dispersion = sig_hat / sqrt(6 n) exactly
        # (|v|^2/sigma^2 ~ chi^2(3n) -> Var(sig_hat) ~ sigma^2/(6n)), on populated bins.
        mask = weight > 0
        expected = jnp.where(mask, sig_hat / jnp.sqrt(6.0 * n_bin), 0.0)
        np.testing.assert_allclose(
            np.asarray(se[mask]), np.asarray(expected[mask]), rtol=1e-10
        )

    def test_empty_bins_masked(self):
        key = jax.random.PRNGKey(3)
        n = 500
        # Concentrate all radii in [0,1] so outer bins are empty / under n_min.
        directions = jax.random.normal(key, (n, 3))
        directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)
        r = jax.random.uniform(jax.random.PRNGKey(4), (n,), minval=0.0, maxval=1.0)
        pos = r[:, None] * directions
        vel = jax.random.normal(jax.random.PRNGKey(5), (n, 3))
        gid = jnp.zeros((n,), dtype=jnp.int32)
        r_edges = jnp.array([0.0, 1.0, 2.0, 3.0, 10.0])  # outer bins empty

        sig_hat, se, weight, n_bin = di.binned_sigma1d(
            pos,
            vel,
            gid,
            1,
            r_edges,
            n_min=10_000,  # force everything under-populated
        )
        # No NaNs anywhere.
        for arr in (sig_hat, se, weight, n_bin):
            assert not bool(jnp.any(jnp.isnan(arr)))
        # All weights zero (n_min huge), and outer bins definitely empty.
        assert bool(jnp.all(weight == 0.0))
        assert bool(weight[0, -1] == 0.0)
        assert bool(n_bin[0, -1] == 0)


class TestBinnedSigmaBeta:
    def test_isotropic_beta_near_zero(self):
        key = jax.random.PRNGKey(10)
        n = 60_000
        pos = _uniform_radius_positions(key, n)
        vel = jax.random.normal(jax.random.PRNGKey(11), (n, 3)) * 1.5
        r_edges = jnp.linspace(0.0, 5.0, 6)

        out = di.binned_sigma_beta(pos, vel, r_edges, component_id=None, n_min=50)
        beta_hat, weight, _ = out.beta_hat, out.weight, out.n
        assert beta_hat.shape == (1, 5)
        # SE of beta ~ a few / sqrt(n); for isotropic, |beta| should be tiny.
        for k in range(5):
            if weight[0, k] > 0:
                assert jnp.abs(beta_hat[0, k]) < 0.05, (k, float(beta_hat[0, k]))

    def test_radial_anisotropy_positive_beta(self):
        key = jax.random.PRNGKey(20)
        n = 60_000
        pos = _uniform_radius_positions(key, n)
        r_hat = pos / jnp.linalg.norm(pos, axis=1, keepdims=True)

        # Build a velocity field with sigma_r > sigma_t (radially biased).
        sig_r, sig_t = 3.0, 1.0
        kr, kt = jax.random.split(jax.random.PRNGKey(21))
        v_r_mag = jax.random.normal(kr, (n,)) * sig_r
        # tangential: random in plane perpendicular to r_hat, each component sig_t.
        rand = jax.random.normal(kt, (n, 3)) * sig_t
        v_t = rand - (jnp.sum(rand * r_hat, axis=1, keepdims=True)) * r_hat
        vel_radial = v_r_mag[:, None] * r_hat + v_t
        r_edges = jnp.linspace(0.0, 5.0, 6)

        out = di.binned_sigma_beta(vel=vel_radial, pos=pos, r_edges=r_edges, n_min=50)
        for k in range(5):
            if out.weight[0, k] > 0:
                assert out.beta_hat[0, k] > 0.2, (k, float(out.beta_hat[0, k]))

        # Tangentially biased: sigma_t > sigma_r -> beta < 0.
        sig_r2, sig_t2 = 1.0, 3.0
        kr2, kt2 = jax.random.split(jax.random.PRNGKey(22))
        v_r_mag2 = jax.random.normal(kr2, (n,)) * sig_r2
        rand2 = jax.random.normal(kt2, (n, 3)) * sig_t2
        v_t2 = rand2 - (jnp.sum(rand2 * r_hat, axis=1, keepdims=True)) * r_hat
        vel_tan = v_r_mag2[:, None] * r_hat + v_t2
        out2 = di.binned_sigma_beta(vel=vel_tan, pos=pos, r_edges=r_edges, n_min=50)
        for k in range(5):
            if out2.weight[0, k] > 0:
                assert out2.beta_hat[0, k] < -0.2, (k, float(out2.beta_hat[0, k]))


class TestChi2Loglike:
    def test_perfect_model_gives_zero_loglike(self):
        sig_hat = jnp.array([1.0, 2.0, 3.0])
        se = jnp.array([0.1, 0.1, 0.1])
        weight = jnp.array([1.0, 1.0, 1.0])
        loglike = di.gaussian_loglike((sig_hat, se, weight), lambda theta: sig_hat)
        val = loglike(jnp.array(0.0))
        np.testing.assert_allclose(np.asarray(val), 0.0, atol=1e-12)

    def test_gradient_flows_to_predict_params(self):
        sig_hat = jnp.array([1.0, 2.0, 3.0])
        se = jnp.array([0.1, 0.2, 0.3])
        weight = jnp.array([1.0, 1.0, 1.0])
        # predict_fn = theta * ones (so optimum is somewhere away from sig_hat).
        loglike = di.gaussian_loglike(
            (sig_hat, se, weight), lambda theta: theta * jnp.ones(3)
        )
        g = jax.grad(loglike)(jnp.array(0.5))
        assert jnp.isfinite(g)
        assert jnp.abs(g) > 0.0


class TestBinnedNumberDensity:
    """Frozen per-shell counts N_k -- the data of a Poisson profile likelihood."""

    def test_counts_match_known_bins(self):
        # 5 stars in [0,1), 3 in [1,2), 2 in [2,3) -> counts (5, 3, 2).
        r = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 1.1, 1.2, 1.3, 2.5, 2.6])
        dirs = jax.random.normal(jax.random.PRNGKey(0), (r.shape[0], 3))
        dirs = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)
        pos = r[:, None] * dirs
        r_edges = jnp.array([0.0, 1.0, 2.0, 3.0])

        counts = di.binned_number_density(pos, r_edges)
        assert counts.shape == (3,)
        np.testing.assert_array_equal(np.asarray(counts), np.array([5.0, 3.0, 2.0]))

    def test_out_of_range_excluded(self):
        # Radii outside [r0, rK] do not count.
        r = jnp.array([0.5, 1.5, 5.0])  # 5.0 is beyond the last edge (3.0)
        dirs = jax.random.normal(jax.random.PRNGKey(1), (3, 3))
        dirs = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)
        pos = r[:, None] * dirs
        r_edges = jnp.array([0.0, 1.0, 2.0, 3.0])

        counts = di.binned_number_density(pos, r_edges)
        assert float(jnp.sum(counts)) == 2.0  # the 5.0-radius star is dropped

    def test_no_nans_and_traceable(self):
        pos = _uniform_radius_positions(jax.random.PRNGKey(2), 1000)
        r_edges = jnp.linspace(0.0, 5.0, 9)
        counts = jax.jit(di.binned_number_density)(pos, r_edges)
        assert counts.shape == (8,)
        assert not bool(jnp.any(jnp.isnan(counts)))
        assert float(jnp.sum(counts)) == 1000.0  # all in range


class TestPoissonLoglike:
    """Per-bin Poisson log-likelihood; data frozen, gradient through predict only."""

    def test_gradient_zero_at_analytic_mle(self):
        # mu = theta * base; loglike maximized at theta_hat = sum(N) / sum(base)
        # (d/dtheta sum[N log(theta base) - theta base] = sum[N/theta - base] = 0).
        counts = jnp.array([10.0, 7.0, 3.0, 1.0])
        base = jnp.array([4.0, 3.0, 2.0, 1.0])
        weight = jnp.ones(4)
        ll = di.poisson_loglike((counts, weight), lambda th: th * base)

        th_star = float(jnp.sum(counts) / jnp.sum(base))
        g = jax.grad(ll)(jnp.array(th_star))
        assert abs(float(g)) < 1e-8, float(g)
        # Below the optimum the loglike is still increasing (positive slope).
        g_below = jax.grad(ll)(jnp.array(0.5 * th_star))
        assert float(g_below) > 0.0

    def test_recovers_scale_via_adam(self):
        # Deterministic "expected" counts at theta_true -> MLE must hit theta_true.
        base = jnp.array([20.0, 12.0, 6.0, 2.0])
        theta_true = 1.7
        counts = theta_true * base  # the expected (noise-free) counts
        weight = jnp.ones(4)
        # Reparametrize theta = exp(z) to stay positive.
        ll = di.poisson_loglike((counts, weight), lambda z: jnp.exp(z) * base)
        negloglike = lambda z: -ll(z[0])
        z_hat, trace = di.mle_adam(negloglike, jnp.array([0.0]), n_steps=800, lr=3e-2)
        np.testing.assert_allclose(float(jnp.exp(z_hat[0])), theta_true, rtol=1e-3)

    def test_zero_count_bins_no_nan(self):
        counts = jnp.array([5.0, 0.0, 0.0])
        weight = jnp.ones(3)
        ll = di.poisson_loglike((counts, weight), lambda th: th * jnp.ones(3))
        val = ll(jnp.array(2.0))
        g = jax.grad(ll)(jnp.array(2.0))
        assert jnp.isfinite(val) and jnp.isfinite(g)

    def test_weight_masks_bins(self):
        # A zero-weight bin must not influence the loglike at all.
        counts = jnp.array([10.0, 1000.0])
        base = jnp.array([5.0, 5.0])
        ll_masked = di.poisson_loglike(
            (counts, jnp.array([1.0, 0.0])), lambda th: th * base
        )
        ll_only0 = di.poisson_loglike(
            (counts[:1], jnp.array([1.0])), lambda th: th * base[:1]
        )
        np.testing.assert_allclose(
            float(ll_masked(jnp.array(1.3))),
            float(ll_only0(jnp.array(1.3))),
            rtol=1e-12,
        )

    def test_fisher_cov_runs_on_poisson(self):
        # Observed-Hessian Fisher is PD for a well-identified 1-param Poisson fit.
        base = jnp.array([20.0, 12.0, 6.0, 2.0])
        counts = 1.7 * base
        ll = di.poisson_loglike((counts, jnp.ones(4)), lambda z: jnp.exp(z) * base)
        cov = di.fisher_cov(lambda z: -ll(z[0]), jnp.array([float(jnp.log(1.7))]))
        assert cov.shape == (1, 1) and float(cov[0, 0]) > 0.0


class TestPoissonFisherInformation:
    """Reverse-mode (jacrev) Poisson expected information F = J^T diag(w/mu) J.

    The Poisson sibling of :func:`fisher_information_gn`; reverse-mode only, so it
    survives the diffrax-ODE ``custom_vjp`` in the King profile that B11 fits
    (``jax.hessian`` would crash forward-mode over that ``custom_vjp``)."""

    def test_linear_model_matches_closed_form(self):
        # mu(z) = A z (>0 at z_hat) -> J = A -> F = A^T diag(1/mu) A.
        A = jnp.array([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0], [0.2, 0.8]])
        z_hat = jnp.array([2.0, 3.0])
        mu = A @ z_hat
        expected = A.T @ jnp.diag(1.0 / mu) @ A
        F = di.poisson_fisher_information(lambda z: A @ z, z_hat)
        np.testing.assert_allclose(np.asarray(F), np.asarray(expected), atol=1e-10)

    def test_weight_downweights_bins(self):
        A = jnp.array([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0]])
        z_hat = jnp.array([2.0, 3.0])
        mu = A @ z_hat
        w = jnp.array([1.0, 0.0, 1.0])  # drop the middle bin
        expected = A.T @ jnp.diag(w / mu) @ A
        F = di.poisson_fisher_information(lambda z: A @ z, z_hat, weight=w)
        np.testing.assert_allclose(np.asarray(F), np.asarray(expected), atol=1e-10)

    def test_symmetric_pd_for_full_rank(self):
        A = jnp.array([[2.0, 1.0], [1.0, 3.0], [0.0, 1.0], [1.0, 1.0]])
        z_hat = jnp.array([1.5, 2.0])
        F = di.poisson_fisher_information(lambda z: A @ z, z_hat)
        np.testing.assert_allclose(np.asarray(F), np.asarray(F.T), atol=1e-12)
        assert bool(jnp.all(jnp.linalg.eigvalsh(0.5 * (F + F.T)) > 0))

    def test_empty_mu_bins_no_nan(self):
        # A truncation fit has bins where mu==0 and dmu/dz==0 (beyond the edge):
        # 1/mu would be inf and 0*inf -> NaN. The flooring must keep F finite, with
        # those zero-mu/zero-Jacobian bins contributing nothing.
        def predict_mu(z):
            full = z[0] * jnp.array([4.0, 3.0, 2.0, 1.0])
            mask = jnp.array([1.0, 1.0, 0.0, 0.0])  # last two bins truncated to 0
            return full * mask

        z_hat = jnp.array([1.5])
        F = di.poisson_fisher_information(predict_mu, z_hat)
        assert jnp.isfinite(F).all()
        # equals the closed form over the populated bins only.
        base = jnp.array([4.0, 3.0])
        mu = 1.5 * base
        expected = jnp.sum(base**2 / mu)
        np.testing.assert_allclose(float(F[0, 0]), float(expected), rtol=1e-10)


class TestAdamMLE:
    def test_recovers_quadratic_minimum(self):
        negloglike = lambda z: (z[0] - 3.0) ** 2
        z0 = jnp.array([0.0])
        z_hat, trace = di.mle_adam(negloglike, z0, n_steps=600, lr=5e-2)
        assert trace.shape == (600,)
        np.testing.assert_allclose(np.asarray(z_hat[0]), 3.0, atol=1e-3)
        # trace should be monotone-ish decreasing and plateau near 0.
        assert float(trace[-1]) < float(trace[0])
        assert float(trace[-1]) < 1e-5

    def test_fisher_cov_quadratic(self):
        negloglike = lambda z: (z[0] - 3.0) ** 2
        z_hat = jnp.array([3.0])
        cov = di.fisher_cov(negloglike, z_hat)
        # Hessian = 2 -> cov = 0.5.
        np.testing.assert_allclose(np.asarray(cov), np.array([[0.5]]), atol=1e-8)

    def test_fisher_raises_on_non_pd(self):
        # Indefinite Hessian: f(z) = -z^2 -> Hessian = -2 (negative).
        negloglike = lambda z: -(z[0] ** 2)
        z_hat = jnp.array([0.0])
        try:
            di.fisher_cov(negloglike, z_hat)
        except ValueError:
            return
        raise AssertionError("fisher_cov should raise ValueError on non-PD Hessian")


class TestFisherInformationGN:
    """Gauss-Newton (reverse-mode only) Fisher information F = J^T J [+ Hess].

    Used on the B2 demo loss because jax.hessian crashes through the diffrax
    ODE inside from_imf (forward-mode over a custom_vjp). The Jacobian of the
    standardized residual vector is reverse-mode (jacrev) only.
    """

    def test_linear_model_returns_AtA(self):
        # r(z) = A z - b  =>  J = A  =>  F = J^T J = A^T A exactly.
        A = jnp.array(
            [[1.0, 2.0, 0.0], [0.0, 1.0, -1.0], [3.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
        )
        b = jnp.array([0.5, -1.0, 2.0, 0.3])
        residual_fn = lambda z: A @ z - b
        z_hat = jnp.array([0.1, -0.2, 0.3])
        F = di.fisher_information_gn(residual_fn, z_hat)
        np.testing.assert_allclose(np.asarray(F), np.asarray(A.T @ A), atol=1e-10)
        # Independent of z_hat for a linear residual.
        F2 = di.fisher_information_gn(residual_fn, jnp.array([5.0, -3.0, 1.0]))
        np.testing.assert_allclose(np.asarray(F2), np.asarray(A.T @ A), atol=1e-10)

    def test_extra_negloglike_adds_its_hessian(self):
        # Linear residual contributes A^T A; a quadratic extra term contributes
        # its (constant) Hessian H. Total must be A^T A + H.
        A = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
        residual_fn = lambda z: A @ z
        # extra = 0.5 z^T H z  (H spd, diagonal) -> Hessian = H.
        H = jnp.diag(jnp.array([4.0, 9.0, 16.0]))
        extra = lambda z: 0.5 * z @ (H @ z)
        z_hat = jnp.array([0.3, -0.1, 0.7])
        F = di.fisher_information_gn(residual_fn, z_hat, extra_negloglike=extra)
        np.testing.assert_allclose(np.asarray(F), np.asarray(A.T @ A + H), atol=1e-10)

    def test_returns_symmetric_pd_for_well_posed(self):
        # Full-rank residual Jacobian -> J^T J is symmetric PD.
        A = jnp.array(
            [[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0], [1.0, 0.0, 1.0]]
        )
        residual_fn = lambda z: A @ z - jnp.ones(4)
        F = di.fisher_information_gn(residual_fn, jnp.zeros(3))
        np.testing.assert_allclose(np.asarray(F), np.asarray(F.T), atol=1e-12)
        eigvals = jnp.linalg.eigvalsh(0.5 * (F + F.T))
        assert bool(jnp.all(eigvals > 0)), np.asarray(eigvals)

    def test_nonlinear_residual_uses_jacrev_at_z_hat(self):
        # r(z) = [z0^2, z0 z1]  =>  J(z) = [[2 z0, 0], [z1, z0]].
        residual_fn = lambda z: jnp.array([z[0] ** 2, z[0] * z[1]])
        z_hat = jnp.array([1.5, -0.5])
        J = jnp.array([[2.0 * 1.5, 0.0], [-0.5, 1.5]])
        F = di.fisher_information_gn(residual_fn, z_hat)
        np.testing.assert_allclose(np.asarray(F), np.asarray(J.T @ J), atol=1e-10)


class TestRunNuts:
    """Vendored blackjax NUTS wrapper (window adaptation -> NUTS -> draws + div count).

    Sampled on a known correlated 2-D Gaussian; the wrapper must recover the mean
    (within a few SE of the n_samples draws), the covariance (within ~15%), and
    report 0 divergences (a well-conditioned Gaussian has none).
    """

    def test_recovers_correlated_gaussian(self):
        mu = jnp.array([1.5, -0.7])
        # Correlated 2-D covariance (rho = 0.6), well-conditioned.
        cov = jnp.array([[1.0, 0.6 * 1.0 * 2.0], [0.6 * 1.0 * 2.0, 4.0]])
        prec = jnp.linalg.inv(cov)

        def logdensity_fn(z):
            d = z - mu
            return -0.5 * d @ (prec @ d)

        key = jax.random.PRNGKey(0)
        z0 = jnp.zeros(2)
        out = di.run_nuts(logdensity_fn, z0, key, n_warmup=400, n_samples=2000)

        assert out.samples.shape == (2000, 2)
        # 0 divergences on a well-conditioned Gaussian.
        assert int(out.n_divergent) == 0, int(out.n_divergent)

        samp = out.samples
        post_mean = jnp.mean(samp, axis=0)
        # Mean within a few SE (SE = sqrt(diag(cov)/n_eff); use n_samples as a
        # conservative n_eff floor -> 4 SE is generous but still a real check).
        se = jnp.sqrt(jnp.diag(cov) / samp.shape[0])
        for i in range(2):
            dev = float(jnp.abs(post_mean[i] - mu[i]))
            assert dev < 4.0 * float(se[i]), (i, dev, float(se[i]))

        # Covariance within ~15% (entrywise relative, on the scale of the entry).
        post_cov = jnp.cov(samp.T)
        for i in range(2):
            for k in range(2):
                rel = float(jnp.abs(post_cov[i, k] - cov[i, k]) / jnp.abs(cov[i, k]))
                assert rel < 0.15, (i, k, float(post_cov[i, k]), float(cov[i, k]), rel)


class TestReparam:
    def test_logit_expit_roundtrip(self):
        lo, hi = -2.0, 5.0
        xs = jnp.array([-1.999, -1.0, 0.0, 2.3, 4.999])
        z = di.logit(xs, lo, hi)
        back = di.expit(z, lo, hi)
        np.testing.assert_allclose(
            np.asarray(back), np.asarray(xs), rtol=1e-10, atol=1e-10
        )

    def test_expit_in_bounds(self):
        lo, hi = 0.5, 3.5
        zs = jnp.array([-1e3, -10.0, 0.0, 10.0, 1e3])
        xs = di.expit(zs, lo, hi)
        assert bool(jnp.all(xs > lo))
        assert bool(jnp.all(xs < hi))

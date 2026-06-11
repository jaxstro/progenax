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

# tests/unit/test_demo_inference.py -> parents[0]=unit, [1]=tests, [2]=repo root.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
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

        sig_hat, se, weight, n_bin = di.binned_sigma1d(pos, vel, gid, 1, r_edges, n_min=30)
        # SE of a dispersion estimate = sig_hat / sqrt(2 n) exactly, on populated bins.
        mask = weight > 0
        expected = jnp.where(mask, sig_hat / jnp.sqrt(2.0 * n_bin), 0.0)
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
            pos, vel, gid, 1, r_edges, n_min=10_000  # force everything under-populated
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
        beta_hat, weight, n_bin = out.beta_hat, out.weight, out.n
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


class TestReparam:
    def test_logit_expit_roundtrip(self):
        lo, hi = -2.0, 5.0
        xs = jnp.array([-1.999, -1.0, 0.0, 2.3, 4.999])
        z = di.logit(xs, lo, hi)
        back = di.expit(z, lo, hi)
        np.testing.assert_allclose(np.asarray(back), np.asarray(xs), rtol=1e-10, atol=1e-10)

    def test_expit_in_bounds(self):
        lo, hi = 0.5, 3.5
        zs = jnp.array([-1e3, -10.0, 0.0, 10.0, 1e3])
        xs = di.expit(zs, lo, hi)
        assert bool(jnp.all(xs > lo))
        assert bool(jnp.all(xs < hi))

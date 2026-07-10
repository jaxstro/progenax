"""Physics validation for the differentiable mass-segregation observables.

Validates the soft observables in ``progenax.diagnostics.segregation_approx`` against
**external / exact** oracles (never self-consistency), per
the internal design note:

- Oracle 1 (central): soft -> exact as the softness (tau, beta) -> 0.
- Oracle 2: monotonic response + rank-correlation with the exact Allison+2009 Lambda_MSR
  across a controlled segregation sweep.
- Oracle 3: hand-constructed regimes (unsegregated -> null, segregated -> strong,
  inverse -> opposite sign).
- Oracle 4: differentiability (autodiff vs finite-difference; no NaN on degenerate input).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial import cKDTree

from progenax.diagnostics import compute_lambda_msr
from progenax.diagnostics.segregation_approx import (
    lambda_msr_approx,
    radial_concentration_approx,
    sigma_m_approx,
)


# ----------------------------------------------------------------------------
# Cluster builders with a controllable segregation strength.
# ----------------------------------------------------------------------------
def _cluster(key, N=400, n_massive=20, core_scale=0.05, inverse=False):
    """Bimodal-mass cluster. core_scale=1.0 ~ unsegregated; small => segregated.

    If inverse, massive stars are placed on a wide rim instead of a tight core.
    """
    k1, k2 = jax.random.split(key)
    if inverse:
        # Inverse: low-mass stars centrally concentrated, massive stars on the wide
        # outskirts of the SAME cluster (both centred at the origin).
        halo = jax.random.normal(k1, (N - n_massive, 3)) * 0.4
        special = jax.random.normal(k2, (n_massive, 3)) * 2.5
    else:
        halo = jax.random.normal(k1, (N - n_massive, 3)) * 1.0
        special = jax.random.normal(k2, (n_massive, 3)) * core_scale
    positions = jnp.concatenate([halo, special], axis=0)
    masses = jnp.concatenate([jnp.full(N - n_massive, 0.5), jnp.full(n_massive, 10.0)])
    return positions, masses


# ----------------------------------------------------------------------------
# Exact (non-differentiable) oracles -- the tau,beta -> 0 limits of the surrogates.
# ----------------------------------------------------------------------------
def _exact_radial(xy, massive):
    center = xy[massive].mean(axis=0)
    r = np.sqrt(((xy - center) ** 2).sum(axis=1))
    return r[massive].mean() / r.mean()


def _exact_lambda_nn(xy, massive):
    """Hard 1-NN ratio: mean(d_1NN)_all / mean(d_1NN)_massive (the surrogate's limit)."""
    tree = cKDTree(xy)
    d, _ = tree.query(xy, k=2)  # [:,0] is self (0); [:,1] is the true 1-NN
    nn = d[:, 1]
    return nn.mean() / nn[massive].mean()


def _exact_sigma(xy, massive, k=6):
    tree = cKDTree(xy)
    d, _ = tree.query(xy, k=k + 1)
    r_k = d[:, k]
    sigma = (k - 1) / (np.pi * r_k**2)
    return np.corrcoef(massive.astype(float), np.log(sigma))[0, 1]


# ============================================================================
# Oracle 1 -- soft converges to exact as the softness -> 0 (the central claim).
# ============================================================================
class TestHardLimitConvergence:
    def test_radial_converges_to_exact(self):
        pos, m = _cluster(jax.random.PRNGKey(0), core_scale=0.05)
        xy = np.asarray(pos[:, :2])
        exact = _exact_radial(xy, np.asarray(m) > 2.0)
        errs = [
            abs(float(radial_concentration_approx(pos, m, m_cut=2.0, tau=t)) - exact)
            for t in (0.5, 0.1, 0.02, 0.005)
        ]
        assert errs[-1] < errs[0]  # monotone improvement
        assert errs[-1] < 1e-3  # exact at the sharp limit

    def test_sigma_converges_to_exact(self):
        pos, m = _cluster(jax.random.PRNGKey(1), core_scale=0.05)
        xy = np.asarray(pos[:, :2])
        exact = _exact_sigma(xy, np.asarray(m) > 2.0, k=6)
        errs = [
            abs(float(sigma_m_approx(pos, m, m_cut=2.0, tau=t, k=6)) - exact)
            for t in (0.5, 0.1, 0.02, 0.005)
        ]
        assert errs[-1] < errs[0]
        assert errs[-1] < 1e-2

    def test_lambda_converges_to_exact_nn_ratio(self):
        pos, m = _cluster(jax.random.PRNGKey(2), core_scale=0.05)
        xy = np.asarray(pos[:, :2])
        exact = _exact_lambda_nn(xy, np.asarray(m) > 2.0)
        errs = [
            abs(float(lambda_msr_approx(pos, m, m_cut=2.0, tau=t, beta=b)) - exact)
            for t, b in ((0.5, 0.3), (0.1, 0.1), (0.02, 0.03), (0.005, 0.01))
        ]
        assert errs[-1] < errs[0]
        assert errs[-1] < 5e-2


# ============================================================================
# Oracle 2 -- monotonic response + rank-correlation with exact Lambda_MSR.
# ============================================================================
class TestSegregationSweep:
    def _sweep(self):
        scales = np.linspace(0.05, 1.0, 8)  # tight core -> diffuse (unsegregated)
        lam, rad, sig, exact = [], [], [], []
        for i, s in enumerate(scales):
            pos, m = _cluster(jax.random.PRNGKey(100 + i), core_scale=float(s))
            lam.append(float(lambda_msr_approx(pos, m, m_cut=2.0, tau=0.3, beta=0.1)))
            rad.append(float(radial_concentration_approx(pos, m, m_cut=2.0, tau=0.3)))
            sig.append(float(sigma_m_approx(pos, m, m_cut=2.0, tau=0.3, k=6)))
            exact.append(
                compute_lambda_msr(np.asarray(pos), np.asarray(m), N_massive=20)[0]
            )
        return scales, np.array(lam), np.array(rad), np.array(sig), np.array(exact)

    def test_monotonic_in_segregation_strength(self):
        from scipy.stats import spearmanr

        scales, lam, rad, sig, _ = self._sweep()
        # tighter core (smaller scale) = more segregated:
        assert spearmanr(scales, lam).correlation < -0.8  # Lambda up as scale down
        assert spearmanr(scales, rad).correlation > 0.8  # C up as scale up
        assert spearmanr(scales, sig).correlation < -0.8  # S up as scale down

    def test_rank_correlates_with_exact_lambda_msr(self):
        from scipy.stats import spearmanr

        _, lam, rad, sig, exact = self._sweep()
        assert abs(spearmanr(exact, lam).correlation) > 0.8
        assert abs(spearmanr(exact, rad).correlation) > 0.8
        assert abs(spearmanr(exact, sig).correlation) > 0.8


# ============================================================================
# Oracle 3 -- hand-constructed regimes.
# ============================================================================
class TestRegimes:
    def test_unsegregated_null(self):
        lam, rad, sig = [], [], []
        for s in range(6):
            pos, m = _cluster(jax.random.PRNGKey(s), core_scale=1.0)
            lam.append(float(lambda_msr_approx(pos, m, m_cut=2.0, tau=0.3, beta=0.1)))
            rad.append(float(radial_concentration_approx(pos, m, m_cut=2.0, tau=0.3)))
            sig.append(float(sigma_m_approx(pos, m, m_cut=2.0, tau=0.3, k=6)))
        assert abs(np.mean(lam) - 1.0) < 0.2
        assert abs(np.mean(rad) - 1.0) < 0.2
        assert abs(np.mean(sig)) < 0.15

    def test_segregated_strong(self):
        pos, m = _cluster(jax.random.PRNGKey(7), core_scale=0.05)
        assert float(lambda_msr_approx(pos, m, m_cut=2.0, tau=0.3, beta=0.1)) > 1.3
        assert float(radial_concentration_approx(pos, m, m_cut=2.0, tau=0.3)) < 0.7
        assert float(sigma_m_approx(pos, m, m_cut=2.0, tau=0.3, k=6)) > 0.3

    def test_inverse_segregation_opposite_sign(self):
        pos, m = _cluster(jax.random.PRNGKey(8), inverse=True)
        # massive stars on a wide offset rim => less concentrated, sparser locally
        assert float(radial_concentration_approx(pos, m, m_cut=2.0, tau=0.3)) > 1.2
        assert float(sigma_m_approx(pos, m, m_cut=2.0, tau=0.3, k=6)) < 0.0
        assert float(lambda_msr_approx(pos, m, m_cut=2.0, tau=0.3, beta=0.1)) < 1.0


# ============================================================================
# Oracle 4 -- differentiability.
# ============================================================================
class TestDifferentiability:
    @pytest.mark.parametrize(
        "fn",
        [
            lambda p, m, mc: radial_concentration_approx(p, m, m_cut=mc, tau=0.8),
            lambda p, m, mc: lambda_msr_approx(p, m, m_cut=mc, tau=0.8, beta=0.1),
            lambda p, m, mc: sigma_m_approx(p, m, m_cut=mc, tau=0.8, k=6),
        ],
        ids=["radial", "lambda", "sigma"],
    )
    def test_grad_m_cut_matches_fd(self, fn):
        pos, m = _cluster(jax.random.PRNGKey(3), core_scale=0.1)
        g = jax.grad(lambda mc: fn(pos, m, mc))(2.0)
        eps = 1e-4
        fd = (float(fn(pos, m, 2.0 + eps)) - float(fn(pos, m, 2.0 - eps))) / (2 * eps)
        assert jnp.isfinite(g)
        np.testing.assert_allclose(float(g), fd, rtol=3e-4, atol=1e-6)

    def test_grad_positions_finite(self):
        pos, m = _cluster(jax.random.PRNGKey(4), core_scale=0.1)
        for fn in (
            lambda p: radial_concentration_approx(p, m, m_cut=2.0, tau=0.5),
            lambda p: lambda_msr_approx(p, m, m_cut=2.0, tau=0.5, beta=0.1),
            lambda p: sigma_m_approx(p, m, m_cut=2.0, tau=0.5, k=6),
        ):
            g = jax.grad(lambda p: fn(p).sum() if fn(p).ndim else fn(p))(pos)
            assert jnp.all(jnp.isfinite(g))

    def test_no_nan_on_degenerate_inputs(self):
        # All-equal masses (no massive population) and coincident points.
        pos = (
            jnp.zeros((50, 3))
            + jax.random.normal(jax.random.PRNGKey(5), (50, 3)) * 1e-6
        )
        m = jnp.full(50, 1.0)
        for val in (
            radial_concentration_approx(pos, m, m_cut=2.0, tau=0.5),
            lambda_msr_approx(pos, m, m_cut=2.0, tau=0.5, beta=0.1),
            sigma_m_approx(pos, m, m_cut=2.0, tau=0.5, k=6),
        ):
            assert jnp.isfinite(val)

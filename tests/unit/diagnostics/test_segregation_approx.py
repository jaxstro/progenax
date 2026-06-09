"""Unit tests for differentiable mass-segregation observables.

RED-first per docs/plans/2026-06-09-differentiable-segregation-observable-design.md.
Covers the shared soft-mass-cut kernel and the radial-concentration observable.
Oracles live in tests/validation/test_segregation_approx_physics.py.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


# --------------------------------------------------------------------------
# Fixtures: hand-constructed configurations with a known segregation state.
# --------------------------------------------------------------------------
def _segregated_cluster(key, N=400, n_massive=20):
    """Massive stars in a tight central core, low-mass stars in a wide halo.

    Returns (positions[N,3], masses[N]). Massive stars are the LAST n_massive
    entries (so tests must not rely on ordering).
    """
    k1, k2 = jax.random.split(key)
    halo = jax.random.normal(k1, (N - n_massive, 3)) * 1.0
    core = jax.random.normal(k2, (n_massive, 3)) * 0.05
    positions = jnp.concatenate([halo, core], axis=0)
    masses = jnp.concatenate(
        [jnp.full(N - n_massive, 0.5), jnp.full(n_massive, 10.0)]
    )
    return positions, masses


def _unsegregated_cluster(key, N=400, n_massive=20):
    """Massive stars drawn from the SAME spatial distribution as low-mass."""
    k1, k2 = jax.random.split(key)
    positions = jax.random.normal(k1, (N, 3)) * 1.0
    masses = jnp.full(N, 0.5)
    idx = jax.random.choice(k2, N, (n_massive,), replace=False)
    masses = masses.at[idx].set(10.0)
    return positions, masses


# --------------------------------------------------------------------------
# Shared soft-mass-cut kernel.
# --------------------------------------------------------------------------
class TestSoftMassWeights:
    def test_importable(self):
        from progenax.diagnostics.segregation_approx import soft_mass_weights

        assert callable(soft_mass_weights)

    def test_weights_in_unit_interval(self):
        from progenax.diagnostics.segregation_approx import soft_mass_weights

        masses = jnp.array([0.1, 0.5, 1.0, 5.0, 20.0])
        w = soft_mass_weights(masses, m_cut=2.0, tau=0.5)
        assert w.shape == (5,)
        assert jnp.all(w > 0.0) and jnp.all(w < 1.0)

    def test_monotonic_increasing_in_mass(self):
        from progenax.diagnostics.segregation_approx import soft_mass_weights

        masses = jnp.linspace(0.1, 50.0, 64)
        w = soft_mass_weights(masses, m_cut=5.0, tau=1.0)
        assert jnp.all(jnp.diff(w) >= -1e-9)

    def test_hard_limit_approaches_indicator(self):
        """As tau -> 0, w -> 1[m > m_cut]."""
        from progenax.diagnostics.segregation_approx import soft_mass_weights

        masses = jnp.array([1.0, 3.0, 8.0])  # m_cut=5 -> indicator [0,0,1]
        w = soft_mass_weights(masses, m_cut=5.0, tau=1e-4)
        np.testing.assert_allclose(np.asarray(w), [0.0, 0.0, 1.0], atol=1e-3)

    def test_differentiable_in_m_cut(self):
        from progenax.diagnostics.segregation_approx import soft_mass_weights

        masses = jnp.array([1.0, 4.0, 9.0])
        g = jax.grad(lambda mc: jnp.sum(soft_mass_weights(masses, mc, 1.0)))(3.0)
        assert jnp.isfinite(g)


# --------------------------------------------------------------------------
# Radial-concentration observable.
# --------------------------------------------------------------------------
class TestRadialConcentration:
    def test_returns_scalar(self):
        from progenax.diagnostics.segregation_approx import radial_concentration_approx

        pos, m = _segregated_cluster(jax.random.PRNGKey(0))
        C = radial_concentration_approx(pos, m, m_cut=2.0, tau=0.5)
        assert C.shape == ()

    def test_segregated_below_one(self):
        """Massive stars in the core => mass-weighted mean radius < unweighted => C < 1."""
        from progenax.diagnostics.segregation_approx import radial_concentration_approx

        pos, m = _segregated_cluster(jax.random.PRNGKey(1))
        C = radial_concentration_approx(pos, m, m_cut=2.0, tau=0.5)
        assert float(C) < 0.8

    def test_unsegregated_near_one(self):
        from progenax.diagnostics.segregation_approx import radial_concentration_approx

        vals = []
        for s in range(8):
            pos, m = _unsegregated_cluster(jax.random.PRNGKey(s))
            vals.append(float(radial_concentration_approx(pos, m, m_cut=2.0, tau=0.5)))
        assert abs(np.mean(vals) - 1.0) < 0.15

    def test_segregated_below_unsegregated(self):
        from progenax.diagnostics.segregation_approx import radial_concentration_approx

        ps, ms = _segregated_cluster(jax.random.PRNGKey(2))
        pu, mu = _unsegregated_cluster(jax.random.PRNGKey(2))
        Cs = radial_concentration_approx(ps, ms, m_cut=2.0, tau=0.5)
        Cu = radial_concentration_approx(pu, mu, m_cut=2.0, tau=0.5)
        assert float(Cs) < float(Cu)

    def test_projects_to_2d_by_default(self):
        """Default project_to_2d=True ignores z; an explicit 3D call differs."""
        from progenax.diagnostics.segregation_approx import radial_concentration_approx

        pos, m = _segregated_cluster(jax.random.PRNGKey(3))
        C2d = radial_concentration_approx(pos, m, m_cut=2.0, tau=0.5)
        C3d = radial_concentration_approx(pos, m, m_cut=2.0, tau=0.5, project_to_2d=False)
        assert not np.isclose(float(C2d), float(C3d))

    def test_differentiable_in_m_cut_matches_fd(self):
        from progenax.diagnostics.segregation_approx import radial_concentration_approx

        pos, m = _segregated_cluster(jax.random.PRNGKey(4))

        def f(mc):
            return radial_concentration_approx(pos, m, m_cut=mc, tau=0.8)

        g = jax.grad(f)(2.0)
        eps = 1e-4
        fd = (float(f(2.0 + eps)) - float(f(2.0 - eps))) / (2 * eps)
        assert jnp.isfinite(g)
        np.testing.assert_allclose(float(g), fd, rtol=1e-4, atol=1e-6)

    def test_jit_and_vmap(self):
        from progenax.diagnostics.segregation_approx import radial_concentration_approx

        pos, m = _segregated_cluster(jax.random.PRNGKey(5))
        jitted = jax.jit(
            lambda p, mm: radial_concentration_approx(p, mm, m_cut=2.0, tau=0.5)
        )
        assert jnp.isfinite(jitted(pos, m))
        batch_pos = jnp.stack([pos, pos])
        batch_m = jnp.stack([m, m])
        out = jax.vmap(
            lambda p, mm: radial_concentration_approx(p, mm, m_cut=2.0, tau=0.5)
        )(batch_pos, batch_m)
        assert out.shape == (2,)


# --------------------------------------------------------------------------
# Soft Lambda_MSR observable (MST-ratio surrogate, closed-form random baseline).
# --------------------------------------------------------------------------
class TestLambdaMSRApprox:
    def test_returns_scalar(self):
        from progenax.diagnostics.segregation_approx import lambda_msr_approx

        pos, m = _segregated_cluster(jax.random.PRNGKey(0))
        lam = lambda_msr_approx(pos, m, m_cut=2.0, tau=0.5, beta=0.05)
        assert lam.shape == ()

    def test_segregated_above_one(self):
        """Massive stars clumped => short massive NN-length => Lambda > 1.

        The (N-1)<d_1NN> MST proxy is a *local-density* measure, so its dynamic range
        is milder than the global radial concentration -- it saturates near ~1.5-1.6
        even for strong segregation. We assert a clear separation from unity, not a
        large magnitude (the Fisher-information figure quantifies the sensitivity gap).
        """
        from progenax.diagnostics.segregation_approx import lambda_msr_approx

        pos, m = _segregated_cluster(jax.random.PRNGKey(1))
        lam = lambda_msr_approx(pos, m, m_cut=2.0, tau=0.5, beta=0.05)
        assert float(lam) > 1.3

    def test_unsegregated_near_one(self):
        from progenax.diagnostics.segregation_approx import lambda_msr_approx

        vals = []
        for s in range(8):
            pos, m = _unsegregated_cluster(jax.random.PRNGKey(s))
            vals.append(float(lambda_msr_approx(pos, m, m_cut=2.0, tau=0.5, beta=0.05)))
        assert abs(np.mean(vals) - 1.0) < 0.2

    def test_segregated_above_unsegregated(self):
        from progenax.diagnostics.segregation_approx import lambda_msr_approx

        ps, ms = _segregated_cluster(jax.random.PRNGKey(2))
        pu, mu = _unsegregated_cluster(jax.random.PRNGKey(2))
        ls = lambda_msr_approx(ps, ms, m_cut=2.0, tau=0.5, beta=0.05)
        lu = lambda_msr_approx(pu, mu, m_cut=2.0, tau=0.5, beta=0.05)
        assert float(ls) > float(lu)

    def test_differentiable_in_m_cut_matches_fd(self):
        from progenax.diagnostics.segregation_approx import lambda_msr_approx

        pos, m = _segregated_cluster(jax.random.PRNGKey(4))

        def f(mc):
            return lambda_msr_approx(pos, m, m_cut=mc, tau=0.8, beta=0.05)

        g = jax.grad(f)(2.0)
        eps = 1e-4
        fd = (float(f(2.0 + eps)) - float(f(2.0 - eps))) / (2 * eps)
        assert jnp.isfinite(g)
        np.testing.assert_allclose(float(g), fd, rtol=2e-4, atol=1e-6)

    def test_jit_and_vmap(self):
        from progenax.diagnostics.segregation_approx import lambda_msr_approx

        pos, m = _segregated_cluster(jax.random.PRNGKey(5))
        jitted = jax.jit(
            lambda p, mm: lambda_msr_approx(p, mm, m_cut=2.0, tau=0.5, beta=0.05)
        )
        assert jnp.isfinite(jitted(pos, m))
        out = jax.vmap(
            lambda p, mm: lambda_msr_approx(p, mm, m_cut=2.0, tau=0.5, beta=0.05)
        )(jnp.stack([pos, pos]), jnp.stack([m, m]))
        assert out.shape == (2,)

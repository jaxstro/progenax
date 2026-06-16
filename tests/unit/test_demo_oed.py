"""Stage-1 OED demo unit tests (Task 1: predicted observable + per-star Fisher blocks)."""
import sys
import pathlib

import jax
import jax.numpy as jnp
import pytest
import progenax  # noqa: F401  -- enables float64
from jaxstro.units import STELLAR

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
import _demo_oed as oed  # noqa: E402


def test_predict_sigma_shape_and_units():
    th = oed.theta_truth()                      # (3,) = (r_a, M, r_h)
    sig = oed.predict_sigma(th, oed.R_BINS, STELLAR.G)   # (3, K) channels x bins
    assert sig.shape == (3, oed.R_BINS.shape[0])
    assert jnp.all(sig > 0)
    # isotropic-ish check: at small R, los ~ pm_r ~ pm_t within 30%
    inner = sig[:, 0]
    assert jnp.max(inner) / jnp.min(inner) < 1.5


def test_per_star_blocks_shape_and_symmetry():
    th = oed.theta_truth()
    Mb, sig = oed.per_star_blocks(th, oed.R_BINS, oed.EPS, STELLAR.G)
    K = oed.R_BINS.shape[0]
    assert Mb.shape == (3, K, 3, 3)             # channel, bin, P, P
    # each block is symmetric PSD rank-1: M = 2 J J^T / denom
    assert jnp.allclose(Mb, jnp.swapaxes(Mb, -1, -2), atol=1e-12)
    # diagonal entries non-negative
    assert jnp.all(jnp.diagonal(Mb, axis1=-2, axis2=-1) >= -1e-12)


# --- Task 2: design allocation, completeness, additive Fisher F = Sum n*c*M ---


def test_fisher_additivity_and_linearity():
    th = oed.theta_truth()
    Mb, _ = oed.per_star_blocks(th, oed.R_BINS, oed.EPS, STELLAR.G)
    K = oed.R_BINS.shape[0]
    z = jnp.zeros(3 * K)                         # uniform softmax
    F1 = oed.fisher(z, Mb, oed.completeness(oed.R_BINS), N_total=1000.0)
    F2 = oed.fisher(z, Mb, oed.completeness(oed.R_BINS), N_total=2000.0)
    # F linear in N_total at fixed design fractions
    assert jnp.allclose(F2, 2.0 * F1, rtol=1e-10)
    assert F1.shape == (3, 3)
    assert jnp.allclose(F1, F1.T, atol=1e-10)


def test_completeness_rolls_off_outward():
    c = oed.completeness(oed.R_BINS)
    assert c[0] > c[-1]                          # core more complete than outskirts
    assert jnp.all((c > 0) & (c <= 1.0))


# --- Task 2.5: dimensionless (ln-theta) Fisher (ADR 0011) ---


def test_fisher_dimensionless_well_conditioned():
    th = oed.theta_truth()
    Mb, _ = oed.per_star_blocks(th, oed.R_BINS, oed.EPS, STELLAR.G)
    cb = oed.completeness(oed.R_BINS)
    z = jnp.zeros(3 * oed.R_BINS.shape[0])
    F = oed.fisher(z, Mb, cb, 4000.0, oed.PRIOR_DIAG)
    assert jnp.linalg.cond(F) < 1e3                       # dimensionless (was ~1.7e9 raw)
    frac_sigma_ra = jnp.linalg.inv(F)[0, 0] ** 0.5        # FRACTIONAL precision on r_a
    assert 0.01 < frac_sigma_ra < 0.5                     # ~0.12 at the working mock


# --- Task 3: c / D / A optimality criteria + AD-vs-FD gradient gate ---


def _crits():
    th = oed.theta_truth()
    Mb, _ = oed.per_star_blocks(th, oed.R_BINS, oed.EPS, STELLAR.G)
    cb = oed.completeness(oed.R_BINS)
    return Mb, cb


def test_criteria_values_positive():
    Mb, cb = _crits()
    z = jnp.zeros(3 * oed.R_BINS.shape[0])
    F = oed.fisher(z, Mb, cb, 2000.0, oed.PRIOR_DIAG)
    assert oed.c_criterion(F) > 0                       # marginal var of r_a
    assert jnp.isfinite(oed.d_criterion(F))             # -logdet
    assert oed.a_criterion(F) > 0                       # tr F^-1


def test_criteria_grads_AD_vs_FD():
    Mb, cb = _crits()
    z = jax.random.normal(jax.random.PRNGKey(0), (3 * oed.R_BINS.shape[0],)) * 0.1
    for crit in (oed.c_criterion, oed.d_criterion, oed.a_criterion):
        loss = lambda zz: crit(oed.fisher(zz, Mb, cb, 2000.0, oed.PRIOR_DIAG))
        g_ad = jax.grad(loss)(z)
        # central FD on a few coords
        eps = 1e-5
        for i in (0, 5, 17, 31):
            zp = z.at[i].add(eps); zm = z.at[i].add(-eps)
            g_fd = (loss(zp) - loss(zm)) / (2 * eps)
            assert jnp.allclose(g_ad[i], g_fd, rtol=1e-4, atol=1e-8), (crit.__name__, i)


# --- Task 4: optax multi-start optimizer ---


def test_optimizer_reduces_c_criterion():
    Mb, cb = _crits()
    z0 = jnp.zeros(3 * oed.R_BINS.shape[0])
    F0 = oed.fisher(z0, Mb, cb, 2000.0, oed.PRIOR_DIAG)
    res = oed.optimize_design(oed.c_criterion, Mb, cb, 2000.0,
                              key=jax.random.PRNGKey(1), n_starts=4, n_steps=300)
    Fopt = oed.fisher(res.z, Mb, cb, 2000.0, oed.PRIOR_DIAG)
    assert oed.c_criterion(Fopt) < oed.c_criterion(F0)   # design beats uniform
    assert res.trace[-1] <= res.trace[0]


def test_optimizer_allocation_normalized():
    Mb, cb = _crits()
    res = oed.optimize_design(oed.c_criterion, Mb, cb, 2000.0,
                              key=jax.random.PRNGKey(2), n_starts=2, n_steps=100)
    n = 2000.0 * jax.nn.softmax(res.z)
    assert jnp.allclose(jnp.sum(n), 2000.0, rtol=1e-6)   # budget conserved (pre-completeness)


# --- Task 5: sky projection + calibration ensemble (the gate) ---


def test_project_to_sky_components():
    pos = jnp.array([[3.0, 0.0, 1.0], [0.0, 2.0, -1.0]])
    vel = jnp.array([[0.0, 5.0, 7.0], [3.0, 0.0, -2.0]])
    R, v_los, v_pm_r, v_pm_t = oed.project_to_sky(pos, vel)
    assert jnp.allclose(R, jnp.array([3.0, 2.0]))
    assert jnp.allclose(v_los, jnp.array([7.0, -2.0]))      # = v_z
    # star 1 at phi=0: pm_r = vx, pm_t = vy
    assert jnp.allclose(v_pm_r[0], 0.0) and jnp.allclose(v_pm_t[0], 5.0)


@pytest.mark.slow
def test_fisher_calibration_matches_realized_scatter():
    """Realized Var(r_a_hat) over mock draws ~ (F^-1)_{r_a,r_a} at the uniform design."""
    cal = oed.calibrate_fisher(z=jnp.zeros(3 * oed.R_BINS.shape[0]),
                               N_total=4000.0, n_draws=64, key=jax.random.PRNGKey(7))
    # tolerance set by MC error on a variance from 64 draws (~sqrt(2/64)~18%): allow 35%
    assert jnp.abs(cal.realized_var_ra - cal.fisher_var_ra) / cal.fisher_var_ra < 0.35

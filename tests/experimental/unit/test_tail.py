"""Soft-sigmoid dense-tail mask (spec §3.6).

Tail membership is a differentiable sigmoid w = σ(κ(s − s_t)); the mass-weighted
tail fraction f_tail_actual = Σ w ρ / Σ ρ must, in the sharp limit κ→∞, equal the
mass fraction above s_t (the quantity AC6 compares to BM19 f_dense).
"""

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.experimental


def test_tail_weight_half_at_transition():
    """w(s_t) = σ(0) = 0.5 exactly."""
    from gravoturb_fdf.field.tail import tail_weights

    assert float(tail_weights(jnp.array(2.0), s_t=2.0, kappa=8.0)) == pytest.approx(0.5)


def test_tail_weight_monotone_increasing():
    """w increases monotonically with s."""
    from gravoturb_fdf.field.tail import tail_weights

    s = jnp.linspace(-3.0, 6.0, 50)
    w = tail_weights(s, s_t=1.5, kappa=4.0)
    assert jnp.all(jnp.diff(w) > 0)
    assert float(w[0]) < 0.05 and float(w[-1]) > 0.95


def test_f_tail_actual_sharp_limit_equals_mass_fraction():
    """κ→∞: f_tail_actual → Σ_{s>s_t} ρ / Σ ρ (hard mass fraction above s_t)."""
    from gravoturb_fdf.field.tail import f_tail_actual

    s = jnp.array([0.0, 1.0, 2.0, 3.0])
    rho = jnp.exp(s)
    s_t = 1.5
    hard = float(jnp.sum(jnp.where(s > s_t, rho, 0.0)) / jnp.sum(rho))
    soft = float(f_tail_actual(s, rho, s_t=s_t, kappa=200.0))
    assert soft == pytest.approx(hard, abs=1e-3)


def test_f_tail_actual_matches_definition():
    """f_tail_actual reproduces Σ w ρ / Σ ρ on a known array."""
    from gravoturb_fdf.field.tail import f_tail_actual, tail_weights

    s = jnp.array([-1.0, 0.5, 2.0, 4.0])
    rho = jnp.exp(s)
    w = tail_weights(s, s_t=1.0, kappa=3.0)
    expected = float(jnp.sum(w * rho) / jnp.sum(rho))
    assert float(f_tail_actual(s, rho, s_t=1.0, kappa=3.0)) == pytest.approx(expected)


def test_f_tail_actual_differentiable():
    """f_tail_actual is differentiable in the soft-mask parameters."""
    from gravoturb_fdf.field.tail import f_tail_actual

    s = jnp.linspace(-2.0, 5.0, 200)
    rho = jnp.exp(s)
    g_kappa = float(jax.grad(lambda k: f_tail_actual(s, rho, 1.5, k))(4.0))
    g_st = float(jax.grad(lambda t: f_tail_actual(s, rho, t, 4.0))(1.5))
    assert jnp.isfinite(g_kappa) and jnp.isfinite(g_st)
    assert g_st < 0.0  # raising the threshold removes mass from the tail

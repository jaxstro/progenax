"""Tests for the per-channel scalar reductions (design D4: params->IC audit).

Each reduction maps an IC array to a finite scalar, is sensitive to its own
physical channel (radius / speed / mass), and is differentiable (these feed
``jax.grad`` in the audit engine) — including a finite gradient at zero thanks
to the ``+ 1e-30`` guard inside the ``sqrt``.
"""
import jax
import jax.numpy as jnp
import pytest

import progenax  # noqa: F401  (enables float64 at import time)

from tests.validation.grad_audit.reductions import (
    mean_radius,
    mean_speed,
    mean_mass,
    identity_sum,
)


# --- (a) each reduction returns a finite scalar --------------------------------

def test_mean_radius_finite_scalar():
    out = mean_radius(jnp.ones((10, 3)))
    assert out.shape == ()
    assert jnp.isfinite(out)


def test_mean_speed_finite_scalar():
    out = mean_speed(jnp.ones((10, 3)))
    assert out.shape == ()
    assert jnp.isfinite(out)


def test_mean_mass_finite_scalar():
    out = mean_mass(jnp.ones((10,)))
    assert out.shape == ()
    assert jnp.isfinite(out)


def test_identity_sum_finite_scalar():
    out = identity_sum(jnp.arange(5.0))
    assert out.shape == ()
    assert jnp.isfinite(out)


# --- (b) each reduction scales with its own channel ----------------------------

def test_mean_radius_scales_with_positions():
    pos = jnp.ones((10, 3))
    assert float(mean_radius(2.0 * pos)) == pytest.approx(2.0 * float(mean_radius(pos)))


def test_mean_speed_scales_with_velocities():
    vel = jnp.ones((10, 3))
    assert float(mean_speed(3.0 * vel)) == pytest.approx(3.0 * float(mean_speed(vel)))


def test_mean_mass_scales_with_masses():
    masses = jnp.linspace(0.5, 5.0, 10)
    assert float(mean_mass(4.0 * masses)) == pytest.approx(4.0 * float(mean_mass(masses)))


def test_identity_sum_value():
    x = jnp.arange(5.0)  # 0 + 1 + 2 + 3 + 4 = 10
    assert float(identity_sum(x)) == pytest.approx(10.0)


# --- (c) the 1e-30 guard: finite value (and gradient) on zeros -----------------

def test_mean_radius_finite_on_zeros():
    out = mean_radius(jnp.zeros((5, 3)))
    assert jnp.isfinite(out)


def test_mean_speed_finite_on_zeros():
    out = mean_speed(jnp.zeros((5, 3)))
    assert jnp.isfinite(out)


def test_mean_radius_gradient_finite_on_zeros():
    """The 1e-30 guard must keep d(mean_radius)/d(positions) finite at zero
    (a bare sqrt(0) would give a NaN gradient)."""
    g = jax.grad(lambda p: mean_radius(p))(jnp.zeros((5, 3)))
    assert jnp.all(jnp.isfinite(g))


def test_mean_speed_gradient_finite_on_zeros():
    g = jax.grad(lambda v: mean_speed(v))(jnp.zeros((5, 3)))
    assert jnp.all(jnp.isfinite(g))

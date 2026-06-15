"""Unit tests for the Tout+1996 ZAMS stellar relations (progenax.stellar).

Anchors are the PDF-verified solar (M=1, Z=0.02) values from
docs/core-papers/tout1996_zams_coefficients_verified.md:
    L = 0.6977 Lsun, R = 0.8882 Rsun, T_eff ~ 5600 K, log g ~ 4.54.
"""
import jax
import jax.numpy as jnp
import pytest

import progenax  # noqa: F401 — enables float64 on import
from progenax.stellar import (
    zams_luminosity,
    zams_radius,
    zams_effective_temperature,
    zams_surface_gravity,
    inverse_zams_luminosity,
)


class TestZAMSLuminosity:
    def test_sun_anchor(self):
        # Tout+1996 Sun ZAMS L = 0.6977 Lsun (verified P1)
        assert zams_luminosity(jnp.array(1.0)) == pytest.approx(0.698, abs=0.01)

    def test_broadcasts_over_array(self):
        L = zams_luminosity(jnp.array([0.5, 1.0, 10.0]))
        assert L.shape == (3,)
        assert jnp.all(L[1:] > L[:-1])  # monotone increasing

    def test_differentiable(self):
        g = jax.grad(lambda m: zams_luminosity(m))(jnp.array(1.0))
        assert jnp.isfinite(g) and g > 0


class TestZAMSRadius:
    def test_sun_anchor(self):
        # Tout+1996 Sun ZAMS R = 0.8882 Rsun (verified P1)
        assert zams_radius(jnp.array(1.0)) == pytest.approx(0.888, abs=0.01)

    def test_broadcasts_over_array(self):
        R = zams_radius(jnp.array([0.5, 1.0, 10.0]))
        assert R.shape == (3,)
        assert jnp.all(R > 0)
        # 10 Msun ZAMS radius is a few Rsun
        assert 2.0 < R[2] < 8.0

    def test_differentiable(self):
        g = jax.grad(lambda m: zams_radius(m))(jnp.array(1.0))
        assert jnp.isfinite(g)


class TestZAMSEffectiveTemperature:
    def test_sun_anchor(self):
        # Stefan-Boltzmann from the verified L,R -> ~5600 K (pinned with abs=150)
        teff = zams_effective_temperature(jnp.array(1.0))
        assert teff == pytest.approx(5600.0, abs=150.0)

    def test_broadcasts_over_array(self):
        teff = zams_effective_temperature(jnp.array([0.5, 1.0, 10.0]))
        assert teff.shape == (3,)
        assert jnp.all(teff > 0)
        assert jnp.all(teff[1:] > teff[:-1])  # hotter at higher mass

    def test_differentiable(self):
        g = jax.grad(lambda m: zams_effective_temperature(m))(jnp.array(1.0))
        assert jnp.isfinite(g)


class TestZAMSSurfaceGravity:
    def test_sun_anchor(self):
        # log g = log10(G M / R^2) in cgs -> ~4.54 dex for the ZAMS Sun
        logg = zams_surface_gravity(jnp.array(1.0))
        assert logg == pytest.approx(4.54, abs=0.1)

    def test_broadcasts_over_array(self):
        logg = zams_surface_gravity(jnp.array([0.5, 1.0, 10.0]))
        assert logg.shape == (3,)
        assert jnp.all(jnp.isfinite(logg))

    def test_differentiable(self):
        g = jax.grad(lambda m: zams_surface_gravity(m))(jnp.array(1.0))
        assert jnp.isfinite(g)


class TestInverseZAMSLuminosity:
    def test_round_trip(self):
        m = jnp.array([0.5, 1.0, 5.0, 20.0])
        m_rec = inverse_zams_luminosity(zams_luminosity(m))
        assert jnp.allclose(m_rec, m, rtol=1e-5)

    def test_scalar_in_scalar_out(self):
        L = zams_luminosity(jnp.array(1.0))
        m = inverse_zams_luminosity(L)
        assert jnp.ndim(m) == 0
        assert m == pytest.approx(1.0, rel=1e-5)

    def test_differentiable(self):
        g = jax.grad(lambda L: inverse_zams_luminosity(L)[0])(jnp.array([100.0]))
        assert jnp.isfinite(g)

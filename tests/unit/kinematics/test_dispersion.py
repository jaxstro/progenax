"""Unit tests for the differentiable dispersion forward models.

Phase 0 Task 1: scaffold — exports present + NamedTuple field layout.
Phase 0 Task 2: jeans_dispersion 3-D — isotropic closed form, GM scaling
invariants, r_a domain guard, jit smoke.
"""

import jax
import jax.numpy as jnp
import progenax
from progenax import jeans_dispersion, project_dispersion
from progenax.kinematics.dispersion import DispersionProfile, ProjectedDispersion
from progenax.profiles import PlummerProfile


def test_exports_and_namedtuples():
    assert {"jeans_dispersion", "project_dispersion"} <= set(progenax.__all__)
    assert DispersionProfile._fields == ("r", "sigma_r", "sigma_t", "sigma_1d", "beta")
    assert ProjectedDispersion._fields == ("R", "sigma_los", "sigma_pm_r", "sigma_pm_t", "Sigma")


def test_plummer_isotropic_closed_form():
    prof = PlummerProfile(r_h=1.0)
    r = jnp.array([0.3, 0.7, 1.0, 2.0])
    dp = jeans_dispersion(prof, None, r, M=400.0, G=0.00449)
    truth = jnp.sqrt(0.00449 * 400.0 / (6.0 * jnp.sqrt(r**2 + prof.a**2)))
    assert jnp.allclose(dp.sigma_1d, truth, rtol=3e-3)
    assert jnp.allclose(dp.beta, 0.0, atol=1e-10)
    assert jnp.allclose(dp.sigma_r, dp.sigma_t, rtol=1e-6)


def test_gm_scaling_invariants():
    prof = PlummerProfile(r_h=1.0)
    r = jnp.array([1.0])
    s1 = jeans_dispersion(prof, 2.0, r, 400.0, 0.00449).sigma_r
    assert jnp.allclose(
        jeans_dispersion(prof, 2.0, r, 800.0, 0.00449).sigma_r**2, 2 * s1**2, rtol=1e-4
    )
    assert jnp.allclose(
        jeans_dispersion(prof, 2.0, r, 400.0, 2 * 0.00449).sigma_r**2, 2 * s1**2, rtol=1e-4
    )


def test_r_a_domain_guard():
    import pytest

    prof = PlummerProfile(r_h=1.0)
    with pytest.raises(ValueError):  # r_a < 0.75 a is unphysical for Plummer OM
        jeans_dispersion(prof, 0.1 * prof.a, jnp.array([1.0]), 400.0, 0.00449)


def test_jit_smoke():
    prof = PlummerProfile(r_h=1.0)
    f = jax.jit(lambda ra: jeans_dispersion(prof, ra, jnp.array([1.0]), 400.0, 0.00449).sigma_r)
    assert jnp.isfinite(f(2.0)).all()

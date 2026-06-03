# tests/unit/profiles/test_profile_api.py
"""Functional profile API: factory + sampling + analytic potentials for all 3 profiles."""
import jax
import jax.numpy as jnp
import pytest
from jaxstro.units import STELLAR
from progenax.profiles.api import make_profile, sample_density_profile, compute_profile_potential
from progenax.profiles.plummer import PlummerProfile
from progenax.profiles.king import KingProfile
from progenax.profiles.eff import EFFProfile


@pytest.mark.parametrize("name,cls,kw", [
    ("plummer", PlummerProfile, {}),
    ("king", KingProfile, {"W0": 7.0}),
    ("eff", EFFProfile, {"gamma": 3.0, "r_t": 15.0}),
])
def test_make_profile_dispatch(name, cls, kw):
    p = make_profile(name, R_half=1.0, **kw)
    assert isinstance(p, cls)


def test_make_profile_unknown_raises():
    with pytest.raises(ValueError, match="Unknown profile"):
        make_profile("hernquist", R_half=1.0)


@pytest.mark.parametrize("name,kw", [
    ("plummer", {}), ("king", {"W0": 7.0}), ("eff", {"gamma": 3.0, "r_t": 15.0}),
])
def test_sample_density_profile_shape(name, kw):
    pos = sample_density_profile(jax.random.PRNGKey(0), 200, name, R_half=1.0, **kw)
    assert pos.shape == (200, 3) and jnp.all(jnp.isfinite(pos))


@pytest.mark.parametrize("name,kw", [
    ("plummer", {}), ("king", {"W0": 7.0}),
    ("eff", {"gamma": 3.0, "r_t": 15.0}), ("eff", {"gamma": 4.0, "r_t": 15.0}),
])
def test_compute_profile_potential_negative_and_finite(name, kw):
    pos = jnp.array([[0.5, 0.0, 0.0], [0.0, 1.5, 0.0], [2.0, 0.0, 0.0]])
    phi = compute_profile_potential(pos, name, M_total=1000.0, R_half=1.0, G=STELLAR.G, **kw)
    assert phi.shape == (3,)
    assert jnp.all(phi < 0) and jnp.all(jnp.isfinite(phi)), f"{name}: {phi}"


def test_compute_profile_potential_unknown_raises():
    with pytest.raises(ValueError, match="Unknown profile"):
        compute_profile_potential(jnp.zeros((2, 3)), "hernquist", M_total=1.0, R_half=1.0, G=STELLAR.G)

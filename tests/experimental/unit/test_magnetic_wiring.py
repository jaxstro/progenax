"""W1: MagneticSpec + build_cluster_ic(magnetic=...) scaffold and coupling guards (ADR-0060).

magnetic=None ⇒ byte-identical legacy path (enforced by the existing byte-identity pins in the
full suite). When set, μ_Φ-primary magnetism is a gas-cloud property: it requires
VelocitySpec(mode='physical') + a GasSpec (so M_cl = M_star/sfe and c_s exist) — decision (a).
"""

import jax
import jax.numpy as jnp
import pytest

from gravoturb.cluster import build_cluster_ic
from gravoturb.specs import (
    CloudSpec,
    CompositionSpec,
    GasSpec,
    GeometrySpec,
    MagneticSpec,
    VelocitySpec,
)
from jaxstro.units import STELLAR

from progenax import PlummerProfile

pytestmark = [pytest.mark.experimental, pytest.mark.unit]


# --------------------------------------------------------------------------- #
# MagneticSpec constructor validation
# --------------------------------------------------------------------------- #
def test_magnetic_spec_valid_and_pytree():
    m = MagneticSpec(mu_phi=2.0)
    assert float(m.mu_phi) == 2.0
    assert m.realize == "field"            # default: full RMHD-ready vector field
    assert m.mean_field_axis == 2          # default z (Extension-A line of sight)
    assert m.anisotropy == "theory"        # default: sourced Hu & Lazarian closure


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(mu_phi=0.0), "mu_phi"),
        (dict(mu_phi=-1.0), "mu_phi"),
        (dict(mu_phi=1.0, realize="bogus"), "realize"),
        (dict(mu_phi=1.0, mean_field_axis=3), "mean_field_axis"),
        (dict(mu_phi=1.0, field_slope=0.0), "field_slope"),
        (dict(mu_phi=1.0, anisotropy="bogus"), "anisotropy"),
        (dict(mu_phi=1.0, anisotropy="fixed"), "anisotropy_value"),   # fixed needs a value
        (dict(mu_phi=1.0, anisotropy="theory", anisotropy_value=3.0), "anisotropy_value"),
    ],
)
def test_magnetic_spec_rejects_bad_physics(kwargs, match):
    with pytest.raises((ValueError, TypeError), match=match):
        MagneticSpec(**kwargs)


def test_magnetic_spec_fixed_anisotropy_ok():
    m = MagneticSpec(mu_phi=1.0, anisotropy="fixed", anisotropy_value=2.5)
    assert float(m.anisotropy_value) == 2.5


# --------------------------------------------------------------------------- #
# (a) coupling guard at the builder boundary
# --------------------------------------------------------------------------- #
def _masses():
    return jnp.linspace(0.5, 5.0, 40)


def _physical_gas_kwargs():
    return dict(
        cloud=CloudSpec(mach=8.0, b=0.5, alpha=1.8, beta=3.0),
        geometry=GeometrySpec(profile=PlummerProfile(r_h=1.0), box_size=4.0, shape=(16, 16, 16)),
        velocity=VelocitySpec(beta_v=4.0, mode="physical", c_s=0.2),
        composition=CompositionSpec(),
        gas=GasSpec(sfe=0.3),
        G=STELLAR.G, units=STELLAR, key=jax.random.PRNGKey(0),
    )


def test_magnetic_requires_physical_velocity_mode():
    kw = _physical_gas_kwargs()
    kw["velocity"] = VelocitySpec(beta_v=4.0, Q_target=0.5)  # virial_target
    kw["gas"] = None
    with pytest.raises(ValueError, match="physical"):
        build_cluster_ic(_masses(), magnetic=MagneticSpec(mu_phi=2.0), **kw)


def test_magnetic_requires_gas():
    kw = _physical_gas_kwargs()
    kw["gas"] = None
    with pytest.raises(ValueError, match="gas"):
        build_cluster_ic(_masses(), magnetic=MagneticSpec(mu_phi=2.0), **kw)


def test_magnetic_none_builds_and_matches_absent():
    # magnetic=None must run the untouched legacy path (byte-identity backstop; the pinned
    # hashes in the full suite are the exhaustive gate).
    kw = _physical_gas_kwargs()
    a = build_cluster_ic(_masses(), **kw)
    b = build_cluster_ic(_masses(), magnetic=None, **kw)
    for la, lb in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        if isinstance(la, str):          # Physics carries string leaves (coupling, mode, ...)
            assert la == lb
        else:
            assert jnp.array_equal(la, lb)

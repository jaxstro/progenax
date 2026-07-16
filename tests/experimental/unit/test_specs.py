"""Constructor validation of the gravoturb parameter specs (Phase 0.5).

Bad physics fails loudly at spec construction — never deep inside the pipeline.
"""

import pytest
from gravoturb.specs import CloudSpec, CompositionSpec, GeometrySpec, VelocitySpec

from progenax import PlummerProfile

pytestmark = [pytest.mark.experimental, pytest.mark.unit]


def test_cloud_spec_valid_and_pytree():
    c = CloudSpec(mach=8.0, b=0.5, alpha=1.8, beta=3.0)
    assert float(c.mach) == 8.0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(mach=-1.0, b=0.5, alpha=1.8, beta=3.0), "mach"),
        (dict(mach=8.0, b=0.0, alpha=1.8, beta=3.0), "b must"),
        (dict(mach=8.0, b=1.5, alpha=1.8, beta=3.0), "b must"),
        (dict(mach=8.0, b=0.5, alpha=1.0, beta=3.0), "alpha"),
        (dict(mach=8.0, b=0.5, alpha=1.8, beta=0.0), "beta"),
    ],
)
def test_cloud_spec_rejects_bad_physics(kwargs, match):
    with pytest.raises(ValueError, match=match):
        CloudSpec(**kwargs)


def test_geometry_spec_guards():
    GeometrySpec(profile=PlummerProfile(r_h=0.5), box_size=4.0, shape=(16, 16, 16))
    with pytest.raises(ValueError, match="box_size"):
        GeometrySpec(profile=PlummerProfile(r_h=0.5), box_size=0.0, shape=(16, 16, 16))
    with pytest.raises(ValueError, match="shape"):
        GeometrySpec(profile=PlummerProfile(r_h=0.5), box_size=4.0, shape=(4, 16, 16))
    with pytest.raises(TypeError, match="SpatialProfile"):
        GeometrySpec(profile=object(), box_size=4.0, shape=(16, 16, 16))


def test_velocity_spec_guards():
    VelocitySpec(beta_v=4.0, Q_target=0.5)
    VelocitySpec(beta_v=4.0, Q_target=0.0)  # cold collapse (v=0 IC) is physical
    with pytest.raises(ValueError, match="Q_target"):
        VelocitySpec(beta_v=4.0, Q_target=-0.1)
    with pytest.raises(ValueError, match="beta_v"):
        VelocitySpec(beta_v=-1.0, Q_target=0.5)


def test_specs_constructible_with_traced_values():
    """Traced construction (vmap/jit/grad sweeps) is supported: value checks skip
    tracers (parity with the pre-spec flat signature) — review regression fix."""
    import jax
    import jax.numpy as jnp

    machs = jax.vmap(lambda m: CloudSpec(mach=m, b=0.5, alpha=1.8, beta=3.0).mach)(
        jnp.array([4.0, 8.0]))
    assert machs.shape == (2,)
    g = jax.grad(lambda q: VelocitySpec(beta_v=4.0, Q_target=q).Q_target * 2.0)(0.5)
    assert float(g) == 2.0


def test_velocity_spec_physical_mode_valid():
    """Phase 2: mode='physical' takes c_s [km/s] + eta_v (default 1); Q becomes emergent."""
    v = VelocitySpec(beta_v=4.0, mode="physical", c_s=0.2)
    assert v.mode == "physical"
    assert v.Q_target is None
    assert float(v.c_s) == 0.2
    assert float(v.eta_v) == 1.0  # default: sigma_star = mach * c_s
    v2 = VelocitySpec(beta_v=4.0, mode="physical", c_s=0.2, eta_v=0.5)
    assert float(v2.eta_v) == 0.5


@pytest.mark.parametrize(
    "kwargs, match",
    [
        # no silent precedence: Q_target + physical mode is a constructor error
        (dict(beta_v=4.0, Q_target=0.5, mode="physical", c_s=0.2), "Q_target"),
        (dict(beta_v=4.0, mode="physical"), "c_s"),          # physical requires c_s
        (dict(beta_v=4.0, mode="physical", c_s=-0.2), "c_s"),
        (dict(beta_v=4.0, mode="physical", c_s=0.2, eta_v=0.0), "eta_v"),
        (dict(beta_v=4.0, Q_target=0.5, c_s=0.2), "c_s"),    # physical-mode knob leaks
        (dict(beta_v=4.0, Q_target=0.5, eta_v=0.8), "eta_v"),
        (dict(beta_v=4.0), "Q_target"),                       # virial_target requires Q
        (dict(beta_v=4.0, Q_target=0.5, mode="bogus"), "mode"),
    ],
)
def test_velocity_spec_mode_guards(kwargs, match):
    with pytest.raises(ValueError, match=match):
        VelocitySpec(**kwargs)


def test_velocity_spec_physical_traced_construction():
    """c_s / eta_v are differentiable leaves: traced construction supported (main parity)."""
    import jax

    g = jax.grad(
        lambda c: VelocitySpec(beta_v=4.0, mode="physical", c_s=c).c_s * 3.0
    )(0.2)
    assert float(g) == 3.0


def test_cloud_spec_from_larson_closure():
    """Cloud-level inputs (M_ecl, SFE, rho_cl) close through the released Larson chain:
    mach from sigma_v(R_cloud)/c_s, beta from Kim & Ryu, b from environment, box = 2 R_cloud."""
    import jax.numpy as jnp
    from gravoturb.specs import cloud_spec_from_larson

    from progenax.cluster.turbulence import (
        b_from_environment,
        cloud_radius_from_density,
        spectral_slope_from_mach,
        turbulent_mach_from_cloud,
    )

    M_ecl, sfe, rho_cl, alpha = 1000.0, 0.3, 100.0, 1.8
    cloud, box_size = cloud_spec_from_larson(M_ecl=M_ecl, sfe=sfe, rho_cl=rho_cl, alpha=alpha)

    R = cloud_radius_from_density(jnp.asarray(M_ecl), sfe, rho_cl)
    assert float(box_size) == pytest.approx(2.0 * float(R), rel=1e-12)
    assert float(cloud.mach) == pytest.approx(float(turbulent_mach_from_cloud(R)), rel=1e-12)
    assert float(cloud.beta) == pytest.approx(
        float(spectral_slope_from_mach(cloud.mach)), rel=1e-12)
    assert float(cloud.b) == pytest.approx(
        float(b_from_environment(jnp.log10(rho_cl))), rel=1e-12)
    assert float(cloud.alpha) == alpha
    # explicit b overrides the environment mapping
    cloud_b, _ = cloud_spec_from_larson(M_ecl=M_ecl, sfe=sfe, rho_cl=rho_cl, alpha=alpha, b=0.4)
    assert float(cloud_b.b) == 0.4


def test_composition_spec_guards():
    CompositionSpec()  # multi_freefall default, f_sub derived
    CompositionSpec(placement="two_population", f_sub=0.3)
    assert float(CompositionSpec().mask_sharpness) == 8.0
    with pytest.raises(ValueError, match="f_sub"):
        CompositionSpec(placement="two_population", f_sub=1.2)
    with pytest.raises(ValueError, match="mask_sharpness"):
        CompositionSpec(mask_sharpness=0.0)


def test_top_level_api_exports():
    import gravoturb

    for name in ("build_cluster_ic", "ClusterIC", "CloudSpec", "GeometrySpec",
                 "VelocitySpec", "CompositionSpec"):
        assert hasattr(gravoturb, name)

"""Constructor validation of the gravoturb parameter specs (Phase 0.5).

Bad physics fails loudly at spec construction — never deep inside the pipeline.
"""

import pytest

from progenax import PlummerProfile

from gravoturb.specs import CloudSpec, CompositionSpec, GeometrySpec, VelocitySpec

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
    with pytest.raises(ValueError, match="Q_target"):
        VelocitySpec(beta_v=4.0, Q_target=0.0)
    with pytest.raises(ValueError, match="beta_v"):
        VelocitySpec(beta_v=-1.0, Q_target=0.5)


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

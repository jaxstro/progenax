"""Conformance tests for the composition protocols (Batch 0, F6).

Guards the protocol-based composition architecture: every shipped profile /
velocity-DF / IMF must satisfy its protocol, and the protocols must actually
discriminate (reject a non-conforming class). SpatialProfile and VelocityDF are
method-only protocols (so ``issubclass`` works); IMFProtocol has data members
(``m_min``/``m_max``) so it requires instance checks.
"""

import pytest

import progenax.imf as IMF
from progenax import (
    EFFProfile,
    EFFVelocityDF,
    KingProfile,
    KingVelocityDF,
    PlummerProfile,
    PlummerVelocityDF,
)
from progenax.protocols import IMFProtocol, SpatialProfile, VelocityDF


@pytest.mark.parametrize("cls", [PlummerProfile, KingProfile, EFFProfile])
def test_spatial_profiles_conform(cls):
    assert issubclass(cls, SpatialProfile)


@pytest.mark.parametrize("cls", [PlummerVelocityDF, KingVelocityDF, EFFVelocityDF])
def test_velocity_dfs_conform(cls):
    assert issubclass(cls, VelocityDF)


def test_velocity_df_protocol_single_source():
    """Audit A1: kinematics.api must re-export the canonical protocols.VelocityDF,
    not a structural duplicate (which would silently drift)."""
    from progenax.kinematics.api import VelocityDF as ApiExport
    from progenax.protocols import VelocityDF as Canonical

    assert ApiExport is Canonical


@pytest.mark.parametrize(
    "imf",
    [
        IMF.Maschberger(),
        IMF.ChabrierIMF(),
        IMF.PowerLawIMF(exponents=[2.35], breakpoints=[], m_min=0.1, m_max=100.0),
        IMF.TaperedPowerLaw(),
        IMF.Schechter(),
    ],
)
def test_imfs_conform(imf):
    assert isinstance(imf, IMFProtocol)


def test_plummer_instances_conform():
    assert isinstance(PlummerProfile(r_h=1.0), SpatialProfile)
    assert isinstance(PlummerVelocityDF(r_h=1.0), VelocityDF)


def test_protocols_reject_nonconforming():
    """Discrimination: the protocols are not vacuously true."""

    class NotAProfile:  # missing sample_positions / characteristic_radius
        pass

    class NotAnIMF:  # has the data members but none of the methods
        m_min = 0.1
        m_max = 1.0

    assert not issubclass(NotAProfile, SpatialProfile)
    assert not isinstance(NotAnIMF(), IMFProtocol)

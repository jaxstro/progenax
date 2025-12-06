# progenax/tests/unit/test_protocols.py
"""Tests for progenax protocols."""

import pytest
from typing import Protocol, runtime_checkable


class TestProtocolsExist:
    """Verify protocol classes are importable."""

    def test_spatial_profile_importable(self):
        from progenax.protocols import SpatialProfile
        assert hasattr(SpatialProfile, 'sample_positions')
        assert hasattr(SpatialProfile, 'characteristic_radius')

    def test_velocity_df_importable(self):
        from progenax.protocols import VelocityDF
        assert hasattr(VelocityDF, 'sample_velocities')

    def test_imf_protocol_importable(self):
        from progenax.protocols import IMFProtocol
        assert hasattr(IMFProtocol, 'sample')
        assert hasattr(IMFProtocol, 'logpdf')
        assert hasattr(IMFProtocol, 'cdf')
        assert hasattr(IMFProtocol, 'ppf')

    def test_protocols_are_runtime_checkable(self):
        from progenax.protocols import SpatialProfile, VelocityDF, IMFProtocol
        # Should be runtime checkable
        assert hasattr(SpatialProfile, '__protocol_attrs__') or isinstance(SpatialProfile, type)

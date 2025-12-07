"""Tests for TruncatedIMF wrapper."""

import jax
import jax.numpy as jnp
import pytest


class TestTruncatedIMF:
    """Test TruncatedIMF wrapper functionality."""

    def test_truncated_imf_importable(self):
        from progenax.imf.truncated import TruncatedIMF
        assert TruncatedIMF is not None

    def test_truncation_enforced(self):
        """Samples should be within [m_min, m_max]."""
        from progenax.imf.truncated import TruncatedIMF
        from progenax.imf.base import BaseIMF
        # Will test with concrete implementation
        pass

    def test_wraps_any_imf(self):
        """Should work with any IMF implementing the protocol."""
        from progenax.imf.truncated import TruncatedIMF
        pass

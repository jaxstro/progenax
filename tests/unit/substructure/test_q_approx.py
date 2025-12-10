"""Tests for JAX-native Q parameter approximation."""

import pytest


class TestImports:
    def test_imports(self):
        """Module should be importable."""
        from progenax.diagnostics.q_approx import (
            q_approx_naive,
            q_approx_fast,
            q_approx,
            DEFAULT_CALIBRATION,
        )
        assert callable(q_approx_naive)
        assert callable(q_approx_fast)
        assert callable(q_approx)

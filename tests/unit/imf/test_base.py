"""Tests for IMF base infrastructure."""

import jax
import jax.numpy as jnp
import pytest


class TestBaseIMFExists:
    """Test BaseIMF is importable and has required methods."""

    def test_base_imf_importable(self):
        from progenax.imf.base import BaseIMF
        assert BaseIMF is not None

    def test_has_required_methods(self):
        from progenax.imf.base import BaseIMF
        assert hasattr(BaseIMF, 'logpdf')
        assert hasattr(BaseIMF, 'cdf')
        assert hasattr(BaseIMF, 'ppf')
        assert hasattr(BaseIMF, 'sample')
        assert hasattr(BaseIMF, 'mean_mass')


class TestPPFNewton:
    """Test Newton PPF solver with custom_jvp."""

    def test_ppf_newton_exists(self):
        from progenax.imf.base import _ppf_newton
        assert callable(_ppf_newton)

    def test_ppf_is_differentiable(self):
        """PPF should have gradients via custom_jvp."""
        from progenax.imf.base import BaseIMF

        # Create a simple test IMF (will need concrete implementation)
        # For now just check the infrastructure exists
        pass


class TestBaseIMFSamplingModes:
    """Test different sampling modes."""

    def test_sample_n_mode(self):
        """N mode: exactly N masses."""
        from progenax.imf.base import BaseIMF
        # Will test with concrete implementation
        pass

    def test_sample_m_total_mode(self):
        """M_total mode: masses until total mass reached."""
        from progenax.imf.base import BaseIMF
        pass

    def test_sample_fixed_n_mode(self):
        """Fixed-N mode: exactly N masses summing to M_total."""
        from progenax.imf.base import BaseIMF
        pass

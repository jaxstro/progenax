"""Tests for IMF sampling modes."""
import jax
import jax.numpy as jnp
import pytest
from progenax.imf.smooth import Maschberger


class TestSampleMTotalJIT:
    """sample_m_total should be JIT-compatible."""

    def test_sample_m_total_jittable(self, key):
        """sample_m_total can be JIT-compiled."""
        imf = Maschberger()

        @jax.jit
        def sample_wrapper(k):
            masses, n_live = imf.sample_m_total(k, m_total=100.0, n_max=500)
            return masses, n_live

        masses, n_live = sample_wrapper(key)
        assert masses.shape == (500,)
        # n_live should be a JAX array, not Python int
        assert hasattr(n_live, 'shape'), "n_live should be a JAX array"

    def test_sample_m_total_n_live_is_array(self, key):
        """n_live returned is a JAX scalar array, not Python int."""
        imf = Maschberger()
        masses, n_live = imf.sample_m_total(key, m_total=100.0, n_max=500)

        # Should be a 0-dimensional JAX array
        assert hasattr(n_live, 'shape'), "n_live should be a JAX array"
        assert n_live.shape == (), f"n_live should be scalar, got shape {n_live.shape}"

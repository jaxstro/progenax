"""Tests for gradient flow through IMF parameters."""
import jax
import jax.numpy as jnp
import pytest
from progenax.imf import ChabrierIMF, Maschberger


class TestParameterGradients:
    """Verify gradients flow through IMF parameters (not just u)."""

    def test_chabrier_grad_wrt_alpha(self):
        """Gradient w.r.t. alpha is non-zero through ppf."""
        u = jnp.array([0.3, 0.5, 0.7])

        def loss(alpha):
            imf = ChabrierIMF(alpha=alpha)
            masses = imf.ppf(u)
            return jnp.sum(masses)

        grad_val = jax.grad(loss)(2.35)
        assert jnp.isfinite(grad_val), f"Gradient is {grad_val}"
        assert jnp.abs(grad_val) > 1e-6, f"Gradient is effectively zero: {grad_val}"

    def test_chabrier_grad_wrt_sigma(self):
        """Gradient w.r.t. sigma is non-zero through ppf."""
        u = jnp.array([0.3, 0.5, 0.7])

        def loss(sigma):
            imf = ChabrierIMF(sigma=sigma)
            masses = imf.ppf(u)
            return jnp.sum(masses)

        grad_val = jax.grad(loss)(0.69)
        assert jnp.isfinite(grad_val), f"Gradient is {grad_val}"
        assert jnp.abs(grad_val) > 1e-6, f"Gradient is effectively zero: {grad_val}"

    def test_maschberger_grad_wrt_mu(self):
        """Gradient w.r.t. mu is non-zero through ppf."""
        u = jnp.array([0.3, 0.5, 0.7])

        def loss(mu):
            imf = Maschberger(mu=mu)
            masses = imf.ppf(u)
            return jnp.sum(masses)

        grad_val = jax.grad(loss)(0.2)
        assert jnp.isfinite(grad_val), f"Gradient is {grad_val}"
        assert jnp.abs(grad_val) > 1e-6, f"Gradient is effectively zero: {grad_val}"

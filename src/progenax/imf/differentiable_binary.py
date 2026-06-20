"""Differentiable binary population models for gradient-based inference.

Provides smooth-threshold relaxations of binary assignment, enabling
jax.grad to flow through binary fraction and mass-ratio parameters.

The key technique is the reparameterization trick: pre-draw fixed uniform
random numbers, then replace the hard threshold comparison with a steep
sigmoid. At low temperature, each star is essentially binary or single,
but the gradient is nonzero — concentrated on marginal stars near the
threshold.

# TODO: Future refactor — split binary.py into a binary/ subpackage
# (mass_ratio.py, fraction.py, model.py, differentiable.py).
# Not worth doing now with April 28 proposal deadline.

References:
    Moe & Di Stefano (2017) ApJS 230, 15 — mass-dependent binary statistics
    Kingma & Welling (2014) ICLR — reparameterization trick
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class DifferentiableBinaryFraction(eqx.Module):
    """Smooth mass-dependent binary fraction via logistic regression.

    Models f_b(m) = sigmoid(a + b * log10(m) + c * log10(m)^2), a smooth
    approximation to the Moe & Di Stefano (2017) step function.

    Fully differentiable w.r.t. parameters (a, b, c).

    Args:
        a: log-odds intercept
        b: log-odds linear slope with log10(mass)
        c: log-odds quadratic term (captures flattening at low mass)
    """

    a: float
    b: float
    c: float = 0.0

    def __call__(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Compute binary fraction for given mass(es).

        Args:
            m: primary mass in Msun (scalar or array)

        Returns:
            f_b in (0, 1)
        """
        log_m = jnp.log10(jnp.maximum(m, 1e-10))
        log_odds = self.a + self.b * log_m + self.c * log_m**2
        return jax.nn.sigmoid(log_odds)

    def probability(self, masses, radii=None):
        """BinaryFractionModel protocol: f_bin(masses) (radii ignored)."""
        return self(masses)

    @classmethod
    def from_moe2017(cls) -> "DifferentiableBinaryFraction":
        """Factory: a smooth quadratic-logit fit to the MassDependentBinaryFraction steps.

        Weighted least-squares fit of logit(f_b) = a + b*log10(m) + c*log10(m)^2 to the
        MassDependentBinaryFraction step values (the Moe+2017 Table 13-derived multiplicity
        fractions; see that class). The quadratic term captures the low-mass flattening.
        Agreement is ≲3% at representative (bin-centre) masses, rising to ~6% near the step
        discontinuities (a continuous curve cannot match a step exactly at its edge).

        Returns:
            DifferentiableBinaryFraction with the fitted (a, b, c)
        """
        return cls(a=-0.2799, b=1.4170, c=0.4755)


class DifferentiableBinaryModel(eqx.Module):
    """Full differentiable binary population model.

    Uses smooth-threshold relaxation for binary assignment and
    reparameterized power-law sampling for mass ratios. All parameters
    are differentiable via jax.grad.

    The model has two components:
    1. Binary fraction: f_b(m; a, b) via DifferentiableBinaryFraction
    2. Mass-ratio distribution: q ~ power-law with mass-dependent slope
       gamma(m) = c + d * log10(m)

    Pre-drawn uniform samples (u_binary, u_q) are held fixed — they
    represent the stochastic realization of a particular cluster.
    Changing the parameters changes which stars are binary and what
    companions they have, while holding the realization fixed.

    Args:
        binary_fraction: DifferentiableBinaryFraction instance
        gamma_intercept: power-law q slope intercept (c)
        gamma_slope: power-law q slope mass dependence (d)
        temperature: sigmoid sharpness (lower = closer to hard threshold)
    """

    binary_fraction: DifferentiableBinaryFraction
    gamma_intercept: float
    gamma_slope: float
    temperature: float

    def sample_systems(
        self,
        m1: Float[Array, "N"],
        u_binary: Float[Array, "N"],
        u_q: Float[Array, "N"],
    ) -> tuple:
        """Compute secondary masses and soft binary weights.

        Uses the reparameterization trick: fixed uniform draws are
        transformed through differentiable functions of the parameters.

        Args:
            m1: primary masses (N,)
            u_binary: fixed uniform draws for binary decision (N,)
            u_q: fixed uniform draws for mass ratio (N,)

        Returns:
            m2: secondary masses (N,) — always positive (even for "singles")
            soft_weights: smooth binary indicators (N,) in [0, 1]
        """
        # Mass-dependent binary fraction
        f_b = self.binary_fraction(m1)

        # Smooth threshold (reparameterized Bernoulli)
        soft_weights = jax.nn.sigmoid((f_b - u_binary) / self.temperature)

        # Mass-dependent q distribution (reparameterized power law)
        # gamma(m) = c + d * log10(m)
        gamma = self.gamma_intercept + self.gamma_slope * jnp.log10(
            jnp.maximum(m1, 1e-10)
        )
        # Inverse CDF of power law p(q) ~ q^gamma on [0, 1]:
        # q = u^(1/(gamma+1))
        # Clamp gamma > -0.99 to avoid division by zero
        gamma_safe = jnp.maximum(gamma, -0.99)
        q = u_q ** (1.0 / (gamma_safe + 1.0))

        # Clamp q to [0.01, 1] for physical validity
        q = jnp.clip(q, 0.01, 1.0)
        m2 = q * m1

        return m2, soft_weights

    @classmethod
    def moe2017(cls, temperature: float = 0.01) -> "DifferentiableBinaryModel":
        """Factory with parameters fit to Moe+2017.

        Binary fraction: quadratic-logit fit to Table 13 via
            DifferentiableBinaryFraction.from_moe2017()
            (a=-0.2799, b=1.4170, c=0.4755).
        Mass-ratio slope: linear fit to gamma(m) (c=0.1907, d=-0.7521)

        Args:
            temperature: sigmoid sharpness. Lower = closer to hard
                threshold. Default 0.01 gives <1% marginal stars.

        Returns:
            DifferentiableBinaryModel with Moe+2017-calibrated parameters
        """
        return cls(
            binary_fraction=DifferentiableBinaryFraction.from_moe2017(),
            gamma_intercept=0.1907,
            gamma_slope=-0.7521,
            temperature=temperature,
        )

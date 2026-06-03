"""Moe & Di Stefano (2017) full mass-dependent mass-ratio model.

Split from ``binary.py`` (file-length limit). Reference: Moe & Di Stefano (2017)
ApJS 230, 15.
"""

from __future__ import annotations

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray


class MoeDiStefano2017(eqx.Module):
    """Mass-dependent mass-ratio distribution from Moe & Di Stefano (2017).

    Reference:
        Moe & Di Stefano (2017) ApJS 230, 15
        Table 10: Intrinsic mass-ratio distributions

    This implements the full mass-dependent model where:
        - γ (power-law exponent) depends on primary mass
        - f_twin (twin excess) depends on primary mass and period
        - The distribution transitions smoothly across mass ranges

    Model (simplified, period-averaged):
        - M1 < 0.8 Msun: γ ≈ 0.4, f_twin ≈ 0.05 (M-dwarfs)
        - 0.8 < M1 < 1.2 Msun: γ ≈ 0.3, f_twin ≈ 0.10 (Solar-type)
        - 1.2 < M1 < 3.5 Msun: γ ≈ 0.0, f_twin ≈ 0.08 (A/F stars)
        - M1 > 3.5 Msun: γ ≈ -0.5, f_twin ≈ 0.03 (OB stars)

    Parameters:
        q_min: Minimum mass ratio (default: 0.1)
        sigma_twin: Width of twin peak (default: 0.03)
    """

    q_min: float = 0.1
    sigma_twin: float = 0.03

    def _gamma_of_mass(self, m1: Float[Array, "..."]) -> Float[Array, "..."]:
        """Power-law exponent as function of primary mass (Moe+17 Table 10)."""
        # Piecewise linear interpolation
        gamma = jnp.where(
            m1 < 0.8,
            0.4,  # M-dwarfs
            jnp.where(
                m1 < 1.2,
                0.3,  # Solar-type
                jnp.where(
                    m1 < 3.5,
                    0.0,  # A/F stars
                    -0.5,  # OB stars
                ),
            ),
        )
        return gamma

    def _ftwin_of_mass(self, m1: Float[Array, "..."]) -> Float[Array, "..."]:
        """Twin excess fraction as function of primary mass (Moe+17 Table 10)."""
        f_twin = jnp.where(
            m1 < 0.8,
            0.05,  # M-dwarfs
            jnp.where(
                m1 < 1.2,
                0.10,  # Solar-type (highest twin excess)
                jnp.where(
                    m1 < 3.5,
                    0.08,  # A/F stars
                    0.03,  # OB stars (lowest twin excess)
                ),
            ),
        )
        return f_twin

    def sample_given_primary(
        self, key: PRNGKeyArray, m1: Float[Array, "n"]
    ) -> Float[Array, "n"]:
        """Sample mass ratios given primary masses.

        This is the key method: q distribution depends on M1.

        Args:
            key: JAX random key
            m1: Primary masses (n,)

        Returns:
            Mass ratios q ∈ [q_min, 1] with shape (n,)
        """
        n = m1.shape[0]
        key1, key2, key3 = jax.random.split(key, 3)

        # Get mass-dependent parameters
        gamma = self._gamma_of_mass(m1)
        f_twin = self._ftwin_of_mass(m1)

        # Decide which component: power-law or twin
        is_twin = jax.random.uniform(key1, (n,)) < f_twin

        # Sample power-law component
        u_pl = jax.random.uniform(key2, (n,))

        def sample_powerlaw(gamma_val, u_val):
            """Sample from power-law q^gamma."""
            q0 = self.q_min

            def neq_m1():
                g = gamma_val
                norm = (1.0 - q0 ** (g + 1.0)) / (g + 1.0)
                inner = u_val * (g + 1.0) * norm + q0 ** (g + 1.0)
                return inner ** (1.0 / (g + 1.0))

            def eq_m1():
                norm = -jnp.log(q0)
                return q0 * jnp.exp(u_val * norm)

            return jax.lax.cond(jnp.abs(gamma_val + 1.0) < 1e-10, eq_m1, neq_m1)

        q_powerlaw = jax.vmap(sample_powerlaw)(gamma, u_pl)

        # Sample twin component (truncated Gaussian centered at q=1, truncated to [q_min, 1])
        # Use inverse CDF sampling for correct distribution
        z_min = (self.q_min - 1.0) / self.sigma_twin
        z_max = 0.0  # (1.0 - 1.0) / sigma = 0
        # CDF values at boundaries
        cdf_min = 0.5 * (1.0 + jax.scipy.special.erf(z_min / jnp.sqrt(2.0)))
        cdf_max = 0.5  # Φ(0) = 0.5
        # Sample uniform in [cdf_min, cdf_max], then inverse CDF
        u_twin = jax.random.uniform(key3, (n,))
        u_scaled = cdf_min + u_twin * (cdf_max - cdf_min)
        # Inverse CDF: Φ^(-1)(u) = √2 * erfinv(2u - 1)
        z_samples = jnp.sqrt(2.0) * jax.scipy.special.erfinv(2.0 * u_scaled - 1.0)
        q_twin = 1.0 + z_samples * self.sigma_twin

        # Select based on component
        q = jnp.where(is_twin, q_twin, q_powerlaw)
        return q

    def pdf_given_primary(
        self, q: Float[Array, "..."], m1: float
    ) -> Float[Array, "..."]:
        """PDF of mass ratio given primary mass."""
        gamma = self._gamma_of_mass(jnp.asarray(m1))
        f_twin = self._ftwin_of_mass(jnp.asarray(m1))

        # Power-law normalization
        g = gamma
        q0 = self.q_min
        pl_norm = jax.lax.cond(
            jnp.abs(g + 1.0) < 1e-10,
            lambda: -jnp.log(q0),
            lambda: (1.0 - q0 ** (g + 1.0)) / (g + 1.0),
        )
        pl_pdf = q**g / pl_norm

        # Gaussian twin peak
        z_min = (q0 - 1.0) / self.sigma_twin
        cdf_min = 0.5 * (1.0 + jax.scipy.special.erf(z_min / jnp.sqrt(2.0)))
        gauss_norm = 0.5 - cdf_min
        gauss_pdf = (
            jnp.exp(-0.5 * ((q - 1.0) / self.sigma_twin) ** 2)
            / (self.sigma_twin * jnp.sqrt(2.0 * jnp.pi))
            / gauss_norm
        )

        combined = (1.0 - f_twin) * pl_pdf + f_twin * gauss_pdf
        in_range = (q >= q0) & (q <= 1.0)
        return jnp.where(in_range, combined, 0.0)



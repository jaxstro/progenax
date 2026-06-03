"""Binary-fraction models (split from binary.py)."""

from __future__ import annotations

from typing import Callable, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray


# =============================================================================
# Binary Fraction Models
# =============================================================================


class ConstantBinaryFraction(eqx.Module):
    """Constant binary fraction independent of mass.

    Reference:
        Raghavan et al. (2010) ApJS 190, 1
        Overall multiplicity fraction ~46% for solar-type stars.

    Parameters:
        f_bin: Binary fraction (default: 0.5)
    """

    f_bin: float = 0.5

    def __call__(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Return binary fraction (constant)."""
        return jnp.full_like(m, self.f_bin)


class MassDependentBinaryFraction(eqx.Module):
    """Mass-dependent binary fraction from Moe & Di Stefano (2017).

    Reference:
        Moe & Di Stefano (2017) ApJS 230, 15 - Table 13
        "Close Binary Fraction as Function of Primary Mass"

    Model (period-integrated companion frequency):
        - M < 0.1 Msun: f_bin ≈ 0.22 (VLM/brown dwarfs)
        - 0.1 < M < 0.5 Msun: f_bin ≈ 0.26 (M-dwarfs)
        - 0.5 < M < 1.0 Msun: f_bin ≈ 0.44 (K/G-dwarfs)
        - 1.0 < M < 2.0 Msun: f_bin ≈ 0.50 (F/A-stars)
        - 2.0 < M < 5.0 Msun: f_bin ≈ 0.60 (B-stars)
        - 5.0 < M < 10 Msun: f_bin ≈ 0.80 (early B)
        - M > 10 Msun: f_bin ≈ 0.90 (O-stars)

    Note: These are companion frequencies, not strict binary fractions.
    Higher-order multiples (triples, etc.) are common at high masses.
    """

    def __call__(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Return binary fraction as function of mass."""
        return jnp.where(
            m < 0.1,
            0.22,
            jnp.where(
                m < 0.5,
                0.26,
                jnp.where(
                    m < 1.0,
                    0.44,
                    jnp.where(
                        m < 2.0,
                        0.50,
                        jnp.where(
                            m < 5.0,
                            0.60,
                            jnp.where(m < 10.0, 0.80, 0.90),
                        ),
                    ),
                ),
            ),
        )



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
    """Mass-dependent MULTIPLICITY fraction (probability a primary has ≥1 companion).

    These are the **multiplicity fractions** f = 1 − (single-star fraction). For
    M ≥ 0.8 Msun they are derived from Moe & Di Stefano (2017) ApJS 230, 15, Table 13
    (the single-star fraction F_{n=0;q>0.1} row): e.g. solar 1−0.60=0.40, B-star 1−0.41≈0.59,
    O-star 1−0.06≈0.94. Below 0.8 Msun the values come from M-dwarf surveys (Raghavan et al.
    2010; Duchêne & Kraus 2013). They are NOT the "close binary fraction" (Table 13's
    f_{logP<3.7}) nor the total multiplicity *frequency* f_mult (which exceeds 1 for massive
    stars because of triples/quadruples); they are the fraction of primaries with a companion.

    Model (multiplicity fraction vs primary mass):
        - M < 0.1 Msun: f ≈ 0.22 (VLM/brown dwarfs; M-dwarf surveys)
        - 0.1 < M < 0.5 Msun: f ≈ 0.26 (M-dwarfs; surveys)
        - 0.5 < M < 1.0 Msun: f ≈ 0.44 (K/G-dwarfs; Raghavan 2010 ≈0.44, Moe solar 1−0.60=0.40)
        - 1.0 < M < 2.0 Msun: f ≈ 0.50 (F/A-stars)
        - 2.0 < M < 5.0 Msun: f ≈ 0.60 (B-stars; Moe A/late-B 1−0.41≈0.59)
        - 5.0 < M < 10 Msun: f ≈ 0.80 (early B; Moe mid/early-B 1−{0.24,0.16}≈0.76–0.84)
        - M > 10 Msun: f ≈ 0.90 (O-stars; Moe O 1−0.06≈0.94)

    Note: pairing one companion per "binary" system; higher-order multiples (triples, etc.)
    are common at high masses but are not modelled here (use the full Moe model for that).
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



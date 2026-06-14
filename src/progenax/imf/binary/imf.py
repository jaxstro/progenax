"""Binary IMF class (split from binary.py)."""

from __future__ import annotations

from typing import Callable, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from ..base import BaseIMF
from .mass_ratio import (
    MassRatioProtocol,
    FlatMassRatio,
    PowerLawMassRatio,
    TwinPeakedMassRatio,
)
from .moe_di_stefano import MoeDiStefano2017
from .binary_fraction import (
    ConstantBinaryFraction,
    MassDependentBinaryFraction,
)

# Type aliases
BinaryFractionCallable = Callable[[Float[Array, "..."]], Float[Array, "..."]]
MassRatioSamplerCallable = Callable[
    [PRNGKeyArray, Float[Array, "n"]], Float[Array, "n"]
]


# =============================================================================
# Binary IMF Class
# =============================================================================


class BinaryIMF(eqx.Module):
    """Initial Mass Function for binary star populations.

    Combines a primary-mass IMF with a mass-ratio distribution and
    binary fraction model to generate complete binary populations.

    Default Configuration (Moe & Di Stefano 2017):
        - Mass-ratio distribution: MoeDiStefano2017 (mass-dependent γ and twin excess)
        - Binary fraction: MassDependentBinaryFraction (increases with primary mass)

    This default follows the comprehensive observational constraints from
    Moe & Di Stefano (2017) ApJS 230, 15, which represents the most complete
    characterization of binary statistics to date.

    Reference:
        Kroupa (1995) MNRAS 277, 1491 - IMF-consistent binary populations
        Moe & Di Stefano (2017) ApJS 230, 15 - Modern comprehensive model
        Raghavan et al. (2010) ApJS 190, 1 - Solar-type binary statistics
        Sana et al. (2012) Science 337, 444 - O-star binary statistics

    Parameters:
        primary_imf: IMF for primary stars (must implement BaseIMF interface)
        q_distribution: Mass-ratio distribution. Options:
            - MoeDiStefano2017() [DEFAULT] - Mass-dependent from Moe+17
            - FlatMassRatio() - Uniform q ∈ [q_min, 1]
            - PowerLawMassRatio(gamma) - p(q) ∝ q^γ
            - TwinPeakedMassRatio() - Flat + Gaussian twin peak
            - Custom callable: f(key, m1) -> q array
        binary_fraction: Binary fraction model. Options:
            - MassDependentBinaryFraction() [DEFAULT] - From Moe+17 Table 13
            - ConstantBinaryFraction(f_bin) - Constant fraction
            - float - Shorthand for ConstantBinaryFraction
            - Custom callable: f(m) -> f_bin array

    Examples:
        >>> from progenax.imf import PowerLawIMF
        >>> from progenax.imf.binary import BinaryIMF
        >>>
        >>> # Default: full Moe+17 mass-dependent model
        >>> imf = BinaryIMF(primary_imf=PowerLawIMF.kroupa())
        >>> # Equivalent to:
        >>> # imf = BinaryIMF(
        >>> #     primary_imf=PowerLawIMF.kroupa(),
        >>> #     q_distribution=MoeDiStefano2017(),
        >>> #     binary_fraction=MassDependentBinaryFraction(),
        >>> # )
        >>>
        >>> # Simple constant binary fraction
        >>> imf = BinaryIMF(
        ...     primary_imf=PowerLawIMF.kroupa(),
        ...     binary_fraction=0.5,  # 50% binaries
        ... )
        >>>
        >>> # Custom mass-ratio sampler
        >>> def my_q_sampler(key, m1):
        ...     # Custom logic: q depends on m1
        ...     return jax.random.uniform(key, m1.shape, minval=0.3, maxval=1.0)
        >>> imf = BinaryIMF(
        ...     primary_imf=PowerLawIMF.kroupa(),
        ...     q_distribution=my_q_sampler,
        ... )
        >>>
        >>> # Custom binary fraction function
        >>> def my_f_bin(m):
        ...     # Custom: 40% for low mass, 80% for high mass
        ...     return jnp.where(m < 1.0, 0.4, 0.8)
        >>> imf = BinaryIMF(
        ...     primary_imf=PowerLawIMF.kroupa(),
        ...     binary_fraction=my_f_bin,
        ... )
    """

    primary_imf: BaseIMF
    q_distribution: Union[
        MassRatioProtocol,
        MoeDiStefano2017,
        MassRatioSamplerCallable,
        None,
    ] = None
    binary_fraction: Union[
        ConstantBinaryFraction,
        MassDependentBinaryFraction,
        BinaryFractionCallable,
        float,
        None,
    ] = None

    def _get_q_distribution(
        self,
    ) -> Union[MassRatioProtocol, MoeDiStefano2017, MassRatioSamplerCallable]:
        """Get mass-ratio distribution, defaulting to MoeDiStefano2017."""
        if self.q_distribution is None:
            return MoeDiStefano2017()
        return self.q_distribution

    def _get_binary_fraction_model(
        self,
    ) -> Union[
        ConstantBinaryFraction,
        MassDependentBinaryFraction,
        BinaryFractionCallable,
        float,
    ]:
        """Get binary fraction model, defaulting to MassDependentBinaryFraction."""
        if self.binary_fraction is None:
            return MassDependentBinaryFraction()
        return self.binary_fraction

    def _get_binary_fraction(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Get binary fraction for given masses."""
        f_bin_model = self._get_binary_fraction_model()
        if isinstance(f_bin_model, float):
            return jnp.full_like(m, f_bin_model)
        return f_bin_model(m)

    def sample_primaries(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample n primary masses from the IMF."""
        return self.primary_imf.sample(key, n)

    def sample_mass_ratios(
        self, key: PRNGKeyArray, m1: Float[Array, "n"]
    ) -> Float[Array, "n"]:
        """Sample mass ratios given primary masses.

        For MoeDiStefano2017, the distribution depends on primary mass.
        For other distributions, it's independent.
        For custom callables, passes (key, m1) directly.
        """
        n = m1.shape[0]
        q_dist = self._get_q_distribution()

        # Check if it's a callable (custom sampler)
        if callable(q_dist) and not isinstance(
            q_dist,
            (
                MoeDiStefano2017,
                FlatMassRatio,
                PowerLawMassRatio,
                TwinPeakedMassRatio,
            ),
        ):
            return q_dist(key, m1)
        elif isinstance(q_dist, MoeDiStefano2017):
            return q_dist.sample_given_primary(key, m1)
        else:
            return q_dist.sample(key, n)

    def sample_systems(
        self, key: PRNGKeyArray, n: int
    ) -> Tuple[Float[Array, "n"], Float[Array, "n"], Bool[Array, "n"]]:
        """Sample n stellar systems (singles + binaries).

        Returns:
            Tuple of:
                - m1: Primary masses (n,)
                - m2: Secondary masses (n,) - 0 for single stars
                - is_binary: Boolean mask (n,) - True for binaries
        """
        key1, key2, key3 = jax.random.split(key, 3)

        # Sample primary masses
        m1 = self.sample_primaries(key1, n)

        # Decide which are binaries
        f_bin = self._get_binary_fraction(m1)
        is_binary = jax.random.uniform(key2, (n,)) < f_bin

        # Sample mass ratios
        q = self.sample_mass_ratios(key3, m1)

        # Compute secondary masses (0 for singles)
        m2 = jnp.where(is_binary, m1 * q, 0.0)

        return m1, m2, is_binary

    def sample_all_masses(
        self, key: PRNGKeyArray, n: int
    ) -> Tuple[Float[Array, "..."], Bool[Array, "n"]]:
        """Sample n systems and return all stellar masses.

        Returns:
            Tuple of:
                - masses: All stellar masses (flattened, primaries + secondaries)
                - is_binary: Boolean mask indicating which systems were binaries

        Note: The returned masses array has length n + n_binary where
        n_binary is the number of binary systems.
        """
        m1, m2, is_binary = self.sample_systems(key, n)

        # Concatenate primary and secondary masses
        # Filter out m2 = 0 (singles)
        all_masses = jnp.concatenate([m1, m2[is_binary]])

        return all_masses, is_binary

    def mean_system_mass(self) -> float:
        """Expected total mass per system.

        Returns:
            E[M_total] = E[M1] × (1 + f_bin × E[q])

        Note: This assumes binary fraction and q are independent of M1.
        For mass-dependent models, this is approximate.
        """
        mean_m1 = self.primary_imf.mean_mass()
        f_bin_model = self._get_binary_fraction_model()
        q_dist = self._get_q_distribution()

        # Get average binary fraction (approximate for mass-dependent)
        if isinstance(f_bin_model, float):
            avg_f_bin = f_bin_model
        else:
            # Sample estimate
            test_masses = jnp.logspace(-1, 2, 100)
            avg_f_bin = jnp.mean(self._get_binary_fraction(test_masses))

        # Get average q (approximate for mass-dependent)
        if isinstance(q_dist, MoeDiStefano2017):
            # Assume average q ≈ 0.5 for mass-dependent case
            avg_q = 0.5
        elif hasattr(q_dist, "q_min"):
            # For power-law or flat: E[q] ≈ (1 + q_min) / 2 for flat
            avg_q = (1.0 + q_dist.q_min) / 2.0
        else:
            avg_q = 0.5

        return mean_m1 * (1.0 + avg_f_bin * avg_q)

    def binary_fraction_overall(self) -> float:
        """Return overall binary fraction.

        For mass-dependent models, this integrates over the IMF.
        """
        f_bin_model = self._get_binary_fraction_model()
        if isinstance(f_bin_model, float):
            return f_bin_model
        else:
            # Sample estimate: integrate f_bin(M) × IMF(M) dM
            key = jax.random.PRNGKey(0)
            m_samples = self.primary_imf.sample(key, 10000)
            f_bin_samples = self._get_binary_fraction(m_samples)
            return float(jnp.mean(f_bin_samples))

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def moe2017(cls, primary_imf: BaseIMF) -> "BinaryIMF":
        """Create BinaryIMF with full Moe & Di Stefano (2017) model.

        This is the default configuration, but this factory method makes
        it explicit and documents the reference.

        Reference:
            Moe & Di Stefano (2017) ApJS 230, 15
            "Mind Your Ps and Qs: The Interrelation between Period (P) and
            Mass-ratio (q) Distributions of Binary Stars"

        Args:
            primary_imf: IMF for primary stars

        Returns:
            BinaryIMF with mass-dependent q and f_bin
        """
        return cls(
            primary_imf=primary_imf,
            q_distribution=MoeDiStefano2017(),
            binary_fraction=MassDependentBinaryFraction(),
        )

    @classmethod
    def simple(
        cls,
        primary_imf: BaseIMF,
        binary_fraction: float = 0.5,
        q_min: float = 0.1,
    ) -> "BinaryIMF":
        """Create BinaryIMF with simple constant parameters.

        Good for quick tests or when detailed binary statistics aren't needed.

        Reference:
            Raghavan et al. (2010) ApJS 190, 1 - f_bin ≈ 0.46 for FGK stars
            Duchêne & Kraus (2013) ARA&A 51, 269 - General review

        Args:
            primary_imf: IMF for primary stars
            binary_fraction: Constant binary fraction (default: 0.5)
            q_min: Minimum mass ratio (default: 0.1)

        Returns:
            BinaryIMF with flat q distribution and constant f_bin
        """
        return cls(
            primary_imf=primary_imf,
            q_distribution=FlatMassRatio(q_min=q_min),
            binary_fraction=binary_fraction,
        )

    @classmethod
    def massive_stars(
        cls,
        primary_imf: BaseIMF,
        gamma: float = -0.1,
        binary_fraction: float = 0.69,
    ) -> "BinaryIMF":
        """Create BinaryIMF tuned for massive star populations.

        Reference:
            Sana et al. (2012) Science 337, 444
            "Binary Interaction Dominates the Evolution of Massive Stars"
            O-stars: f_bin ≈ 0.69, γ ≈ -0.1 (slight preference for unequal)

        Args:
            primary_imf: IMF for primary stars
            gamma: Power-law exponent for q distribution (default: -0.1, Sana 2012 κ=-0.1)
            binary_fraction: Binary fraction (default: 0.69, Sana 2012 intrinsic f_bin=0.69±0.09)

        Returns:
            BinaryIMF configured for OB star populations
        """
        return cls(
            primary_imf=primary_imf,
            q_distribution=PowerLawMassRatio(gamma=gamma, q_min=0.1),
            binary_fraction=binary_fraction,
        )


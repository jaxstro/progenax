# progenax/src/progenax/protocols.py
"""
Protocol classes for progenax type safety.

Defines interfaces for spatial profiles, velocity distributions, and IMFs.
These protocols enable composition: mix Plummer positions with King velocities.
"""

from typing import Protocol, runtime_checkable
from jaxtyping import Array, Float, PRNGKeyArray


@runtime_checkable
class SpatialProfile(Protocol):
    """
    Protocol for spatial density profiles.

    Implementations must provide position sampling and characteristic radius.
    Used by build_spatial_ic() for generic IC assembly.

    Example implementations:
        - PlummerProfile: Plummer (1911) density
        - KingProfile: King (1966) models
        - EFFProfile: Elson-Fall-Freeman (1987)
    """

    def sample_positions(
        self,
        masses: Float[Array, "N"],
        key: PRNGKeyArray,
    ) -> Float[Array, "N 3"]:
        """
        Sample 3D positions from density profile.

        Args:
            masses: Particle masses (N,) [M_sun]
            key: JAX random key

        Returns:
            Cartesian positions (N, 3) in length units
        """
        ...

    def characteristic_radius(self) -> Float[Array, ""]:
        """
        Return characteristic radius for softening computation.

        For Plummer: r_h (half-mass radius)
        For King: r_t (tidal radius)
        """
        ...


@runtime_checkable
class VelocityDF(Protocol):
    """
    Protocol for velocity distribution functions.

    Implementations must sample velocities given positions and masses.
    Enables composability: mix Plummer positions + King velocities.
    """

    def sample_velocities(
        self,
        positions: Float[Array, "N 3"],
        masses: Float[Array, "N"],
        key: PRNGKeyArray,
        G: float | None = None,
    ) -> Float[Array, "N 3"]:
        """
        Sample velocities from distribution function.

        Args:
            positions: Particle positions (N, 3)
            masses: Particle masses (N,)
            key: JAX random key
            G: Gravitational constant. If None, uses progenax.DEFAULT_UNITS.G
               (~0.00450 for stellar dynamics in pc³ Msun⁻¹ Myr⁻²)

        Returns:
            Cartesian velocities (N, 3) in velocity units
        """
        ...


@runtime_checkable
class IMFProtocol(Protocol):
    """
    Protocol for Initial Mass Functions.

    All IMF classes must implement these methods for consistent API
    and protocol-based composition (e.g., TruncatedIMF wrapping).

    Attributes:
        m_min: Minimum mass in distribution [M_sun]
        m_max: Maximum mass in distribution [M_sun]
    """

    m_min: float
    m_max: float

    def logpdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Log probability density (normalized over [m_min, m_max])."""
        ...

    def cdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Cumulative distribution function."""
        ...

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Percent point function (inverse CDF). Differentiable on (0, 1).

        u may be clamped to (eps, 1-eps) internally so gradients stay finite at the
        open boundary of unbounded-support distributions.
        """
        ...

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Draw n samples using reparameterization trick."""
        ...

    def mean_mass(self) -> float:
        """Expected mass E[m] over domain."""
        ...


@runtime_checkable
class PeriodDistribution(Protocol):
    """Protocol for binary orbital-period distributions (period in days).

    Implementations expose the standard sample/pdf/cdf/ppf quartet.

    Example implementations:
        - LogUniformPeriod (Öpik 1924)
        - LogNormalPeriod (Duquennoy & Mayor 1991)
        - SanaOBPeriod (Sana et al. 2012)
    """

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample n periods [days]."""
        ...

    def pdf(self, P: Float[Array, "..."]) -> Float[Array, "..."]:
        """Probability density at period P [days]."""
        ...

    def cdf(self, P: Float[Array, "..."]) -> Float[Array, "..."]:
        """Cumulative distribution at period P [days]."""
        ...

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Percent point function (inverse CDF). Differentiable on (0, 1).

        u may be clamped to (eps, 1-eps) internally so gradients stay finite at the
        open boundary of unbounded-support distributions.
        """
        ...


@runtime_checkable
class EccentricityDistribution(Protocol):
    """Protocol for unconditional binary-eccentricity distributions.

    Implementations expose the standard sample/pdf/cdf/ppf quartet.

    Example implementations:
        - ThermalEccentricity (f(e) = 2e; Ambartsumian 1937 / Heggie 1975)
        - UniformEccentricity
    """

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample n eccentricities."""
        ...

    def pdf(self, e: Float[Array, "..."]) -> Float[Array, "..."]:
        """Probability density at eccentricity e."""
        ...

    def cdf(self, e: Float[Array, "..."]) -> Float[Array, "..."]:
        """Cumulative distribution at eccentricity e."""
        ...

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Percent point function (inverse CDF). Differentiable on (0, 1).

        u may be clamped to (eps, 1-eps) internally so gradients stay finite at the
        open boundary of unbounded-support distributions.
        """
        ...


@runtime_checkable
class ConditionalEccentricityDistribution(Protocol):
    """Protocol for period-CONDITIONAL eccentricity distributions, p(e | P).

    Sampling takes (key, periods); n is implied by ``periods.shape``. Distinct
    from the unconditional :class:`EccentricityDistribution` (sample(key, n)).

    Example implementations:
        - MoeEccentricity (period-dependent circular->thermal heuristic)
    """

    def sample(
        self, key: PRNGKeyArray, periods: Float[Array, "N"]
    ) -> Float[Array, "N"]:
        """Sample one eccentricity per period [days]."""
        ...


@runtime_checkable
class MassPeriodEccentricityDistribution(Protocol):
    """Protocol for period- AND mass-conditional eccentricity distributions,
    p(e | P, M1).

    Sampling takes (key, periods, masses).

    Example implementations:
        - MoeEccentricity (Moe & Di Stefano 2017 e^η(logP, M1))
        - LogisticThermalEccentricity (accepts masses but ignores them)
    """

    def sample(
        self,
        key: PRNGKeyArray,
        periods: Float[Array, "N"],
        masses: Float[Array, "N"],
    ) -> Float[Array, "N"]:
        """Sample one eccentricity per (period [days], primary mass [Msun])."""
        ...


@runtime_checkable
class BinaryFractionModel(Protocol):
    """Protocol for binary-fraction models — uniform across mass and environment.

    `probability(masses, radii=None) -> f_bin` returns the per-star binary fraction.
    Mass-based models (ConstantBinaryFraction, MassDependentBinaryFraction,
    DifferentiableBinaryFraction) ignore `radii`; RadialBinaryFraction ignores
    `masses`; CombinedBinaryFraction modulates one by the other.

    Example implementations:
        - ConstantBinaryFraction, MassDependentBinaryFraction (Moe Table 13)
        - DifferentiableBinaryFraction
        - RadialBinaryFraction (phenomenological f_b(r))
        - CombinedBinaryFraction (mass x radial)
    """

    def probability(
        self,
        masses: Float[Array, "N"],
        radii: Float[Array, "N"] | None = None,
    ) -> Float[Array, "N"]:
        """Binary fraction f_bin in [0, 1] for each star."""
        ...


__all__ = [
    "SpatialProfile",
    "VelocityDF",
    "IMFProtocol",
    "PeriodDistribution",
    "EccentricityDistribution",
    "ConditionalEccentricityDistribution",
    "MassPeriodEccentricityDistribution",
    "BinaryFractionModel",
]

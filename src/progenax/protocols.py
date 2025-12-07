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
    Used by build_ic() for generic IC assembly.

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
            G: Gravitational constant. If None, uses jaxstro.units.DEFAULT.G
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
        """Percent point function (inverse CDF). Differentiable."""
        ...

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Draw n samples using reparameterization trick."""
        ...

    def mean_mass(self) -> float:
        """Expected mass E[m] over domain."""
        ...


__all__ = ["SpatialProfile", "VelocityDF", "IMFProtocol"]

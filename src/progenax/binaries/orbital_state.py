"""Binary orbital state for IC generation.

Port from gravax-legacy with explicit G parameter for progenax.
All functions take explicit G parameter (NOT get_G() defaults).
"""

from __future__ import annotations
from typing import Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float

from .kepler import KeplerElements
from .kepler_period import period_to_semimajor_axis


class BinaryOrbitalState(eqx.Module):
    """Binary orbital state for IC generation.

    Combines component masses with Keplerian orbital elements.
    This is an IC-specific container - for general orbital mechanics,
    use KeplerElements directly.

    Attributes:
        m1: Primary mass [M_sun]
        m2: Secondary mass [M_sun]
        elements: KeplerElements (a, e, i, Omega, omega, M0). The period/mean motion
            is derived on demand from (a, m1+m2, G) in to_state, not cached here.

    Example:
        >>> from jaxstro.units import PLANETARY
        >>> # PLANETARY.G is per-year, so convert the period days -> years.
        >>> state = BinaryOrbitalState.from_log_period(
        ...     m1=1.0, m2=0.5, logP_days=2.0, e=0.3,
        ...     inc=0.1, Omega=0.0, omega=0.0, M_anom=0.0,
        ...     G=PLANETARY.G, day_in_time_units=1.0 / 365.25
        ... )
        >>> r1, v1, r2, v2 = state.to_resolved_positions(G=PLANETARY.G)
    """
    m1: Float[Array, ""]
    m2: Float[Array, ""]
    elements: KeplerElements

    @classmethod
    def from_log_period(
        cls,
        m1: float,
        m2: float,
        logP_days: float,
        e: float,
        inc: float = 0.0,
        Omega: float = 0.0,
        omega: float = 0.0,
        M_anom: float = 0.0,
        *,
        G: float,
        day_in_time_units: float = 1.0,
    ) -> "BinaryOrbitalState":
        """Create binary state from log10(period/days) format.

        This is the standard format from binary population synthesis.

        Args:
            m1: Primary mass [M_sun]
            m2: Secondary mass [M_sun]
            logP_days: log10(P / day)
            e: Eccentricity [0, 1)
            inc: Inclination [rad]
            Omega: Longitude of ascending node [rad]
            omega: Argument of periapsis [rad]
            M_anom: Mean anomaly at epoch [rad]
            G: Gravitational constant (REQUIRED, no default)
            day_in_time_units: Conversion factor days -> code time units

        Returns:
            BinaryOrbitalState ready for IC generation
        """
        # Convert period
        P = (10.0 ** logP_days) * day_in_time_units

        # Derive semi-major axis from Kepler's 3rd law
        M_total = m1 + m2
        a = period_to_semimajor_axis(P, M_total, G)

        # Create orbital elements
        elements = KeplerElements(
            a=a, e=e, i=inc, Omega=Omega, omega=omega, M0=M_anom
        )

        return cls(
            m1=jnp.asarray(m1),
            m2=jnp.asarray(m2),
            elements=elements,
        )

    @classmethod
    def from_semi_major_axis(
        cls,
        m1: float,
        m2: float,
        a: float,
        e: float,
        inc: float = 0.0,
        Omega: float = 0.0,
        omega: float = 0.0,
        M_anom: float = 0.0,
        *,
        G: float,
    ) -> "BinaryOrbitalState":
        """Create binary state from semi-major axis directly.

        Args:
            m1: Primary mass [M_sun]
            m2: Secondary mass [M_sun]
            a: Semi-major axis [length units]
            e: Eccentricity [0, 1)
            inc: Inclination [rad]
            Omega: Longitude of ascending node [rad]
            omega: Argument of periapsis [rad]
            M_anom: Mean anomaly at epoch [rad]
            G: Gravitational constant (REQUIRED, no default)

        Returns:
            BinaryOrbitalState ready for IC generation
        """
        elements = KeplerElements(
            a=a, e=e, i=inc, Omega=Omega, omega=omega, M0=M_anom
        )

        return cls(
            m1=jnp.asarray(m1),
            m2=jnp.asarray(m2),
            elements=elements,
        )

    def to_resolved_positions(
        self,
        G: float,
    ) -> Tuple[Float[Array, "3"], Float[Array, "3"], Float[Array, "3"], Float[Array, "3"]]:
        """Get resolved barycentric positions and velocities.

        Args:
            G: Gravitational constant (REQUIRED, no default)

        Returns:
            r1, v1, r2, v2: Position and velocity of each component.
                COM is at origin (m1*r1 + m2*r2 = 0).
        """
        # Pass masses directly (no float conversion for vmap compatibility)
        return self.elements.to_binary_state(
            m1=self.m1,
            m2=self.m2,
            G=G,
        )


def _make_elements_from_inputs(
    m1: Float[Array, ""],
    m2: Float[Array, ""],
    logP_days: Float[Array, ""],
    e: Float[Array, ""],
    inc: Float[Array, ""],
    Omega: Float[Array, ""],
    omega: Float[Array, ""],
    M_anom: Float[Array, ""],
    *,
    G: float,
    day_in_time_units: float = 1.0,
) -> BinaryOrbitalState:
    """Internal: build a BinaryOrbitalState from standard binary-sampler outputs."""
    return BinaryOrbitalState.from_log_period(
        m1=m1, m2=m2, logP_days=logP_days, e=e,
        inc=inc, Omega=Omega, omega=omega, M_anom=M_anom,
        G=G, day_in_time_units=day_in_time_units,
    )


def _elements_to_resolved_state(
    elem: BinaryOrbitalState,
    G: float,
) -> Tuple[Float[Array, "3"], Float[Array, "3"], Float[Array, "3"], Float[Array, "3"]]:
    """Internal: resolved barycentric state for one BinaryOrbitalState (COM at origin)."""
    return elem.to_resolved_positions(G=G)


def batch_elements_to_resolved(
    m1: Float[Array, "N"],
    m2: Float[Array, "N"],
    logP_days: Float[Array, "N"],
    e: Float[Array, "N"],
    inc: Float[Array, "N"],
    Omega: Float[Array, "N"],
    omega: Float[Array, "N"],
    M_anom: Float[Array, "N"],
    *,
    G: float,
    day_in_time_units: float = 1.0,
) -> Tuple[Float[Array, "N 3"], Float[Array, "N 3"], Float[Array, "N 3"], Float[Array, "N 3"]]:
    """Vectorized wrapper to get resolved (r1, v1, r2, v2) for N binaries.

    Args:
        m1, m2: Component masses [N]
        logP_days: log10(P / day) [N]
        e: Eccentricity [N]
        inc: Inclination [rad] [N]
        Omega: Longitude of ascending node [rad] [N]
        omega: Argument of periapsis [rad] [N]
        M_anom: Mean anomaly [rad] [N]
        G: Gravitational constant (REQUIRED, no default)
        day_in_time_units: Conversion factor days -> code time units

    Returns:
        r1, v1, r2, v2: Arrays of shape [N, 3]
    """
    # Vectorize the element creation
    make_fn = jax.vmap(
        lambda m1, m2, lp, e, i, O, o, M: _make_elements_from_inputs(
            m1, m2, lp, e, i, O, o, M, G=G, day_in_time_units=day_in_time_units
        ),
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0),
    )
    elems = make_fn(m1, m2, logP_days, e, inc, Omega, omega, M_anom)

    # Vectorize state conversion
    to_state = jax.vmap(lambda el: _elements_to_resolved_state(el, G=G))
    r1, v1, r2, v2 = to_state(elems)

    return r1, v1, r2, v2


__all__ = ["BinaryOrbitalState", "batch_elements_to_resolved"]

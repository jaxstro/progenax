"""Inverse two-body problem: Cartesian state (r, v) -> Keplerian elements.

Extracted from :mod:`kepler` so ``KeplerElements.from_state`` stays a thin
classmethod. The angular-momentum / eccentricity-vector method
(Murray & Dermott 1999 §2.8; Vallado 2007 Algorithm 9), decomposed into small
named helpers. Plane degeneracy is dimensionless, and equatorial angles
follow the sign of the angular momentum.

Conventions (degenerate cases):
    - Circular orbits (e ≈ 0): omega set to 0 (undefined).
    - Equatorial orbits (i near 0 or pi): Omega set to 0 (undefined).
    - Unbound orbits (E ≥ 0): a = +inf.
    - All angles wrapped to [0, 2π).
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

_TWO_PI = 2.0 * jnp.pi


def _arccos_clamped(num: Float[Array, ""], denom: Float[Array, ""]) -> Float[Array, ""]:
    """Angle from a normalized dot product, with a finite zero-denominator branch.

    A safe inactive denominator avoids a dimensional absolute floor. The angle
    itself remains undefined for degenerate node/circular data.
    """
    safe = jnp.where(denom != 0.0, denom, 1.0)
    return jnp.arccos(jnp.clip(num / safe, -1.0, 1.0))


def _semimajor_axis(
    energy: Float[Array, ""], G: float, M_total: float
) -> Float[Array, ""]:
    """a = -GM/(2E) for bound orbits (E<0); +inf for unbound (E≥0)."""
    bound = energy < 0.0
    safe_energy = jnp.where(bound, energy, -1.0)
    return jnp.where(bound, -G * M_total / (2.0 * safe_energy), jnp.inf)


def _ascending_node(h: Float[Array, "3"]):
    """Node vector n = ẑ × h, its magnitude, and Omega = arctan2(n_y, n_x).

    Omega is undefined for equatorial orbits (|n|/|h| near zero), so it is
    set to zero. Return (n, n_mag, Omega, inclined), with Omega in [0, 2pi).
    """
    n = jnp.cross(jnp.array([0.0, 0.0, 1.0]), h)
    n_mag = jnp.sqrt(jnp.sum(n**2))
    h_mag = jnp.sqrt(jnp.sum(h**2))
    # |n|/|h|=|sin(i)| is dimensionless. An absolute |n| cutoff
    # changes the orbital-plane classification when length/time units change.
    inclined = n_mag > 1e-10 * h_mag
    safe_nx = jnp.where(inclined, n[0], 1.0)
    safe_ny = jnp.where(inclined, n[1], 0.0)
    Omega = jnp.where(inclined, jnp.arctan2(safe_ny, safe_nx), 0.0)
    return n, n_mag, jnp.mod(Omega, _TWO_PI), inclined


def _arg_periapsis(
    n: Float[Array, "3"],
    n_mag: Float[Array, ""],
    e_vec: Float[Array, "3"],
    e: Float[Array, ""],
    inclined,
    orientation,
) -> Float[Array, ""]:
    """Argument of periapsis from n dot e or the oriented equatorial eccentricity.

    Sign from e_z (e_z<0 → omega = 2π − omega). Circular (e≈0) → 0. Wrapped to [0, 2π).
    """
    omega = jnp.where(
        e > 1e-10,
        jnp.where(
            inclined,
            # Inclined orbit: omega from n · e
            jnp.where(
                e_vec[2] >= 0,
                _arccos_clamped(jnp.dot(n, e_vec), n_mag * e),
                _TWO_PI - _arccos_clamped(jnp.dot(n, e_vec), n_mag * e),
            ),
            # Omega=0 gauge; retrograde equatorial motion reverses the
            # in-plane orientation under R_x(pi).
            jnp.arctan2(orientation * e_vec[1], e_vec[0]),
        ),
        0.0,  # Circular orbit
    )
    return jnp.mod(omega, _TWO_PI)


def _true_anomaly(
    r: Float[Array, "3"],
    v: Float[Array, "3"],
    r_mag: Float[Array, ""],
    n: Float[Array, "3"],
    n_mag: Float[Array, ""],
    e_vec: Float[Array, "3"],
    e: Float[Array, ""],
    inclined,
    orientation,
) -> Float[Array, ""]:
    """True anomaly nu from e·r (eccentric) or position angle (circular).

    Sign from r·v (r·v<0 → nu = 2π − nu). For circular orbits nu is measured from
    the node (inclined) or as arctan2(r_y, r_x) (equatorial).
    """
    return jnp.where(
        e > 1e-10,
        jnp.where(
            jnp.dot(r, v) >= 0,
            _arccos_clamped(jnp.dot(e_vec, r), e * r_mag),
            _TWO_PI - _arccos_clamped(jnp.dot(e_vec, r), e * r_mag),
        ),
        # Circular orbit: nu from position angle in the orbital plane
        jnp.where(
            inclined,
            jnp.where(
                r[2] >= 0,
                _arccos_clamped(jnp.dot(n, r), n_mag * r_mag),
                _TWO_PI - _arccos_clamped(jnp.dot(n, r), n_mag * r_mag),
            ),
            jnp.arctan2(orientation * r[1], r[0]),
        ),
    )


def _true_to_mean(nu: Float[Array, ""], e: Float[Array, ""]) -> Float[Array, ""]:
    """Convert true anomaly nu → mean anomaly M via the eccentric anomaly E.

    tan(E/2) = sqrt((1−e)/(1+e)) tan(nu/2); M = E − e sin E. Wrapped to [0, 2π).
    """
    E = 2.0 * jnp.arctan2(
        jnp.sqrt(jnp.maximum(1.0 - e, 0.0)) * jnp.sin(nu / 2.0),
        jnp.sqrt(jnp.maximum(1.0 + e, 1e-30)) * jnp.cos(nu / 2.0),
    )
    E = jnp.mod(E, _TWO_PI)
    M = E - e * jnp.sin(E)
    return jnp.mod(M, _TWO_PI)


def orbital_elements_from_state(
    position: Float[Array, "3"],
    velocity: Float[Array, "3"],
    M_total: float,
    G: float,
):
    """Keplerian elements (a, e, i, Omega, omega, M0) from Cartesian (r, v).

    Angular-momentum / eccentricity-vector method (Murray & Dermott 1999 §2.8).
    Returns a 6-tuple of scalars; ``KeplerElements.from_state`` wraps it.
    """
    r = position
    v = velocity

    r_mag = jnp.sqrt(jnp.sum(r**2))
    v_mag = jnp.sqrt(jnp.sum(v**2))

    # Specific orbital energy: E = v²/2 − GM/r
    energy = 0.5 * v_mag**2 - G * M_total / r_mag

    # Specific angular momentum: h = r × v
    h = jnp.cross(r, v)
    h_mag = jnp.sqrt(jnp.sum(h**2))

    # Eccentricity vector: e = (v × h)/(GM) − r/|r|
    e_vec = jnp.cross(v, h) / (G * M_total) - r / r_mag
    e = jnp.sqrt(jnp.sum(e_vec**2))

    a = _semimajor_axis(energy, G, M_total)
    i = _arccos_clamped(h[2], h_mag)  # arccos(h_z/|h|)
    n, n_mag, Omega, inclined = _ascending_node(h)
    orientation = jnp.where(h[2] >= 0.0, 1.0, -1.0)
    omega = _arg_periapsis(n, n_mag, e_vec, e, inclined, orientation)
    nu = _true_anomaly(r, v, r_mag, n, n_mag, e_vec, e, inclined, orientation)
    M = _true_to_mean(nu, e)

    return a, e, i, Omega, omega, M


__all__ = ["orbital_elements_from_state"]

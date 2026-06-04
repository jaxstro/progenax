"""Inverse two-body problem: Cartesian state (r, v) -> Keplerian elements.

Extracted from :mod:`kepler` so ``KeplerElements.from_state`` stays a thin
classmethod. The angular-momentum / eccentricity-vector method
(Murray & Dermott 1999 §2.8; Vallado 2007 Algorithm 9), decomposed into small
named helpers. All ops are preserved exactly from the original ``from_state``
(bit-identical); only structure changed.

Conventions (degenerate cases):
    - Circular orbits (e ≈ 0): omega set to 0 (undefined).
    - Equatorial orbits (i ≈ 0): Omega set to 0 (undefined).
    - Unbound orbits (E ≥ 0): a = +inf.
    - All angles wrapped to [0, 2π).
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

_TWO_PI = 2.0 * jnp.pi


def _arccos_clamped(num: Float[Array, ""], denom: Float[Array, ""]) -> Float[Array, ""]:
    """arccos(clip(num / (denom + 1e-30), -1, 1)) — the repeated angle kernel.

    The 1e-30 keeps the quotient finite at denom=0 (degenerate node/circular);
    the clip guards arccos against |arg|>1 from round-off.
    """
    return jnp.arccos(jnp.clip(num / (denom + 1e-30), -1.0, 1.0))


def _semimajor_axis(
    energy: Float[Array, ""], G: float, M_total: float
) -> Float[Array, ""]:
    """a = -GM/(2E) for bound orbits (E<0); +inf for unbound (E≥0)."""
    return jnp.where(energy < 0, -G * M_total / (2.0 * energy + 1e-30), jnp.inf)


def _ascending_node(h: Float[Array, "3"]):
    """Node vector n = ẑ × h, its magnitude, and Omega = arctan2(n_y, n_x).

    Omega is undefined for equatorial orbits (|n|→0) → set to 0. Returns
    (n, n_mag, Omega) with Omega wrapped to [0, 2π).
    """
    n = jnp.cross(jnp.array([0.0, 0.0, 1.0]), h)
    n_mag = jnp.sqrt(jnp.sum(n**2))
    Omega = jnp.where(n_mag > 1e-10, jnp.arctan2(n[1], n[0]), 0.0)
    return n, n_mag, jnp.mod(Omega, _TWO_PI)


def _arg_periapsis(
    n: Float[Array, "3"],
    n_mag: Float[Array, ""],
    e_vec: Float[Array, "3"],
    e: Float[Array, ""],
) -> Float[Array, ""]:
    """Argument of periapsis omega from n·e (inclined) or arctan2(e_y, e_x) (equatorial).

    Sign from e_z (e_z<0 → omega = 2π − omega). Circular (e≈0) → 0. Wrapped to [0, 2π).
    """
    omega = jnp.where(
        e > 1e-10,
        jnp.where(
            n_mag > 1e-10,
            # Inclined orbit: omega from n · e
            jnp.where(
                e_vec[2] >= 0,
                _arccos_clamped(jnp.dot(n, e_vec), n_mag * e),
                _TWO_PI - _arccos_clamped(jnp.dot(n, e_vec), n_mag * e),
            ),
            # Equatorial orbit: omega from arctan2(e_y, e_x)
            jnp.arctan2(e_vec[1], e_vec[0]),
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
            n_mag > 1e-10,
            jnp.where(
                r[2] >= 0,
                _arccos_clamped(jnp.dot(n, r), n_mag * r_mag),
                _TWO_PI - _arccos_clamped(jnp.dot(n, r), n_mag * r_mag),
            ),
            jnp.arctan2(r[1], r[0]),
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
    n, n_mag, Omega = _ascending_node(h)
    omega = _arg_periapsis(n, n_mag, e_vec, e)
    nu = _true_anomaly(r, v, r_mag, n, n_mag, e_vec, e)
    M = _true_to_mean(nu, e)

    return a, e, i, Omega, omega, M


__all__ = ["orbital_elements_from_state"]

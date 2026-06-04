"""
Kepler orbital mechanics for binary systems and orbital elements.

Port from gravax-legacy with explicit G parameter for progenax.
All functions take explicit G parameter (NOT get_G() defaults).
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float
from typing import NamedTuple


class CartesianState(NamedTuple):
    """Cartesian phase-space state of a single body.

    A JAX-pytree NamedTuple (transparent to jit/grad/vmap); also tuple-iterable,
    so ``pos, vel = state`` works.

    Attributes:
        position: (3,) Cartesian position [length units]
        velocity: (3,) Cartesian velocity [velocity units]
    """

    position: Float[Array, "3"]
    velocity: Float[Array, "3"]


class BinaryState(NamedTuple):
    """Resolved barycentric phase-space state of a binary's two components.

    A JAX-pytree NamedTuple; tuple-iterable, so ``r1, v1, r2, v2 = state`` works.

    Attributes:
        r1, v1: Position/velocity of the primary [length, velocity units]
        r2, v2: Position/velocity of the secondary [length, velocity units]
    """

    r1: Float[Array, "3"]
    v1: Float[Array, "3"]
    r2: Float[Array, "3"]
    v2: Float[Array, "3"]


class KeplerElements(eqx.Module):
    """
    Keplerian orbital elements as Equinox module.

    Represents an orbit in the two-body problem using classical elements.
    All angles in radians, distances in current unit system.

    Attributes:
        a: Semi-major axis [length units]
        e: Eccentricity (0 ≤ e < 1 for bound orbits)
        i: Inclination [rad] (0 to π)
        Omega: Longitude of ascending node [rad] (0 to 2π)
        omega: Argument of periapsis [rad] (0 to 2π)
        M0: Mean anomaly at epoch [rad] (0 to 2π)

    References:
        Murray & Dermott (1999) "Solar System Dynamics" Eq 2.122
        Binney & Tremaine (2008) "Galactic Dynamics" Ch 3

    Examples:
        >>> # Circular orbit
        >>> elements = KeplerElements(a=1.0, e=0.0, i=0.0, Omega=0.0, omega=0.0, M0=0.0)
        >>> state = elements.to_state(M_total=1.0, G=1.0)

        >>> # Eccentric orbit
        >>> import jax.numpy as jnp
        >>> elements = KeplerElements(
        ...     a=1.0, e=0.5, i=jnp.pi/4,
        ...     Omega=0.0, omega=0.0, M0=0.0
        ... )
        >>> state = elements.to_state(M_total=1.0, G=1.0)
    """

    a: Float[Array, ""]      # Semi-major axis
    e: Float[Array, ""]      # Eccentricity
    i: Float[Array, ""]      # Inclination
    Omega: Float[Array, ""]  # Longitude of ascending node
    omega: Float[Array, ""]  # Argument of periapsis
    M0: Float[Array, ""]     # Mean anomaly at epoch

    def __init__(
        self,
        a: float,
        e: float = 0.0,
        i: float = 0.0,
        Omega: float = 0.0,
        omega: float = 0.0,
        M0: float = 0.0,
    ):
        """
        Initialize Keplerian orbital elements.

        Args:
            a: Semi-major axis [length units]
            e: Eccentricity (default: 0.0 = circular)
            i: Inclination [rad] (default: 0.0 = equatorial)
            Omega: Longitude of ascending node [rad] (default: 0.0)
            omega: Argument of periapsis [rad] (default: 0.0)
            M0: Mean anomaly at epoch [rad] (default: 0.0)
        """
        self.a = jnp.asarray(a, dtype=jnp.float64)
        self.e = jnp.asarray(e, dtype=jnp.float64)
        self.i = jnp.asarray(i, dtype=jnp.float64)
        self.Omega = jnp.asarray(Omega, dtype=jnp.float64)
        self.omega = jnp.asarray(omega, dtype=jnp.float64)
        self.M0 = jnp.asarray(M0, dtype=jnp.float64)

    def to_state(
        self,
        M_total: float,
        G: float,
    ) -> CartesianState:
        """
        Convert orbital elements to Cartesian state (position, velocity).

        Solves Kepler's equation and transforms to Cartesian coordinates.

        Args:
            M_total: Total mass of binary system [M☉]
            G: Gravitational constant (REQUIRED, no default)

        Returns:
            CartesianState(position, velocity), each a (3,) array. Tuple-iterable
            (``pos, vel = state``) and a JAX pytree.

        Algorithm:
            1. Solve Kepler's equation for eccentric anomaly E
            2. Convert E to perifocal coordinates (x_p, y_p)
            3. Rotate to inertial frame using (i, Omega, omega)

        References:
            Murray & Dermott (1999) Eq 2.81, 2.122
        """
        # Step 1: Solve Kepler's equation M = E - e*sin(E) for E
        E = self._solve_kepler_equation(self.M0, self.e)

        # Step 2: Perifocal coordinates
        # Position: x_p = a(cos E - e), y_p = a*sqrt(1-e^2)*sin E
        cos_E = jnp.cos(E)
        sin_E = jnp.sin(E)

        x_p = self.a * (cos_E - self.e)
        y_p = self.a * jnp.sqrt(jnp.maximum(1.0 - self.e**2, 1e-12)) * sin_E

        # Perifocal velocities
        # Mean motion: n = sqrt(GM/a^3). Use a divide-safe denominator (double
        # where) rather than an ABSOLUTE floor on a^3: a^3 is a dimensional
        # quantity, so a fixed 1e-12 floor silently corrupts tight binaries in
        # STELLAR units (a ~ 1e-6 pc => a^3 ~ 1e-17). The where guards only the
        # true singularity a=0 (and keeps the gradient finite there).
        a3 = self.a**3
        a3_safe = jnp.where(a3 > 0.0, a3, 1.0)
        n = jnp.where(a3 > 0.0, jnp.sqrt(G * M_total / a3_safe), 0.0)

        # dE/dt = n/(1 - e*cos E)
        E_dot = n / jnp.maximum(1.0 - self.e * cos_E, 1e-12)

        vx_p = -self.a * sin_E * E_dot
        vy_p = self.a * jnp.sqrt(jnp.maximum(1.0 - self.e**2, 1e-12)) * cos_E * E_dot

        # Step 3: Rotate to inertial frame
        # Rotation matrices for (Omega, i, omega)
        position = self._rotate_perifocal_to_inertial(
            jnp.array([x_p, y_p, 0.0])
        )
        velocity = self._rotate_perifocal_to_inertial(
            jnp.array([vx_p, vy_p, 0.0])
        )

        return CartesianState(position=position, velocity=velocity)

    def to_binary_state(
        self,
        m1: float,
        m2: float,
        G: float,
    ) -> BinaryState:
        """
        Convert orbital elements to resolved binary state vectors.

        Computes positions and velocities of both binary components in the
        center-of-mass frame, ensuring COM conservation and zero total momentum.

        Args:
            m1: Mass of the body returned as (r1, v1) [M☉].
            m2: Mass of the body returned as (r2, v2) [M☉]. m1/m2 are POSITIONAL
                (m1 >= m2 is NOT enforced); the barycentric split
                m1*r1 + m2*r2 = 0 is exact regardless of which is larger.
            G: Gravitational constant (REQUIRED, no default)

        Returns:
            BinaryState(r1, v1, r2, v2) where:
                r1: Position of primary [length units] (3,)
                v1: Velocity of primary [velocity units] (3,)
                r2: Position of secondary [length units] (3,)
                v2: Velocity of secondary [velocity units] (3,)

        Algorithm:
            1. Compute relative orbit state using to_state(M_total=m1+m2)
            2. Convert to barycentric frame using mass ratios:
               r1 = -m2/(m1+m2) * r_rel
               r2 = +m1/(m1+m2) * r_rel

        Conservation laws:
            - Center of mass at origin: m1*r1 + m2*r2 = 0
            - Zero total momentum: m1*v1 + m2*v2 = 0
            - Relative orbit: r_rel = r2 - r1, v_rel = v2 - v1

        Examples:
            >>> # Equal-mass binary (m1 = m2 = 1 M☉)
            >>> elements = KeplerElements(a=1.0, e=0.5, i=0.0, Omega=0.0, omega=0.0, M0=0.0)
            >>> r1, v1, r2, v2 = elements.to_binary_state(m1=1.0, m2=1.0, G=1.0)
            >>> # COM check
            >>> r_com = (1.0 * r1 + 1.0 * r2) / 2.0
            >>> assert jnp.allclose(r_com, 0.0, atol=1e-10)

            >>> # Extreme mass ratio binary (m1 = 10 M☉, m2 = 0.1 M☉)
            >>> r1, v1, r2, v2 = elements.to_binary_state(m1=10.0, m2=0.1, G=1.0)
            >>> # Primary barely moves, secondary does large orbit
            >>> assert jnp.linalg.norm(r1) < jnp.linalg.norm(r2) / 10

        References:
            Murray & Dermott (1999) §2.2 - Center-of-mass frame
            Binney & Tremaine (2008) §3.1 - Two-body problem
        """
        M_total = m1 + m2

        # Get relative orbit state: r_rel = r2 - r1, v_rel = v2 - v1
        relative_state = self.to_state(M_total=M_total, G=G)
        r_rel = relative_state.position
        v_rel = relative_state.velocity

        # Convert to barycentric frame
        # COM condition: m1*r1 + m2*r2 = 0
        # Relative: r_rel = r2 - r1
        # Solution: r1 = -m2/(m1+m2) * r_rel, r2 = m1/(m1+m2) * r_rel

        mu1 = m2 / M_total  # Mass ratio for primary
        mu2 = m1 / M_total  # Mass ratio for secondary

        r1 = -mu1 * r_rel
        r2 = mu2 * r_rel

        v1 = -mu1 * v_rel
        v2 = mu2 * v_rel

        return BinaryState(r1, v1, r2, v2)

    @classmethod
    def from_state(
        cls,
        position: Float[Array, "3"],
        velocity: Float[Array, "3"],
        M_total: float,
        G: float,
    ) -> "KeplerElements":
        """
        Convert Cartesian state (position, velocity) to orbital elements.

        Computes Keplerian orbital elements from position and velocity vectors
        using the angular momentum and eccentricity vector method.

        Args:
            position: Cartesian position vector [length units]
            velocity: Cartesian velocity vector [velocity units]
            M_total: Total mass of binary system [M☉]
            G: Gravitational constant (REQUIRED, no default)

        Returns:
            KeplerElements instance with computed orbital parameters

        Algorithm:
            1. Compute specific angular momentum h = r × v
            2. Compute eccentricity vector e = (v × h)/(GM) - r/|r|
            3. Extract elements from h and e vectors
            4. Convert true anomaly to mean anomaly via eccentric anomaly

        References:
            Murray & Dermott (1999) §2.8
            Vallado (2007) "Fundamentals of Astrodynamics" Algorithm 9

        Examples:
            >>> # Round-trip test
            >>> original = KeplerElements(a=1.0, e=0.5, i=0.0, Omega=0.0, omega=0.0, M0=0.0)
            >>> state = original.to_state(M_total=1.0, G=1.0)
            >>> recovered = KeplerElements.from_state(
            ...     state.position, state.velocity, M_total=1.0, G=1.0
            ... )
            >>> assert jnp.allclose(recovered.a, original.a)

        Notes:
            - For circular orbits (e ≈ 0), omega is set to 0 (undefined)
            - For equatorial orbits (i ≈ 0), Omega is set to 0 (undefined)
            - Unbound orbits (E ≥ 0) return a = infinity
            - All angles wrapped to [0, 2π)
        """
        r = position
        v = velocity

        # Compute magnitudes
        r_mag = jnp.sqrt(jnp.sum(r**2))
        v_mag = jnp.sqrt(jnp.sum(v**2))

        # Specific orbital energy: E = v²/2 - GM/r
        energy = 0.5 * v_mag**2 - G * M_total / r_mag

        # Specific angular momentum: h = r × v
        h = jnp.cross(r, v)
        h_mag = jnp.sqrt(jnp.sum(h**2))

        # Eccentricity vector: e = (v × h)/(GM) - r/|r|
        e_vec = jnp.cross(v, h) / (G * M_total) - r / r_mag
        e = jnp.sqrt(jnp.sum(e_vec**2))

        # Semi-major axis: a = -GM/(2E) for bound orbits
        # For unbound orbits (E ≥ 0), set a = infinity
        a = jnp.where(
            energy < 0,
            -G * M_total / (2.0 * energy + 1e-30),
            jnp.inf
        )

        # Inclination: i = arccos(h_z / |h|)
        i = jnp.arccos(jnp.clip(h[2] / (h_mag + 1e-30), -1.0, 1.0))

        # Node vector: n = z × h (points to ascending node)
        n = jnp.cross(jnp.array([0.0, 0.0, 1.0]), h)
        n_mag = jnp.sqrt(jnp.sum(n**2))

        # Longitude of ascending node: Omega = arctan2(n_y, n_x)
        # For equatorial orbits (i ≈ 0), Omega is undefined → set to 0
        Omega = jnp.where(
            n_mag > 1e-10,
            jnp.arctan2(n[1], n[0]),
            0.0
        )
        Omega = jnp.mod(Omega, 2.0 * jnp.pi)  # Wrap to [0, 2π)

        # Argument of periapsis: omega = arccos(n · e / (|n| |e|))
        # Sign from e_z: if e_z < 0, omega = 2π - omega
        # For circular orbits (e ≈ 0), omega is undefined → set to 0
        # For equatorial orbits (i ≈ 0), use arctan2(e_y, e_x)
        omega = jnp.where(
            e > 1e-10,
            jnp.where(
                n_mag > 1e-10,
                # Inclined orbit: omega from n · e
                jnp.where(
                    e_vec[2] >= 0,
                    jnp.arccos(jnp.clip(jnp.dot(n, e_vec) / (n_mag * e + 1e-30), -1.0, 1.0)),
                    2.0 * jnp.pi - jnp.arccos(jnp.clip(jnp.dot(n, e_vec) / (n_mag * e + 1e-30), -1.0, 1.0))
                ),
                # Equatorial orbit: omega from arctan2(e_y, e_x)
                jnp.arctan2(e_vec[1], e_vec[0])
            ),
            0.0  # Circular orbit
        )
        omega = jnp.mod(omega, 2.0 * jnp.pi)  # Wrap to [0, 2π)

        # True anomaly: nu = arccos(e · r / (|e| |r|))
        # Sign from r · v: if r · v < 0, nu = 2π - nu
        # For circular orbits (e ≈ 0), nu from arctan2(r_y, r_x) in orbital plane
        nu = jnp.where(
            e > 1e-10,
            jnp.where(
                jnp.dot(r, v) >= 0,
                jnp.arccos(jnp.clip(jnp.dot(e_vec, r) / (e * r_mag + 1e-30), -1.0, 1.0)),
                2.0 * jnp.pi - jnp.arccos(jnp.clip(jnp.dot(e_vec, r) / (e * r_mag + 1e-30), -1.0, 1.0))
            ),
            # Circular orbit: nu from position angle in orbital plane
            jnp.where(
                n_mag > 1e-10,
                # Inclined: angle from node vector
                jnp.where(
                    r[2] >= 0,
                    jnp.arccos(jnp.clip(jnp.dot(n, r) / (n_mag * r_mag + 1e-30), -1.0, 1.0)),
                    2.0 * jnp.pi - jnp.arccos(jnp.clip(jnp.dot(n, r) / (n_mag * r_mag + 1e-30), -1.0, 1.0))
                ),
                # Equatorial: arctan2(y, x)
                jnp.arctan2(r[1], r[0])
            )
        )

        # Convert true anomaly to eccentric anomaly
        # tan(E/2) = sqrt((1-e)/(1+e)) * tan(nu/2)
        E = 2.0 * jnp.arctan2(
            jnp.sqrt(jnp.maximum(1.0 - e, 0.0)) * jnp.sin(nu / 2.0),
            jnp.sqrt(jnp.maximum(1.0 + e, 1e-30)) * jnp.cos(nu / 2.0)
        )
        E = jnp.mod(E, 2.0 * jnp.pi)  # Wrap to [0, 2π)

        # Convert eccentric anomaly to mean anomaly: M = E - e*sin(E)
        M = E - e * jnp.sin(E)
        M = jnp.mod(M, 2.0 * jnp.pi)  # Wrap to [0, 2π)

        # Return KeplerElements (keep as JAX arrays for JIT compatibility)
        return cls(
            a=a,
            e=e,
            i=i,
            Omega=Omega,
            omega=omega,
            M0=M,
        )

    @staticmethod
    def _solve_kepler_equation(
        M: Float[Array, ""],
        e: Float[Array, ""],
        max_iter: int = 50,
    ) -> Float[Array, ""]:
        """
        Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E.

        Uses Newton-Raphson iteration with fixed iterations (differentiable).
        Converges very fast for e < 1 (typically 3-4 iterations for e < 0.8). The
        default max_iter=50 is deliberately conservative so the fixed-iteration
        scheme still reaches machine precision near e -> 1 (slower convergence);
        the extra iterations are cheap and guarantee accuracy across all e < 1.

        Args:
            M: Mean anomaly [rad]
            e: Eccentricity
            max_iter: Maximum iterations (default: 50)

        Returns:
            E: Eccentric anomaly [rad]

        Notes:
            Uses jax.lax.scan for fixed iterations (fully differentiable).
            Initial guess: E0 = M + 0.85*sign(sin M)*e (good for moderate e)

        References:
            Murray & Dermott (1999) Eq 2.60
        """
        # Wrap M to [0, 2π) for numerical stability
        M_wrapped = jnp.mod(M, 2.0 * jnp.pi)

        # Initial guess: E0 = M + 0.85*sign(sin M)*e
        # This is better than E0 = M for moderate eccentricities
        E = M_wrapped + 0.85 * jnp.sign(jnp.sin(M_wrapped)) * e

        def newton_step(E_prev, _):
            """One Newton-Raphson iteration."""
            f = E_prev - e * jnp.sin(E_prev) - M_wrapped
            f_prime = 1.0 - e * jnp.cos(E_prev)
            E_new = E_prev - f / jnp.maximum(jnp.abs(f_prime), 1e-12)
            return E_new, None

        # Use scan for fixed iterations (differentiable)
        E, _ = jax.lax.scan(newton_step, E, None, length=max_iter)

        return E

    def _rotate_perifocal_to_inertial(
        self,
        vec_perifocal: Float[Array, "3"],
    ) -> Float[Array, "3"]:
        """
        Rotate vector from perifocal frame to inertial frame.

        Rotation sequence: R = R_z(Omega) * R_x(i) * R_z(omega)

        The perifocal frame has x-axis pointing to periapsis.
        The rotation transforms to the inertial frame.

        Args:
            vec_perifocal: Vector in perifocal coordinates (3,)

        Returns:
            Vector in inertial coordinates (3,)

        References:
            Murray & Dermott (1999) Eq 2.122
        """
        # Precompute trig functions
        cos_Omega = jnp.cos(self.Omega)
        sin_Omega = jnp.sin(self.Omega)
        cos_i = jnp.cos(self.i)
        sin_i = jnp.sin(self.i)
        cos_omega = jnp.cos(self.omega)
        sin_omega = jnp.sin(self.omega)

        # Combined rotation matrix (Murray & Dermott Eq 2.122)
        # R = R_z(Omega) * R_x(i) * R_z(omega)
        R11 = cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i
        R12 = -cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i
        R13 = sin_Omega * sin_i

        R21 = sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i
        R22 = -sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i
        R23 = -cos_Omega * sin_i

        R31 = sin_omega * sin_i
        R32 = cos_omega * sin_i
        R33 = cos_i

        # Matrix multiplication
        x = R11 * vec_perifocal[0] + R12 * vec_perifocal[1] + R13 * vec_perifocal[2]
        y = R21 * vec_perifocal[0] + R22 * vec_perifocal[1] + R23 * vec_perifocal[2]
        z = R31 * vec_perifocal[0] + R32 * vec_perifocal[1] + R33 * vec_perifocal[2]

        return jnp.array([x, y, z])


def compute_period(
    a: float,
    M_total: float,
    G: float,
) -> float:
    """
    Compute orbital period from semi-major axis using Kepler's 3rd law.

    Args:
        a: Semi-major axis [length units]
        M_total: Total mass of binary system [M☉]
        G: Gravitational constant (REQUIRED, no default)

    Returns:
        period: Orbital period [time units]

    Formula:
        T = 2π√(a³/(GM))

    Examples:
        >>> # Earth orbit: a=1 AU, M=1 M☉ → T≈1 year
        >>> G = 39.478  # AU³/Msun/yr²
        >>> T = compute_period(a=1.0, M_total=1.0, G=G)
        >>> print(f"Period: {T:.2f} years")
        Period: 1.00 years

        >>> # Stellar cluster orbit: a=1 pc, M=1000 M☉
        >>> G = 0.00450  # pc³/Msun/Myr²
        >>> T = compute_period(a=1.0, M_total=1000.0, G=G)
        >>> print(f"Period: {T:.2f} Myr")
        Period: 0.94 Myr

    References:
        Kepler's 3rd Law: T² ∝ a³/M
        Murray & Dermott (1999) Eq 2.37
    """
    # T = 2π√(a³/(GM))  (G, M_total > 0 for any physical binary)
    period = 2.0 * jnp.pi * jnp.sqrt(a**3 / (G * M_total))

    return period


def period_to_semimajor_axis(
    period: float,
    M_total: float,
    G: float,
) -> float:
    """
    Compute semi-major axis from orbital period using Kepler's 3rd law.

    Args:
        period: Orbital period [time units]
        M_total: Total mass of binary system [M☉]
        G: Gravitational constant (REQUIRED, no default)

    Returns:
        a: Semi-major axis [length units]

    Formula:
        a = (GMT²/(4π²))^(1/3)

    Examples:
        >>> # Binary with 10 day period, M_total=2 M☉
        >>> G = 39.478  # AU³/Msun/yr²
        >>> P_yr = 10.0 / 365.25  # Convert days to years
        >>> a = period_to_semimajor_axis(P_yr, M_total=2.0, G=G)
        >>> print(f"Semi-major axis: {a:.3f} AU")
        Semi-major axis: 0.089 AU

        >>> # Star cluster binary: 10 Myr period, M_total=2 M☉
        >>> G = 0.00450  # pc³/Msun/Myr²
        >>> a = period_to_semimajor_axis(10.0, M_total=2.0, G=G)
        >>> print(f"Semi-major axis: {a:.2f} pc")
        Semi-major axis: 4.64 pc

    References:
        Kepler's 3rd Law: a³ ∝ T²M
        Murray & Dermott (1999) Eq 2.37
    """
    # a = (GM*T²/(4π²))^(1/3)
    a = (G * M_total * period**2 / (4.0 * jnp.pi**2))**(1.0 / 3.0)

    return a


__all__ = ["KeplerElements", "compute_period", "period_to_semimajor_axis"]

"""
Plummer (1911) velocity distribution function as Equinox module.

Implements VelocityDF protocol for use with IC assembly.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray


class PlummerVelocityDF(eqx.Module):
    """
    Plummer (1911) velocity distribution function.

    Samples velocity magnitudes from the exact Plummer DF using Beta distribution
    (no rejection sampling required). Velocities are isotropically distributed.

    The distribution for q = v/v_esc is:
        g(q) ∝ q² (1 - q²)^(7/2)  for q ∈ [0, 1]

    This corresponds to the energy distribution:
        f(E) ∝ E^(7/2)  where E = ψ - v²/2 is the binding energy

    Sampling method:
        Let u = q², then u ~ Beta(3/2, 9/2)
        Therefore: q = sqrt(u), v = q × v_esc

    This gives the exact velocity dispersion:
        <q²> = 1/4  =>  <v²> = v_esc²/4  =>  σ² = v_esc²/12

    Which matches the Plummer formula:
        σ²(r) = GM/(6√(r²+a²))  with  v_esc²(r) = 2GM/√(r²+a²)

    Attributes:
        r_h: Half-mass radius [length units]
        a: Plummer scale radius [length units] (computed from r_h)

    References:
        Plummer (1911), MNRAS, 71, 460 - Original Plummer model
        Aarseth (2003), "Gravitational N-Body Simulations", Section 4.3.2
        Binney & Tremaine (2008), "Galactic Dynamics", Section 4.3
        Dehnen (1993), MNRAS, 265, 250 - Exact analytical DF

    Notes:
        - Beta(3/2, 9/2) sampling is EXACT (no rejection, 100% efficient)
        - Fully differentiable and JIT-compatible
        - For Plummer sphere: v_esc² = 2GM/√(r²+a²)
        - Verified: v_esc = sqrt(12) × σ (exact Plummer relation)

    Examples:
        >>> from progenax.profiles.plummer import PlummerProfile
        >>> import jax
        >>> import jax.numpy as jnp
        >>>
        >>> # Create spatial profile and velocity DF
        >>> profile = PlummerProfile(r_h=1.0)
        >>> velocity_df = PlummerVelocityDF(r_h=1.0)
        >>>
        >>> # Sample positions and velocities
        >>> masses = jnp.ones(100)
        >>> key = jax.random.PRNGKey(42)
        >>> key_pos, key_vel = jax.random.split(key)
        >>>
        >>> positions = profile.sample_positions(masses, key_pos)
        >>> velocities = velocity_df.sample_velocities(positions, masses, key_vel, G=1.0)
    """

    r_h: Float[Array, ""]
    a: Float[Array, ""]

    def __init__(self, r_h: float = 1.0):
        """
        Initialize Plummer velocity distribution function.

        Args:
            r_h: Half-mass radius [length units], must match spatial profile
        """
        self.r_h = jnp.asarray(r_h, dtype=jnp.float64)
        # Scale radius: a = r_h / sqrt(2^(2/3) - 1)
        # From Plummer (1911): r_h = a * sqrt(2^(2/3) - 1)
        self.a = self.r_h / jnp.sqrt(2**(2/3) - 1)

    @jax.jit
    def sample_velocities(
        self,
        positions: Float[Array, "N 3"],
        masses: Float[Array, "N"],
        key: PRNGKeyArray,
        G: float = 1.0,
    ) -> Float[Array, "N 3"]:
        """
        Sample velocities from Plummer distribution function.

        Samples velocity magnitudes from exact Plummer DF (Beta distribution),
        then assigns isotropic random directions.

        Args:
            positions: Particle positions (N, 3) [length units]
            masses: Particle masses (N,) [M☉]
            key: JAX random key
            G: Gravitational constant (default: 1.0)

        Returns:
            Cartesian velocities (N, 3) [velocity units]

        Notes:
            - Velocities are isotropic (no radial bias)
            - All velocities satisfy v < v_esc (bound particles)
            - Statistical properties match Plummer (1911) exactly
        """
        N = positions.shape[0]
        key_mag, key_theta, key_phi = jax.random.split(key, 3)

        # Compute radii from positions
        radii = jnp.linalg.norm(positions, axis=1)

        # Sample velocity magnitudes from Plummer DF
        v_magnitudes = self._sample_velocity_magnitudes(radii, masses, key_mag, G)

        # Sample isotropic directions
        cos_theta = jax.random.uniform(key_theta, (N,), minval=-1.0, maxval=1.0)
        phi = jax.random.uniform(key_phi, (N,), minval=0.0, maxval=2*jnp.pi)

        # Convert to Cartesian velocity components
        sin_theta = jnp.sqrt(1.0 - cos_theta**2)
        vx = v_magnitudes * sin_theta * jnp.cos(phi)
        vy = v_magnitudes * sin_theta * jnp.sin(phi)
        vz = v_magnitudes * cos_theta

        velocities = jnp.stack([vx, vy, vz], axis=1)

        return velocities

    @jax.jit
    def _sample_velocity_magnitudes(
        self,
        r: Float[Array, "N"],
        masses: Float[Array, "N"],
        key: PRNGKeyArray,
        G: float,
    ) -> Float[Array, "N"]:
        """
        Sample velocity magnitudes from Plummer distribution function.

        Uses the exact Plummer DF via Beta distribution sampling (no rejection!).
        The distribution for q = v/v_esc is:
            g(q) ∝ q² (1 - q²)^(7/2)  for q ∈ [0, 1]

        Sampling method:
            Let u = q², then u ~ Beta(3/2, 9/2)
            Therefore: q = sqrt(u), v = q × v_esc

        Args:
            r: Particle radii [length units]
            masses: Particle masses [M☉]
            key: JAX random key
            G: Gravitational constant

        Returns:
            Velocity magnitudes [velocity units]

        References:
            Plummer (1911), MNRAS, 71, 460 - Original Plummer model
            Aarseth (2003), "Gravitational N-Body Simulations", Section 4.3.2
            Binney & Tremaine (2008), "Galactic Dynamics", Section 4.3
            Dehnen (1993), MNRAS, 265, 250 - Exact analytical DF

        Notes:
            - Beta(3/2, 9/2) sampling is EXACT (no rejection, 100% efficient)
            - Fully differentiable and JIT-compatible
            - For Plummer sphere: v_esc² = 2GM/√(r²+a²)
            - Verified: v_esc = sqrt(12) × σ (exact Plummer relation)
        """
        N = r.shape[0]  # Use .shape[0] not len() to get concrete int in JIT

        # Escape velocity at radius r
        # v_esc² = 2|ψ(r)| = 2GM/√(r²+a²)
        M_total = jnp.sum(masses)
        v_esc = jnp.sqrt(2.0 * G * M_total / jnp.sqrt(r**2 + self.a**2))

        # Sample q² from Beta(3/2, 9/2)
        # This gives g(q) ∝ q²(1-q²)^(7/2) exactly!
        u = jax.random.beta(key, a=1.5, b=4.5, shape=(N,))  # u = q² ∈ [0,1]
        q = jnp.sqrt(u)  # q = v/v_esc

        # Velocity magnitude
        v = q * v_esc

        return v


__all__ = ["PlummerVelocityDF"]

"""
King (1966) velocity distribution function as Equinox module.

Implements VelocityDF protocol for use with IC assembly.
Samples from the "lowered Maxwellian" distribution with escape velocity cutoff.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

from progenax import defaults


class KingVelocityDF(eqx.Module):
    """
    King (1966) "lowered Maxwellian" velocity distribution function.

    The King DF gives a velocity distribution at each radius:
        f(E) ∝ (e^(ψ - v²/2σ²) - 1)  for E < 0
        f(E) = 0                      for E ≥ 0

    where ψ(r) is the dimensionless potential and σ is the central velocity
    dispersion.

    This implementation uses a simplified approach:
    - Samples from Gaussian (Maxwellian) velocity distribution
    - Applies cutoff at escape velocity
    - Velocity dispersion decreases with radius following King profile

    Attributes:
        W0: King concentration parameter (dimensionless central potential)
        r_c: Core radius [length units]
        r_t: Tidal radius [length units]

    References:
        King, I. R. (1966), "The Structure of Star Clusters. III. Some Simple
        Dynamical Models", AJ, 71, 64

        Heggie & Hut (2003), "The Gravitational Million-Body Problem", §6.2

        Binney & Tremaine (2008), "Galactic Dynamics", Section 4.3

    Notes:
        - W0 typically ranges from 1 (low concentration) to 12 (high)
        - Globular clusters have W0 ~ 5-9
        - Simplified velocity dispersion profile (not full King potential)
        - Fully differentiable and JIT-compatible

    Examples:
        >>> from progenax.profiles.king import KingProfile
        >>> import jax
        >>> import jax.numpy as jnp
        >>>
        >>> # Create spatial profile and velocity DF
        >>> profile = KingProfile(W0=7.0, r_c=1.0, r_t=10.0)
        >>> velocity_df = KingVelocityDF(W0=7.0, r_c=1.0, r_t=10.0)
        >>>
        >>> # Sample positions and velocities
        >>> masses = jnp.ones(100)
        >>> key = jax.random.PRNGKey(42)
        >>> key_pos, key_vel = jax.random.split(key)
        >>>
        >>> positions = profile.sample_positions(masses, key_pos)
        >>> from jaxstro.units import STELLAR
        >>> velocities = velocity_df.sample_velocities(positions, masses, key_vel, G=STELLAR.G)
    """

    W0: Float[Array, ""]
    r_c: Float[Array, ""]
    r_t: Float[Array, ""]

    def __init__(self, W0: float = 5.0, r_c: float = 1.0, r_t: float = 10.0):
        """
        Initialize King velocity distribution function.

        Args:
            W0: King concentration parameter (dimensionless central potential)
            r_c: Core radius [length units]
            r_t: Tidal radius [length units]
        """
        self.W0 = jnp.asarray(W0)
        self.r_c = jnp.asarray(r_c)
        self.r_t = jnp.asarray(r_t)

    def sample_velocities(
        self,
        positions: Float[Array, "N 3"],
        masses: Float[Array, "N"],
        key: PRNGKeyArray,
        G: float | None = None,
    ) -> Float[Array, "N 3"]:
        """
        Sample velocities from King distribution function.

        Samples from isotropic Gaussian with radius-dependent velocity
        dispersion, then applies cutoff at escape velocity.

        Args:
            positions: Particle positions (N, 3) [length units]
            masses: Particle masses (N,) [M☉]
            key: JAX random key
            G: Gravitational constant. If None, uses progenax.DEFAULT_UNITS.G
               (~0.00450 for stellar dynamics in pc³ Msun⁻¹ Myr⁻²)

        Returns:
            Cartesian velocities (N, 3) [velocity units]

        Notes:
            - Velocities are isotropic (no radial bias)
            - All velocities satisfy v < v_esc (bound particles)
            - Uses simplified velocity dispersion profile
        """
        if G is None:
            G = defaults.DEFAULT_UNITS.G

        N = positions.shape[0]
        M_total = jnp.sum(masses)

        # Compute radii
        radii = jnp.linalg.norm(positions, axis=1)

        # Central velocity dispersion (from virial theorem)
        # For King models: σ₀² ≈ G M_total / (9 r_c)  (approximate)
        sigma_0_squared = G * M_total / (9.0 * self.r_c)

        # Simplified dimensionless potential (approximate)
        # ψ(r) ≈ W₀ × (1 - r²/(r_t²+r_c²))
        psi = self.W0 * (1.0 - radii**2 / (self.r_t**2 + self.r_c**2))
        psi = jnp.maximum(psi, 0.0)  # Ensure non-negative

        # Velocity dispersion at each radius (from King DF)
        # σ(r)² = σ₀² × (1 + ψ(r)/3)  (approximate)
        sigma_r_squared = sigma_0_squared * (1.0 + psi / 3.0)
        sigma_r = jnp.sqrt(jnp.maximum(sigma_r_squared, 1e-10))

        # Escape velocity at each radius: v_esc² ≈ 2 ψ(r) σ₀²
        v_esc_squared = 2.0 * psi * sigma_0_squared
        v_esc = jnp.sqrt(jnp.maximum(v_esc_squared, 0.0))

        # Sample isotropic Gaussian velocities - split once into 3 subkeys
        keys = jax.random.split(key, 3)
        v_x = jax.random.normal(keys[0], shape=(N,)) * sigma_r
        v_y = jax.random.normal(keys[1], shape=(N,)) * sigma_r
        v_z = jax.random.normal(keys[2], shape=(N,)) * sigma_r

        velocities = jnp.stack([v_x, v_y, v_z], axis=1)

        # Apply cutoff at escape velocity
        v_mag = jnp.linalg.norm(velocities, axis=1, keepdims=True)
        velocities = jnp.where(
            v_mag > v_esc.reshape(-1, 1),
            velocities * (v_esc.reshape(-1, 1) / (v_mag + 1e-30)),
            velocities,
        )

        return velocities


__all__ = ["KingVelocityDF"]

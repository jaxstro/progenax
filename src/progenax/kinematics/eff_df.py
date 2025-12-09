"""
EFF (Elson-Fall-Freeman) isotropic velocity distribution as Equinox module.

Implements VelocityDF protocol for use with IC assembly.

The EFF profile has no analytic distribution function (unlike Plummer), so we
use isotropic Gaussian velocities with a velocity scale estimated from virial
equilibrium.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray


class EFFVelocityDF(eqx.Module):
    """
    Isotropic Gaussian velocity distribution for EFF profiles.

    Since the EFF profile has no analytic distribution function, we assign
    isotropic Gaussian velocities with a velocity scale estimated from:
        σ ≈ √(G M_total / (6 a))

    This provides a reasonable initial velocity distribution. Global virial
    ratio rescaling is handled by the higher-level kinematics API.

    Attributes:
        a: Scale radius [length units], must match spatial profile
        gamma: Power-law index (for documentation, not used in velocity sampling)
        r_t: Tidal radius [length units], must match spatial profile

    References:
        Elson, Fall & Freeman (1987), ApJ, 323, 54

    Notes:
        - Velocities are isotropic Gaussian (no radial dependence)
        - Fully differentiable and JIT-compatible
        - For virial rescaling, use progenax.kinematics.sample_velocities_pipeline

    Examples:
        >>> from progenax.profiles.eff import EFFProfile
        >>> import jax
        >>> import jax.numpy as jnp
        >>>
        >>> # Create spatial profile and velocity DF
        >>> profile = EFFProfile(a=1.0, gamma=3.0, r_t=10.0)
        >>> velocity_df = EFFVelocityDF(a=1.0, gamma=3.0, r_t=10.0)
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

    a: Float[Array, ""]
    gamma: Float[Array, ""]
    r_t: Float[Array, ""]

    def __init__(self, a: float = 1.0, gamma: float = 3.0, r_t: float = 10.0):
        """
        Initialize EFF velocity distribution function.

        Args:
            a: Scale radius [length units], must match spatial profile
            gamma: Power-law index (for documentation)
            r_t: Tidal radius [length units], must match spatial profile
        """
        self.a = jnp.asarray(a)
        self.gamma = jnp.asarray(gamma)
        self.r_t = jnp.asarray(r_t)

    def sample_velocities(
        self,
        positions: Float[Array, "N 3"],
        masses: Float[Array, "N"],
        key: PRNGKeyArray,
        G: float | None = None,
    ) -> Float[Array, "N 3"]:
        """
        Sample velocities from isotropic Gaussian distribution.

        Velocities are drawn from 3D Gaussian with velocity scale:
            σ ≈ √(G M_total / (6 a))

        This provides a reasonable initial distribution. Global virial ratio
        rescaling is handled by the higher-level kinematics API.

        Args:
            positions: Particle positions (N, 3) [length units]
            masses: Particle masses (N,) [mass units]
            key: JAX random key for reproducible sampling
            G: Gravitational constant. If None, uses jaxstro.units.STELLAR.G
               (~0.00450 for stellar dynamics in pc³ Msun⁻¹ Myr⁻²)

        Returns:
            Cartesian velocities (N, 3) [velocity units]

        Notes:
            - Velocities are isotropic (no radial dependence)
            - All velocities are finite
            - Mean velocity is zero (no bulk motion)
        """
        if G is None:
            from jaxstro.units import STELLAR
            G = STELLAR.G

        N = positions.shape[0]
        M_total = jnp.sum(masses)

        # Velocity scale from virial theorem estimate
        # σ ≈ √(G M_total / (6 a))
        sigma = jnp.sqrt(G * M_total / (6.0 * self.a))

        # Isotropic Gaussian velocities - split once into 3 subkeys
        keys = jax.random.split(key, 3)
        v_x = jax.random.normal(keys[0], shape=(N,)) * sigma
        v_y = jax.random.normal(keys[1], shape=(N,)) * sigma
        v_z = jax.random.normal(keys[2], shape=(N,)) * sigma

        velocities = jnp.stack([v_x, v_y, v_z], axis=1)

        return velocities


__all__ = ["EFFVelocityDF"]

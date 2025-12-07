# progenax/src/progenax/profiles/plummer.py
"""
Plummer (1911) density profile as Equinox module.

Implements SpatialProfile protocol for use with IC assembly.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray


class PlummerProfile(eqx.Module):
    """
    Plummer (1911) spherical density profile.

    ρ(r) = (3M / 4πa³) × (1 + r²/a²)^(-5/2)

    Attributes:
        r_h: Half-mass radius [length units]
        a: Scale radius [length units] (computed from r_h)

    References:
        Plummer (1911) MNRAS 71, 460

    Examples:
        >>> profile = PlummerProfile(r_h=1.0)  # 1 pc
        >>> masses = jnp.ones(100)
        >>> key = jax.random.PRNGKey(42)
        >>> positions = profile.sample_positions(masses, key)
    """

    r_h: Float[Array, ""]
    a: Float[Array, ""]

    def __init__(self, r_h: float = 1.0):
        """
        Initialize Plummer profile.

        Args:
            r_h: Half-mass radius [length units]
        """
        self.r_h = jnp.asarray(r_h, dtype=jnp.float64)
        # Scale radius from half-mass radius
        # From Plummer CDF: M(<r_h)/M = 0.5 = r_h³ / (r_h² + a²)^(3/2)
        # Solving: a = r_h * sqrt((1 - 0.5^(2/3)) / 0.5^(2/3))
        self.a = self.r_h * jnp.sqrt((1.0 - 0.5**(2/3)) / 0.5**(2/3))

    def sample_positions(
        self,
        masses: Float[Array, "N"],
        key: PRNGKeyArray,
    ) -> Float[Array, "N 3"]:
        """
        Sample 3D positions from Plummer density profile.

        Uses inverse CDF for radii + isotropic angles.

        Args:
            masses: Particle masses (N,) [Msun]
            key: JAX random key

        Returns:
            Cartesian positions (N, 3) [length units]
        """
        N = len(masses)
        key_r, key_theta, key_phi = jax.random.split(key, 3)

        # Sample radii via inverse CDF
        u = jax.random.uniform(key_r, (N,))
        radii = self._sample_radii(u)

        # Sample isotropic angles
        cos_theta = jax.random.uniform(key_theta, (N,), minval=-1.0, maxval=1.0)
        phi = jax.random.uniform(key_phi, (N,), minval=0.0, maxval=2*jnp.pi)

        # Convert to Cartesian
        sin_theta = jnp.sqrt(1.0 - cos_theta**2)
        x = radii * sin_theta * jnp.cos(phi)
        y = radii * sin_theta * jnp.sin(phi)
        z = radii * cos_theta

        positions = jnp.stack([x, y, z], axis=1)

        return positions

    @jax.jit
    def _sample_radii(self, u: Float[Array, "N"]) -> Float[Array, "N"]:
        """
        Sample radii from Plummer profile via inverse CDF.

        Plummer cumulative mass: M(<r) / M_total = r³ / (r² + a²)^(3/2)

        Inverse: r = a × sqrt(u^(2/3) / (1 - u^(2/3)))

        Args:
            u: Uniform random [0,1]

        Returns:
            Radii [length units]

        References:
            Verified against PLUMMER_FIXES.md (v0.3.0)
        """
        # Inverse CDF formula
        # r = a × sqrt(u^(2/3) / (1 - u^(2/3)))
        u_23 = jnp.power(u, 2.0/3.0)
        radii = self.a * jnp.sqrt(u_23 / (1.0 - u_23))

        return radii

    def characteristic_radius(self) -> Float[Array, ""]:
        """
        Return characteristic radius (r_h for Plummer).

        Returns:
            Half-mass radius [length units]
        """
        return self.r_h


__all__ = ["PlummerProfile"]

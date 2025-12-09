# progenax/src/progenax/profiles/uniform.py
"""
Uniform sphere density profile as Equinox module.

Implements SpatialProfile protocol for use with IC assembly.
This is the CW04 '3D0' reference distribution for Q parameter validation.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray


class UniformSphereProfile(eqx.Module):
    """
    Uniform density sphere profile.

    ρ(r) = ρ₀ for r ≤ R, 0 otherwise

    This is the CW04 '3D0' distribution with Q ≈ 0.79.

    Attributes:
        R: Outer radius [length units]

    References:
        Cartwright & Whitworth (2004) MNRAS 348, 589 - Table 1, '3D0'

    Examples:
        >>> profile = UniformSphereProfile(R=1.0)  # 1 pc
        >>> masses = jnp.ones(100)
        >>> key = jax.random.PRNGKey(42)
        >>> positions = profile.sample_positions(masses, key)
    """

    R: Float[Array, ""]

    def __init__(self, R: float = 1.0):
        """
        Initialize uniform sphere profile.

        Args:
            R: Outer radius [length units]. Must be positive.

        Raises:
            ValueError: If R <= 0
        """
        if R <= 0:
            raise ValueError(f"Outer radius R must be positive, got {R}")
        R_arr = jnp.asarray(R, dtype=jnp.float64)
        object.__setattr__(self, "R", R_arr)

    def sample_positions(
        self,
        masses: Float[Array, "N"],
        key: PRNGKeyArray,
    ) -> Float[Array, "N 3"]:
        """
        Sample 3D positions from uniform density sphere.

        Uses inverse CDF for radii + isotropic angles.

        For uniform density in 3D:
            - CDF: M(<r) / M = (r/R)³
            - Inverse: r = R × u^(1/3)

        Args:
            masses: Particle masses (N,) [Msun]. Note: Only the array
                length is used to determine N; mass values are not used
                for position sampling in this profile.
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
        Sample radii from uniform sphere via inverse CDF.

        For uniform density ρ = const:
            M(<r) / M = r³ / R³
            Inverse: r = R × u^(1/3)

        Args:
            u: Uniform random [0,1]

        Returns:
            Radii [length units]
        """
        # Inverse CDF: r = R × u^(1/3)
        # Clamp u to prevent issues at boundaries
        eps = 1e-10
        u_clamped = jnp.clip(u, eps, 1.0)

        radii = self.R * jnp.power(u_clamped, 1.0 / 3.0)

        return radii

    def characteristic_radius(self) -> Float[Array, ""]:
        """
        Return characteristic radius (R for uniform sphere).

        For uniform sphere, half-mass radius r_h = R × (0.5)^(1/3) ≈ 0.794 R.

        Returns:
            Outer radius R [length units]
        """
        return self.R

    def half_mass_radius(self) -> Float[Array, ""]:
        """
        Return half-mass radius.

        For uniform sphere: r_h = R × 0.5^(1/3) ≈ 0.794 R

        Returns:
            Half-mass radius [length units]
        """
        return self.R * jnp.power(0.5, 1.0 / 3.0)

    def density(self, r: Float[Array, "..."]) -> Float[Array, "..."]:
        """
        Unnormalized density profile ρ(r) = 1 for r ≤ R, 0 otherwise.

        The uniform density profile normalized would be:
            ρ(r) = 3M / (4πR³) for r ≤ R

        This method returns the unnormalized form (step function).

        Args:
            r: Radial distances [length units]. Can be any shape.

        Returns:
            Unnormalized density at each radius (same shape as input)
        """
        return jnp.where(r <= self.R, 1.0, 0.0)


__all__ = ["UniformSphereProfile"]

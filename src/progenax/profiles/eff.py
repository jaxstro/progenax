# progenax/src/progenax/profiles/eff.py
"""
EFF (Elson-Fall-Freeman 1987) cluster profile.

Implements the EFF truncated power-law density profile commonly used for
young compact star clusters.

References:
    Elson, Fall & Freeman (1987), ApJ, 323, 54 - Original EFF paper
    Cabrera-Ziri et al. (2016), MNRAS, 457, 809 - EFF fits to young clusters
    Krumholz et al. (2019), ARA&A, 57, 227 - Star cluster structure review
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class EFFProfile(eqx.Module):
    """
    EFF (Elson-Fall-Freeman 1987) truncated power-law profile.

    Density profile:
        ρ(r) = ρ_0 × (1 + r²/a²)^{-γ/2}  for r ≤ r_t
             = 0                          for r > r_t

    This profile is commonly used for young massive clusters and has no
    analytic distribution function (velocities must be assigned separately).

    Attributes:
        a: Scale radius [length units] (core-like region)
        gamma: Power-law index (concentration parameter)
               - γ=2.0: Shallow, extended profile
               - γ=3.0: Intermediate (typical for young clusters)
               - γ=4.0: Steep, concentrated profile
        r_t: Tidal/truncation radius [length units]

    References:
        Elson, Fall & Freeman (1987), ApJ, 323, 54, Eq. 1
    """

    a: Float[Array, ""]
    gamma: Float[Array, ""]
    r_t: Float[Array, ""]

    def __init__(self, a: float = 1.0, gamma: float = 3.0, r_t: float = 10.0):
        """
        Initialize EFF profile.

        Args:
            a: Scale radius [length units] (core-like region)
               Typical values: 0.3-1.0 pc for young massive clusters
            gamma: Power-law index (concentration parameter)
                   Typical value: γ=3.0 for young clusters
            r_t: Tidal/truncation radius [length units]
                 Typical values: 5-20 pc for young compact clusters
        """
        self.a = jnp.asarray(a, dtype=jnp.float64)
        self.gamma = jnp.asarray(gamma, dtype=jnp.float64)
        self.r_t = jnp.asarray(r_t, dtype=jnp.float64)

    def sample_positions(
        self,
        masses: Float[Array, "N"],
        key: PRNGKeyArray,
    ) -> Float[Array, "N 3"]:
        """
        Sample particle positions from EFF density profile.

        Uses numerical inverse CDF sampling since the EFF profile has no
        closed-form CDF for general γ.

        Args:
            masses: Particle masses [mass units]
            key: JAX random key for reproducible sampling

        Returns:
            Particle positions [length units]
        """
        N = len(masses)

        # Sample radii via numerical inverse CDF
        key, subkey = jax.random.split(key)
        radii = self._sample_radii(subkey, N)

        # Isotropic angles
        key, subkey = jax.random.split(key)
        theta = jnp.arccos(1.0 - 2.0 * jax.random.uniform(subkey, shape=(N,)))
        key, subkey = jax.random.split(key)
        phi = 2.0 * jnp.pi * jax.random.uniform(subkey, shape=(N,))

        # Convert to Cartesian
        x = radii * jnp.sin(theta) * jnp.cos(phi)
        y = radii * jnp.sin(theta) * jnp.sin(phi)
        z = radii * jnp.cos(theta)

        return jnp.stack([x, y, z], axis=1)

    def _sample_radii(self, key: PRNGKeyArray, N: int) -> Float[Array, "N"]:
        """
        Sample radii from EFF profile using numerical inverse CDF.

        The cumulative mass M(<r) = 4π ∫₀^r ρ(r') r'² dr' has no closed form
        for general γ, so we compute it numerically and invert via interpolation.

        Args:
            key: JAX random key
            N: Number of particles to sample

        Returns:
            Radii following EFF profile [length units]
        """
        # Create grid for cumulative mass function
        N_grid = 1000
        r_grid = jnp.linspace(0.0, self.r_t, N_grid)

        # Compute density on grid
        rho_grid = self._density_unnormalized(r_grid)

        # Integrand: 4π r² ρ(r)
        integrand = 4.0 * jnp.pi * r_grid**2 * rho_grid

        # Cumulative integral using trapezoid rule
        dr = r_grid[1] - r_grid[0]
        M_cumulative = jnp.cumsum(integrand) * dr

        # Normalize to [0, 1]
        M_total_computed = M_cumulative[-1]
        M_normalized = M_cumulative / (M_total_computed + 1e-30)

        # Generate uniform random numbers
        u = jax.random.uniform(key, shape=(N,))

        # Inverse CDF: interpolate to find r where M_normalized = u
        r_sampled = jnp.interp(u, M_normalized, r_grid)

        # Ensure r ≤ r_t (strict truncation)
        r_sampled = jnp.clip(r_sampled, 0.0, self.r_t)

        return r_sampled

    @jax.jit
    def _density_unnormalized(self, r: Float[Array, "N"]) -> Float[Array, "N"]:
        """
        EFF density profile (unnormalized, ρ_0=1).

        Args:
            r: Radii where to evaluate density [length units]

        Returns:
            Density at each radius (unnormalized)
        """
        rho = 1.0 / jnp.power(1.0 + r**2 / self.a**2, self.gamma / 2.0)

        # Truncate at tidal radius
        rho = jnp.where(r <= self.r_t, rho, 0.0)

        return rho

    def characteristic_radius(self) -> Float[Array, ""]:
        """
        Return characteristic radius (tidal radius for EFF).

        Returns:
            Tidal radius [length units]
        """
        return self.r_t


__all__ = ["EFFProfile"]

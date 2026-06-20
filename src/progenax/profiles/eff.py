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

    3-D volume density:
        rho(r) = rho_0 * (1 + r^2/a^2)^{-gamma/2}  for r <= r_t
               = 0                                   for r > r_t

    Provenance note: Elson, Fall & Freeman (1987) Eq. 1 defines this functional
    form as the *projected surface brightness* mu(r), with gamma the surface
    (projected) slope (their median ~2.6). Here we adopt the same form as the
    3-D *volume* density -- the standard N-body / IC-code convention -- so this
    ``gamma`` is a 3-D density slope, offset by ~1 from EFF87's surface slope
    (Abel projection of r^-gamma gives a surface slope gamma-1). At gamma=5 the
    form reduces exactly to Plummer. No closed-form DF; velocities are assigned
    via Eddington inversion in kinematics.EFFVelocityDF.

    The CDF is precomputed at initialization for efficient sampling.

    Attributes:
        a: Scale radius [length units]
        gamma: 3-D density power-law slope (rho ~ r^-gamma at r >> a)
               - gamma=3.0: typical young-cluster 3-D slope
               - gamma=5.0: reduces to the Plummer profile
        r_t: Tidal/truncation radius [length units]
        _r_grid: Precomputed radial grid for CDF interpolation
        _cdf_grid: Precomputed CDF values on grid

    References:
        Elson, Fall & Freeman (1987), ApJ, 323, 54 (Eq. 1 = surface brightness,
        used here as the 3-D volume density; see docs bibliography note).
    """

    a: Float[Array, ""]
    gamma: Float[Array, ""]
    r_t: Float[Array, ""]
    _r_grid: Float[Array, "n_grid"]
    _cdf_grid: Float[Array, "n_grid"]

    def __init__(
        self,
        a: float = 1.0,
        gamma: float = 3.0,
        r_t: float = 10.0,
        n_grid: int = 1000,
    ):
        """
        Initialize EFF profile with precomputed CDF.

        Args:
            a: Scale radius [length units] (core-like region)
               Typical values: 0.3-1.0 pc for young massive clusters
            gamma: Power-law index (concentration parameter)
                   Typical value: gamma=3.0 for young clusters
            r_t: Tidal/truncation radius [length units]
                 Typical values: 5-20 pc for young compact clusters
            n_grid: Number of grid points for CDF interpolation (default: 1000)
        """
        a_arr = jnp.asarray(a, dtype=jnp.float64)
        gamma_arr = jnp.asarray(gamma, dtype=jnp.float64)
        r_t_arr = jnp.asarray(r_t, dtype=jnp.float64)

        # Build radial grid for CDF — sqrt-stretched (r = r_t * u^2) to
        # concentrate points in the core, consistent with King/Michie (audit R4).
        # Measured EFF core error at r_t/a=100, 0.3a: +4.2% (linear) -> <1%
        # (stretched). `compute_profile_potential` (profiles/api.py) reads this
        # same _r_grid and is kept consistent with a matching non-uniform
        # enclosed-mass integral. Smooth in r_t -> differentiable.
        u_grid = jnp.linspace(0.0, 1.0, n_grid)
        r_grid = r_t_arr * u_grid**2

        # Compute density on grid: rho(r) = (1 + r^2/a^2)^(-gamma/2)
        rho_grid = jnp.power(1.0 + (r_grid / a_arr) ** 2, -gamma_arr / 2.0)
        # Truncate at tidal radius (already satisfied by grid construction)
        rho_grid = jnp.where(r_grid <= r_t_arr, rho_grid, 0.0)

        # Integrand: 4*pi*r^2*rho(r)
        integrand = 4.0 * jnp.pi * r_grid**2 * rho_grid

        # Cumulative mass via the NON-UNIFORM trapezoid rule (2nd-order): the
        # sqrt-stretched grid has variable spacing, so each trapezoid is weighted
        # by its own width diff(r_grid). (The old cumsum(integrand)*dr was a
        # 1st-order Riemann sum mislabeled "trapezoid" — audit M5.)
        M_cum = jnp.concatenate(
            [
                jnp.zeros(1, dtype=integrand.dtype),
                jnp.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * jnp.diff(r_grid)),
            ]
        )

        # Normalize to [0, 1] for CDF
        cdf_grid = M_cum / (M_cum[-1] + 1e-30)

        # Store using object.__setattr__ (future-proof Equinox pattern)
        object.__setattr__(self, "a", a_arr)
        object.__setattr__(self, "gamma", gamma_arr)
        object.__setattr__(self, "r_t", r_t_arr)
        object.__setattr__(self, "_r_grid", r_grid)
        object.__setattr__(self, "_cdf_grid", cdf_grid)

    def sample_positions(
        self,
        masses: Float[Array, "N"],
        key: PRNGKeyArray,
    ) -> Float[Array, "N 3"]:
        """
        Sample particle positions from EFF density profile.

        Uses precomputed CDF for efficient inverse transform sampling.

        Args:
            masses: Particle masses [mass units]. Note: Only the array
                length is used to determine N; mass values are not used
                for position sampling in this profile.
            key: JAX random key for reproducible sampling

        Returns:
            Particle positions [length units]
        """
        N = len(masses)

        # Sample radii via precomputed inverse CDF
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
        Sample radii from precomputed CDF via inverse transform.

        Args:
            key: JAX random key
            N: Number of particles to sample

        Returns:
            Radii following EFF profile [length units]
        """
        # Generate uniform random numbers
        u = jax.random.uniform(key, shape=(N,))

        # Inverse CDF: interpolate to find r where CDF = u
        r_sampled = jnp.interp(u, self._cdf_grid, self._r_grid)

        # Ensure r <= r_t (strict truncation)
        r_sampled = jnp.clip(r_sampled, 0.0, self.r_t)

        return r_sampled

    def density(self, r: Float[Array, "..."]) -> Float[Array, "..."]:
        """
        Unnormalized density profile rho(r) = (1 + r^2/a^2)^{-gamma/2}.

        The EFF density profile normalized would be:
            rho(r) = rho_0 * (1 + r^2/a^2)^{-gamma/2}  for r <= r_t
                   = 0                                   for r > r_t

        This method returns the unnormalized form (rho_0=1), useful for
        plotting and analysis with jaxstroviz.

        Args:
            r: Radial distances [length units]. Can be any shape.

        Returns:
            Unnormalized density at each radius (same shape as input)
        """
        rho = jnp.power(1.0 + (r / self.a) ** 2, -self.gamma / 2.0)
        # Truncate at tidal radius
        return jnp.where(r <= self.r_t, rho, 0.0)

    def characteristic_radius(self) -> Float[Array, ""]:
        """
        Return characteristic radius (tidal radius for EFF).

        Returns:
            Tidal radius [length units]
        """
        return self.r_t


__all__ = ["EFFProfile"]

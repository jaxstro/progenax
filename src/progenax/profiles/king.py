# progenax/src/progenax/profiles/king.py
"""
King (1966) density and potential profiles.

Implements the King (1966) dynamical model for star clusters with tidal cutoff,
widely used for globular cluster simulations. This module contains the core
profile functions (density, potential, cumulative mass) and the KingProfile
class implementing the SpatialProfile protocol.

The King model is characterized by:
- Lowered Maxwellian distribution function
- Tidal truncation at radius r_t
- Concentration parameter W0 (dimensionless central potential)

References:
    King, I. R. (1966), "The Structure of Star Clusters. III. Some Simple
    Dynamical Models", AJ, 71, 64

    Binney & Tremaine (2008), "Galactic Dynamics" (2nd ed.), Section 4.3

Notes:
    - W0 typically ranges from 1 (low concentration) to 12 (high concentration)
    - Globular clusters have W0 ~ 5-9
    - Models with W0 > 12 are unstable (core collapse)
"""

from typing import Tuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


# ==============================================================================
# King K-function (Exact Density-Potential Relation)
# ==============================================================================


@jax.jit
def king_K_function(W: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    King's K function for exact density-potential relation.

    K(W) = erf(sqrt(W)) - (2/sqrt(pi)) sqrt(W) exp(-W)

    This function appears in the exact King (1966) density formula:
        rho(r)/rho_0 = [K(W0) - K(W0 - psi(r))] / K(W0)

    Args:
        W: Dimensionless potential parameter (can be scalar or array)

    Returns:
        K(W) value(s)

    References:
        King (1966), AJ, 71, 64, Eq. 12
        Binney & Tremaine (2008), "Galactic Dynamics", Eq. 4.113

    Note:
        For W < 0, returns 0 (not physical in King models).
        Uses jax.scipy.special.erf for the error function.
    """
    # Clamp BEFORE sqrt/exp so the backward pass never differentiates sqrt at W=0
    # (the classic where-NaN trap, audit C2). The true derivative is
    # dK/dW = (2/sqrt(pi)) sqrt(W) e^{-W}, which is 0 at W=0, so W<=0 selects a
    # constant-0 branch with finite (zero) gradient.
    W_pos = jnp.where(W > 0.0, W, 1.0)  # never feed 0/negative to sqrt
    sqrt_W = jnp.sqrt(W_pos)

    # K(W) = erf(sqrt(W)) - (2/sqrt(pi)) sqrt(W) exp(-W)
    term1 = jax.scipy.special.erf(sqrt_W)
    term2 = (2.0 / jnp.sqrt(jnp.pi)) * sqrt_W * jnp.exp(-W_pos)
    K_pos = term1 - term2

    # W <= 0 is unphysical in King models: K(W) -> 0 (value and gradient).
    K = jnp.where(W > 0.0, K_pos, 0.0)

    return K


# ==============================================================================
# King Profile ODE Solution (Poisson Equation)
# ==============================================================================


def _king_poisson_rhs(xi: float, y: Float[Array, "2"], args: tuple) -> Float[Array, "2"]:
    """
    Right-hand side of King's dimensionless Poisson equation.

    The King (1966) model satisfies:
        d^2 psi/d xi^2 + (2/xi) d psi/d xi = -rho_tilde(psi)

    where xi = r/r_c is dimensionless radius and rho_tilde(psi) is the dimensionless
    density from integrating the King distribution function.

    We convert to first-order system:
        y[0] = psi(xi)
        y[1] = d psi/d xi

    Then:
        dy[0]/d xi = y[1]
        dy[1]/d xi = -rho_tilde(psi) - (2/xi) y[1]

    Args:
        xi: Dimensionless radius xi = r/r_c
        y: State vector [psi, d psi/d xi]
        args: (W0,) - concentration parameter

    Returns:
        Derivative [d psi/d xi, d^2 psi/d xi^2]

    References:
        King (1966), AJ, 71, 64, Eq. 9-10
        Binney & Tremaine (2008), "Galactic Dynamics", Section 4.3.2
    """
    (W0,) = args
    psi, dpsi_dxi = y[0], y[1]

    # Dimensionless density from King DF integration
    # rho_tilde(psi) = [K(W0) - K(W0 - psi)] / K(W0)
    K_W0 = king_K_function(W0)
    K_W0_minus_psi = king_K_function(W0 - psi)

    rho_tilde = jnp.where(K_W0 > 1e-10, (K_W0 - K_W0_minus_psi) / K_W0, 0.0)

    # Poisson equation: d^2 psi/d xi^2 = -rho_tilde(psi) - (2/xi) d psi/d xi
    # Handle singularity at xi=0 using L'Hopital: lim_{xi->0} (2/xi) d psi/d xi = 0
    d2psi_dxi2 = jnp.where(
        xi > 1e-6,
        -rho_tilde - (2.0 / xi) * dpsi_dxi,
        -rho_tilde,  # At center, use psi''(0) = -rho(0)
    )

    return jnp.array([dpsi_dxi, d2psi_dxi2])


def solve_king_profile(
    W0: float, xi_max: float = 100.0, n_points: int = 500
) -> Tuple[Float[Array, "n_points"], Float[Array, "n_points"]]:
    """
    Solve King's Poisson equation numerically using diffrax.

    Integrates from xi=0 (center) outward until psi(xi) -> 0 (tidal radius).

    Boundary conditions (King 1966, Eq. 10):
        psi(0) = W0  (central potential)
        d psi/d xi|_0 = 0  (symmetry at center)

    Args:
        W0: King concentration parameter
        xi_max: Maximum dimensionless radius to integrate to
        n_points: Number of points in output grid

    Returns:
        xi_grid: Dimensionless radii xi = r/r_c
        psi_grid: Dimensionless potential psi(xi)

    References:
        King (1966), AJ, 71, 64
        Binney & Tremaine (2008), Section 4.3.2

    Note:
        JIT-compatible when ``n_points`` (and ``xi_max``) are static: they set the
        ``linspace`` size and are closed over, so ``jax.jit(solve_king_profile)(W0)``
        traces fine (W0 may be a tracer). Uses Tsit5 (Runge-Kutta 5th order) from
        diffrax for robustness.
    """
    # Initial conditions
    y0 = jnp.array([W0, 0.0])  # [psi(0), d psi/d xi|_0]

    # Integration domain
    xi_span = (1e-6, xi_max)  # Start slightly off center to avoid singularity

    # Create ODE term
    term = diffrax.ODETerm(_king_poisson_rhs)

    # Use Tsit5 (adaptive Runge-Kutta)
    solver = diffrax.Tsit5()

    # Adaptive step size controller
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-10)

    # Save at specified points
    saveat = diffrax.SaveAt(ts=jnp.linspace(xi_span[0], xi_span[1], n_points))

    # Solve ODE
    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=xi_span[0],
        t1=xi_span[1],
        dt0=1e-4,
        y0=y0,
        args=(W0,),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=100000,
    )

    xi_grid = solution.ts
    psi_grid = solution.ys[:, 0]  # Extract psi(xi)

    # Ensure psi >= 0 (truncate at tidal radius where psi -> 0)
    psi_grid = jnp.maximum(psi_grid, 0.0)

    return xi_grid, psi_grid


def _find_tidal_radius(
    xi_grid: Float[Array, "n_points"],
    psi_grid: Float[Array, "n_points"],
) -> Float[Array, ""]:
    """
    Find dimensionless tidal radius where psi first crosses zero.

    Args:
        xi_grid: Dimensionless radii from ODE solution
        psi_grid: Dimensionless potential from ODE solution

    Returns:
        xi_t: Dimensionless tidal radius where psi(xi_t) = 0

    Note:
        Uses linear interpolation for precise crossing point.
        If no crossing found, returns last grid point.
    """
    # Find where psi drops to zero (or below due to numerics)
    crossing_mask = psi_grid <= 0
    has_crossing = jnp.any(crossing_mask)
    first_zero_idx = jnp.argmax(crossing_mask)

    # Linear interpolation for precise xi_t
    idx = jnp.maximum(first_zero_idx - 1, 0)
    psi0, psi1 = psi_grid[idx], psi_grid[first_zero_idx]
    xi0, xi1 = xi_grid[idx], xi_grid[first_zero_idx]
    t = psi0 / (psi0 - psi1 + 1e-30)
    xi_t = xi0 + t * (xi1 - xi0)

    # If no crossing, use last point. Return the array (do NOT cast to float):
    # float() concretizes the tracer and breaks jit/grad through from_W0_rc (C2).
    xi_t = jnp.where(has_crossing, xi_t, xi_grid[-1])
    return xi_t


# ==============================================================================
# KingProfile Class (SpatialProfile implementation)
# ==============================================================================


class KingProfile(eqx.Module):
    """
    King (1966) spherical density profile.

    Implements SpatialProfile protocol for IC assembly.

    The CDF is precomputed at initialization for efficient sampling.

    Attributes:
        W0: King concentration parameter (dimensionless)
        r_c: Core radius [length units]
        r_t: Tidal (truncation) radius [length units]
        xi_grid: Pre-computed dimensionless radii from ODE solver
        psi_grid: Pre-computed dimensionless potential from ODE solver
        _r_grid: Precomputed radial grid for CDF interpolation
        _cdf_grid: Precomputed CDF values on grid

    References:
        King (1966), AJ, 71, 64

    Examples:
        # Recommended: Use from_W0_rc for self-consistent model
        >>> profile = KingProfile.from_W0_rc(W0=7.0, r_c=1.0)

        # Or manually with pre-computed ODE solution
        >>> xi_grid, psi_grid = solve_king_profile(W0=7.0)
        >>> profile = KingProfile(W0=7.0, r_c=1.0, r_t=10.0,
        ...                       xi_grid=xi_grid, psi_grid=psi_grid)
        >>> masses = jnp.ones(100)
        >>> key = jax.random.PRNGKey(42)
        >>> positions = profile.sample_positions(masses, key)
    """

    W0: Float[Array, ""]
    r_c: Float[Array, ""]
    r_t: Float[Array, ""]
    xi_grid: Float[Array, "n_points"]
    psi_grid: Float[Array, "n_points"]
    _r_grid: Float[Array, "n_grid"]
    _cdf_grid: Float[Array, "n_grid"]

    def __init__(
        self,
        W0: float,
        r_c: float,
        r_t: float,
        xi_grid: Float[Array, "n_points"],
        psi_grid: Float[Array, "n_points"],
        n_grid: int = 1000,
    ):
        """
        Initialize King profile with precomputed CDF.

        Args:
            W0: King concentration parameter (typical 1-12)
            r_c: Core radius [length units]
            r_t: Tidal radius [length units]
            xi_grid: Pre-computed dimensionless radii from solve_king_profile()
            psi_grid: Pre-computed dimensionless potential from solve_king_profile()
            n_grid: Number of grid points for CDF interpolation (default: 1000)
        """
        W0_arr = jnp.asarray(W0, dtype=jnp.float64)
        r_c_arr = jnp.asarray(r_c, dtype=jnp.float64)
        r_t_arr = jnp.asarray(r_t, dtype=jnp.float64)
        xi_grid_arr = jnp.asarray(xi_grid, dtype=jnp.float64)
        psi_grid_arr = jnp.asarray(psi_grid, dtype=jnp.float64)

        # Build radial grid for CDF
        r_grid = jnp.linspace(0.0, r_t_arr, n_grid)
        xi_grid_local = r_grid / r_c_arr

        # Compute density on grid via interpolation of ODE solution
        psi_vals = jnp.interp(
            xi_grid_local,
            xi_grid_arr,
            psi_grid_arr,
            left=W0_arr,
            right=0.0
        )

        # King density: rho(r)/rho_0 = [K(W0) - K(W0 - psi)] / K(W0)
        K_W0 = king_K_function(W0_arr)
        K_W0_minus_psi = king_K_function(W0_arr - psi_vals)
        rho_grid = jnp.where(
            K_W0 > 1e-10,
            (K_W0 - K_W0_minus_psi) / K_W0,
            0.0
        )

        # Truncate at tidal radius
        rho_grid = jnp.where(r_grid <= r_t_arr, rho_grid, 0.0)

        # Integrand: 4*pi*r^2*rho(r)
        integrand = 4.0 * jnp.pi * r_grid**2 * rho_grid

        # Cumulative mass via the trapezoid rule (2nd-order). The old
        # cumsum(integrand)*dr was a 1st-order left/right-Riemann sum mislabeled
        # "trapezoid" and biased the sampled radial distribution (audit M5).
        dr = r_grid[1] - r_grid[0]
        M_cum = jnp.concatenate([
            jnp.zeros(1, dtype=integrand.dtype),
            jnp.cumsum(0.5 * (integrand[1:] + integrand[:-1])) * dr,
        ])

        # Normalize to [0, 1] for CDF
        cdf_grid = M_cum / (M_cum[-1] + 1e-30)

        # Store using object.__setattr__ (future-proof Equinox pattern)
        object.__setattr__(self, "W0", W0_arr)
        object.__setattr__(self, "r_c", r_c_arr)
        object.__setattr__(self, "r_t", r_t_arr)
        object.__setattr__(self, "xi_grid", xi_grid_arr)
        object.__setattr__(self, "psi_grid", psi_grid_arr)
        object.__setattr__(self, "_r_grid", r_grid)
        object.__setattr__(self, "_cdf_grid", cdf_grid)

    @classmethod
    def from_W0_rc(
        cls,
        W0: float,
        r_c: float,
        xi_max: float = 100.0,
        n_ode_points: int = 500,
        n_grid: int = 1000,
    ) -> "KingProfile":
        """
        Create self-consistent King profile where r_t is derived from W0.

        This is the RECOMMENDED constructor. The tidal radius is computed
        from where the potential psi(xi) crosses zero, ensuring a physically
        self-consistent King model.

        Args:
            W0: King concentration parameter (typical 1-12)
            r_c: Core radius [length units]
            xi_max: Maximum dimensionless radius for ODE integration (default: 100)
            n_ode_points: Number of ODE solution points (default: 500)
            n_grid: Number of grid points for CDF interpolation (default: 1000)

        Returns:
            KingProfile with self-consistent r_t derived from W0

        Examples:
            >>> profile = KingProfile.from_W0_rc(W0=7.0, r_c=1.0)
            >>> print(f"Tidal radius: {profile.r_t:.2f}")
        """
        xi_grid, psi_grid = solve_king_profile(W0, xi_max=xi_max, n_points=n_ode_points)
        xi_t = _find_tidal_radius(xi_grid, psi_grid)
        r_t = r_c * xi_t
        return cls(
            W0=W0,
            r_c=r_c,
            r_t=r_t,
            xi_grid=xi_grid,
            psi_grid=psi_grid,
            n_grid=n_grid,
        )

    def sample_positions(
        self,
        masses: Float[Array, "N"],
        key: PRNGKeyArray,
    ) -> Float[Array, "N 3"]:
        """
        Sample particle positions from King density profile.

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
            Radii following King profile [length units]
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
        Unnormalized density profile rho(r)/rho_0.

        The King density is:
            rho(r)/rho_0 = [K(W0) - K(W0 - psi(r))] / K(W0)

        where K is the King K-function and psi(r) is the dimensionless
        potential obtained from interpolating the ODE solution.

        This method returns the unnormalized form (rho_0=1), useful for
        plotting and analysis with jaxstroviz.

        Args:
            r: Radial distances [length units]. Can be any shape.

        Returns:
            Unnormalized density at each radius (same shape as input)
        """
        xi = r / self.r_c
        psi_vals = jnp.interp(xi, self.xi_grid, self.psi_grid, left=self.W0, right=0.0)

        K_W0 = king_K_function(self.W0)
        K_W0_minus_psi = king_K_function(self.W0 - psi_vals)

        rho = jnp.where(K_W0 > 1e-10, (K_W0 - K_W0_minus_psi) / K_W0, 0.0)

        # Truncate at tidal radius
        return jnp.where(r <= self.r_t, rho, 0.0)

    def characteristic_radius(self) -> Float[Array, ""]:
        """
        Return characteristic radius (tidal radius for King).

        Returns:
            Tidal radius [length units]
        """
        return self.r_t


__all__ = ["KingProfile", "solve_king_profile", "king_K_function"]

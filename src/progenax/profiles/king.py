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
- Concentration parameter W₀ (dimensionless central potential)

References:
    King, I. R. (1966), "The Structure of Star Clusters. III. Some Simple
    Dynamical Models", AJ, 71, 64

    Binney & Tremaine (2008), "Galactic Dynamics" (2nd ed.), Section 4.3

Notes:
    - W₀ typically ranges from 1 (low concentration) to 12 (high concentration)
    - Globular clusters have W₀ ~ 5-9
    - Models with W₀ > 12 are unstable (core collapse)
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

    K(W) = erf(√W) - (2/√π) √W exp(-W)

    This function appears in the exact King (1966) density formula:
        ρ(r)/ρ₀ = [K(W₀) - K(W₀ - ψ(r))] / K(W₀)

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
    # Ensure non-negative (King models have ψ ≥ 0)
    W_safe = jnp.maximum(W, 0.0)
    sqrt_W = jnp.sqrt(W_safe)

    # K(W) = erf(√W) - (2/√π) √W exp(-W)
    term1 = jax.scipy.special.erf(sqrt_W)
    term2 = (2.0 / jnp.sqrt(jnp.pi)) * sqrt_W * jnp.exp(-W_safe)

    K = term1 - term2

    # For W ≈ 0, K(W) → 0, handle numerically
    K = jnp.where(W_safe < 1e-10, 0.0, K)

    return K


# ==============================================================================
# King Profile ODE Solution (Poisson Equation)
# ==============================================================================


def _king_poisson_rhs(xi: float, y: Float[Array, "2"], args: tuple) -> Float[Array, "2"]:
    """
    Right-hand side of King's dimensionless Poisson equation.

    The King (1966) model satisfies:
        d²ψ/dξ² + (2/ξ) dψ/dξ = -ρ̃(ψ)

    where ξ = r/r_c is dimensionless radius and ρ̃(ψ) is the dimensionless
    density from integrating the King distribution function.

    We convert to first-order system:
        y[0] = ψ(ξ)
        y[1] = dψ/dξ

    Then:
        dy[0]/dξ = y[1]
        dy[1]/dξ = -ρ̃(ψ) - (2/ξ) y[1]

    Args:
        xi: Dimensionless radius ξ = r/r_c
        y: State vector [ψ, dψ/dξ]
        args: (W0,) - concentration parameter

    Returns:
        Derivative [dψ/dξ, d²ψ/dξ²]

    References:
        King (1966), AJ, 71, 64, Eq. 9-10
        Binney & Tremaine (2008), "Galactic Dynamics", Section 4.3.2
    """
    (W0,) = args
    psi, dpsi_dxi = y[0], y[1]

    # Dimensionless density from King DF integration
    # ρ̃(ψ) = [K(W₀) - K(W₀ - ψ)] / K(W₀)
    K_W0 = king_K_function(W0)
    K_W0_minus_psi = king_K_function(W0 - psi)

    rho_tilde = jnp.where(K_W0 > 1e-10, (K_W0 - K_W0_minus_psi) / K_W0, 0.0)

    # Poisson equation: d²ψ/dξ² = -ρ̃(ψ) - (2/ξ) dψ/dξ
    # Handle singularity at ξ=0 using L'Hôpital: lim_{ξ→0} (2/ξ) dψ/dξ = 0
    d2psi_dxi2 = jnp.where(
        xi > 1e-6,
        -rho_tilde - (2.0 / xi) * dpsi_dxi,
        -rho_tilde,  # At center, use ψ''(0) = -ρ(0)
    )

    return jnp.array([dpsi_dxi, d2psi_dxi2])


def solve_king_profile(
    W0: float, xi_max: float = 100.0, n_points: int = 500
) -> Tuple[Float[Array, "n_points"], Float[Array, "n_points"]]:
    """
    Solve King's Poisson equation numerically using diffrax.

    Integrates from ξ=0 (center) outward until ψ(ξ) → 0 (tidal radius).

    Boundary conditions (King 1966, Eq. 10):
        ψ(0) = W₀  (central potential)
        dψ/dξ|₀ = 0  (symmetry at center)

    Args:
        W0: King concentration parameter
        xi_max: Maximum dimensionless radius to integrate to
        n_points: Number of points in output grid

    Returns:
        xi_grid: Dimensionless radii ξ = r/r_c
        psi_grid: Dimensionless potential ψ(ξ)

    References:
        King (1966), AJ, 71, 64
        Binney & Tremaine (2008), Section 4.3.2

    Note:
        Cannot be JIT-compiled due to n_points (concrete value needed for linspace).
        Uses Tsit5 (Runge-Kutta 5th order) from diffrax for robustness.
    """
    # Initial conditions
    y0 = jnp.array([W0, 0.0])  # [ψ(0), dψ/dξ|₀]

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
    psi_grid = solution.ys[:, 0]  # Extract ψ(ξ)

    # Ensure ψ ≥ 0 (truncate at tidal radius where ψ → 0)
    psi_grid = jnp.maximum(psi_grid, 0.0)

    return xi_grid, psi_grid


# ==============================================================================
# KingProfile Class (SpatialProfile implementation)
# ==============================================================================


class KingProfile(eqx.Module):
    """
    King (1966) spherical density profile.

    Implements SpatialProfile protocol for IC assembly.

    Attributes:
        W0: King concentration parameter (dimensionless)
        r_c: Core radius [length units]
        r_t: Tidal (truncation) radius [length units]
        xi_grid: Pre-computed dimensionless radii from ODE solver
        psi_grid: Pre-computed dimensionless potential from ODE solver

    References:
        King (1966), AJ, 71, 64

    Examples:
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

    def __init__(
        self,
        W0: float,
        r_c: float,
        r_t: float,
        xi_grid: Float[Array, "n_points"],
        psi_grid: Float[Array, "n_points"],
    ):
        """
        Initialize King profile.

        Args:
            W0: King concentration parameter (typical 1-12)
            r_c: Core radius [length units]
            r_t: Tidal radius [length units]
            xi_grid: Pre-computed dimensionless radii from solve_king_profile()
            psi_grid: Pre-computed dimensionless potential from solve_king_profile()
        """
        self.W0 = jnp.asarray(W0, dtype=jnp.float64)
        self.r_c = jnp.asarray(r_c, dtype=jnp.float64)
        self.r_t = jnp.asarray(r_t, dtype=jnp.float64)
        self.xi_grid = jnp.asarray(xi_grid, dtype=jnp.float64)
        self.psi_grid = jnp.asarray(psi_grid, dtype=jnp.float64)

    def sample_positions(
        self,
        masses: Float[Array, "N"],
        key: PRNGKeyArray,
    ) -> Float[Array, "N 3"]:
        """
        Sample particle positions from King density profile.

        Uses numerical inverse CDF sampling since the King profile has no
        closed-form CDF.

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
        Sample radii from King profile using numerical inverse CDF.

        The cumulative mass M(<r) has no closed form, so we compute it
        numerically and invert via interpolation.

        Args:
            key: JAX random key
            N: Number of particles to sample

        Returns:
            Radii following King profile [length units]
        """
        # Create grid for cumulative mass function
        N_grid = 1000
        r_grid = jnp.linspace(0.0, self.r_t, N_grid)
        xi_grid_local = r_grid / self.r_c

        # Compute density on grid via interpolation of ODE solution
        psi_vals = jnp.interp(
            xi_grid_local,
            self.xi_grid,
            self.psi_grid,
            left=self.W0,
            right=0.0
        )

        # King density: ρ(r)/ρ₀ = [K(W₀) - K(W₀ - ψ)] / K(W₀)
        K_W0 = king_K_function(self.W0)
        K_W0_minus_psi = king_K_function(self.W0 - psi_vals)
        rho_grid = jnp.where(
            K_W0 > 1e-10,
            (K_W0 - K_W0_minus_psi) / K_W0,
            0.0
        )

        # Truncate at tidal radius
        rho_grid = jnp.where(r_grid <= self.r_t, rho_grid, 0.0)

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

    def characteristic_radius(self) -> Float[Array, ""]:
        """
        Return characteristic radius (tidal radius for King).

        Returns:
            Tidal radius [length units]
        """
        return self.r_t


__all__ = ["KingProfile", "solve_king_profile", "king_K_function"]

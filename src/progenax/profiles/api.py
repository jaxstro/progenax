# progenax/src/progenax/profiles/api.py
"""
Functional API for spatial density profiles.

Provides profile-agnostic functions for:
- Creating profile instances via factory function
- Sampling positions from density profiles
- Computing analytic gravitational potentials

This API enables the cluster IC generator to work with any supported
density profile without depending on specific profile class implementations.

Example:
    >>> from progenax.profiles.api import sample_density_profile, compute_profile_potential
    >>> import jax
    >>>
    >>> key = jax.random.PRNGKey(42)
    >>> positions = sample_density_profile(key, N_stars=1000, profile="plummer", R_half=1.0)
    >>>
    >>> # Compute potential at those positions
    >>> from jaxstro.units import STELLAR
    >>> phi = compute_profile_potential(
    ...     positions, profile="plummer", M_total=1000.0, R_half=1.0, G=STELLAR.G
    ... )
"""

from typing import Literal, Union

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, PRNGKeyArray

from progenax.profiles.eff import EFFProfile
from progenax.profiles.king import KingProfile, king_lowered_maxwellian_density
from progenax.profiles.plummer import PlummerProfile

# Type alias for supported profile names
ProfileName = Literal["plummer", "king", "eff"]


def make_profile(
    name: ProfileName,
    R_half: float,
    **kwargs,
) -> Union[PlummerProfile, KingProfile, EFFProfile]:
    """
    Factory function for creating profile instances.

    This is the primary entry point for creating density profiles. It provides
    a uniform interface where R_half is the external scale parameter, with
    profile-specific shape parameters passed via kwargs.

    Args:
        name: Profile type - "plummer", "king", or "eff"
        R_half: Half-mass radius (for Plummer) or characteristic radius (for King/EFF)
                in length units [pc in stellar units].

                - **Plummer**: This is the actual half-mass radius r_h
                - **King**: This is treated as the core radius r_c (not half-mass)
                - **EFF**: This is treated as the scale radius a (not half-mass)

        **kwargs: Profile-specific parameters:

            **King profile**:
                - W0: float = 5.0 - King concentration parameter (typical 1-12)
                - n_grid: int = 1000 - CDF interpolation grid points

            **EFF profile**:
                - gamma: float = 3.0 - Power-law index (concentration)
                - r_t: float = 10*R_half - Tidal/truncation radius
                - n_grid: int = 1000 - CDF interpolation grid points

    Returns:
        Profile instance (PlummerProfile, KingProfile, or EFFProfile)

    Raises:
        ValueError: If profile name is not recognized

    Examples:
        >>> # Plummer profile with 1 pc half-mass radius
        >>> profile = make_profile("plummer", R_half=1.0)

        >>> # King profile with W0=7 (globular cluster typical)
        >>> profile = make_profile("king", R_half=1.0, W0=7.0)

        >>> # EFF profile with gamma=3 (young cluster typical)
        >>> profile = make_profile("eff", R_half=1.0, gamma=3.0, r_t=15.0)

    Notes:
        For King and EFF profiles, the mapping from R_half to internal parameters
        is simplified: R_half is used directly as r_c (King) or a (EFF). For
        precise half-mass radius control, the user should compute the appropriate
        r_c or a value externally.
    """
    name_lower = name.lower()

    if name_lower == "plummer":
        return PlummerProfile(r_h=R_half)

    elif name_lower == "king":
        W0 = kwargs.get("W0", 5.0)
        n_grid = kwargs.get("n_grid", 1000)
        # Use R_half as core radius r_c
        # Tidal radius is derived self-consistently from W0
        return KingProfile.from_W0_rc(W0=W0, r_c=R_half, n_grid=n_grid)

    elif name_lower == "eff":
        gamma = kwargs.get("gamma", 3.0)
        # Default tidal radius is 10x the scale radius
        r_t = kwargs.get("r_t", 10.0 * R_half)
        n_grid = kwargs.get("n_grid", 1000)
        # Use R_half as scale radius a
        return EFFProfile(a=R_half, gamma=gamma, r_t=r_t, n_grid=n_grid)

    else:
        raise ValueError(
            f"Unknown profile type: '{name}'. "
            f"Supported profiles: 'plummer', 'king', 'eff'"
        )


def sample_density_profile(
    key: PRNGKeyArray,
    N_stars: int,
    profile: ProfileName,
    R_half: float,
    **kwargs,
) -> Float[Array, "N 3"]:
    """
    Sample N_stars positions from the chosen density profile.

    This function creates a profile instance and samples positions from its
    density distribution using inverse CDF sampling.

    Args:
        key: JAX random key for reproducibility
        N_stars: Number of stars (positions) to sample
        profile: Profile type - "plummer", "king", or "eff"
        R_half: Half-mass/characteristic radius in length units [pc]
        **kwargs: Profile-specific parameters (passed to make_profile)
                  See make_profile() docstring for details.

    Returns:
        positions: Array of shape (N_stars, 3) in length units [pc]
                   Positions are sampled from the density profile's
                   radial distribution with isotropic angular distribution.

    Examples:
        >>> import jax
        >>> key = jax.random.PRNGKey(42)
        >>>
        >>> # Sample 1000 positions from Plummer profile
        >>> positions = sample_density_profile(key, 1000, "plummer", R_half=1.0)
        >>> positions.shape
        (1000, 3)

        >>> # Sample from King profile with specific concentration
        >>> positions = sample_density_profile(key, 1000, "king", R_half=1.0, W0=7.0)

    Notes:
        - All profiles sample radii via inverse CDF for efficiency
        - Angular distribution is isotropic (uniform on sphere)
        - The masses argument required by profile.sample_positions() is
          filled with ones internally (mass values don't affect spatial sampling)
    """
    # Create profile instance
    profile_instance = make_profile(profile, R_half, **kwargs)

    # Create dummy masses array (only length matters for position sampling)
    masses = jnp.ones(N_stars)

    # Sample positions from profile
    return profile_instance.sample_positions(masses, key)


def compute_profile_potential(
    positions: Float[Array, "N 3"],
    profile: ProfileName,
    M_total: float,
    R_half: float,
    G: float,
    **kwargs,
) -> Float[Array, "N"]:
    """
    Compute analytic gravitational potential at given positions.

    This function computes Phi(r) for the specified density profile.
    The potential is normalized to the total mass M_total and gravitational
    constant G.

    Args:
        positions: Particle positions, shape (N, 3) [length units]
        profile: Profile type - "plummer", "king", or "eff"
        M_total: Total cluster mass [mass units, typically Msun]
        R_half: Half-mass/characteristic radius [length units, typically pc]
        G: Gravitational constant in consistent units
           (use jaxstro.units.STELLAR.G for stellar units)
        **kwargs: Profile-specific parameters (passed to make_profile)

    Returns:
        phi: SPECIFIC gravitational potential (per unit mass) at each position,
             shape (N,). All values are negative (bound system).
             Units: [length units]^2 / [time units]^2 (e.g. pc^2/Myr^2 in
             STELLAR units) — consistent with specific kinetic energy 0.5*v^2,
             so E = 0.5*v^2 + phi is a specific orbital energy.

    Examples:
        >>> from jaxstro.units import STELLAR
        >>> import jax.numpy as jnp
        >>>
        >>> positions = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        >>> phi = compute_profile_potential(
        ...     positions, "plummer", M_total=1000.0, R_half=1.0, G=STELLAR.G
        ... )

    Notes:
        **Plummer potential** (analytic):
            Phi(r) = -G * M_total / sqrt(r^2 + a^2)
            where a = R_half * sqrt(2^(2/3) - 1) is the scale radius.

        **King potential** (exact relative potential):
            Phi(r) = -sigma^2 * psi(r), with psi the dimensionless King ODE
            potential (psi(r_t) = 0) and sigma^2 = G M / (9 r_c mu(W0)) the
            self-consistent velocity scale (matches KingVelocityDF).

        **EFF potential** (true spherical potential):
            Phi(r) = -G [ M(<r)/r + 4 pi int_r^rt rho s ds ], using the exact
            enclosed mass from the profile's density grid (interior monopole +
            outer-shell term).

        The potential is computed per-particle and vectorized for efficiency.
    """
    # Compute radii from positions
    r = jnp.linalg.norm(positions, axis=1)

    profile_lower = profile.lower()

    if profile_lower == "plummer":
        # Plummer scale radius from half-mass radius
        # a = r_h * sqrt((1 - 0.5^(2/3)) / 0.5^(2/3)) ≈ 0.7664 * r_h
        a = R_half * jnp.sqrt((1.0 - 0.5 ** (2 / 3)) / 0.5 ** (2 / 3))

        # Exact Plummer potential: Phi(r) = -G*M / sqrt(r^2 + a^2)
        phi = -G * M_total / jnp.sqrt(r**2 + a**2)

    elif profile_lower == "king":
        # True King relative potential V(r) = -sigma^2 * psi(r), with psi(r_t)=0.
        # psi is the dimensionless potential from the King ODE; sigma is the
        # self-consistent velocity scale sigma^2 = G M / (9 r_c mu(W0)), matching
        # KingVelocityDF so energies are consistent with the sampled velocities.
        profile_instance = make_profile(profile, R_half, **kwargs)
        # profile_lower == "king" => make_profile returned a KingProfile.
        assert isinstance(profile_instance, KingProfile)
        rho0 = king_lowered_maxwellian_density(profile_instance.W0)
        rho_tilde = jnp.where(
            rho0 > 1e-10,
            king_lowered_maxwellian_density(profile_instance.psi_grid) / rho0,
            0.0,
        )
        mu = jnp.trapezoid(
            rho_tilde * profile_instance.xi_grid**2, profile_instance.xi_grid
        )
        sigma_sq = G * M_total / (9.0 * profile_instance.r_c * mu)
        xi = r / profile_instance.r_c
        psi = jnp.interp(
            xi,
            profile_instance.xi_grid,
            profile_instance.psi_grid,
            left=profile_instance.W0,
            right=0.0,
        )
        phi = -sigma_sq * psi

    elif profile_lower == "eff":
        # True EFF potential from the exact enclosed mass on the profile grid:
        #   Phi(r) = -G [ M(<r)/r + 4 pi int_r^rt rho s ds ]
        # (interior monopole + outer-shell term). Uses the same density grid as
        # the sampler, and is jit/grad-safe in gamma (no Python branch on gamma).
        profile_instance = make_profile(profile, R_half, **kwargs)
        # profile_lower == "eff" => make_profile returned an EFFProfile.
        assert isinstance(profile_instance, EFFProfile)
        rgrid = profile_instance._r_grid
        rho_t = (1.0 + (rgrid / profile_instance.a) ** 2) ** (
            -profile_instance.gamma / 2.0
        )
        # EFFProfile uses a NON-UNIFORM (sqrt-stretched) _r_grid (audit R4), so
        # the enclosed-mass / outer-shell integrals must weight each trapezoid by
        # its own width diff(rgrid) — a single dr would mis-integrate the core.
        # This keeps M_enc_frac == profile._cdf_grid (the test pins it).
        dr = jnp.diff(rgrid)

        def _cumtrap_nonuniform(y):
            return jnp.concatenate(
                [
                    jnp.zeros(1, dtype=y.dtype),
                    jnp.cumsum(0.5 * (y[1:] + y[:-1]) * dr),
                ]
            )

        I2 = _cumtrap_nonuniform(
            rho_t * rgrid**2
        )  # propto M(<r); I2[-1] propto M_total
        M_enc_frac = I2 / (I2[-1] + 1e-30)  # M(<r)/M_total (= profile CDF)
        J_outer = _cumtrap_nonuniform(rho_t * rgrid)
        J_outer = J_outer[-1] - J_outer  # int_r^rt rho_t s ds
        phi_grid = (
            -G
            * M_total
            * (
                M_enc_frac / jnp.maximum(rgrid, 1e-3 * profile_instance.a)
                + J_outer / (I2[-1] + 1e-30)
            )
        )
        phi = jnp.interp(r, rgrid, phi_grid)

    else:
        raise ValueError(
            f"Unknown profile type: '{profile}'. "
            f"Supported profiles: 'plummer', 'king', 'eff'"
        )

    return phi


__all__ = [
    "ProfileName",
    "make_profile",
    "sample_density_profile",
    "compute_profile_potential",
]

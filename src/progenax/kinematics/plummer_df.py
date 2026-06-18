"""
Plummer (1911) velocity distribution function as Equinox module.

Implements VelocityDF protocol for use with IC assembly.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

from progenax.kinematics.eddington import (
    assign_om_directions,
    sample_speed_from_f_table,
)

# Resolution of the analytic Osipkov-Merritt speed inverse-CDF table.
_N_OM_E = 1000


def _plummer_om_f_table(rho_a, n_e=_N_OM_E):
    """Analytic Osipkov-Merritt Plummer DF table (Merritt 1985, Eq. 45), dimensionless.

    x = Q/sigma_0^2 in [0, 6]; rho_a = (r_0/r_a)^2 = (a/r_a)^2. The OM DF is

        f_I(x) ∝ x^{7/2} [1 - rho_a + (63/4) rho_a x^{-2}]
               = (1 - rho_a) x^{7/2} + (63/4) rho_a x^{3/2}     (>= 0 for r_a >= 0.75 a).

    The overall positive constant is irrelevant for inverse-CDF sampling.
    """
    x = jnp.linspace(0.0, 6.0, n_e)
    f = (1.0 - rho_a) * x**3.5 + (63.0 / 4.0) * rho_a * x**1.5
    return x, f


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
        Merritt (1985), AJ, 90, 1027, Eq. 42 - explicit isotropic Plummer DF f(E) ∝ (−E)^(7/2)

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
        >>> from jaxstro.units import STELLAR
        >>> velocities = velocity_df.sample_velocities(positions, masses, key_vel, G=STELLAR.G)
    """

    r_h: Float[Array, ""]
    a: Float[Array, ""]
    anisotropy_radius: Float[Array, ""] | None
    _om_E_grid: Float[Array, "n_e"] | None
    _om_f_grid: Float[Array, "n_e"] | None

    def __init__(self, r_h: float = 1.0, anisotropy_radius: float | None = None):
        """
        Initialize Plummer velocity distribution function.

        Args:
            r_h: Half-mass radius [length units], must match spatial profile
            anisotropy_radius: Osipkov-Merritt radius r_a [length units], or None for the
                isotropic DF. When set, velocities follow the analytic OM Plummer DF
                (Merritt 1985, Eq. 45) with beta(r) = r^2/(r^2 + r_a^2). Requires
                r_a >= 0.75 a (Eq. 46); smaller r_a makes the DF negative (refused).
        """
        self.r_h = jnp.asarray(r_h)
        # Scale radius from half-mass radius
        # From Plummer (1911): M(<r)/M = r³/(r²+a²)^(3/2)
        # At r = r_h: 0.5 = r_h³/(r_h²+a²)^(3/2)
        # Solving: a = r_h * sqrt(2^(2/3) - 1) ≈ 0.7664 * r_h
        self.a = self.r_h * jnp.sqrt(2**(2/3) - 1)

        if anisotropy_radius is None:
            self.anisotropy_radius = None
            self._om_E_grid = None
            self._om_f_grid = None
        else:
            self.anisotropy_radius = jnp.asarray(anisotropy_radius)
            # Eager non-negativity guard for a concrete r_a (Merritt 1985, Eq. 46). Under
            # tracing (e.g. jax.grad w.r.t. r_a) the check is skipped and the caller owns
            # the r_a >= 0.75 a bound; the table itself stays traced-safe (jnp only).
            if isinstance(anisotropy_radius, (int, float)):
                a_val = float(self.a)
                if anisotropy_radius < 0.75 * a_val:
                    raise ValueError(
                        f"Plummer Osipkov-Merritt DF requires anisotropy_radius >= 0.75 a "
                        f"= {0.75 * a_val:.4f} (Merritt 1985, Eq. 46); got "
                        f"{anisotropy_radius}. Smaller r_a makes the phase-space DF negative."
                    )
            rho_a = (self.a / self.anisotropy_radius) ** 2  # (r_0/r_a)^2, traced-safe
            E_grid, f_grid = _plummer_om_f_table(rho_a)
            self._om_E_grid = E_grid
            self._om_f_grid = f_grid

    def sample_velocities(
        self,
        positions: Float[Array, "N 3"],
        masses: Float[Array, "N"],
        key: PRNGKeyArray,
        G: float,
    ) -> Float[Array, "N 3"]:
        """
        Sample velocities from Plummer distribution function.

        Samples velocity magnitudes from exact Plummer DF (Beta distribution),
        then assigns isotropic random directions.

        Args:
            positions: Particle positions (N, 3) [length units]
            masses: Particle masses (N,) [M☉]
            key: JAX random key
            G: Gravitational constant (REQUIRED, explicit-units policy). E.g.
               ``STELLAR.G`` ~0.00450 in pc³ Msun⁻¹ Myr⁻².

        Returns:
            Cartesian velocities (N, 3) [velocity units]

        Notes:
            - Velocities are isotropic (no radial bias)
            - All velocities satisfy v < v_esc (bound particles)
            - Statistical properties match Plummer (1911) exactly
        """
        N = positions.shape[0]
        radii = jnp.linalg.norm(positions, axis=1)

        if self.anisotropy_radius is None:
            # Isotropic Plummer DF via exact Beta(3/2, 9/2) speed sampling.
            key_mag, key_theta, key_phi = jax.random.split(key, 3)
            v_magnitudes = self._sample_velocity_magnitudes(radii, masses, key_mag, G)
            cos_theta = jax.random.uniform(key_theta, (N,), minval=-1.0, maxval=1.0)
            phi = jax.random.uniform(key_phi, (N,), minval=0.0, maxval=2 * jnp.pi)
            sin_theta = jnp.sqrt(1.0 - cos_theta**2)
            vx = v_magnitudes * sin_theta * jnp.cos(phi)
            vy = v_magnitudes * sin_theta * jnp.sin(phi)
            vz = v_magnitudes * cos_theta
            return jnp.stack([vx, vy, vz], axis=1)

        # Osipkov-Merritt anisotropic Plummer DF (Merritt 1985, Eq. 45). Dimensionless
        # units: sigma_0^2 = G M / (6 a); psi(r) = 6 (1 + r^2/a^2)^{-1/2}. Sample the
        # speed from s^2 f(psi - s^2/2) (shared inverse-CDF), then assign OM directions.
        M_total = jnp.sum(masses)
        sigma0 = jnp.sqrt(G * M_total / (6.0 * self.a))
        psi_r = 6.0 / jnp.sqrt(1.0 + (radii / self.a) ** 2)

        key_speed, key_dir = jax.random.split(key)
        speed_keys = jax.random.split(key_speed, N)
        s = jax.vmap(
            lambda k, p: sample_speed_from_f_table(k, p, self._om_E_grid, self._om_f_grid)
        )(speed_keys, psi_r)
        speeds = sigma0 * s
        return assign_om_directions(key_dir, positions, speeds, self.anisotropy_radius)

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
            Merritt (1985), AJ, 90, 1027, Eq. 42 - explicit isotropic Plummer DF f(E) ∝ (−E)^(7/2)

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

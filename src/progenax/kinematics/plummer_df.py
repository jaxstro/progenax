"""
Plummer (1911) velocity distribution function as Equinox module.

Implements VelocityDF protocol for use with IC assembly.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxstro.numerics.sampling import inverse_cdf_draw
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray

from progenax.kinematics.eddington import assign_om_directions
from progenax.numerics import require_positive


def _om_speed_quantile(uniform, psi, rho_a):
    """Invert the OM auxiliary-speed CDF (Merritt 1985, Eq. 45).

    uniform is in [0,1], psi=Psi/sigma0^2 in (0,6], rho_a=(a/r_a)^2
    in [0,16/9]. The returned speed is in units of sigma0.

    Evaluate the analytic DF on q=s/sqrt(2 psi) in [0,1]. The common
    psi^(3/2) factor cancels on normalization. Inversion uses 256 points;
    its derivatives are piecewise smooth between interpolation knots.
    """
    q = jnp.linspace(0.0, 1.0, 256)
    A = (1.0 - rho_a) * psi**2
    B = (63.0 / 4.0) * rho_a
    # Scale out amplitude before integration; A can be negative for r_a<a.
    weight = q**2 * (1.0 - q**2) ** 1.5 * (A * (1.0 - q**2) ** 2 + B) / (psi**2 + B)
    return jnp.sqrt(2.0 * psi) * inverse_cdf_draw(weight, q, uniform)


class PlummerVelocityDF(eqx.Module):
    """
    Velocity distribution for the Plummer density model.

    Samples velocity magnitudes from the exact Plummer DF using Beta distribution
    in the default isotropic branch. An optional
    anisotropy radius selects the Osipkov-Merritt branch. The relations below
    describe the isotropic branch.

    The distribution for q = v/v_esc is:
        g(q) ∝ q² (1 - q²)^(7/2)  for q ∈ [0, 1]

    The isotropic phase-space distribution function is:
        f(ℰ) ∝ ℰ^(7/2), where ℰ = ψ - v²/2 is the specific binding energy.
    The mass distribution per unit energy also includes the density of states.

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
        - The isotropic conditional speed law is sampled through a Beta variate.
        - Finite samples fluctuate around the analytical moments.
        - Isotropic fixed-draw scale derivatives are checked for r_h, G, and total mass.
        - These checks do not establish derivatives of population expectations.
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

    def __init__(
        self, r_h: ArrayLike = 1.0, anisotropy_radius: ArrayLike | None = None
    ):
        """
        Initialize Plummer velocity distribution function.

        Args:
            r_h: Positive half-mass radius [length units], must match spatial profile
            anisotropy_radius: Osipkov-Merritt radius r_a [length units], or None for the
                isotropic DF. When set, velocities follow the analytic OM Plummer DF
                (Merritt 1985, Eq. 45) with beta(r) = r^2/(r^2 + r_a^2). Requires
                r_a >= 0.75 a (Eq. 46). Concrete scalar inputs are checked; callers
                must enforce this domain for traced parameters.
        """
        require_positive(r_h, "PlummerVelocityDF r_h")
        self.r_h = jnp.asarray(r_h)
        # Scale radius from half-mass radius
        # From Plummer (1911): M(<r)/M = r³/(r²+a²)^(3/2)
        # At r = r_h: 0.5 = r_h³/(r_h²+a²)^(3/2)
        # Solving: a = r_h * sqrt(2^(2/3) - 1) ≈ 0.7664 * r_h
        self.a = self.r_h * jnp.sqrt(2 ** (2 / 3) - 1)

        if anisotropy_radius is None:
            self.anisotropy_radius = None
        else:
            self.anisotropy_radius = jnp.asarray(anisotropy_radius)
            # Eager non-negativity guard for a concrete r_a (Merritt 1985, Eq. 46). Under
            # tracing (e.g. jax.grad w.r.t. r_a) the check is skipped and the caller owns
            # the r_a >= 0.75 a bound; the table itself stays traced-safe (jnp only).
            if not isinstance(
                self.anisotropy_radius, jax.core.Tracer
            ) and not isinstance(self.a, jax.core.Tracer):
                a_val = float(self.a)
                if float(self.anisotropy_radius) < 0.75 * a_val:
                    raise ValueError(
                        f"Plummer Osipkov-Merritt DF requires anisotropy_radius >= 0.75 a "
                        f"= {0.75 * a_val:.4f} (Merritt 1985, Eq. 46); got "
                        f"{anisotropy_radius}. Smaller r_a makes the phase-space DF negative."
                    )

    def sample_velocities(
        self,
        positions: Float[Array, "N 3"],
        masses: Float[Array, "N"],
        key: PRNGKeyArray,
        G: float,
    ) -> Float[Array, "N 3"]:
        """
        Sample velocities from Plummer distribution function.

        The default branch uses the exact isotropic Beta speed law. The OM
        branch numerically inverts its analytic DF and applies a Cartesian
        direction stretch.

        Args:
            positions: Particle positions (N, 3) [length units]
            masses: Particle masses (N,) [M☉]
            key: JAX random key
            G: Gravitational constant (REQUIRED, explicit-units policy). E.g.
               ``STELLAR.G`` ~0.00450 in pc³ Msun⁻¹ Myr⁻².

        Returns:
            Cartesian velocities (N, 3) [velocity units]

        Notes:
            - The default branch assigns isotropic directions; a finite anisotropy_radius
              selects the separate Osipkov-Merritt construction.
            - All velocities satisfy v < v_esc (bound particles)
            - Finite samples fluctuate around the analytical moments.
        """
        N = positions.shape[0]
        r_squared = jnp.sum(positions**2, axis=1)

        if self.anisotropy_radius is None:
            # Isotropic Plummer DF via exact Beta(3/2, 9/2) speed sampling.
            key_mag, key_theta, key_phi = jax.random.split(key, 3)
            v_magnitudes = self._sample_velocity_magnitudes(
                r_squared, masses, key_mag, G
            )
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
        psi_r = 6.0 / jnp.sqrt(1.0 + r_squared / self.a**2)

        key_speed, key_dir = jax.random.split(key)
        speed_keys = jax.random.split(key_speed, N)
        rho_a = (self.a / self.anisotropy_radius) ** 2
        s = jax.vmap(lambda k, p: _om_speed_quantile(jax.random.uniform(k), p, rho_a))(
            speed_keys, psi_r
        )
        speeds = sigma0 * s
        return assign_om_directions(key_dir, positions, speeds, self.anisotropy_radius)

    def _sample_velocity_magnitudes(
        self,
        r_squared: Float[Array, "N"],
        masses: Float[Array, "N"],
        key: PRNGKeyArray,
        G: float,
    ) -> Float[Array, "N"]:
        """
        Sample velocity magnitudes from Plummer distribution function.

        Uses the isotropic Plummer conditional speed law via Beta distribution sampling.
        The distribution for q = v/v_esc is:
            g(q) ∝ q² (1 - q²)^(7/2)  for q ∈ [0, 1]

        Sampling method:
            Let u = q², then u ~ Beta(3/2, 9/2)
            Therefore: q = sqrt(u), v = q × v_esc

        Args:
            r_squared: Squared particle radii [length units squared]
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
            - The isotropic conditional speed law is sampled through a Beta variate.
            - Finite samples fluctuate around the analytical moments.
            - Isotropic fixed-draw scale derivatives are checked separately from population expectations.
            - For Plummer sphere: v_esc² = 2GM/√(r²+a²)
            - Verified: v_esc = sqrt(12) × σ (exact Plummer relation)
        """
        N = r_squared.shape[0]

        # Escape velocity at radius r
        # v_esc² = 2|ψ(r)| = 2GM/√(r²+a²)
        M_total = jnp.sum(masses)
        v_esc = jnp.sqrt(2.0 * G * M_total / jnp.sqrt(r_squared + self.a**2))

        # Sample q² from Beta(3/2, 9/2)
        # This gives g(q) ∝ q²(1-q²)^(7/2) exactly!
        u = jax.random.beta(key, a=1.5, b=4.5, shape=(N,))  # u = q² ∈ [0,1]
        q = jnp.sqrt(u)  # q = v/v_esc

        # Velocity magnitude
        v = q * v_esc

        return v


__all__ = ["PlummerVelocityDF"]

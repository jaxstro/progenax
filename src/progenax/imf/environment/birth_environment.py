"""BirthEnvironment inference target (split from environment.py)."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from .coefficients import (
    JERABKOVA_COEFFICIENTS,
    MARKS_COEFFICIENTS,
    MARKS_TABLE3_COEFFICIENTS,
    DEFAULT_SFE,
)
from .density import (
    compute_r_half,
    compute_rho_ecl,
    compute_rho_cl,
    compute_log_rho_cl_6,
)


# =============================================================================
# BirthEnvironment - Primary inference target
# =============================================================================

class BirthEnvironment(eqx.Module):
    """Physical birth environment with SFE parameter.

    Represents conditions during star formation that influence the IMF.
    All fields are JAX arrays for gradient-based inference.

    Attributes:
        metallicity: [Fe/H], default 0.0 (solar). Calibrated range [-2.5, +0.5]
        log_mecl: log₁₀(M_ecl / M☉). Calibrated range [3, 8]
        sfe: Star formation efficiency ε = M_ecl / M_cl, default 0.33
        log_rho_cl: Optional override for log₁₀(ρ_cl / 10⁶ M☉ pc⁻³)

    Example:
        >>> # Solar neighborhood cluster
        >>> env = BirthEnvironment.solar()
        >>> params = env_to_imf_params(env)
        >>> print(f"α₃ = {float(params.alpha3):.2f}")  # ~2.3

        >>> # Dense, metal-poor globular cluster with low SFE
        >>> env = BirthEnvironment.from_cluster_mass(
        ...     M_ecl=1e6, FeH=-1.5, sfe=0.1
        ... )
        >>> params = env_to_imf_params(env)
        >>> print(f"α₃ = {float(params.alpha3):.2f}")  # Top-heavy!
    """

    metallicity: Float[Array, ""]
    log_mecl: Float[Array, ""]
    sfe: Float[Array, ""]
    log_rho_cl: Float[Array, ""] | None = None

    def __init__(
        self,
        metallicity: Float[Array, ""] | None = None,
        log_mecl: Float[Array, ""] | None = None,
        sfe: Float[Array, ""] | None = None,
        log_rho_cl: Float[Array, ""] | None = None,
    ):
        """Initialize BirthEnvironment with defaults.

        Args:
            metallicity: [Fe/H], default 0.0 (solar)
            log_mecl: log₁₀(M_ecl/M☉), required
            sfe: Star formation efficiency, default 0.33
            log_rho_cl: Optional density override
        """
        if log_mecl is None:
            raise ValueError("log_mecl is required")

        self.metallicity = metallicity if metallicity is not None else jnp.array(0.0)
        self.log_mecl = log_mecl
        self.sfe = sfe if sfe is not None else jnp.array(DEFAULT_SFE)
        self.log_rho_cl = log_rho_cl

    @classmethod
    def from_cluster_mass(
        cls,
        M_ecl: float,
        FeH: float = 0.0,
        sfe: float = DEFAULT_SFE,
        log_rho_cl: float | None = None,
    ) -> BirthEnvironment:
        """Create from embedded cluster mass with solar metallicity default.

        Args:
            M_ecl: Embedded cluster mass [M☉]
            FeH: Metallicity [Fe/H], default 0.0 (solar)
            sfe: Star formation efficiency, default 0.33
            log_rho_cl: Optional log₁₀(ρ_cl / 10⁶ M☉ pc⁻³) override

        Returns:
            BirthEnvironment with computed log_mecl
        """
        return cls(
            metallicity=jnp.array(FeH),
            log_mecl=jnp.log10(jnp.array(M_ecl)),
            sfe=jnp.array(sfe),
            log_rho_cl=jnp.array(log_rho_cl) if log_rho_cl is not None else None,
        )

    @classmethod
    def solar(cls) -> BirthEnvironment:
        """Solar neighborhood: 1000 M☉ cluster, solar metallicity, default SFE."""
        return cls(
            metallicity=jnp.array(0.0),
            log_mecl=jnp.array(3.0),  # 1000 M☉
            sfe=jnp.array(DEFAULT_SFE),
        )

    @classmethod
    def massive_gc(cls, FeH: float = -1.5) -> BirthEnvironment:
        """Typical massive globular cluster birth conditions."""
        return cls(
            metallicity=jnp.array(FeH),
            log_mecl=jnp.array(6.0),  # 10^6 M☉
            sfe=jnp.array(DEFAULT_SFE),
        )

    @classmethod
    def ngc_7078(cls) -> BirthEnvironment:
        """NGC 7078 (M15) - most top-heavy in Marks+2012 Table 1.

        From Marks+2012:
            [Fe/H] = -2.16
            ρ_cl = 258.13 × 10⁶ M☉/pc³
            α₃ = 0.76 (extremely top-heavy!)
        """
        return cls(
            metallicity=jnp.array(-2.16),
            log_mecl=jnp.array(6.5),
            sfe=jnp.array(DEFAULT_SFE),
            log_rho_cl=jnp.log10(jnp.array(258.13)),  # log₁₀(ρ/10⁶)
        )

    @classmethod
    def ngc_104(cls) -> BirthEnvironment:
        """NGC 104 (47 Tuc) - well-studied globular cluster."""
        return cls(
            metallicity=jnp.array(-0.76),
            log_mecl=jnp.array(6.0),
            sfe=jnp.array(DEFAULT_SFE),
            log_rho_cl=jnp.log10(jnp.array(9.54)),
        )

    # -------------------------------------------------------------------------
    # Turbulence Properties (for FDF parameter derivation)
    # -------------------------------------------------------------------------

    def cloud_radius(self) -> Float[Array, ""]:
        """Parent cloud radius [pc] from (M_ecl, SFE, ρ_cl).

        Derives the size of the parent molecular cloud that formed the cluster,
        using a spherical geometry:

            R_cloud = (3 M_gas / (4π ρ_cl))^(1/3)

        where M_gas = M_ecl / SFE.

        Returns
        -------
        R_cloud : Float[Array, ""]
            Parent cloud radius [pc].

        Notes
        -----
        This is NOT the stellar half-mass radius r_h (which is much smaller).
        The fractal density structure is imprinted at the cloud scale.

        The cloud density is either:
        - From log_rho_cl if provided at construction
        - Computed from M_ecl and SFE via Marks+2012 r_h-M scaling

        Expected ranges:
            | M_ecl   | R_cloud |
            |---------|---------|
            | 10³ M☉  | ~1.5 pc |
            | 10⁴ M☉  | ~2.5 pc |
            | 10⁵ M☉  | ~4.0 pc |
            | 10⁶ M☉  | ~6.5 pc |

        Examples
        --------
        >>> env = BirthEnvironment.from_cluster_mass(M_ecl=1e4)
        >>> print(f"R_cloud = {float(env.cloud_radius()):.2f} pc")
        """
        from progenax.cluster.turbulence import cloud_radius_from_density

        M_ecl = jnp.power(10.0, self.log_mecl)

        # Get cloud density
        if self.log_rho_cl is not None:
            # Use provided density (log₁₀(ρ_cl / 10⁶))
            rho_cl = jnp.power(10.0, self.log_rho_cl + 6.0)
        else:
            # Compute from Marks+2012 scaling
            rho_cl = compute_rho_cl(M_ecl, self.sfe)

        return cloud_radius_from_density(M_ecl, self.sfe, rho_cl)

    def turbulent_mach(
        self,
        c_s: float = 0.2,
        sigma_v0: float = 1.0,
        alpha: float = 0.5,
    ) -> Float[Array, ""]:
        """Gas turbulent Mach number from Larson velocity-size relation.

        Uses the parent cloud radius (NOT stellar r_h) because the fractal
        density structure is imprinted by gas turbulence before star formation.

            M = σ_v(R_cloud) / c_s

        where σ_v = σ_v0 × (R_cloud)^α (Larson 1981).

        Parameters
        ----------
        c_s : float, optional
            Sound speed [km/s]. Default 0.2 (cold GMC at T ~ 10 K).
        sigma_v0 : float, optional
            Larson normalization [km/s] at 1 pc. Default 1.0.
        alpha : float, optional
            Larson exponent. Default 0.5.

        Returns
        -------
        mach : Float[Array, ""]
            Turbulent Mach number.

        Notes
        -----
        Physical ranges for typical star-forming clouds (using Larson relation):

        =========  =========  =====  =========
        Cluster    R_cloud    σ_v    Mach
        =========  =========  =====  =========
        Small OC   ~1.5 pc    ~1.2   ~6
        Large OC   ~2.5 pc    ~1.6   ~8
        YMC        ~4.0 pc    ~2.0   ~10
        GC         ~6.5 pc    ~2.5   ~13
        =========  =========  =====  =========

        These are MUCH more realistic than virial-based estimates which give
        M ~ 20-400 for the same cluster masses.

        References
        ----------
        .. [1] Larson (1981) MNRAS 194, 809 - Velocity-size relation
        .. [2] Solomon et al. (1987) ApJ 319, 730 - GMC properties
        .. [3] Federrath et al. (2010) A&A 512, A81 - Turbulence-density relation

        Examples
        --------
        >>> env = BirthEnvironment.from_cluster_mass(M_ecl=1e4)
        >>> print(f"Mach = {float(env.turbulent_mach()):.1f}")  # ~8
        """
        from progenax.cluster.turbulence import turbulent_mach_from_cloud

        R_cloud = self.cloud_radius()
        return turbulent_mach_from_cloud(R_cloud, c_s, sigma_v0, alpha)

    def sigma_ln_rho(
        self,
        b: float | None = None,
        c_s: float = 0.2,
        sigma_v0: float = 1.0,
        alpha: float = 0.5,
    ) -> Float[Array, ""]:
        """σ_ln_ρ from Federrath+2010 density-Mach relation.

        The variance of the log-density field in supersonic turbulence:
            σ²_ln_ρ = ln(1 + b² M²)

        Uses Larson velocity-size relation to derive Mach number from
        parent cloud properties.

        Parameters
        ----------
        b : float or None, optional
            Turbulence driving parameter. If None (default), derives b from
            cloud density via b_from_environment().
            - b ≈ 1/3 (0.33): Solenoidal (incompressible) driving
            - b ≈ 1.0: Compressive (irrotational) driving
            - b ≈ 0.4: Natural mixture (common manual value)
        c_s : float, optional
            Sound speed [km/s]. Default 0.2.
        sigma_v0 : float, optional
            Larson normalization [km/s] at 1 pc. Default 1.0.
        alpha : float, optional
            Larson exponent. Default 0.5.

        Returns
        -------
        sigma_ln_rho : Float[Array, ""]
            Standard deviation of log-density field.

        Notes
        -----
        Physical ranges (using Larson relation):

        =========  =========
        Cluster    σ_ln_ρ
        =========  =========
        Small OC   ~1.1
        Large OC   ~1.3
        YMC        ~1.4
        GC         ~1.6
        =========  =========

        When b=None (default), the driving parameter is derived from cloud
        density using b_from_environment(). Low-density clouds get more
        solenoidal driving (b≈0.33), high-density cores get more compressive
        driving (b≈0.7). This is TENTATIVE and may be overridden by explicit b.

        References
        ----------
        .. [1] Federrath et al. (2010) A&A 512, A81, Eq. 14
        .. [2] Federrath (2013) MNRAS 436, 1245 - Driving modes

        Examples
        --------
        >>> env = BirthEnvironment.from_cluster_mass(M_ecl=1e4)
        >>> # Use environment-derived b (default)
        >>> print(f"σ_ln_ρ = {float(env.sigma_ln_rho()):.2f}")
        >>> # Or override with explicit b
        >>> print(f"σ_ln_ρ (b=0.4) = {float(env.sigma_ln_rho(b=0.4)):.2f}")
        """
        from progenax.cluster.turbulence import (
            sigma_ln_rho_from_mach,
            b_from_environment,
        )

        # Derive b from environment if not provided
        if b is None:
            log_rho = self._get_log_rho_cl()
            b = float(b_from_environment(log_rho))

        mach = self.turbulent_mach(c_s, sigma_v0, alpha)
        return sigma_ln_rho_from_mach(mach, b)

    def _get_log_rho_cl(self) -> Float[Array, ""]:
        """Get log₁₀ of cloud density [M☉/pc³] for b derivation."""
        if self.log_rho_cl is not None:
            # User provided explicit cloud density (in log₁₀(ρ / 10⁶) units)
            # Convert to log₁₀(ρ) by adding 6
            return self.log_rho_cl + 6.0
        else:
            # Derive from Marks+2012 r_h-M relation
            # compute_rho_cl is defined in this module
            M_ecl = jnp.power(10.0, self.log_mecl)
            sfe = self.sfe if self.sfe is not None else 0.33
            rho_cl = compute_rho_cl(M_ecl, sfe)
            return jnp.log10(rho_cl)

    def spectral_slope(
        self,
        c_s: float = 0.2,
        sigma_v0: float = 1.0,
        alpha: float = 0.5,
    ) -> Float[Array, ""]:
        """Power spectrum slope β from turbulence regime.

        Interpolates between:
            - Subsonic (M << 1): Kolmogorov β = 11/3 ≈ 3.67
            - Supersonic (M >> 1): Burgers β ≈ 4.0

        Uses Larson velocity-size relation to derive Mach number.

        Parameters
        ----------
        c_s : float, optional
            Sound speed [km/s]. Default 0.2.
        sigma_v0 : float, optional
            Larson normalization [km/s] at 1 pc. Default 1.0.
        alpha : float, optional
            Larson exponent. Default 0.5.

        Returns
        -------
        beta : Float[Array, ""]
            Power spectrum slope P(k) ∝ k^{-β}.

        Notes
        -----
        For star-forming clouds with M >> 1, expect β ≈ 4.

        References
        ----------
        .. [1] Kolmogorov (1941) - Incompressible turbulence
        .. [2] Burgers (1948) - Shock-dominated turbulence

        Examples
        --------
        >>> env = BirthEnvironment.from_cluster_mass(M_ecl=1e4)
        >>> print(f"β = {float(env.spectral_slope()):.2f}")  # ~4.0
        """
        from progenax.cluster.turbulence import spectral_slope_from_mach

        mach = self.turbulent_mach(c_s, sigma_v0, alpha)
        return spectral_slope_from_mach(mach)



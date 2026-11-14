"""BirthEnvironment inference target (split from environment.py)."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from .coefficients import (
    DEFAULT_SFE,
)
from .density import (
    compute_rho_cl,
)

# =============================================================================
# BirthEnvironment - Primary inference target
# =============================================================================


class BirthEnvironment(eqx.Module):
    """Physical birth environment with SFE parameter.

    Represents conditions during star formation that influence the IMF.
    All fields are JAX arrays for gradient-based inference.

    Attributes:
        metallicity: [Fe/H], default 0.0 (solar). Mapping clips to [-2.5, +0.5]
            when clamp_domain=True; this is not an empirical validity certificate.
        log_mecl: log₁₀(M_ecl / M☉). Mapping clips to [3, 8] by default.
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

            R_cloud = (3 M_cl / (4π ρ_cl))^(1/3)

        where M_cl = M_ecl / SFE is the total birth-cloud mass.

        Returns
        -------
        R_cloud : Float[Array, ""]
            Parent cloud radius [pc].

        Notes
        -----
        This spherical radius is distinct from the stellar half-mass radius r_h.
        With the default density prescription, rho_cl=3 M_ecl/(8 pi r_h^3 SFE),
        the formula reduces to R_cloud=2^(1/3) r_h and is independent of SFE.
        An explicit density override changes that relation.

        The cloud density is either:
        - From log_rho_cl if provided at construction
        - Computed from M_ecl and SFE via Marks+2012 r_h-M scaling

        Default density prescription (no override):
            | M_ecl   | R_cloud |
            |---------|---------|
            | 10³ M☉  | ~0.31 pc |
            | 10⁴ M☉  | ~0.42 pc |
            | 10⁵ M☉  | ~0.56 pc |
            | 10⁶ M☉  | ~0.76 pc |

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
        Default density prescription and Larson parameters (no override):

        =========  =========  =====  =========
        Cluster    R_cloud    σ_v    Mach
        =========  =========  =====  =========
        10³ M☉     ~0.31 pc   ~0.56  ~2.8
        10⁴ M☉     ~0.42 pc   ~0.65  ~3.2
        10⁵ M☉     ~0.56 pc   ~0.75  ~3.8
        10⁶ M☉     ~0.76 pc   ~0.87  ~4.4
        =========  =========  =====  =========

        sigma_v is a 3-D turbulent velocity dispersion in km/s. These values
        follow the adopted cluster density prescription; they do not validate
        a molecular-cloud population or an explicit density override.

        References
        ----------
        .. [1] Larson (1981) MNRAS 194, 809 - Velocity-size relation
        .. [2] Solomon et al. (1987) ApJ 319, 730 - GMC properties
        .. [3] Federrath et al. (2010) A&A 512, A81 - Turbulence-density relation

        Examples
        --------
        >>> env = BirthEnvironment.from_cluster_mass(M_ecl=1e4)
        >>> print(f"Mach = {float(env.turbulent_mach()):.1f}")  # ~3.2
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
        The derived b remains a JAX scalar: JIT and parameter derivatives
        include both its density dependence and the Mach-number dependence.

        References
        ----------
        .. [1] Federrath et al. (2010) A&A 512, A81, Eq. 19
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
            b_from_environment,
            sigma_ln_rho_from_mach,
        )

        # Derive b from environment if not provided
        if b is None:
            log_rho = self._get_log_rho_cl()
            b_value = b_from_environment(log_rho)
        else:
            b_value = jnp.asarray(b)

        mach = self.turbulent_mach(c_s, sigma_v0, alpha)
        return sigma_ln_rho_from_mach(mach, b_value)

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
        """DENSITY power-spectrum slope β from the turbulence regime (Kim & Ryu 2005).

        The 3D density spectrum P_3D(k)∝k^{-β} FLATTENS with Mach number (β decreases),
        because supersonic turbulence concentrates mass into sheets and filaments — the
        opposite of the velocity (Kolmogorov→Burgers) cascade. Uses the Larson
        velocity-size relation to derive the Mach number, then
        ``cluster.turbulence.spectral_slope_from_mach``.

        Note: β is a diagnostic of the gas field's spatial structure; it does NOT enter
        the IMF slopes (those depend on ρ_cl and [Fe/H]; see Marks+2012 / Jerabkova+2018).

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
            Density power spectrum slope P_3D(k) ∝ k^{-β}, in [2, 11/3].

        References
        ----------
        .. [1] Kim & Ryu (2005) ApJ 630, L45 - density power spectrum vs Mach

        Examples
        --------
        >>> env = BirthEnvironment.from_cluster_mass(M_ecl=1e4)
        >>> print(f"β = {float(env.spectral_slope()):.2f}")  # ~3.18 (mildly supersonic)
        """
        from progenax.cluster.turbulence import spectral_slope_from_mach

        mach = self.turbulent_mach(c_s, sigma_v0, alpha)
        return spectral_slope_from_mach(mach)

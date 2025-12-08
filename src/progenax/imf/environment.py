"""
Environment-conditioned Initial Mass Functions.

IMF parameters that depend on local astrophysical environment:
- Metallicity (low Z → top-heavy IMF)
- Gas density/temperature (Jeans mass dependence)
- Star formation rate (starburst conditions)

References:
    Marks et al. (2012), MNRAS, 422, 2246 - α(ρ) relation
    Jerabkova et al. (2018), A&A, 620, A39 - IGIMF with metallicity dependence
    Adams & Fatuzzo (1996), ApJ, 464, 256 - Jeans mass scaling
    Chabrier (2003), PASP, 115, 763 - Characteristic mass interpretation
"""

from __future__ import annotations

from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .power_law import PowerLawIMF

# ============================================================================
# Physical Constants (from jaxstro core)
# ============================================================================

from jaxstro import constants as C

# Re-export jaxstro constants for local use (CGS units)
G_CGS = C.G_CGS          # Gravitational constant [cm³ g⁻¹ s⁻²]
K_B_CGS = C.K_B          # Boltzmann constant [erg K⁻¹]
M_P_CGS = C.M_P          # Proton mass [g]
M_SUN_CGS = C.MSUN_G     # Solar mass [g]

# Mean molecular weight for molecular cloud (H₂ dominated)
MU_MOL = 2.33


# ============================================================================
# Environment Parameter Classes
# ============================================================================


class GasEnvironment(eqx.Module):
    """Local gas environment parameters.

    Describes the physical conditions in a molecular cloud or star-forming region
    that affect the IMF shape.

    Attributes:
        n_H: Hydrogen number density [cm⁻³] (default: 10⁴, typical GMC)
        T_gas: Gas temperature [K] (default: 10, cold molecular cloud)
        Z: Metallicity [Z_sun] (default: 1.0, solar)

    Examples:
        >>> # Solar neighborhood GMC
        >>> env = GasEnvironment(n_H=1e4, T_gas=10.0, Z=1.0)

        >>> # Dense cluster-forming clump
        >>> env = GasEnvironment(n_H=1e6, T_gas=20.0, Z=0.3)

        >>> # Primordial (Pop III) conditions
        >>> env = GasEnvironment(n_H=1e4, T_gas=200.0, Z=0.0)
    """

    n_H: float = 1e4  # [cm⁻³]
    T_gas: float = 10.0  # [K]
    Z: float = 1.0  # [Z_sun]

    @property
    def rho(self) -> float:
        """Mass density [g cm⁻³]."""
        return self.n_H * MU_MOL * M_P_CGS

    @property
    def log_n(self) -> float:
        """Log₁₀ of number density."""
        return jnp.log10(self.n_H)

    @classmethod
    def solar_neighborhood(cls) -> "GasEnvironment":
        """Typical solar neighborhood GMC conditions.

        Returns:
            Environment with n=10⁴ cm⁻³, T=10 K, Z=1.0
        """
        return cls(n_H=1e4, T_gas=10.0, Z=1.0)

    @classmethod
    def dense_clump(cls) -> "GasEnvironment":
        """Dense cluster-forming clump conditions.

        Returns:
            Environment with n=10⁶ cm⁻³, T=20 K, Z=1.0
        """
        return cls(n_H=1e6, T_gas=20.0, Z=1.0)

    @classmethod
    def low_metallicity(cls, Z: float = 0.1) -> "GasEnvironment":
        """Low metallicity environment (halo, dwarf galaxy).

        Args:
            Z: Metallicity in solar units [Z_sun]

        Returns:
            Environment with n=10⁴ cm⁻³, T=10 K, specified Z
        """
        return cls(n_H=1e4, T_gas=10.0, Z=Z)

    @classmethod
    def starburst(cls) -> "GasEnvironment":
        """Starburst galaxy conditions.

        High density, elevated temperature from turbulence/feedback.

        Returns:
            Environment with n=10⁵ cm⁻³, T=50 K, Z=1.5
        """
        return cls(n_H=1e5, T_gas=50.0, Z=1.5)

    @classmethod
    def primordial(cls) -> "GasEnvironment":
        """Primordial (Pop III) star-forming conditions.

        Zero metallicity, H₂ cooling limited to ~200 K.

        Returns:
            Environment with n=10⁴ cm⁻³, T=200 K, Z=0.0
        """
        return cls(n_H=1e4, T_gas=200.0, Z=0.0)


# ============================================================================
# Jeans Mass Scaling
# ============================================================================


def jeans_mass(n_H: float, T: float, mu: float = MU_MOL) -> float:
    """Jeans mass for gravitational instability.

    M_J = (π^{5/2} / 6) × (c_s³ / (G^{3/2} × ρ^{1/2}))

    where c_s = sqrt(k_B T / (μ m_p)) is the sound speed.

    Args:
        n_H: Hydrogen number density [cm⁻³]
        T: Gas temperature [K]
        mu: Mean molecular weight (default: 2.33 for H₂)

    Returns:
        Jeans mass [M_sun]

    References:
        Jeans (1902), Phil. Trans. R. Soc., 199, 1
    """
    rho = n_H * mu * M_P_CGS  # [g cm⁻³]
    c_s = jnp.sqrt(K_B_CGS * T / (mu * M_P_CGS))  # [cm s⁻¹]

    # M_J = (π^{5/2}/6) × c_s³ / (G^{3/2} × ρ^{1/2})
    M_J_cgs = (jnp.pi ** 2.5 / 6.0) * (c_s ** 3) / (G_CGS ** 1.5 * jnp.sqrt(rho))

    return M_J_cgs / M_SUN_CGS


def characteristic_mass_from_jeans(
    n_H: float,
    T: float,
    efficiency: float = 0.1,
) -> float:
    """Characteristic stellar mass from Jeans mass scaling.

    The characteristic mass m_c of the IMF is thought to be set by the
    thermal Jeans mass in the cloud, with some efficiency factor:

        m_c ≈ ε × M_J(ρ, T)

    The efficiency factor (~0.1) accounts for fragmentation and feedback.

    Args:
        n_H: Hydrogen number density [cm⁻³]
        T: Gas temperature [K]
        efficiency: Star formation efficiency (default: 0.1)

    Returns:
        Characteristic stellar mass [M_sun]

    References:
        Adams & Fatuzzo (1996), ApJ, 464, 256
        Bate & Bonnell (2005), MNRAS, 356, 1201
    """
    M_J = jeans_mass(n_H, T)
    return efficiency * M_J


# ============================================================================
# Bounded Transformations
# ============================================================================


def alpha_bounded(
    f_alpha: float,
    alpha_min: float = 1.5,
    alpha_max: float = 2.7,
) -> float:
    """Map unbounded value to bounded alpha range via sigmoid.

    Uses sigmoid bijector to ensure alpha stays within physical bounds
    regardless of input. Centered at midpoint of range.

    Args:
        f_alpha: Unbounded input (from environment mapping or neural net)
        alpha_min: Lower bound (default: 1.5, extreme top-heavy)
        alpha_max: Upper bound (default: 2.7, bottom-heavy)

    Returns:
        Bounded alpha in (alpha_min, alpha_max)

    Physics:
        - alpha < 2.0: Top-heavy (more massive stars)
        - alpha ~ 2.3: Salpeter/Kroupa standard
        - alpha > 2.5: Bottom-heavy (fewer massive stars)

    Examples:
        >>> # Neural network output → bounded alpha
        >>> f_alpha = 0.0  # Centered
        >>> alpha = alpha_bounded(f_alpha)
        >>> print(f"{alpha:.2f}")  # ~2.1 (midpoint of 1.5-2.7)
        2.10

        >>> # Extreme inputs stay bounded
        >>> alpha_low = alpha_bounded(-100.0)
        >>> alpha_high = alpha_bounded(100.0)
        >>> print(f"{alpha_low:.2f}, {alpha_high:.2f}")  # ~1.5, ~2.7
        1.50, 2.70
    """
    # Sigmoid maps R → (0, 1)
    sigmoid = jax.nn.sigmoid(f_alpha)
    alpha_range = alpha_max - alpha_min
    return alpha_min + sigmoid * alpha_range


def bonnor_ebert_mass(
    T: float,
    P_over_kB: float,
    Z: float = 1.0,
    m_0: float = 0.3,
) -> float:
    """Bonnor-Ebert mass scaling proxy for characteristic mass.

    The BE mass sets the characteristic mass for gravitational fragmentation
    in a pressure-supported cloud.

    Scaling: m_c ∝ T² × P^(-1/2) × f_Z(Z)

    Args:
        T: Gas temperature [K]
        P_over_kB: Pressure/Boltzmann constant [K cm⁻³]
        Z: Metallicity in solar units [Z☉]
        m_0: Normalization at T=10K, P/k=1e5, Z=1 [M☉] (default: 0.3)

    Returns:
        Characteristic mass [M☉]

    Reference scaling:
        - T=10 K, P/k=1e5 K cm⁻³, Z=1 → m_c ≈ 0.3 M☉
        - T=20 K (doubled) → m_c ≈ 1.2 M☉ (4×)
        - P/k=4e5 (4×) → m_c ≈ 0.15 M☉ (0.5×)

    Physics:
        The Bonnor-Ebert mass is the critical mass for gravitational
        instability in an isothermal sphere embedded in external pressure.
        Lower pressure or higher temperature → larger fragments.

    References:
        Bonnor (1956), MNRAS, 116, 351 - BE critical mass

    Examples:
        >>> # Typical GMC conditions
        >>> m_c = bonnor_ebert_mass(T=10.0, P_over_kB=1e5, Z=1.0)
        >>> print(f"{m_c:.2f} Msun")  # ~0.3 Msun
        0.30 Msun

        >>> # Primordial (Pop III) conditions
        >>> m_primordial = bonnor_ebert_mass(T=200.0, P_over_kB=1e5, Z=0.0)
        >>> print(f"{m_primordial:.1f} Msun")  # Much higher due to high T, low Z
        150.0 Msun
    """
    # Temperature scaling: T² relative to T_ref=10 K
    T_factor = (T / 10.0) ** 2

    # Pressure scaling: P^(-1/2) relative to P_ref=1e5
    P_factor = (P_over_kB / 1e5) ** (-0.5)

    # Metallicity factor: weak dependence (less cooling at low Z)
    # f_Z = (Z + 0.01)^(-0.1) gives ~1.26× at Z=0.1 vs Z=1.0
    Z_factor = (Z + 0.01) ** (-0.1)

    return m_0 * T_factor * P_factor * Z_factor


# ============================================================================
# High-Mass Slope Dependence
# ============================================================================


def alpha_marks2012(log_n: float, log_n_crit: float = 6.5) -> float:
    """High-mass IMF slope from density (Marks et al. 2012).

    The high-mass slope α₃ depends on cloud density:

        α₃ = 2.3 - 0.5 × max(0, log₁₀(n) - log_n_crit)

    For log(n) < 6.5: standard Kroupa α₃ = 2.3
    For log(n) > 6.5: top-heavy IMF with lower α₃

    Args:
        log_n: log₁₀ of number density [cm⁻³]
        log_n_crit: Critical density threshold (default: 6.5)

    Returns:
        High-mass slope α₃

    References:
        Marks et al. (2012), MNRAS, 422, 2246, Eq. 2
    """
    delta = jnp.maximum(0.0, log_n - log_n_crit)
    # High density → top-heavy → lower α
    return 2.3 - 0.5 * delta


def alpha_jerabkova2018(
    Z: float,
    log_n: float,
    alpha_solar: float = 2.3,
) -> float:
    """High-mass IMF slope from metallicity and density (Jerabkova+ 2018).

    Extended prescription including both metallicity and density:

        α₃ = α_solar + Δα(Z) + Δα(ρ)

    Lower metallicity and higher density both lead to top-heavy IMF.

    Args:
        Z: Metallicity [Z_sun]
        log_n: log₁₀ of number density [cm⁻³]
        alpha_solar: Solar neighborhood slope (default: 2.3)

    Returns:
        High-mass slope α₃

    References:
        Jerabkova et al. (2018), A&A, 620, A39
    """
    # Metallicity effect: lower Z → lower α (top-heavy)
    # Approximate: Δα ≈ -0.3 × (1 - Z) for Z < 1
    delta_Z = -0.3 * jnp.maximum(0.0, 1.0 - Z)

    # Density effect from Marks+2012
    # Critical density ~10^5.5 cm^-3 (matches observational threshold)
    delta_rho = -0.5 * jnp.maximum(0.0, log_n - 5.5)

    return alpha_solar + delta_Z + delta_rho


def alpha_from_sfr(sfr: float, alpha_solar: float = 2.3) -> float:
    """High-mass IMF slope from star formation rate.

    Empirical relation: starbursts (high SFR) have top-heavy IMFs.

        α₃ ≈ α_solar - 0.4 × log₁₀(SFR / SFR_MW)

    where SFR_MW ≈ 1.9 M_sun/yr.

    Args:
        sfr: Star formation rate [M_sun/yr]
        alpha_solar: Solar neighborhood slope (default: 2.3)

    Returns:
        High-mass slope α₃

    References:
        Gunawardhana et al. (2011), MNRAS, 415, 1647
        Zhang et al. (2018), Nature, 558, 260
    """
    sfr_mw = 1.9  # Milky Way SFR [M_sun/yr]
    delta = -0.4 * jnp.log10(jnp.maximum(sfr, 1e-6) / sfr_mw)
    return alpha_solar + delta


# ============================================================================
# Environment-Conditioned IMF Classes
# ============================================================================


class EnvironmentIMF(eqx.Module):
    """IMF with parameters conditioned on local environment.

    Adjusts the high-mass slope and characteristic mass based on gas density,
    temperature, and metallicity following observational prescriptions.

    The base IMF is a 3-segment power-law (Kroupa-like) with:
        - α₁ = 0.3 (brown dwarfs, m < 0.08 M_sun)
        - α₂ = 1.3 (low-mass stars, 0.08 < m < m_c)
        - α₃ = α(Z, ρ) (high-mass stars, m > m_c)

    where m_c is the characteristic mass (related to Jeans mass).

    Attributes:
        environment: GasEnvironment with local conditions
        alpha_model: Model for high-mass slope ('marks2012', 'jerabkova2018', 'sfr')
        m_min: Minimum stellar mass [M_sun]
        m_max: Maximum stellar mass [M_sun]
        sfr: Star formation rate [M_sun/yr] (for 'sfr' model)

    Examples:
        >>> # Solar neighborhood
        >>> env = GasEnvironment.solar_neighborhood()
        >>> imf = EnvironmentIMF(env)
        >>> masses = imf.sample(jax.random.PRNGKey(42), 1000)

        >>> # Starburst with top-heavy IMF
        >>> env = GasEnvironment.starburst()
        >>> imf = EnvironmentIMF(env, alpha_model='jerabkova2018')
        >>> print(f"High-mass slope: {imf.alpha_high:.2f}")  # Should be < 2.3
    """

    environment: GasEnvironment
    alpha_model: str = eqx.field(static=True, default="jerabkova2018")
    m_min: float = eqx.field(static=True, default=0.01)
    m_max: float = eqx.field(static=True, default=100.0)
    sfr: float = 1.0  # Only used for alpha_model='sfr'
    # Pre-computed underlying IMF (for JIT compatibility)
    _underlying_imf: PowerLawIMF = eqx.field(init=False)

    def __post_init__(self):
        """Pre-compute underlying IMF at initialization."""
        # Compute alpha using the specified model
        if self.alpha_model == "marks2012":
            alpha3 = float(alpha_marks2012(self.environment.log_n))
        elif self.alpha_model == "jerabkova2018":
            alpha3 = float(alpha_jerabkova2018(
                self.environment.Z, self.environment.log_n
            ))
        elif self.alpha_model == "sfr":
            alpha3 = float(alpha_from_sfr(self.sfr))
        else:
            raise ValueError(f"Unknown alpha_model: {self.alpha_model}")

        # Compute characteristic mass
        m_c = float(jnp.clip(
            characteristic_mass_from_jeans(
                self.environment.n_H, self.environment.T_gas
            ),
            0.1, 2.0
        ))

        # Build 3-segment Kroupa-like IMF
        alphas = [0.3, 1.3, alpha3]
        breaks = [0.08, m_c]

        # Store using object.__setattr__ since Module is frozen
        object.__setattr__(
            self, '_underlying_imf',
            PowerLawIMF(alphas, breaks, self.m_min, self.m_max)
        )

    @property
    def alpha_high(self) -> float:
        """High-mass slope α₃ from environment model."""
        return self._underlying_imf.exponents[2]

    @property
    def m_char(self) -> float:
        """Characteristic mass from Jeans mass scaling."""
        return self._underlying_imf.breakpoints[1]

    def _get_underlying_imf(self) -> PowerLawIMF:
        """Return pre-computed underlying IMF."""
        return self._underlying_imf

    # ========================================================================
    # IMF Protocol Implementation
    # ========================================================================

    def logpdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Normalized log-PDF."""
        return self._get_underlying_imf().logpdf(m)

    def cdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Cumulative distribution function."""
        return self._get_underlying_imf().cdf(m)

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Percent point function (inverse CDF)."""
        return self._get_underlying_imf().ppf(u)

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample n masses."""
        return self._get_underlying_imf().sample(key, n)

    def mean_mass(self) -> float:
        """Expected mass."""
        return self._get_underlying_imf().mean_mass()

    # ========================================================================
    # Factory Methods
    # ========================================================================

    @classmethod
    def solar_neighborhood(cls) -> "EnvironmentIMF":
        """Standard solar neighborhood IMF.

        Returns canonical Kroupa IMF (α₃ ≈ 2.3, m_c ≈ 0.5).

        Returns:
            EnvironmentIMF with solar conditions
        """
        return cls(GasEnvironment.solar_neighborhood())

    @classmethod
    def starburst(cls, sfr: float = 100.0) -> "EnvironmentIMF":
        """Starburst environment with top-heavy IMF.

        Args:
            sfr: Star formation rate [M_sun/yr]

        Returns:
            EnvironmentIMF with starburst conditions
        """
        return cls(
            GasEnvironment.starburst(),
            alpha_model="sfr",
            sfr=sfr,
        )

    @classmethod
    def dense_clump(cls) -> "EnvironmentIMF":
        """Dense cluster-forming clump.

        Returns:
            EnvironmentIMF with dense clump conditions
        """
        return cls(GasEnvironment.dense_clump())

    @classmethod
    def low_metallicity(cls, Z: float = 0.1) -> "EnvironmentIMF":
        """Low metallicity environment.

        Args:
            Z: Metallicity [Z_sun]

        Returns:
            EnvironmentIMF with low-Z conditions (top-heavy)
        """
        return cls(GasEnvironment.low_metallicity(Z))

    @classmethod
    def primordial(cls) -> "EnvironmentIMF":
        """Primordial (Pop III) conditions.

        Very top-heavy IMF due to zero metals and high temperature.

        Returns:
            EnvironmentIMF for Pop III stars
        """
        return cls(GasEnvironment.primordial())


# ============================================================================
# Functional Interface for Custom Prescriptions
# ============================================================================


# Type aliases for environment functions
AlphaFunction = Callable[[GasEnvironment], float]
CharMassFunction = Callable[[GasEnvironment], float]


class CustomEnvironmentIMF(eqx.Module):
    """IMF with user-defined environment prescriptions.

    Allows custom functions for α₃(environment) and m_c(environment).

    Attributes:
        environment: GasEnvironment with local conditions
        _alpha_high: Precomputed high-mass slope
        _m_char: Precomputed characteristic mass
        m_min: Minimum stellar mass [M_sun]
        m_max: Maximum stellar mass [M_sun]

    Examples:
        >>> # Custom prescription with steep metallicity dependence
        >>> def my_alpha(env):
        ...     return 2.3 - 0.5 * jnp.maximum(0, 1 - env.Z)
        >>>
        >>> env = GasEnvironment(Z=0.2)
        >>> imf = CustomEnvironmentIMF(env, alpha_fn=my_alpha)
    """

    environment: GasEnvironment
    # Store precomputed values since functions can't be PyTree leaves
    # These are static because they're computed once at init and don't need gradients
    _alpha_high: float = eqx.field(static=True)
    _m_char: float = eqx.field(static=True)
    m_min: float = eqx.field(static=True, default=0.01)
    m_max: float = eqx.field(static=True, default=100.0)

    def __init__(
        self,
        environment: GasEnvironment,
        alpha_fn: Optional[AlphaFunction] = None,
        char_mass_fn: Optional[CharMassFunction] = None,
        m_min: float = 0.01,
        m_max: float = 100.0,
    ):
        """Initialize with custom environment functions.

        Args:
            environment: GasEnvironment with local conditions
            alpha_fn: Function mapping environment → high-mass slope
                     (default: Jerabkova+2018 prescription)
            char_mass_fn: Function mapping environment → characteristic mass
                         (default: Jeans mass scaling)
            m_min: Minimum stellar mass [M_sun]
            m_max: Maximum stellar mass [M_sun]
        """
        # Compute alpha using provided function or default
        if alpha_fn is not None:
            alpha_high = float(alpha_fn(environment))
        else:
            alpha_high = float(
                alpha_jerabkova2018(environment.Z, environment.log_n)
            )

        # Compute characteristic mass using provided function or default
        if char_mass_fn is not None:
            m_char = float(char_mass_fn(environment))
        else:
            m_char = float(
                characteristic_mass_from_jeans(environment.n_H, environment.T_gas)
            )

        # Use object.__setattr__ for frozen equinox module
        object.__setattr__(self, 'environment', environment)
        object.__setattr__(self, 'm_min', m_min)
        object.__setattr__(self, 'm_max', m_max)
        object.__setattr__(self, '_alpha_high', alpha_high)
        object.__setattr__(self, '_m_char', m_char)

    @property
    def alpha_high(self) -> float:
        """High-mass slope α₃."""
        return self._alpha_high

    @property
    def m_char(self) -> float:
        """Characteristic mass m_c."""
        return self._m_char

    def _get_underlying_imf(self) -> PowerLawIMF:
        """Build PowerLawIMF with custom parameters."""
        # Clip m_char to valid range (already Python float from __init__)
        # Use Python min/max to avoid JAX tracer issues in JIT
        m_c = max(0.1, min(2.0, self._m_char))

        alphas = [0.3, 1.3, self._alpha_high]
        breaks = [0.08, m_c]

        return PowerLawIMF(alphas, breaks, self.m_min, self.m_max)

    # IMF Protocol implementation
    def logpdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Normalized log-PDF."""
        return self._get_underlying_imf().logpdf(m)

    def cdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Cumulative distribution function."""
        return self._get_underlying_imf().cdf(m)

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Percent point function (inverse CDF)."""
        return self._get_underlying_imf().ppf(u)

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample n masses."""
        return self._get_underlying_imf().sample(key, n)

    def mean_mass(self) -> float:
        """Expected mass."""
        return self._get_underlying_imf().mean_mass()


# ============================================================================
# Top-Heavy IMF Utilities
# ============================================================================


def is_top_heavy(alpha_high: float, threshold: float = 2.0) -> bool:
    """Check if IMF is top-heavy based on high-mass slope.

    Top-heavy IMFs produce more massive stars relative to low-mass stars.
    Conventionally, α₃ < 2.0 is considered top-heavy (compared to
    Salpeter α = 2.35 or Kroupa α₃ = 2.3).

    Args:
        alpha_high: High-mass slope α₃
        threshold: Slope threshold for top-heavy (default: 2.0)

    Returns:
        True if IMF is top-heavy
    """
    return alpha_high < threshold


def massive_star_fraction(imf, m_threshold: float = 8.0) -> float:
    """Fraction of mass in stars above threshold.

    Useful diagnostic for top-heavy IMFs. Higher fraction = more top-heavy.

    Args:
        imf: Any IMF with sample() method
        m_threshold: Mass threshold [M_sun] (default: 8.0, core-collapse limit)

    Returns:
        Fraction of total mass in stars with m > m_threshold
    """
    # Sample a large population
    key = jax.random.PRNGKey(42)
    masses = imf.sample(key, 100000)

    # Compute mass fractions
    total_mass = jnp.sum(masses)
    massive_mass = jnp.sum(jnp.where(masses > m_threshold, masses, 0.0))

    return float(massive_mass / (total_mass + 1e-30))


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Physical constants
    "G_CGS",
    "K_B_CGS",
    "M_P_CGS",
    "M_SUN_CGS",
    "MU_MOL",
    # Environment parameters
    "GasEnvironment",
    # Jeans mass
    "jeans_mass",
    "characteristic_mass_from_jeans",
    # Bounded transformations
    "alpha_bounded",
    "bonnor_ebert_mass",
    # Alpha prescriptions
    "alpha_marks2012",
    "alpha_jerabkova2018",
    "alpha_from_sfr",
    # IMF classes
    "EnvironmentIMF",
    "CustomEnvironmentIMF",
    # Utilities
    "is_top_heavy",
    "massive_star_fraction",
]

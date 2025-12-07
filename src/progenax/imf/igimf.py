"""Integrated Galactic Initial Mass Function (IGIMF) theory.

The IGIMF accounts for the fact that stars form in clusters, not in isolation.
The galaxy-wide IMF emerges from integrating the stellar IMF over the cluster
mass function, with each cluster having a maximum stellar mass that depends
on the cluster mass.

Key Result: The IGIMF is steeper than the input stellar IMF because:
    - Most clusters are low-mass (ECMF is steep, β ≈ 2)
    - Low-mass clusters can't form massive stars (m_max-M_ecl relation)
    - Galaxy-wide, massive stars are rarer than the stellar IMF predicts

Primary References:
    - Kroupa & Weidner (2003) ApJ 598, 1076
      "Galactic-Field Initial Mass Functions of Massive Stars"
      Key result: Galaxy-wide α ≈ 2.7-3.0 vs stellar α = 2.3

    - Weidner & Kroupa (2004) MNRAS 348, 187
      "Evidence for a fundamental stellar upper mass limit from clustered
      star formation"
      The m_max(M_ecl) relation.

    - Weidner & Kroupa (2005) ApJ 625, 754
      "The Variation of Integrated Star Initial Mass Functions among Galaxies"
      IGIMF dependence on star formation rate.

    - Weidner, Kroupa & Pflamm-Altenburg (2013) MNRAS 434, 84
      "The IGIMF and its implications"
      Comprehensive review with updated parameters.

    - Pflamm-Altenburg, Weidner & Kroupa (2007) ApJ 671, 1550
      "Converting Hα luminosities into star formation rates"
      IGIMF effects on SFR indicators.

Components:
    - EmbeddedClusterMassFunction: Power-law ECMF with β ≈ 2
    - MaxStellarMass: m_max(M_ecl) relations from observations
    - IGIMF: Full IGIMF model integrating stellar IMF over ECMF

Examples:
    >>> from progenax.imf.igimf import IGIMF
    >>> from progenax.imf import PowerLawIMF
    >>> import jax
    >>>
    >>> # IGIMF for Milky Way-like galaxy (SFR ~ 1 Msun/yr)
    >>> stellar_imf = PowerLawIMF.kroupa()
    >>> igimf = IGIMF(stellar_imf=stellar_imf, sfr=1.0)
    >>>
    >>> # Sample stellar masses galaxy-wide
    >>> key = jax.random.PRNGKey(42)
    >>> masses = igimf.sample(key, 10000)
"""

from __future__ import annotations

from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .base import BaseIMF


# =============================================================================
# Embedded Cluster Mass Function (ECMF)
# =============================================================================


class EmbeddedClusterMassFunction(eqx.Module):
    """Embedded Cluster Mass Function (ECMF).

    The ECMF describes the mass distribution of star-forming clusters:
        dN/dM_ecl ∝ M_ecl^(-β)

    Reference:
        Lada & Lada (2003) ARA&A 41, 57
        "Embedded Clusters in Molecular Clouds"
        β ≈ 2.0 from observations of embedded clusters.

        Weidner, Kroupa & Pflamm-Altenburg (2013) MNRAS 434, 84
        β = 2.0 ± 0.1 as the canonical value.

    Parameters:
        beta: Power-law exponent (default: 2.0)
        M_ecl_min: Minimum cluster mass [M☉] (default: 5.0)
                   Smallest embedded clusters observed.
        M_ecl_max: Maximum cluster mass [M☉] (default: depends on SFR)
                   Set by star formation rate via M_ecl_max(SFR) relation.
    """

    beta: float = 2.0
    M_ecl_min: float = 5.0  # Minimum embedded cluster mass [M☉]
    M_ecl_max: float = 1e6  # Maximum embedded cluster mass [M☉]

    def pdf(self, M_ecl: Float[Array, "..."]) -> Float[Array, "..."]:
        """ECMF probability density: p(M_ecl) ∝ M_ecl^(-β)."""
        # Normalization for power-law
        b = self.beta
        M_min = self.M_ecl_min
        M_max = self.M_ecl_max

        # For β ≠ 1: Z = (M_max^(1-β) - M_min^(1-β)) / (1-β)
        e = 1.0 - b
        Z = jnp.where(
            jnp.abs(e) < 1e-10,
            jnp.log(M_max / M_min),
            (M_max**e - M_min**e) / e,
        )

        pdf_unnorm = M_ecl ** (-b)
        in_range = (M_ecl >= M_min) & (M_ecl <= M_max)
        return jnp.where(in_range, pdf_unnorm / Z, 0.0)

    def cdf(self, M_ecl: Float[Array, "..."]) -> Float[Array, "..."]:
        """ECMF cumulative distribution."""
        b = self.beta
        M_min = self.M_ecl_min
        M_max = self.M_ecl_max
        e = 1.0 - b

        Z = jnp.where(
            jnp.abs(e) < 1e-10,
            jnp.log(M_max / M_min),
            (M_max**e - M_min**e) / e,
        )

        def cdf_scalar(M):
            integral = jnp.where(
                jnp.abs(e) < 1e-10,
                jnp.log(M / M_min),
                (M**e - M_min**e) / e,
            )
            return jnp.clip(integral / Z, 0.0, 1.0)

        M_arr = jnp.asarray(M_ecl)
        if M_arr.ndim == 0:
            return cdf_scalar(M_arr)
        return jax.vmap(cdf_scalar)(M_arr.ravel()).reshape(M_arr.shape)

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Inverse CDF for ECMF."""
        b = self.beta
        M_min = self.M_ecl_min
        M_max = self.M_ecl_max
        e = 1.0 - b

        Z = jnp.where(
            jnp.abs(e) < 1e-10,
            jnp.log(M_max / M_min),
            (M_max**e - M_min**e) / e,
        )

        def ppf_scalar(u_val):
            return jnp.where(
                jnp.abs(e) < 1e-10,
                M_min * jnp.exp(u_val * Z),
                (u_val * e * Z + M_min**e) ** (1.0 / e),
            )

        u_arr = jnp.asarray(u)
        if u_arr.ndim == 0:
            return ppf_scalar(u_arr)
        return jax.vmap(ppf_scalar)(u_arr.ravel()).reshape(u_arr.shape)

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample n cluster masses from ECMF."""
        u = jax.random.uniform(key, (n,))
        return self.ppf(u)

    def mean_mass(self) -> Float[Array, ""]:
        """Mean cluster mass E[M_ecl]."""
        b = self.beta
        M_min = self.M_ecl_min
        M_max = self.M_ecl_max

        # E[M] = ∫ M * p(M) dM = ∫ M^(1-β) / Z dM
        e_mean = 2.0 - b  # exponent for mean integral
        e_norm = 1.0 - b  # exponent for normalization

        # Compute normalization
        if jnp.abs(e_norm) < 1e-10:
            Z = jnp.log(M_max / M_min)
        else:
            Z = (M_max**e_norm - M_min**e_norm) / e_norm

        # Compute mean integral
        if jnp.abs(e_mean) < 1e-10:
            mean_unnorm = jnp.log(M_max / M_min)
        else:
            mean_unnorm = (M_max**e_mean - M_min**e_mean) / e_mean

        return mean_unnorm / Z


# =============================================================================
# Maximum Stellar Mass - Cluster Mass Relation
# =============================================================================


class MaxStellarMass(eqx.Module):
    """Maximum stellar mass as function of embedded cluster mass.

    The m_max(M_ecl) relation encapsulates the observational finding that
    massive stars only form in massive clusters.

    Reference:
        Weidner & Kroupa (2004) MNRAS 348, 187 - Eq. 2
        "Evidence for a fundamental stellar upper mass limit"

        Weidner, Kroupa & Bonnell (2010) MNRAS 401, 275
        "The relation between the most-massive star and its parental
        star cluster mass"

    Three models implemented:
        1. 'weidner04': Original W&K 2004 relation (default)
           m_max = 0.39 * M_ecl^0.56 (capped at m_max_physical)

        2. 'analytical': Simplified analytical relation
           m_max = M_ecl / (1 + M_ecl / m_max_physical)

        3. 'sorted': Optimal sampling (sorted IMF up to cluster mass)
           Stars are drawn in mass-sorted order until sum ≥ M_ecl

    Parameters:
        model: Relation model ('weidner04', 'analytical', 'sorted')
        m_max_physical: Physical upper mass limit [M☉] (default: 150)
                       The most massive star that can form, regardless of cluster.
    """

    model: str = "weidner04"
    m_max_physical: float = 150.0  # Maximum physical stellar mass [M☉]

    def __call__(self, M_ecl: Float[Array, "..."]) -> Float[Array, "..."]:
        """Compute maximum stellar mass for given cluster mass.

        Args:
            M_ecl: Embedded cluster mass(es) [M☉]

        Returns:
            Maximum stellar mass m_max [M☉]
        """
        if self.model == "weidner04":
            return self._weidner04(M_ecl)
        elif self.model == "analytical":
            return self._analytical(M_ecl)
        elif self.model == "sorted":
            return self._sorted_approx(M_ecl)
        else:
            raise ValueError(f"Unknown model: {self.model}")

    def _weidner04(self, M_ecl: Float[Array, "..."]) -> Float[Array, "..."]:
        """Weidner & Kroupa (2004) empirical relation.

        m_max = 0.39 * M_ecl^0.56  (Eq. 2 in W&K 2004)

        Derived from fitting observations of young clusters.
        """
        m_max_raw = 0.39 * M_ecl**0.56
        return jnp.minimum(m_max_raw, self.m_max_physical)

    def _analytical(self, M_ecl: Float[Array, "..."]) -> Float[Array, "..."]:
        """Simplified analytical relation.

        m_max = M_ecl / (1 + M_ecl / m_max_physical)

        Asymptotes to M_ecl for small clusters, m_max_physical for large.
        """
        return M_ecl / (1.0 + M_ecl / self.m_max_physical)

    def _sorted_approx(self, M_ecl: Float[Array, "..."]) -> Float[Array, "..."]:
        """Approximate maximum mass from optimal/sorted sampling.

        For optimal sampling, stars are drawn in descending mass order
        until the total reaches M_ecl. The first star drawn is m_max.

        Approximation: m_max ≈ 0.5 * M_ecl^0.6 (empirical fit)
        """
        m_max_raw = 0.5 * M_ecl**0.6
        return jnp.minimum(m_max_raw, self.m_max_physical)


# =============================================================================
# Maximum Cluster Mass - SFR Relation
# =============================================================================


def max_cluster_mass_from_sfr(
    sfr: float,
    delta_t: float = 10.0,
    beta: float = 2.0,
    M_ecl_min: float = 5.0,
) -> Float[Array, ""]:
    """Maximum cluster mass as function of star formation rate.

    Reference:
        Weidner, Kroupa & Larsen (2004) MNRAS 350, 1503
        "The maximum mass of star clusters"

        Weidner & Kroupa (2005) ApJ 625, 754
        "The dependence of the IGIMF on SFR"

    The maximum cluster mass is set by requiring that, on average,
    one cluster of mass M_ecl_max forms per star formation epoch δt.

    For ECMF ∝ M^(-β) with β=2:
        M_ecl_max ≈ (SFR × δt × (β-1) × M_ecl_min^(β-1))^(1/(β-1))

    For β ≈ 2:
        M_ecl_max ≈ SFR × δt × M_ecl_min / ln(M_ecl_max/M_ecl_min)

    Simplified (Weidner+2004 Eq. 11):
        log10(M_ecl_max) ≈ 0.75 × log10(SFR) + 4.93

    Args:
        sfr: Star formation rate [M☉/yr]
        delta_t: Star formation epoch [Myr] (default: 10)
        beta: ECMF slope (default: 2.0)
        M_ecl_min: Minimum cluster mass [M☉] (default: 5.0)

    Returns:
        Maximum cluster mass M_ecl_max [M☉]
    """
    # Weidner+2004 empirical relation (Eq. 11)
    log_M_max = 0.75 * jnp.log10(sfr + 1e-10) + 4.93
    M_ecl_max = 10**log_M_max

    # Ensure M_ecl_max > M_ecl_min
    return jnp.maximum(M_ecl_max, M_ecl_min * 2.0)


# =============================================================================
# IGIMF Class
# =============================================================================


class IGIMF(eqx.Module):
    """Integrated Galactic Initial Mass Function.

    The IGIMF emerges from integrating the stellar IMF over all star-forming
    clusters in a galaxy, accounting for:
    1. The cluster mass function (ECMF)
    2. The m_max(M_ecl) relation limiting massive stars in small clusters
    3. The M_ecl_max(SFR) relation linking cluster population to SFR

    Key Result: The IGIMF is steeper than the input stellar IMF.
    For Kroupa stellar IMF (α_3 = 2.3), the IGIMF has effective α ≈ 2.7-3.0
    depending on SFR.

    Reference:
        Kroupa & Weidner (2003) ApJ 598, 1076
        Weidner & Kroupa (2005) ApJ 625, 754
        Weidner, Kroupa & Pflamm-Altenburg (2013) MNRAS 434, 84

    Parameters:
        stellar_imf: Base stellar IMF (e.g., Kroupa)
        sfr: Star formation rate [M☉/yr] (default: 1.0, Milky Way-like)
        ecmf_beta: ECMF power-law slope (default: 2.0)
        M_ecl_min: Minimum cluster mass [M☉] (default: 5.0)
        M_ecl_max: Maximum cluster mass [M☉] (default: None, computed from SFR)
        m_max_model: m_max(M_ecl) model ('weidner04', 'analytical', 'sorted')

    Examples:
        >>> from progenax.imf import PowerLawIMF
        >>> from progenax.imf.igimf import IGIMF
        >>>
        >>> # Default: Milky Way-like galaxy
        >>> igimf = IGIMF(stellar_imf=PowerLawIMF.kroupa(), sfr=1.0)
        >>>
        >>> # Starburst galaxy (high SFR)
        >>> igimf_burst = IGIMF(stellar_imf=PowerLawIMF.kroupa(), sfr=100.0)
        >>>
        >>> # Dwarf galaxy (low SFR)
        >>> igimf_dwarf = IGIMF(stellar_imf=PowerLawIMF.kroupa(), sfr=0.001)
    """

    stellar_imf: BaseIMF
    sfr: float = 1.0  # Star formation rate [M☉/yr]
    ecmf_beta: float = 2.0  # ECMF slope
    M_ecl_min: float = 5.0  # Minimum cluster mass [M☉]
    M_ecl_max: Optional[float] = None  # Maximum cluster mass (None = from SFR)
    m_max_model: str = "weidner04"  # m_max(M_ecl) relation
    m_max_physical: float = 150.0  # Physical upper mass limit [M☉]

    def _get_M_ecl_max(self) -> Float[Array, ""]:
        """Get maximum cluster mass, computing from SFR if not specified."""
        if self.M_ecl_max is not None:
            return jnp.asarray(self.M_ecl_max)
        return max_cluster_mass_from_sfr(
            self.sfr, beta=self.ecmf_beta, M_ecl_min=self.M_ecl_min
        )

    def _get_ecmf(self) -> EmbeddedClusterMassFunction:
        """Get ECMF with current parameters."""
        return EmbeddedClusterMassFunction(
            beta=self.ecmf_beta,
            M_ecl_min=self.M_ecl_min,
            M_ecl_max=self._get_M_ecl_max(),
        )

    def _get_m_max_relation(self) -> MaxStellarMass:
        """Get m_max(M_ecl) relation."""
        return MaxStellarMass(
            model=self.m_max_model,
            m_max_physical=self.m_max_physical,
        )

    def sample_cluster(
        self, key: PRNGKeyArray, M_ecl: float
    ) -> Float[Array, "..."]:
        """Sample stars from a single cluster of mass M_ecl.

        Stars are sampled from the stellar IMF, truncated at m_max(M_ecl).

        Args:
            key: JAX random key
            M_ecl: Cluster mass [M☉]

        Returns:
            Array of stellar masses [M☉]
        """
        # Get maximum stellar mass for this cluster
        m_max_relation = self._get_m_max_relation()
        m_max = m_max_relation(jnp.asarray(M_ecl))

        # Sample from stellar IMF (we need to sample enough to fill cluster)
        # Use Poisson estimate: N_stars ≈ M_ecl / mean_stellar_mass
        mean_mass = self.stellar_imf.mean_mass()
        n_expected = int(M_ecl / mean_mass) + 10  # Add buffer

        key1, key2 = jax.random.split(key)
        masses_raw = self.stellar_imf.sample(key1, n_expected)

        # Truncate at m_max
        masses_truncated = jnp.minimum(masses_raw, m_max)

        # Accept stars until we reach cluster mass (stochastic truncation)
        cumsum = jnp.cumsum(masses_truncated)
        mask = cumsum <= M_ecl

        return masses_truncated * mask

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample n stellar masses from the IGIMF.

        This is the main sampling method that:
        1. Samples cluster masses from ECMF
        2. For each cluster, samples stars truncated at m_max(M_ecl)
        3. Collects all stars up to desired count

        Args:
            key: JAX random key
            n: Number of stellar masses to sample

        Returns:
            n stellar masses following the IGIMF [M☉]
        """
        ecmf = self._get_ecmf()
        m_max_relation = self._get_m_max_relation()

        # Strategy: Sample many clusters, collect stars, subsample to n
        # Need to estimate how many clusters to sample
        mean_cluster_mass = ecmf.mean_mass()
        mean_stellar_mass = self.stellar_imf.mean_mass()
        stars_per_cluster = max(mean_cluster_mass / mean_stellar_mass, 1.0)

        # Sample enough clusters to get ~3n stars (with buffer)
        n_clusters = int(3 * n / stars_per_cluster) + 20

        key1, key2, key3 = jax.random.split(key, 3)

        # Sample cluster masses
        cluster_masses = ecmf.sample(key1, n_clusters)

        # For each cluster, get m_max
        m_max_values = m_max_relation(cluster_masses)

        # Sample stellar masses - use vectorized approach
        # For simplicity, sample n_per_cluster stars per cluster
        n_per_cluster = int(stars_per_cluster * 2) + 10

        # Generate all uniform samples at once
        u_samples = jax.random.uniform(key2, (n_clusters, n_per_cluster))

        def sample_cluster_stars(cluster_idx):
            """Sample stars for one cluster."""
            M_ecl = cluster_masses[cluster_idx]
            m_max = m_max_values[cluster_idx]
            u = u_samples[cluster_idx]

            # Transform u -> mass using stellar IMF ppf
            masses_raw = self.stellar_imf.ppf(u)

            # Truncate at m_max
            masses_truncated = jnp.minimum(masses_raw, m_max)

            # Keep stars until cluster mass reached
            cumsum = jnp.cumsum(masses_truncated)
            mask = cumsum <= M_ecl

            # Return masses (zero out rejected)
            return jnp.where(mask, masses_truncated, 0.0)

        # Sample all clusters
        all_masses = jax.vmap(sample_cluster_stars)(jnp.arange(n_clusters))
        all_masses_flat = all_masses.ravel()

        # Sort so non-zeros come first (JIT-compatible approach)
        # Use negative masses so sort puts positive masses first
        sort_key = jnp.where(all_masses_flat > 0, -all_masses_flat, jnp.inf)
        sorted_indices = jnp.argsort(sort_key)
        sorted_masses = all_masses_flat[sorted_indices]

        # Randomly permute the non-zero portion
        # First count how many non-zeros we have (approximately)
        n_available = n_clusters * n_per_cluster  # Upper bound

        # Shuffle the first n_available elements
        perm = jax.random.permutation(key3, n_available)

        # Take first n_available, shuffle, then take first n
        first_chunk = sorted_masses[:n_available]
        shuffled = first_chunk[perm]

        # Return first n (zeros at end if needed, but should be rare)
        return shuffled[:n]

    def effective_slope_high_mass(
        self, m_range: Tuple[float, float] = (10.0, 100.0)
    ) -> Float[Array, ""]:
        """Estimate effective power-law slope at high masses.

        The IGIMF is steeper than the stellar IMF at high masses.
        This method estimates the effective slope by fitting a power-law
        to sampled masses in the given range.

        Args:
            m_range: Mass range for slope estimation [M☉]

        Returns:
            Effective power-law slope α_eff (positive, so IMF ∝ m^(-α))
        """
        # Sample many masses
        key = jax.random.PRNGKey(12345)
        masses = self.sample(key, 100000)

        # Filter to mass range
        in_range = (masses >= m_range[0]) & (masses <= m_range[1])
        masses_selected = masses[in_range]

        # Fit power-law slope using log-histogram
        n_in_range = jnp.sum(in_range)

        log_masses = jnp.log10(jnp.where(in_range, masses, 1.0))
        log_masses_selected = log_masses[in_range]
        hist, bin_edges = jnp.histogram(log_masses_selected, bins=20)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Log-log slope: d(log N) / d(log m) = -(α - 1)
        # So α = 1 - slope
        valid = hist > 0
        n_valid = jnp.sum(valid)

        log_hist = jnp.log10(hist + 1)
        # Simple linear regression
        x = bin_centers[valid]
        y = log_hist[valid]
        slope = jnp.where(
            n_valid >= 3,
            jnp.cov(x, y)[0, 1] / (jnp.var(x) + 1e-10),
            -1.3  # Default to α=2.3 slope
        )

        # α = 1 - slope
        return 1.0 - slope

    def mean_mass(self) -> Float[Array, ""]:
        """Mean stellar mass from IGIMF.

        Returns:
            E[m] in solar masses
        """
        key = jax.random.PRNGKey(0)
        masses = self.sample(key, 50000)
        return jnp.mean(masses[masses > 0])

    def logpdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Approximate log-PDF of IGIMF via kernel density estimation.

        Note: This is an approximation from sampling. For exact IGIMF PDF,
        numerical integration over the cluster mass function is required.
        """
        # Sample reference distribution
        key = jax.random.PRNGKey(42)
        samples = self.sample(key, 50000)
        samples = samples[samples > 0]

        # KDE with log-space bandwidth
        log_samples = jnp.log(samples)
        bandwidth = 0.1  # in log-space

        def kde_logpdf(m_val):
            log_m = jnp.log(m_val)
            # Gaussian kernel in log-space
            kernel = jnp.exp(-0.5 * ((log_m - log_samples) / bandwidth) ** 2)
            density = jnp.mean(kernel) / (bandwidth * jnp.sqrt(2 * jnp.pi))
            # Transform to linear space: p(m) = p(log m) / m
            return jnp.log(density + 1e-30) - jnp.log(m_val + 1e-30)

        m_arr = jnp.asarray(m)
        if m_arr.ndim == 0:
            return kde_logpdf(m_arr)
        return jax.vmap(kde_logpdf)(m_arr.ravel()).reshape(m_arr.shape)

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def milky_way(cls, stellar_imf: Optional[BaseIMF] = None) -> "IGIMF":
        """Create IGIMF for Milky Way-like galaxy.

        Reference:
            Milky Way SFR ≈ 1-3 M☉/yr (Chomiuk & Povich 2011)

        Args:
            stellar_imf: Stellar IMF (default: Kroupa)

        Returns:
            IGIMF configured for Milky Way
        """
        if stellar_imf is None:
            from .power_law import PowerLawIMF

            stellar_imf = PowerLawIMF.kroupa()

        return cls(stellar_imf=stellar_imf, sfr=1.5)

    @classmethod
    def starburst(cls, stellar_imf: Optional[BaseIMF] = None) -> "IGIMF":
        """Create IGIMF for starburst galaxy.

        High SFR → High M_ecl_max → More massive stars → closer to stellar IMF

        Reference:
            Starburst SFR ~ 10-1000 M☉/yr

        Args:
            stellar_imf: Stellar IMF (default: Kroupa)

        Returns:
            IGIMF for starburst conditions
        """
        if stellar_imf is None:
            from .power_law import PowerLawIMF

            stellar_imf = PowerLawIMF.kroupa()

        return cls(stellar_imf=stellar_imf, sfr=100.0)

    @classmethod
    def dwarf_galaxy(cls, stellar_imf: Optional[BaseIMF] = None) -> "IGIMF":
        """Create IGIMF for dwarf galaxy.

        Low SFR → Low M_ecl_max → Few massive stars → steeper than stellar IMF

        Reference:
            Dwarf galaxy SFR ~ 10^-4 to 10^-2 M☉/yr

        Args:
            stellar_imf: Stellar IMF (default: Kroupa)

        Returns:
            IGIMF for dwarf galaxy conditions
        """
        if stellar_imf is None:
            from .power_law import PowerLawIMF

            stellar_imf = PowerLawIMF.kroupa()

        return cls(stellar_imf=stellar_imf, sfr=0.001)


# =============================================================================
# Convenience Functions
# =============================================================================


def igimf_effective_slope(
    sfr: float,
    stellar_alpha: float = 2.3,
    ecmf_beta: float = 2.0,
) -> Float[Array, ""]:
    """Analytical approximation for IGIMF effective slope at high masses.

    Reference:
        Weidner, Kroupa & Pflamm-Altenburg (2013) MNRAS 434, 84 - Eq. 14
        "For SFR → 0, the IGIMF approaches α_3,IGIMF ≈ α_3 + β - 1"

    For β = 2.0 and α_3 = 2.3:
        - High SFR: α_IGIMF → 2.3 (stellar IMF)
        - Low SFR: α_IGIMF → 2.3 + 2.0 - 1 = 3.3

    Args:
        sfr: Star formation rate [M☉/yr]
        stellar_alpha: Stellar IMF high-mass slope (default: 2.3)
        ecmf_beta: ECMF slope (default: 2.0)

    Returns:
        Effective IGIMF slope at high masses
    """
    # Transition SFR (roughly where IGIMF effects become significant)
    sfr_transition = 1.0  # M☉/yr

    # Asymptotic slopes
    alpha_high_sfr = stellar_alpha  # Approaches stellar IMF
    alpha_low_sfr = stellar_alpha + ecmf_beta - 1.0  # Maximum steepening

    # Smooth transition (logistic function in log-SFR)
    log_sfr = jnp.log10(sfr + 1e-10)
    log_sfr_trans = jnp.log10(sfr_transition)
    transition_width = 1.0  # decades

    weight = 1.0 / (1.0 + jnp.exp(-(log_sfr - log_sfr_trans) / transition_width))

    return alpha_low_sfr + weight * (alpha_high_sfr - alpha_low_sfr)


__all__ = [
    "EmbeddedClusterMassFunction",
    "MaxStellarMass",
    "max_cluster_mass_from_sfr",
    "IGIMF",
    "igimf_effective_slope",
]

# progenax/src/progenax/cluster/constants.py
"""Physical constants for FDF and gravoturbulent calculations.

All constants are in units consistent with the jaxstro ecosystem:
- Masses: M☉
- Distances: pc
- Times: Myr
- Velocities: km/s

References
----------
- Larson (1981) MNRAS 194, 809 - Velocity-size relation
- Solomon et al. (1987) ApJ 319, 730 - GMC properties
- Federrath et al. (2010) A&A 512, A81 - Turbulence-density relation
- Goodwin & Whitworth (2004) A&A 413, 929 - Fractal dimension
"""

# =============================================================================
# Gravitational Constant
# =============================================================================

# G in km²/s²/pc/M☉ (for Mach calculation from virial equilibrium)
G_KMS = 4.3e-3


# =============================================================================
# Sound Speed
# =============================================================================

# Sound speed in cold GMC (T ~ 10 K) [km/s]
# c_s = sqrt(k_B T / m_H) ≈ 0.2 km/s for molecular hydrogen at 10 K
C_S_DEFAULT = 0.2


# =============================================================================
# Turbulence Driving Parameter
# =============================================================================

# b parameter (Federrath+2010)
# b ≈ 1/3 for solenoidal (incompressible) driving
# b ≈ 1.0 for compressive driving
# b ≈ 0.4 for natural mixture (default)
B_DEFAULT = 0.4


# =============================================================================
# Power Spectrum Slopes
# =============================================================================

# Kolmogorov slope: incompressible limit
# E(k) ∝ k^(-5/3) → P(k) ∝ k^(-11/3)
BETA_KOLMOGOROV = 11.0 / 3.0  # ≈ 3.67

# Burgers slope: shock-dominated limit
# Velocity discontinuities → steeper spectrum
BETA_BURGERS = 4.0


# =============================================================================
# Larson Velocity-Size Relation
# =============================================================================

# σ_v = σ_v0 × (R / 1 pc)^α (Larson 1981, Solomon+1987)

# Normalization velocity [km/s] at 1 pc
SIGMA_V0_DEFAULT = 1.0

# Power-law exponent (Larson 1981: α ≈ 0.38-0.5)
ALPHA_LARSON = 0.5


# =============================================================================
# Chi Parameter Bounds (Goodwin & Whitworth 2004)
# =============================================================================

# Chi controls small-scale vs large-scale power distribution
# 1.6 = most clumpy (small-scale dominated)
# 3.0 = smoothest (large-scale dominated)
CHI_MIN = 1.6
CHI_MAX = 3.0


__all__ = [
    "G_KMS",
    "C_S_DEFAULT",
    "B_DEFAULT",
    "BETA_KOLMOGOROV",
    "BETA_BURGERS",
    "SIGMA_V0_DEFAULT",
    "ALPHA_LARSON",
    "CHI_MIN",
    "CHI_MAX",
]

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

# Kolmogorov slope: incompressible / transonic DENSITY-spectrum limit.
# E(k) ∝ k^(-5/3) → 3D power-spectral-density P_3D(k) ∝ k^(-11/3). Kim & Ryu (2005)
# measure the transonic (M~1.2) 3D density spectrum slope ≈ -5/3, i.e. β_P3D ≈ 11/3.
BETA_KOLMOGOROV = 11.0 / 3.0  # ≈ 3.67  (transonic ceiling for the density spectrum)

# Burgers VELOCITY slope (shock-dominated). NOTE: this is a *velocity* power-spectrum
# slope; it is NOT the density-spectrum slope of supersonic turbulence (which FLATTENS).
# Retained for legacy imports only; spectral_slope_from_mach no longer uses it.
BETA_BURGERS = 4.0

# Kim & Ryu (2005) DENSITY power spectrum slope vs Mach (3D, P_3D convention).
# Their 3D least-squares slopes E_ρ(k)∝k^{-s}: s = 1.73 (M=1.2), 1.08 (3.4), 0.75 (7.3),
# 0.52 (12). With β_P3D = s + 2, a log-linear least-squares fit gives:
#   β(M) = KIMRYU_BETA_INTERCEPT + KIMRYU_BETA_LOGSLOPE * log10(M),
# clipped to [BETA_DENSITY_FLOOR, BETA_KOLMOGOROV]. β DECREASES with Mach (density
# concentrates into sheets/filaments → shallower spectrum). See per-paper note kim-ryu-2005.
KIMRYU_BETA_INTERCEPT = 3.788
KIMRYU_BETA_LOGSLOPE = -1.203
# 1D strong-shock density limit P_ρ ∝ k^0 → β_P3D = 2 (Saichev & Woyczynski 1996; Kim&Ryu 1D).
BETA_DENSITY_FLOOR = 2.0


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
    "KIMRYU_BETA_INTERCEPT",
    "KIMRYU_BETA_LOGSLOPE",
    "BETA_DENSITY_FLOOR",
    "SIGMA_V0_DEFAULT",
    "ALPHA_LARSON",
    "CHI_MIN",
    "CHI_MAX",
]

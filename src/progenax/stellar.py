"""Tout (1996) ZAMS stellar-structure relations (M -> L, R, T_eff, log g) + inverse.

A differentiable, metallicity-dependent placeholder for the eventual ``startrax``
stellar tracks. Distinct from :func:`progenax.compute_stellar_radii` (Demircan &
Kahraman 1991 empirical *collision* radii for N-body); these are *photometric* ZAMS
relations for CMD / mass-function science.

Reference: Tout et al. (1996), MNRAS 281, 257 (Tables 1 & 2). Coefficients verified
cell-by-cell vs the held PDF (75/75 exact) — see
``docs/core-papers/tout1996_zams_coefficients_verified.md``.

Valid range: 0.1 <= M/Msun <= 100, 1e-4 <= Z <= 0.03 (~5% MS accuracy). ``Z`` is
clipped to [1e-4, 0.03] because the paper forbids metallicity extrapolation — the
rational functions go negative outside that range (Tout+1996 p. 262).
"""
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from jaxstro.constants import LSUN_ERG_S, RSUN_CM, SIGMA_SB, G_CGS, MSUN_G

__all__ = [
    "zams_luminosity",
    "zams_radius",
    "zams_effective_temperature",
    "zams_surface_gravity",
    "inverse_zams_luminosity",
]

_Z_SUN = 0.02  # Tout+1996 reference solar metallicity

# Tout+1996 Table 1 — L(M,Z). Row r = degree-4 polynomial [c0..c4] in log10(Z/Zsun)
# for coefficient (alpha, beta, gamma, delta, epsilon, zeta, eta). PDF-verified (P1).
_TOUT_L_COEFFS = jnp.array([
    [ 0.39704170, -0.32913574,  0.34776688,  0.37470851,  0.09011915],  # alpha
    [ 8.52762600,-24.41225973, 56.43597107, 37.06152575,  5.45624060],  # beta
    [ 0.00025546, -0.00123461, -0.00023246,  0.00045519,  0.00016176],  # gamma
    [ 5.43288900, -8.62157806, 13.44202049, 14.51584135,  3.39793084],  # delta
    [ 5.56357900,-10.32345224, 19.44322980, 18.97361347,  4.16903097],  # epsilon
    [ 0.78866060, -2.90870942,  6.54713531,  4.05606657,  0.53287322],  # zeta
    [ 0.00586685, -0.01704237,  0.03872348,  0.02570041,  0.00383376],  # eta
])

# Tout+1996 Table 2 — R(M,Z). Coefficients (theta, iota, kappa, lambda, mu, xi,
# omicron, pi); nu is a Z-independent scalar. PDF-verified (P1).
_TOUT_R_COEFFS = jnp.array([
    [ 1.71535900,  0.62246212, -0.92557761, -1.16996966, -0.30631491],  # theta
    [ 6.59778800, -0.42450044,-12.13339427,-10.73509484, -2.51487077],  # iota
    [10.08855000, -7.11727086,-31.67119479,-24.24848322, -5.33608972],  # kappa
    [ 1.01249500,  0.32699690, -0.00923418, -0.03876858, -0.00412750],  # lambda
    [ 0.07490166,  0.02410413,  0.07233664,  0.03040467,  0.00197741],  # mu
    [ 3.08223400,  0.94472050, -2.15200882, -2.49219496, -0.63848738],  # xi
    [17.84778000, -7.45345690,-48.96066856,-40.05386135, -9.09331816],  # omicron
    [ 0.00022582, -0.00186899,  0.00388783,  0.00142402, -0.00007671],  # pi
])
_TOUT_R_NU = 0.01077422  # Table 2, Z-independent

_INVERSE_NEWTON_ITERS = 20  # quadratic convergence; ~6-8 reach machine precision


def _metallicity_coeffs(coeff_matrix: Float[Array, "n 5"], Z: float) -> Float[Array, "n"]:
    """Evaluate each row's degree-4 log10(Z/Zsun) polynomial -> per-coefficient scalars."""
    log_Z = jnp.log10(jnp.clip(Z, 1e-4, 0.03) / _Z_SUN)
    basis = log_Z ** jnp.arange(5)            # (5,)
    return coeff_matrix @ basis               # (n,)


def zams_luminosity(mass: Float[Array, "..."], Z: float = 0.02) -> Float[Array, "..."]:
    """ZAMS luminosity [L_sun] from mass [M_sun] and metallicity Z. Tout+1996 Table 1.

    Elementwise over ``mass``; ``Z`` is a scalar (default solar, 0.02). Differentiable.
    """
    M = jnp.asarray(mass, float)
    a, b, g, d, e, z, h = _metallicity_coeffs(_TOUT_L_COEFFS, Z)
    num = a * M**5.5 + b * M**11
    den = g + M**3 + d * M**5 + e * M**7 + z * M**8 + h * M**9.5
    return num / jnp.maximum(den, 1e-10)

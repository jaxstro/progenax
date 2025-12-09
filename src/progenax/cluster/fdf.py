# progenax/src/progenax/cluster/fdf.py
"""
Fractal Displacement Field (FDF) for differentiable cluster substructure.

This module implements a JAX-native, fully differentiable method for generating
fractal substructure in star cluster initial conditions. It replaces the
Goodwin-Whitworth (2004) recursive tree algorithm with a Fourier-mode
displacement field that:

1. Is fully differentiable in clumpiness (chi), blend strength (lambda_frac),
   and amplitude scale (sigma_u)
2. Produces statistically similar structures (same Q_CW, sigma_Sigma/Sigma)
3. Has physically motivated connection to turbulent star formation

Physical basis:
- Turbulent fragmentation in molecular clouds (Larson 1981, Mac Low & Klessen 2004)
- Power-law velocity/density spectra from supersonic turbulence (Federrath+ 2010)

References:
    Goodwin & Whitworth (2004) A&A 413, 929 - Original fractal method (non-JAX)
    Cartwright & Whitworth (2004) MNRAS 348, 589 - Q parameter
    Federrath et al. (2010) A&A 512, A81 - Turbulence spectra
"""

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array, random
from jaxtyping import Float, PRNGKeyArray


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True)
class FractalField:
    """Frozen stochastic structure for displacement field.

    This structure is generated once per realization and frozen via
    `jax.lax.stop_gradient`. No gradients flow through the wavevectors,
    phases, or polarization directions.

    Attributes
    ----------
    k_vecs : Array, shape (M, 3)
        Wavevectors in 1/pc. Magnitudes are log-spaced between k_min and k_max.
    phases : Array, shape (M,)
        Random phases in [0, 2pi] for each mode.
    base_vecs : Array, shape (M, 3)
        Unit polarization vectors for each mode (random directions on S^2).

    Notes
    -----
    The FractalField should be wrapped in `jax.tree_util.tree_map(stop_gradient, ...)`
    before use in differentiable pipelines. This ensures that the stochastic
    structure is fixed while gradients flow through amplitude parameters.
    """

    k_vecs: Float[Array, "M 3"]
    phases: Float[Array, "M"]
    base_vecs: Float[Array, "M 3"]


# Register as JAX pytree (frozen dataclass is already a pytree, but explicit is safer)
jax.tree_util.register_dataclass(
    FractalField,
    data_fields=["k_vecs", "phases", "base_vecs"],
    meta_fields=[],
)


@dataclass(frozen=True)
class FractalDisplacementLayer:
    """Parameters for fractal displacement field layer.

    This dataclass holds all tunable parameters for the FDF method.
    The key differentiable parameters are chi, lambda_frac, and sigma_u.

    Attributes
    ----------
    chi : float
        Clumpiness parameter in [1.5, 3.0]. Controls spectral slope.
        chi=1.5: highly clumpy (more small-scale power)
        chi=3.0: smooth (more large-scale power)
        Calibrated to match Goodwin-Whitworth fractal dimension D.
    lambda_frac : float
        Fractal fraction in [0, 1]. Controls blend strength.
        lambda_frac=0: pure smooth profile
        lambda_frac=1: full displacement applied
    sigma_u : float
        Dimensionless displacement amplitude scale, in units of R_half.
        The actual RMS displacement is approximately sigma_u * R_half / sqrt(2).
        Typical values: 0.1-0.5. Exact mapping to Q_CW is set by calibration.
    n_modes : int
        Number of Fourier modes. More modes = finer structure.
        Default 64 is sufficient for most applications.
    k_min_factor : float
        Minimum wavenumber as fraction of 1/R_half.
        Default 0.5 gives modes on scales ~2*R_half.
    k_max_factor : float
        Maximum wavenumber as fraction of 1/R_half.
        Default 20 gives modes on scales ~R_half/20.
    radial_mode : str
        How to handle radial profile: 'full', 'tangential', or 'remap'.
        Default 'remap' preserves exact radial CDF (recommended).
        'tangential' is expert/experimental.
    virial_ratio : float
        Target Q_vir = K/|U| after velocity assignment.
    coherent_velocities : bool
        If True, velocity field correlates with displacement field.
    lambda_vel : float
        Velocity coherence strength in [0, 1].

    Notes
    -----
    Unlike Goodwin-Whitworth D, chi is differentiable. The mapping
    chi -> Q_CW (Cartwright-Whitworth) is established via calibration.

    The spectral slope is beta(chi) = beta_0 + beta_1*(3 - chi) where beta_0 ~ 2.0
    and beta_1 ~ 1.5 (calibration-dependent).
    """

    chi: float = 2.0
    lambda_frac: float = 1.0
    sigma_u: float = 0.3
    n_modes: int = 64
    k_min_factor: float = 0.5
    k_max_factor: float = 20.0
    radial_mode: Literal["full", "tangential", "remap"] = "remap"
    virial_ratio: float = 0.5
    coherent_velocities: bool = True
    lambda_vel: float = 0.3


# Register as JAX pytree
jax.tree_util.register_dataclass(
    FractalDisplacementLayer,
    data_fields=[
        "chi", "lambda_frac", "sigma_u", "n_modes",
        "k_min_factor", "k_max_factor", "radial_mode",
        "virial_ratio", "coherent_velocities", "lambda_vel",
    ],
    meta_fields=[],
)


# =============================================================================
# Field Initialization
# =============================================================================


def init_fractal_field(
    key: PRNGKeyArray,
    n_modes: int,
    R_half: float,
    k_min_factor: float = 0.5,
    k_max_factor: float = 20.0,
) -> FractalField:
    """Initialize frozen stochastic structure for displacement field.

    Creates a FractalField with log-spaced wavenumbers, random directions,
    random phases, and random polarization vectors.

    Parameters
    ----------
    key : PRNGKey
        JAX random key for reproducibility.
    n_modes : int
        Number of Fourier modes (M). More modes = finer structure.
    R_half : float
        Half-mass radius in pc. Sets scale for k_min, k_max.
    k_min_factor : float, default 0.5
        k_min = k_min_factor / R_half. Default gives modes on scales ~2*R_half.
    k_max_factor : float, default 20.0
        k_max = k_max_factor / R_half. Default gives modes on scales ~R_half/20.

    Returns
    -------
    FractalField
        Frozen stochastic structure with k_vecs, phases, base_vecs.

    Notes
    -----
    This structure should be frozen via stop_gradient before use in
    differentiable pipelines. The k_vecs depend on R_half at initialization,
    but because we apply stop_gradient, changes in R_half during inference
    affect only the amplitude scaling - NOT the internal phase structure.
    """
    key_dir, key_phase, key_pol = random.split(key, 3)

    # Wavenumber range
    k_min = k_min_factor / R_half
    k_max = k_max_factor / R_half

    # Log-spaced wavenumber magnitudes
    t = jnp.linspace(0.0, 1.0, n_modes)
    k_mags = k_min * (k_max / k_min) ** t  # (M,)

    # Random directions on unit sphere (normalize Gaussian vectors)
    raw_dirs = random.normal(key_dir, (n_modes, 3))
    k_dirs = raw_dirs / jnp.linalg.norm(raw_dirs, axis=1, keepdims=True)

    # Wavevectors = magnitude * direction
    k_vecs = k_mags[:, None] * k_dirs  # (M, 3)

    # Random phases in [0, 2*pi]
    phases = random.uniform(key_phase, (n_modes,)) * (2 * jnp.pi)

    # Random polarization directions (unit vectors)
    raw_pol = random.normal(key_pol, (n_modes, 3))
    base_vecs = raw_pol / jnp.linalg.norm(raw_pol, axis=1, keepdims=True)

    return FractalField(
        k_vecs=k_vecs,
        phases=phases,
        base_vecs=base_vecs,
    )

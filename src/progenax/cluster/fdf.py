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


# =============================================================================
# Spectral Constants (from calibration)
# =============================================================================

BETA_0 = 2.0  # Baseline spectral slope
BETA_1 = 1.5  # Slope sensitivity to chi


# =============================================================================
# Amplitude Computation
# =============================================================================


def compute_amplitudes(
    field: FractalField,
    chi: float,
    sigma_u: float,
) -> Float[Array, "M 3"]:
    """Compute mode amplitudes from chi and sigma_u.

    Parameters
    ----------
    field : FractalField
        Frozen field with k_vecs and base_vecs.
    chi : float
        Clumpiness parameter in [1.5, 3.0].
    sigma_u : float
        Displacement amplitude scale in physical units (same units as positions,
        typically pc). The caller should pass sigma_u_physical = dimensionless_sigma_u * R_half.

    Returns
    -------
    a_vecs : Array, shape (M, 3)
        Amplitude vectors for each mode. a_vecs[n] = A_n * base_vecs[n].

    Notes
    -----
    This function is differentiable in chi and sigma_u.
    Gradients do NOT flow through field (should be stop_gradient'd).

    The spectral slope mapping is:
        beta(chi) = beta_0 + beta_1*(3 - chi)

    Mode amplitudes follow:
        A_n proportional to k_n^(-beta/2)

    Normalized so that sum(A_n^2) = sigma_u^2.
    """
    # Wavenumber magnitudes from field
    k_mags = jnp.linalg.norm(field.k_vecs, axis=1)  # (M,)

    # Spectral slope from chi
    # Lower chi → lower beta → shallower slope → more small-scale power (clumpier)
    # Higher chi → higher beta → steeper slope → less small-scale power (smoother)
    beta = BETA_0 + BETA_1 * (chi - 1.5)

    # Unnormalized amplitudes: A_n proportional to k_n^(-beta/2)
    raw_amps = k_mags ** (-0.5 * beta)  # (M,)

    # Normalize so that sum(A_n^2) = sigma_u^2
    norm = jnp.sqrt(jnp.sum(raw_amps ** 2))
    amps = sigma_u * raw_amps / norm  # (M,)

    # Amplitude vectors = scalar amplitude * unit polarization
    a_vecs = amps[:, None] * field.base_vecs  # (M, 3)

    return a_vecs


# =============================================================================
# Displacement Field Evaluation
# =============================================================================


def evaluate_displacement(
    positions: Float[Array, "N 3"],
    field: FractalField,
    a_vecs: Float[Array, "M 3"],
) -> Float[Array, "N 3"]:
    """Evaluate displacement field at given positions.

    Computes u(x) = sum_n a_n cos(k_n . x + phi_n)

    Parameters
    ----------
    positions : Array, shape (N, 3)
        Positions in pc where to evaluate the field.
    field : FractalField
        Frozen field with k_vecs and phases.
    a_vecs : Array, shape (M, 3)
        Amplitude vectors from compute_amplitudes.

    Returns
    -------
    displacements : Array, shape (N, 3)
        Displacement vectors in pc.

    Notes
    -----
    This is fully differentiable in positions and a_vecs.
    """
    # k_n . x_i: shape (N, M)
    # positions: (N, 3), k_vecs: (M, 3)
    dot_products = jnp.einsum("nd,md->nm", positions, field.k_vecs)

    # Add phases: (N, M)
    arguments = dot_products + field.phases[None, :]

    # Cosine terms: (N, M)
    cos_terms = jnp.cos(arguments)

    # Sum over modes: (N, M) @ (M, 3) -> (N, 3)
    displacements = cos_terms @ a_vecs

    return displacements


# =============================================================================
# Displacement Application with Radial Modes
# =============================================================================


def apply_displacement(
    positions: Float[Array, "N 3"],
    displacements: Float[Array, "N 3"],
    lambda_frac: float,
    target_radii: Float[Array, "N"],
    mode: Literal["full", "tangential", "remap"] = "remap",
) -> Float[Array, "N 3"]:
    """Apply displacement field to positions.

    Parameters
    ----------
    positions : Array, shape (N, 3)
        Base positions from smooth profile.
    displacements : Array, shape (N, 3)
        Displacement vectors from evaluate_displacement.
    lambda_frac : float
        Blend fraction in [0, 1].
    target_radii : Array, shape (N,)
        Target radii for 'remap' mode.
    mode : str
        How to handle radial profile:
        - 'full': Just add lambda_frac * displacements (radial CDF changes).
        - 'tangential': Project out radial component, renormalize to original
          radius. EXPERIMENTAL: exact per-star radius preservation.
        - 'remap' (default): Full displacement, then rank-based radial remap
          to exactly match target radial CDF. RECOMMENDED.

    Returns
    -------
    positions_out : Array, shape (N, 3)
        Displaced positions.

    Notes
    -----
    For 'remap' mode: Sorting is piecewise-constant in the permutation;
    gradients flow through the *values* being sorted, not which star is rank k.
    We accept non-smooth gradients w.r.t. permutations and only rely on
    smoothness in the radii values themselves.
    """
    if mode == "full":
        return positions + lambda_frac * displacements

    elif mode == "tangential":
        # Project out radial component
        r = jnp.linalg.norm(positions, axis=1, keepdims=True)
        r_hat = positions / jnp.maximum(r, 1e-10)

        # Tangential displacement
        u_radial = jnp.sum(displacements * r_hat, axis=1, keepdims=True)
        u_tangential = displacements - u_radial * r_hat

        # Apply tangential displacement
        pos_displaced = positions + lambda_frac * u_tangential

        # Renormalize to original radius
        r_new = jnp.linalg.norm(pos_displaced, axis=1, keepdims=True)
        pos_out = pos_displaced * (r / jnp.maximum(r_new, 1e-10))

        return pos_out

    elif mode == "remap":
        # Full displacement
        pos_displaced = positions + lambda_frac * displacements

        # Rank-based radial remap
        r_displaced = jnp.linalg.norm(pos_displaced, axis=1)

        # Sort indices
        idx_displaced = jnp.argsort(r_displaced)
        target_sorted = jnp.sort(target_radii)

        # Map: star at rank k gets target radius at rank k
        r_mapped = jnp.zeros_like(r_displaced)
        r_mapped = r_mapped.at[idx_displaced].set(target_sorted)

        # Rescale directions to new radii
        r_hat = pos_displaced / jnp.maximum(r_displaced[:, None], 1e-10)
        pos_out = r_hat * r_mapped[:, None]

        return pos_out

    else:
        raise ValueError(f"Unknown radial mode: {mode}")

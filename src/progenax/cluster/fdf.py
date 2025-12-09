# progenax/src/progenax/cluster/fdf.py
"""
Fractal Displacement Field (FDF) for differentiable cluster substructure.

This module implements a JAX-native, fully differentiable method for generating
fractal substructure in star cluster initial conditions. It replaces the
Goodwin-Whitworth (2004) recursive tree algorithm with a Fourier-mode
displacement field that:

1. Is fully differentiable in clumpiness (chi), blend strength (lambda_frac),
   and amplitude scale (sigma_u)
2. Creates density perturbations via irrotational displacement (div(u) ≠ 0)
3. Has physically motivated connection to turbulent star formation

Physical basis:
- Turbulent fragmentation in molecular clouds (Larson 1981, Mac Low & Klessen 2004)
- Power-law velocity/density spectra from supersonic turbulence (Federrath+2010)

WARNING - Uncalibrated Parameters
---------------------------------
The spectral envelope parameters (BETA_BASE, SIGMA_LOGK) and default sigma_u
are preliminary heuristics. They are NOT calibrated against:

- MHD simulations of turbulent molecular clouds
- Cartwright & Whitworth (2004) Q(D) measurements

For most applications, prefer the density-field FDF (``fdf_density.py``) which
directly modulates density rather than displacing positions.

References
----------
- Goodwin & Whitworth (2004) A&A 413, 929 - Original fractal method (non-JAX)
- Cartwright & Whitworth (2004) MNRAS 348, 589 - Q parameter definition
- Federrath et al. (2010) A&A 512, A81 - Turbulence spectra
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
        In v2, displacement vectors are parallel to k_vecs (irrotational field).
    phases : Array, shape (M,)
        Random phases in [0, 2pi] for each mode.
    base_vecs : Array, shape (M, 3)
        Unit polarization vectors for each mode (random directions on S^2).
        **LEGACY (v1):** No longer used for displacement direction in v2.
        Kept for API compatibility; displacement is now parallel to k_vecs.

    Notes
    -----
    The FractalField should be wrapped in `jax.tree_util.tree_map(stop_gradient, ...)`
    before use in differentiable pipelines. This ensures that the stochastic
    structure is fixed while gradients flow through amplitude parameters.

    **v2 Change:** Displacement vectors are now parallel to wavevectors (a ∥ k),
    making the field irrotational (u = ∇ψ). This creates actual density
    perturbations via div(u) ≠ 0, rather than incompressible rearrangement.
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

    **Spectral Envelope (v2):**

    The amplitude spectrum uses a χ-dependent lognormal envelope in k-space:
        - chi ≈ CHI_MIN (1.6): envelope peaked at high k → small-scale dominated → clumpy
        - chi ≈ CHI_MAX (3.0): envelope peaked at low k → large-scale dominated → smooth

    Displacement vectors are parallel to wavevectors (irrotational/potential field),
    creating actual density perturbations via div(u) ≠ 0.

    This approach is physically motivated by turbulent star formation: the envelope
    peak controls which scales dominate the density structure.
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
# Spectral Constants (from versioned config)
# =============================================================================

from progenax.cluster.fdf_config import FDF_DISPLACEMENT_DEFAULTS

# Chi range for clumpiness parameter
CHI_MIN = 1.6  # Clumpy end (small-scale dominated)
CHI_MAX = 3.0  # Smooth end (large-scale dominated)

# Lognormal envelope parameters
# NOTE: These are uncalibrated heuristics. See fdf_config.py for details.
BETA_BASE = FDF_DISPLACEMENT_DEFAULTS.beta_base  # Mild baseline power-law slope
SIGMA_LOGK = FDF_DISPLACEMENT_DEFAULTS.sigma_logk  # Envelope width in log-k space

# Legacy constants (kept for backwards compatibility, no longer used)
BETA_0 = 2.0  # Baseline spectral slope (DEPRECATED)
BETA_1 = 1.5  # Slope sensitivity to chi (DEPRECATED)


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
        Frozen field with k_vecs and phases.
    chi : float
        Clumpiness parameter in [CHI_MIN, CHI_MAX] (typically [1.6, 3.0]).
    sigma_u : float
        Displacement amplitude scale in physical units (same units as positions,
        typically pc). The caller should pass sigma_u_physical = dimensionless_sigma_u * R_half.

    Returns
    -------
    a_vecs : Array, shape (M, 3)
        Amplitude vectors for each mode. a_vecs[n] = A_n * k_hat[n].
        Displacement vectors are parallel to wavevectors (potential-like modes).

    Notes
    -----
    This function is differentiable in chi and sigma_u.
    Gradients do NOT flow through field (should be stop_gradient'd).

    **Spectral Envelope (v2):**

    We use a χ-dependent lognormal envelope in k-space with a peak that moves:
        - chi ≈ CHI_MIN (1.6, clumpy): peak at high k → small-scale dominated
        - chi ≈ CHI_MAX (3.0, smooth): peak at low k → large-scale dominated

    The envelope is:
        h(k; χ) = exp(-0.5 * ((log k - log k_peak) / σ_logk)²) * k^(-β_base/2)

    where log_k_peak interpolates linearly from log_k_max (clumpy) to log_k_min (smooth).

    **Compressive Modes:**

    Displacement vectors are parallel to wavevectors: a_vecs[n] ∥ k_vecs[n].
    This makes the field irrotational (u = ∇ψ), so div(u) ≠ 0, creating actual
    density perturbations (clumps and voids).

    The extra factor of k_mag in the amplitude enhances small-scale divergence,
    making density contrast more prominent at small scales.
    """
    # Wavevector magnitudes and unit directions
    k_vecs = field.k_vecs  # (M, 3)
    k_mags = jnp.linalg.norm(k_vecs, axis=1)  # (M,)
    k_hat = k_vecs / jnp.maximum(k_mags[:, None], 1e-12)  # (M, 3) unit vectors

    # Map chi to [0, 1]: 0 = clumpy (small-scale), 1 = smooth (large-scale)
    chi_clamped = jnp.clip(chi, CHI_MIN, CHI_MAX)
    t = (chi_clamped - CHI_MIN) / (CHI_MAX - CHI_MIN)

    # Work in log-k space
    log_k = jnp.log(k_mags + 1e-12)
    log_k_min = jnp.min(log_k)
    log_k_max = jnp.max(log_k)

    # χ-dependent peak position: clumpy → high k, smooth → low k
    log_k_peak = (1.0 - t) * log_k_max + t * log_k_min

    # Lognormal envelope centered on peak
    dlog = (log_k - log_k_peak) / SIGMA_LOGK
    gauss_envelope = jnp.exp(-0.5 * dlog**2)  # (M,)

    # Combine with mild baseline power-law slope
    power_law = k_mags ** (-0.5 * BETA_BASE)  # (M,)
    h = gauss_envelope * power_law

    # Extra k factor to enhance small-scale divergence (density contrast)
    raw = h * k_mags  # (M,)

    # Normalize so RMS displacement magnitude ≈ sigma_u
    norm = jnp.sqrt(jnp.sum(raw**2) + 1e-30)
    amps = sigma_u * raw / jnp.maximum(norm, 1e-12)  # (M,)

    # Amplitude vectors parallel to wavevectors (irrotational/potential field)
    a_vecs = amps[:, None] * k_hat  # (M, 3)

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


# =============================================================================
# Velocity Structure
# =============================================================================


def assign_fractal_velocities(
    key: PRNGKeyArray,
    positions: Float[Array, "N 3"],
    masses: Float[Array, "N"],
    field: FractalField,
    a_vecs: Float[Array, "M 3"],
    frac_params: FractalDisplacementLayer,
    G: float,
    lambda_vel: float = None,
) -> Float[Array, "N 3"]:
    """Assign velocities with optional coherent structure.

    Creates velocities that:
    1. Achieve the target virial ratio Q_vir
    2. Optionally correlate with the displacement field (coherent subclumps)
    3. Have zero center-of-mass motion

    Parameters
    ----------
    key : PRNGKey
        JAX random key.
    positions : Array, shape (N, 3)
        Final positions in pc.
    masses : Array, shape (N,)
        Stellar masses in M_sun.
    field : FractalField
        Frozen displacement field structure.
    a_vecs : Array, shape (M, 3)
        Amplitude vectors.
    frac_params : FractalDisplacementLayer
        Parameters including virial_ratio, coherent_velocities, lambda_vel.
    G : float
        Gravitational constant in units consistent with positions/masses.
    lambda_vel : float, optional
        Override for velocity coherence strength [0, 1].
        If None, uses frac_params.lambda_vel.

    Returns
    -------
    velocities : Array, shape (N, 3)
        Velocity vectors achieving target Q_vir.

    Notes
    -----
    Uses O(N^2) potential energy calculation internally. Acceptable for
    IC generation (N ~ 10^3-10^4) but not for every N-body timestep.
    """
    from progenax.dynamics.virial import compute_potential_energy

    N = masses.shape[0]
    M_total = jnp.sum(masses)

    # Use override if provided, else use frac_params
    lam_vel = lambda_vel if lambda_vel is not None else frac_params.lambda_vel
    lam_vel = jnp.clip(lam_vel, 0.0, 1.0)

    # Compute potential energy for virial scaling
    U_total = compute_potential_energy(positions, masses, G)
    K_target = frac_params.virial_ratio * jnp.abs(U_total)
    sigma_v = jnp.sqrt(2 * K_target / M_total)

    # Base velocities: isotropic Gaussian
    key, subkey = random.split(key)
    v_base = random.normal(subkey, (N, 3)) * sigma_v / jnp.sqrt(3)

    # Coherent perturbation from displacement field
    def add_coherent(v_base):
        # Evaluate displacement at final positions
        u = evaluate_displacement(positions, field, a_vecs)
        u_norm = jnp.linalg.norm(u, axis=1, keepdims=True)
        u_hat = u / jnp.maximum(u_norm, 1e-10)

        # Add coherent component
        v_coherent = lam_vel * sigma_v * u_hat
        return v_base + v_coherent

    def no_coherent(v_base):
        return v_base

    # Apply coherent component if enabled and lambda_vel > 0
    use_coherent = frac_params.coherent_velocities & (lam_vel > 0)
    velocities = jax.lax.cond(use_coherent, add_coherent, no_coherent, v_base)

    # Remove COM motion first
    v_com = jnp.sum(masses[:, None] * velocities, axis=0) / M_total
    velocities = velocities - v_com

    # Rescale to exact target virial ratio (AFTER COM removal)
    K_actual = 0.5 * jnp.sum(masses[:, None] * velocities ** 2)
    scale = jnp.sqrt(K_target / jnp.maximum(K_actual, 1e-12))
    velocities = velocities * scale

    return velocities


# =============================================================================
# Complete IC Generator
# =============================================================================


def generate_fractal_ic(
    key: PRNGKeyArray,
    N_stars: int,
    M_total: float,
    R_half: float,
    profile: str,
    frac_params: FractalDisplacementLayer,
    imf_params,
    field: FractalField = None,
    G: float = None,
):
    """Generate cluster IC with fractal displacement field.

    Parameters
    ----------
    key : PRNGKey
        JAX random key.
    N_stars : int
        Number of stars.
    M_total : float
        Total mass in M_sun.
    R_half : float
        Half-mass radius in pc.
    profile : str
        Density profile type: 'plummer', 'king', or 'eff'.
    frac_params : FractalDisplacementLayer
        FDF parameters (chi, lambda_frac, sigma_u, etc.).
    imf_params : IMF
        IMF instance with .sample(key, n) method.
    field : FractalField, optional
        Pre-initialized FractalField. If None, creates new one.
        For long-running inference loops, precompute and pass in a frozen
        FractalField to avoid reinitialization overhead.
    G : float, optional
        Gravitational constant. If None, uses jaxstro.units.STELLAR.G.

    Returns
    -------
    ClusterState
        Cluster with masses, positions, velocities.

    Notes
    -----
    **Inference pattern**: In long-running inference (HMC, NUTS), you may want
    to precompute a FractalField once and pass it in, rather than reinitializing
    each iteration.

    **Velocity assignment**: Uses O(N^2) potential energy calculation. Acceptable
    for IC generation but not for every N-body timestep.
    """
    from progenax.cluster.core import ClusterState
    from progenax.profiles import sample_density_profile

    if G is None:
        from jaxstro.units import STELLAR
        G = STELLAR.G

    # Split keys
    key_imf, key_pos, key_field, key_vel = random.split(key, 4)

    # Clamp lambda parameters to valid range [0, 1]
    lambda_frac = jnp.clip(frac_params.lambda_frac, 0.0, 1.0)
    lambda_vel = jnp.clip(frac_params.lambda_vel, 0.0, 1.0)

    # ─────────────────────────────────────────────────────────────
    # Step 1: Draw masses from IMF
    # ─────────────────────────────────────────────────────────────
    masses = imf_params.sample(key_imf, N_stars)
    masses = masses * (M_total / jnp.sum(masses))

    # ─────────────────────────────────────────────────────────────
    # Step 2: Sample smooth base profile
    # ─────────────────────────────────────────────────────────────
    positions_base = sample_density_profile(key_pos, N_stars, profile, R_half)
    radii_base = jnp.linalg.norm(positions_base, axis=1)

    # ─────────────────────────────────────────────────────────────
    # Step 3: Initialize and freeze displacement field (or use provided)
    # ─────────────────────────────────────────────────────────────
    if field is None:
        field = init_fractal_field(
            key_field,
            n_modes=frac_params.n_modes,
            R_half=R_half,
            k_min_factor=frac_params.k_min_factor,
            k_max_factor=frac_params.k_max_factor,
        )
    # Freeze stochastic structure (no gradients through wavevectors/phases)
    field = jax.tree_util.tree_map(jax.lax.stop_gradient, field)

    # ─────────────────────────────────────────────────────────────
    # Step 4: Compute amplitudes (differentiable in chi, sigma_u)
    # ─────────────────────────────────────────────────────────────
    sigma_u_physical = frac_params.sigma_u * R_half  # Convert to pc
    a_vecs = compute_amplitudes(field, frac_params.chi, sigma_u_physical)

    # ─────────────────────────────────────────────────────────────
    # Step 5: Evaluate and apply displacement
    # ─────────────────────────────────────────────────────────────
    displacements = evaluate_displacement(positions_base, field, a_vecs)

    positions = apply_displacement(
        positions_base,
        displacements,
        lambda_frac,  # Use clamped value
        target_radii=radii_base,
        mode=frac_params.radial_mode,
    )

    # Recenter positions to COM (finite-N realizations can drift)
    M_total_actual = jnp.sum(masses)
    x_com = jnp.sum(masses[:, None] * positions, axis=0) / M_total_actual
    positions = positions - x_com

    # ─────────────────────────────────────────────────────────────
    # Step 6: Assign velocities
    # ─────────────────────────────────────────────────────────────
    velocities = assign_fractal_velocities(
        key_vel,
        positions,
        masses,
        field,
        a_vecs,
        frac_params,
        G,
        lambda_vel=lambda_vel,  # Use clamped value
    )

    return ClusterState(
        masses=masses,
        positions=positions,
        velocities=velocities,
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Data structures
    "FractalField",
    "FractalDisplacementLayer",
    # Field operations
    "init_fractal_field",
    "compute_amplitudes",
    "evaluate_displacement",
    "apply_displacement",
    # Velocity
    "assign_fractal_velocities",
    # IC Generator
    "generate_fractal_ic",
    # Constants (v2 - lognormal envelope)
    "CHI_MIN",
    "CHI_MAX",
    "BETA_BASE",
    "SIGMA_LOGK",
    # Legacy constants (deprecated)
    "BETA_0",
    "BETA_1",
]

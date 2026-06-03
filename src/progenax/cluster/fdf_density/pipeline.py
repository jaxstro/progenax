"""FDF cluster IC pipeline + D-calibration factory (split from fdf_density.py)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import random
from jax import Array
from jaxtyping import Float, PRNGKeyArray

from progenax import defaults
from progenax.cluster.fdf_config import FDF_HEURISTICS, CHI_MIN, CHI_MAX

if TYPE_CHECKING:
    from progenax.cluster.fdf_config import GravoturbulentEnv, GravoturbulentResult
    from progenax.gravoturb.bm19_model import BM19Result

from .density_field import FractalDensityLayer, TailSubstructureLayer, DensityField3D
from .field_init import init_turbulent_density_field, init_bm19_density_field
from .sampling import sample_positions_from_density, sample_positions_tail


def _resolve_tail_st_and_mode(tail: TailSubstructureLayer):
    """Resolve (sampler mode, s_t) from a TailSubstructureLayer.

    BM19/gravoturbulent carry the tail threshold s_t directly; PN11 carries the
    critical log-density s_crit (routed through the BM19 ``s > s_t`` sampler with
    ``s_t = s_crit``); direct/cluster_type/D_mapping have no BM19 result and fall
    back to the legacy local-overdensity sampler.
    """
    mode = tail.mode
    s_t = None
    if mode in ("bm19", "gravoturbulent") and tail.result is not None:
        s_t = float(tail.result.s_t)
        if mode == "gravoturbulent":
            mode = "bm19"
    elif mode == "pn11" and tail.result is not None:
        s_t = float(tail.result.s_crit)
        mode = "bm19"
    elif mode == "direct" or mode in ("cluster_type", "D_mapping"):
        mode = "pn11_legacy"
    return mode, s_t


def generate_fractal_ic_density(
    key: PRNGKeyArray,
    N_stars: int,
    M_total: float,
    R_half: float,
    imf_params,
    layer: FractalDensityLayer,
    tail: TailSubstructureLayer | None = None,
    env: GravoturbulentEnv | None = None,
    G: float = None,
):
    """Generate cluster IC using density-field fractal method.

    This function creates initial conditions for a star cluster by:
    1. Generating a turbulent gas density field (controlled by FractalDensityLayer)
    2. Sampling star positions from that field

    The position sampling can use either:
    - Pure density sampling (default, when tail=None and env=None)
    - Gravoturbulent tail sampling (when tail or env is provided)

    The tail sampling preferentially places stars in the densest regions of
    the gas field, controlled by f_sub in TailSubstructureLayer.

    Parameters
    ----------
    key : PRNGKey
        JAX random key.
    N_stars : int
        Number of stars.
    M_total : float
        Total cluster mass in M_sun.
    R_half : float
        Half-mass radius in pc.
    imf_params
        IMF instance with .sample(key, n) method.
    layer : FractalDensityLayer
        Turbulence parameters for gas density field (σ_ln_ρ, β).
    tail : TailSubstructureLayer, optional
        Gravoturbulent substructure parameters. If provided, uses two-component
        dense tail + smooth sampling with f_sub controlling the split.
        If None (default), uses standard density-weighted sampling.
    env : GravoturbulentEnv, optional
        If provided, derives tail layer from gravoturbulent theory (Burkhart 2018).
        This OVERRIDES the `tail` parameter with a physics-derived f_sub.
        This is the RECOMMENDED interface when birth cloud properties are known.
    G : float, optional
        Gravitational constant. Uses progenax.DEFAULT_UNITS.G if None.

    Returns
    -------
    ClusterState
        Cluster with masses, positions, velocities.

    Notes
    -----
    The density field is constructed and frozen once via stop_gradient.
    Gradients flow through:
        - sigma_ln_rho (amplitude of density fluctuations)
        - lambda_frac (blend fraction) via blending
        - virial_ratio (velocity scaling)

    Gradients do NOT flow through:
        - Stochastic realization of the field (frozen structure)
        - Cell selection in position sampling (discrete)

    Examples
    --------
    >>> # Standard density sampling (legacy behavior)
    >>> cluster = generate_fractal_ic_density(key, N, M, R, imf, layer)
    >>>
    >>> # With gravoturbulent tail sampling (direct f_sub)
    >>> tail = TailSubstructureLayer(f_sub=0.5)  # YMC-like
    >>> cluster = generate_fractal_ic_density(key, N, M, R, imf, layer, tail=tail)
    >>>
    >>> # With physics-derived f_sub from environment (RECOMMENDED)
    >>> from progenax.cluster.fdf_config import GravoturbulentEnv
    >>> env = GravoturbulentEnv(Sigma=1000, Mach=20, eta_survive=0.85)
    >>> cluster = generate_fractal_ic_density(key, N, M, R, imf, layer, env=env)
    """
    # If env provided, derive tail layer from gravoturbulent theory
    if env is not None:
        from progenax.cluster.fdf_config import tail_layer_from_env

        tail = tail_layer_from_env(env)
    from progenax.cluster.core import ClusterState
    from progenax.dynamics.virial import compute_potential_energy

    if G is None:
        G = defaults.DEFAULT_UNITS.G

    # Split keys
    key_imf, key_field, key_pos, key_vel = random.split(key, 4)

    # Step 1: Sample masses from IMF
    masses = imf_params.sample(key_imf, N_stars)
    masses = masses * (M_total / jnp.sum(masses))

    # Step 2: Initialize and freeze density field
    field = init_turbulent_density_field(key_field, R_half, layer)
    field = jax.tree_util.tree_map(jax.lax.stop_gradient, field)

    # Step 3: Sample positions from density field (tail sampling if a layer is given)
    if tail is not None:
        tail_mode, tail_s_t = _resolve_tail_st_and_mode(tail)
        positions = sample_positions_tail(
            key_pos, field, N_stars, tail.f_sub, mode=tail_mode, s_t=tail_s_t
        )
    else:
        positions = sample_positions_from_density(key_pos, field, N_stars)

    # Step 4: Recenter to center of mass
    M_total_actual = jnp.sum(masses)
    x_com = jnp.sum(masses[:, None] * positions, axis=0) / M_total_actual
    positions = positions - x_com

    # Step 5: Sample velocities from equilibrium DF and rescale to virial ratio
    # Use Plummer velocities as baseline (similar to displacement FDF)
    from progenax.kinematics import PlummerVelocityDF

    df = PlummerVelocityDF(r_h=R_half)
    velocities = df.sample_velocities(positions, masses, key_vel, G=G)

    # Remove COM velocity
    v_com = jnp.sum(masses[:, None] * velocities, axis=0) / M_total_actual
    velocities = velocities - v_com

    # Rescale to target virial ratio
    U = compute_potential_energy(positions, masses, G)
    K_actual = 0.5 * jnp.sum(masses[:, None] * velocities**2)
    K_target = layer.virial_ratio * jnp.abs(U)

    scale = jnp.sqrt(K_target / jnp.maximum(K_actual, 1e-12))
    velocities = velocities * scale

    return ClusterState(
        masses=masses,
        positions=positions,
        velocities=velocities,
    )


# =============================================================================
# Calibration Helper
# =============================================================================


def density_layer_from_D(
    D: float,
    sigma_ln_rho: float,
    lambda_frac: float = 1.0,
    virial_ratio: float = 0.5,
    grid_size: int = 64,
    base_profile: str = "uniform",
) -> FractalDensityLayer:
    """Create FractalDensityLayer from GW-style D parameter.

    UNCALIBRATED: Use env_to_fdf_layer() for physics-based parameters.

    Parameters
    ----------
    D : float
        Target fractal dimension in [1.6, 3.0].
    sigma_ln_rho : float
        Amplitude of log-density fluctuations. REQUIRED - no default.
        Use env_to_fdf_layer() for physics-derived values (~1.1-1.5).
    lambda_frac : float, default 1.0
        Blend fraction [0, 1].
    virial_ratio : float, default 0.5
        Target Q_vir.
    grid_size : int, default 64
        Grid resolution per dimension.
    base_profile : str, default "uniform"
        Base density profile: "uniform" or "plummer".

    Returns
    -------
    FractalDensityLayer
        Configured layer (chi ≈ D, uncalibrated).
    """

    chi = jnp.clip(D, CHI_MIN, CHI_MAX)

    return FractalDensityLayer(
        chi=float(chi),  # Convert to float for dataclass
        sigma_ln_rho=sigma_ln_rho,
        lambda_frac=lambda_frac,
        base_profile=base_profile,
        virial_ratio=virial_ratio,
        grid_size=grid_size,
    )


# =============================================================================
# Module Exports
# =============================================================================


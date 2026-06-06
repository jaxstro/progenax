# progenax/src/progenax/cluster/validation.py
"""
Validation functions for cluster IC generation.

Provides sweep functions for parameter studies and differentiability checks
for mass segregation and virial ratio.

This module contains:
- sweep_mass_segregation_lambda: Sweep λ_seg and compute Λ_MSR, radial profiles
- measure_virial_ratio: Check virial ratio accuracy
- mean_radius_of_massive_jax: JAX-only summary for gradients
- grad_mean_radius_wrt_lambda_seg: Gradient sanity check
- recover_lambda_seg_via_gradient_descent: Toy inverse problem

References:
    Allison et al. (2009), ApJ 700, L99 - Λ_MSR diagnostic
    Cartwright & Whitworth (2004), MNRAS 348, 589 - Q parameter
    Küpper et al. (2011), MNRAS 417, 2300 - σΣ/⟨Σ⟩ relation
"""

from typing import Dict, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from progenax import defaults
from progenax.cluster import (
    ClusterState,
    MassSegregationLayer,
    SpatialStructureParams,
    generate_cluster_ic,
)
from progenax.diagnostics import compute_lambda_msr
from progenax.imf import PowerLawIMF


# =============================================================================
# Mass Segregation Sweep
# =============================================================================


def sweep_mass_segregation_lambda(
    key: jax.random.PRNGKey,
    lambda_values: Sequence[float],
    N_stars: int = 5000,
    M_total: float = 1000.0,
    R_half: float = 1.0,
    n_realizations: int = 20,
    N_massive: int = 10,
) -> Dict:
    """
    Sweep λ_seg and compute Λ_MSR, mean massive radius, and radial profiles.

    For each λ_seg value, generates n_realizations clusters and computes
    summary statistics. Also stores radial histograms for λ=0 and λ=1.

    Args:
        key: JAX random key
        lambda_values: Sequence of λ_seg values to sweep
        N_stars: Number of stars per cluster
        M_total: Total cluster mass [Msun]
        R_half: Half-mass radius [pc]
        n_realizations: Number of realizations per λ_seg
        N_massive: Number of most massive stars for Λ_MSR and ⟨r⟩

    Returns:
        Dictionary with:
        - 'lambda_values': Array of λ_seg values
        - 'lambda_msr_mean': Mean Λ_MSR per λ_seg
        - 'lambda_msr_std': Std Λ_MSR per λ_seg
        - 'r_massive_mean': Mean radius of top N_massive stars per λ_seg
        - 'r_massive_std': Std of mean radius per λ_seg
        - 'radial_hist_ref': Radial histogram for λ=0
        - 'radial_hist_seg': Radial histogram for λ=1
        - 'radial_bins': Bin edges for radial histograms
    """
    imf = PowerLawIMF.kroupa()
    lambda_values = np.array(lambda_values)

    lambda_msr_all = []
    r_massive_all = []

    radial_bins = np.linspace(0, 5 * R_half, 50)
    radial_hist_ref = None
    radial_hist_seg = None

    for lambda_seg in lambda_values:
        msr_realizations = []
        r_massive_realizations = []

        for i in range(n_realizations):
            key, subkey = jax.random.split(key)

            # Generate cluster
            if lambda_seg == 0.0:
                structure_params = SpatialStructureParams(base_profile="plummer")
            else:
                structure_params = SpatialStructureParams(
                    base_profile="plummer",
                    mass_segregation=MassSegregationLayer(lambda_seg=float(lambda_seg)),
                )

            cluster = generate_cluster_ic(
                key=subkey,
                N_stars=N_stars,
                M_total=M_total,
                R_half=R_half,
                imf_params=imf,
                structure_params=structure_params,
            )

            # Convert to NumPy for diagnostics
            positions_np = np.array(cluster.positions)
            masses_np = np.array(cluster.masses)

            # Compute Λ_MSR
            lam_msr, _ = compute_lambda_msr(
                positions_np, masses_np, N_massive=N_massive, N_random_samples=50
            )
            msr_realizations.append(lam_msr)

            # Compute mean radius of massive stars
            radii = np.linalg.norm(positions_np, axis=1)
            massive_idx = np.argsort(-masses_np)[:N_massive]
            r_massive = np.mean(radii[massive_idx])
            r_massive_realizations.append(r_massive)

            # Store radial histograms for first realization
            if i == 0:
                hist, _ = np.histogram(radii, bins=radial_bins, density=True)
                if lambda_seg == 0.0:
                    radial_hist_ref = hist
                elif lambda_seg == 1.0:
                    radial_hist_seg = hist

        lambda_msr_all.append(msr_realizations)
        r_massive_all.append(r_massive_realizations)

    # Compute statistics
    lambda_msr_all = np.array(lambda_msr_all)
    r_massive_all = np.array(r_massive_all)

    return {
        "lambda_values": lambda_values,
        "lambda_msr_mean": np.mean(lambda_msr_all, axis=1),
        "lambda_msr_std": np.std(lambda_msr_all, axis=1),
        "r_massive_mean": np.mean(r_massive_all, axis=1),
        "r_massive_std": np.std(r_massive_all, axis=1),
        "radial_hist_ref": radial_hist_ref,
        "radial_hist_seg": radial_hist_seg,
        "radial_bins": radial_bins,
    }


# =============================================================================
# Virial Ratio Check
# =============================================================================


def measure_virial_ratio(
    key: jax.random.PRNGKey,
    target_Q_vir: float,
    structure_params: SpatialStructureParams,
    N_stars: int = 5000,
    M_total: float = 1000.0,
    R_half: float = 1.0,
    G: Optional[float] = None,
    softening: float = 1e-4,
) -> float:
    """
    Generate one cluster and return actual Q_vir = K/|U|.

    Args:
        key: JAX random key
        target_Q_vir: Target virial ratio (used to set fractal velocities)
        structure_params: Spatial structure parameters
        N_stars: Number of stars
        M_total: Total mass [Msun]
        R_half: Half-mass radius [pc]
        G: Gravitational constant (default: progenax.DEFAULT_UNITS.G)
        softening: Softening length for potential [pc]

    Returns:
        Measured Q_vir = K/|U|
    """
    if G is None:
        G = defaults.DEFAULT_UNITS.G

    imf = PowerLawIMF.kroupa()

    cluster = generate_cluster_ic(
        key=key,
        N_stars=N_stars,
        M_total=M_total,
        R_half=R_half,
        imf_params=imf,
        structure_params=structure_params,
    )

    masses = cluster.masses
    positions = cluster.positions
    velocities = cluster.velocities

    # Kinetic energy
    v2 = jnp.sum(velocities**2, axis=1)
    K = 0.5 * jnp.sum(masses * v2)

    # Potential energy (softened pairwise)
    from progenax.dynamics.virial import compute_potential_energy
    U = compute_potential_energy(positions, masses, G=G, softening=softening)

    Q_vir = float(K / jnp.abs(U))
    return Q_vir


# =============================================================================
# JAX-Only Differentiable Summary
# =============================================================================


def mean_radius_of_massive_jax(
    key: jax.random.PRNGKey,
    lambda_seg: float,
    N_stars: int = 2000,
    N_massive: int = 20,
    M_total: float = 1000.0,
    R_half: float = 1.0,
) -> Array:
    """
    JAX-only summary: mean radius of top N_massive stars as function of λ_seg.

    This function is fully differentiable w.r.t. lambda_seg. It avoids
    NumPy/SciPy and uses only JAX operations.

    Args:
        key: JAX random key
        lambda_seg: Mass segregation parameter [0, 1]
        N_stars: Number of stars
        N_massive: Number of massive stars to average
        M_total: Total mass [Msun]
        R_half: Half-mass radius [pc]

    Returns:
        Mean radius of top N_massive stars (JAX array scalar)
    """
    imf = PowerLawIMF.kroupa()

    if lambda_seg == 0.0:
        structure_params = SpatialStructureParams(base_profile="plummer")
    else:
        structure_params = SpatialStructureParams(
            base_profile="plummer",
            mass_segregation=MassSegregationLayer(lambda_seg=lambda_seg),
        )

    cluster = generate_cluster_ic(
        key=key,
        N_stars=N_stars,
        M_total=M_total,
        R_half=R_half,
        imf_params=imf,
        structure_params=structure_params,
    )

    masses = cluster.masses
    positions = cluster.positions

    # JAX-only: sort by mass (descending), take top N_massive
    mass_order = jnp.argsort(-masses)
    top_indices = mass_order[:N_massive]

    # Compute radii
    radii = jnp.linalg.norm(positions, axis=1)
    r_massive = radii[top_indices]

    return jnp.mean(r_massive)


def grad_mean_radius_wrt_lambda_seg(
    key: jax.random.PRNGKey,
    lambda_seg: float,
    N_stars: int = 2000,
) -> float:
    """
    Return ∂⟨r_massive⟩/∂λ_seg as a JAX scalar.

    Uses jax.grad on mean_radius_of_massive_jax.

    Args:
        key: JAX random key
        lambda_seg: Mass segregation parameter
        N_stars: Number of stars

    Returns:
        Gradient of mean massive radius w.r.t. lambda_seg
    """
    def summary_fn(lam):
        return mean_radius_of_massive_jax(key, lam, N_stars=N_stars)

    grad_fn = jax.grad(summary_fn)
    return float(grad_fn(lambda_seg))


# =============================================================================
# Gradient-Based Parameter Recovery
# =============================================================================


def recover_lambda_seg_via_gradient_descent(
    key: jax.random.PRNGKey,
    lambda_true: float = 0.7,
    n_steps: int = 20,
    step_size: float = 0.5,
    N_stars: int = 2000,
) -> Dict:
    """
    Toy inverse problem: recover λ_seg from a summary statistic.

    Demonstrates that the IC generator is usable inside AD-based inference.

    Algorithm:
        1. Generate a 'target' cluster with lambda_true and compute summary S_true
        2. Define loss(λ) = (S(λ) - S_true)²
        3. Run gradient descent on λ starting from initial guess
        4. Return optimization trajectory

    Args:
        key: JAX random key
        lambda_true: True λ_seg to recover
        n_steps: Number of gradient descent steps
        step_size: Learning rate
        N_stars: Number of stars per cluster

    Returns:
        Dictionary with:
        - 'lambda_history': Array of λ_seg values over iterations
        - 'loss_history': Array of loss values over iterations
        - 'lambda_true': True value
        - 'lambda_final': Final recovered value
    """
    key, target_key, opt_key = jax.random.split(key, 3)

    # Generate target summary
    S_target = mean_radius_of_massive_jax(target_key, lambda_true, N_stars=N_stars)
    S_target = float(S_target)

    # Define loss function
    def loss_fn(lam, step_key):
        S_model = mean_radius_of_massive_jax(step_key, lam, N_stars=N_stars)
        return (S_model - S_target) ** 2

    # Gradient descent
    lambda_history = [0.1]  # Initial guess
    loss_history = []

    lam = 0.1
    for step in range(n_steps):
        opt_key, step_key = jax.random.split(opt_key)

        # Compute loss and gradient
        loss_val = float(loss_fn(lam, step_key))
        loss_history.append(loss_val)

        grad_val = float(jax.grad(lambda l: loss_fn(l, step_key))(lam))

        # Update with clamping to [0, 1]
        lam = lam - step_size * grad_val
        lam = float(jnp.clip(lam, 0.0, 1.0))
        lambda_history.append(lam)

    return {
        "lambda_history": np.array(lambda_history),
        "loss_history": np.array(loss_history),
        "lambda_true": lambda_true,
        "lambda_final": lambda_history[-1],
        "S_target": S_target,
    }


# =============================================================================
# Cluster Generation Helpers for Plotting
# =============================================================================


def generate_cluster_for_plot(
    key: jax.random.PRNGKey,
    lambda_seg: Optional[float] = None,
    Q_vir: float = 0.5,
    N_stars: int = 5000,
    M_total: float = 1000.0,
    R_half: float = 1.0,
) -> ClusterState:
    """
    Generate a single cluster for plotting purposes.

    Args:
        key: JAX random key
        lambda_seg: Mass segregation parameter (None for no segregation)
        Q_vir: Target virial ratio
        N_stars: Number of stars
        M_total: Total mass [Msun]
        R_half: Half-mass radius [pc]

    Returns:
        ClusterState with generated IC
    """
    imf = PowerLawIMF.kroupa()

    if lambda_seg is not None and lambda_seg > 0:
        structure_params = SpatialStructureParams(
            base_profile="plummer",
            mass_segregation=MassSegregationLayer(lambda_seg=lambda_seg),
        )
    else:
        structure_params = SpatialStructureParams(base_profile="plummer")

    return generate_cluster_ic(
        key=key,
        N_stars=N_stars,
        M_total=M_total,
        R_half=R_half,
        imf_params=imf,
        structure_params=structure_params,
    )


__all__ = [
    "sweep_mass_segregation_lambda",
    "measure_virial_ratio",
    "mean_radius_of_massive_jax",
    "grad_mean_radius_wrt_lambda_seg",
    "recover_lambda_seg_via_gradient_descent",
    "generate_cluster_for_plot",
]

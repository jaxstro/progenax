# progenax/src/progenax/cluster/fdf_calibration.py
"""
FDF Calibration: Mapping between GW fractal dimension D and FDF parameters.

This module provides calibration data and helpers for converting between
the user-facing fractal dimension D (Goodwin-Whitworth convention) and
the internal FDF parameters (chi, sigma_u).

v2 Implementation:
    - chi controls the peak position of a lognormal envelope in k-space
    - Lower chi (≈1.6) → peak at high k → small-scale dominated → clumpy
    - Higher chi (≈3.0) → peak at low k → large-scale dominated → smooth
    - Displacement vectors are parallel to wavevectors (irrotational field)
    - This creates actual density perturbations via div(u) ≠ 0

Current stub calibration uses chi ~ D (identity mapping). After v2 spectrum
validation, run calibration sweeps to measure Q(D) and update mappings.

The calibration maps D -> (chi, sigma_u) such that FDF-generated clusters
statistically match GW-generated clusters in Q_CW and sigma_Sigma/Sigma.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float


@dataclass(frozen=True)
class FDFCalibration:
    """Calibration mapping between GW D and FDF parameters.

    Attributes
    ----------
    D_values : Array
        Reference GW fractal dimension values.
    chi_values : Array
        Corresponding chi (clumpiness) values.
    sigma_u_values : Array
        Corresponding sigma_u / R_half values.
    version : str
        Calibration provenance tag. ``"v1_stub_uncalibrated"`` marks the
        placeholder chi~D identity mapping (not yet fit to Goodwin-Whitworth).
    """

    D_values: Float[Array, "K"]
    chi_values: Float[Array, "K"]
    sigma_u_values: Float[Array, "K"]
    version: str = "v1_stub_uncalibrated"

    def chi_from_D(self, D: float) -> float:
        """Interpolate chi from target D.

        Parameters
        ----------
        D : float
            Target fractal dimension in [1.6, 3.0].

        Returns
        -------
        chi : float
            Interpolated clumpiness parameter.
        """
        D_clamped = jnp.clip(D, self.D_values[0], self.D_values[-1])
        return jnp.interp(D_clamped, self.D_values, self.chi_values)

    def sigma_u_from_D(self, D: float) -> float:
        """Interpolate sigma_u from target D.

        Parameters
        ----------
        D : float
            Target fractal dimension in [1.6, 3.0].

        Returns
        -------
        sigma_u : float
            Interpolated displacement amplitude scale.
        """
        D_clamped = jnp.clip(D, self.D_values[0], self.D_values[-1])
        return jnp.interp(D_clamped, self.D_values, self.sigma_u_values)


# =============================================================================
# v2 Stub Calibration Data
# =============================================================================
# This is a placeholder calibration. In production, this will be replaced
# with empirically calibrated values from offline calibration runs.
#
# v2 SPECTRUM CHANGE (2024-12):
# - Displacement vectors now parallel to wavevectors (irrotational field)
# - Amplitude spectrum uses χ-dependent lognormal envelope in k-space
# - Peak moves from high-k (clumpy, D≈1.6) to low-k (smooth, D≈3.0)
# - This creates actual density perturbations via div(u) ≠ 0
#
# Expected behavior after v2 spectrum:
# - Q_CW should monotonically increase with D (clumpy → smooth)
# - Visual progression should be obvious in hero plots
# - chi ~ D mapping may need adjustment after calibration sweep
#
# TODO(post-v2): Run calibration sweep with validate_fdf.py --mode calib
# to measure actual Q(D) curve and update these stub values.

_V1_CALIBRATION = FDFCalibration(
    D_values=jnp.array([1.6, 2.0, 2.4, 2.8, 3.0]),
    chi_values=jnp.array([1.6, 2.0, 2.4, 2.8, 3.0]),  # Identity for v2 stub
    # sigma_u values: larger for clumpy (D=1.6), smaller for smooth (D=3.0)
    # These may need adjustment after v2 spectrum calibration
    sigma_u_values=jnp.array([0.80, 0.55, 0.35, 0.20, 0.05]),
)


def load_fdf_calibration() -> FDFCalibration:
    """Load FDF calibration data.

    Returns
    -------
    FDFCalibration
        Calibration mapping D -> (chi, sigma_u).

    Notes
    -----
    v1 returns a stub calibration with chi ~ D. Emits a ``UserWarning`` (M9) so
    that stub-driven cluster statistics are not mistaken for calibrated results.
    Production calibration will be loaded from progenax.data in future.
    """
    import warnings

    warnings.warn(
        f"FDF calibration '{_V1_CALIBRATION.version}' is an uncalibrated stub "
        "(chi~D identity mapping): FDF-generated cluster statistics (Q_CW, "
        "sigma_Sigma/Sigma) are NOT yet calibrated to Goodwin-Whitworth. Treat "
        "fractal-substructure ICs as qualitative until the calibration sweep "
        "replaces the v1 stub (see progenax/cluster/fdf_calibration.py).",
        UserWarning,
        stacklevel=2,
    )
    return _V1_CALIBRATION


# =============================================================================
# User-Facing API
# =============================================================================


def fractal_layer_from_D(
    D: float,
    virial_ratio: float = 0.5,
    coherent_velocities: bool = True,
    lambda_frac: float = 1.0,
    lambda_vel: float = 0.3,
):
    """Create FractalDisplacementLayer from GW-style D parameter.

    This is the recommended user-facing API for specifying fractal structure
    using the familiar Goodwin-Whitworth fractal dimension convention.

    Parameters
    ----------
    D : float
        Target fractal dimension in [1.6, 3.0] (GW convention).
        D=1.6: highly clumpy
        D=3.0: nearly smooth (homogeneous sphere)
    virial_ratio : float, default 0.5
        Target Q_vir = K/|U|. Use Q_vir < 0.5 for subvirial (collapsing) systems.
    coherent_velocities : bool, default True
        Whether to correlate velocities with displacement structure.
    lambda_frac : float, default 1.0
        Blend fraction (0 = smooth, 1 = full fractal).
    lambda_vel : float, default 0.3
        Velocity coherence strength.

    Returns
    -------
    FractalDisplacementLayer
        Configured layer with calibrated chi and sigma_u.

    Examples
    --------
    >>> from progenax.cluster.fdf_calibration import fractal_layer_from_D
    >>> from progenax.cluster.fdf import generate_fractal_ic
    >>> from progenax.imf import PowerLawIMF
    >>> import jax
    >>>
    >>> key = jax.random.PRNGKey(42)
    >>> imf = PowerLawIMF.kroupa()
    >>> frac = fractal_layer_from_D(D=1.6, virial_ratio=0.3)
    >>> cluster = generate_fractal_ic(
    ...     key, N_stars=1000, M_total=500.0, R_half=0.5,
    ...     profile="plummer", frac_params=frac, imf_params=imf
    ... )
    """
    from progenax.cluster.fdf import FractalDisplacementLayer

    calibration = load_fdf_calibration()

    chi = calibration.chi_from_D(D)
    sigma_u = calibration.sigma_u_from_D(D)

    # Keep as JAX arrays for differentiability (no float() calls!)
    return FractalDisplacementLayer(
        chi=chi,
        lambda_frac=lambda_frac,
        sigma_u=sigma_u,
        virial_ratio=virial_ratio,
        coherent_velocities=coherent_velocities,
        lambda_vel=lambda_vel,
    )

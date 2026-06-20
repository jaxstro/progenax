"""Initial Mass Functions (IMFs) for stellar population synthesis."""

from .base import BaseIMF, IMFProtocol, _ppf_newton
from .binary import (
    BinaryIMF,
    ConstantBinaryFraction,
    FlatMassRatio,
    MassDependentBinaryFraction,
    MassRatioProtocol,
    MoeDiStefano2017,
    MoeDiStefano2017Full,
    MoeJointOrbit,
    MoePeriod,
    PowerLawMassRatio,
    TwinPeakedMassRatio,
)
from .chabrier import ChabrierIMF
from .differentiable import (
    individual_mass_nll,
    log_prob_masses,
    sample_masses_from_params,
)

# Paper-calibrated environment-dependent IMF (v0.3)
from .environment import (
    DEFAULT_SFE,
    # Coefficients
    JERABKOVA_COEFFICIENTS,
    MARKS_COEFFICIENTS,
    MARKS_TABLE3_COEFFICIENTS,
    # Data class
    BirthEnvironment,
    # α₃ functions
    alpha3_jerabkova_generalized,
    alpha3_jerabkova_mecl,
    alpha3_jerabkova_rho,
    alpha3_marks_plane,
    alpha3_marks_table3,
    compute_log_rho_cl_6,
    # Density functions
    compute_r_half,
    compute_rho_cl,
    compute_rho_ecl,
    # Main API
    env_to_imf_params,
    # Low-mass slopes
    lowmass_slopes_metallicity,
    x_hat_marks_plane,
    # x functions
    x_jerabkova_generalized,
    x_jerabkova_rho,
)
from .params import IMFParams
from .power_law import (
    PowerLawIMF,
    estimate_N_max_for_M_total,
    estimate_pool_size,
    prepare_imf_samples,
)
from .smooth import Maschberger, Schechter, TaperedPowerLaw
from .truncated import TruncatedIMF

__all__ = [
    # Base classes
    "IMFProtocol",
    "BaseIMF",
    "_ppf_newton",
    "TruncatedIMF",
    # Power-law IMFs
    "PowerLawIMF",
    "prepare_imf_samples",
    "estimate_N_max_for_M_total",
    "estimate_pool_size",
    # Smooth IMFs
    "Maschberger",
    "TaperedPowerLaw",
    "Schechter",
    "ChabrierIMF",
    # Binary IMFs
    "MassRatioProtocol",
    "FlatMassRatio",
    "PowerLawMassRatio",
    "TwinPeakedMassRatio",
    "MoeDiStefano2017",
    "MoeDiStefano2017Full",
    "MoePeriod",
    "MoeJointOrbit",
    "ConstantBinaryFraction",
    "MassDependentBinaryFraction",
    "BinaryIMF",
    # Differentiable IMF inference
    "IMFParams",
    "log_prob_masses",
    "sample_masses_from_params",
    "individual_mass_nll",
    # Paper-calibrated environment-dependent IMF (v0.3)
    "BirthEnvironment",
    "env_to_imf_params",
    # Density functions
    "compute_r_half",
    "compute_rho_ecl",
    "compute_rho_cl",
    "compute_log_rho_cl_6",
    # x functions
    "x_jerabkova_generalized",
    "x_jerabkova_rho",
    "x_hat_marks_plane",
    # α₃ functions
    "alpha3_jerabkova_generalized",
    "alpha3_jerabkova_mecl",
    "alpha3_jerabkova_rho",
    "alpha3_marks_plane",
    "alpha3_marks_table3",
    # Low-mass slopes
    "lowmass_slopes_metallicity",
    # Coefficients
    "JERABKOVA_COEFFICIENTS",
    "MARKS_COEFFICIENTS",
    "MARKS_TABLE3_COEFFICIENTS",
    "DEFAULT_SFE",
]

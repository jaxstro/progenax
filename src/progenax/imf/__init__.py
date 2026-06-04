"""Initial Mass Functions (IMFs) for stellar population synthesis."""

from .base import IMFProtocol, BaseIMF, _ppf_newton
from .truncated import TruncatedIMF
from .power_law import (
    PowerLawIMF,
    prepare_imf_samples,
    estimate_N_max_for_M_total,
    estimate_pool_size,
)
from .smooth import Maschberger, TaperedPowerLaw, Schechter
from .chabrier import ChabrierIMF
from .binary import (
    MassRatioProtocol,
    FlatMassRatio,
    PowerLawMassRatio,
    TwinPeakedMassRatio,
    MoeDiStefano2017,
    MoeDiStefano2017Full,
    MoePeriod,
    MoeJointOrbit,
    ConstantBinaryFraction,
    MassDependentBinaryFraction,
    BinaryIMF,
)
from .params import IMFParams
from .differentiable import (
    log_prob_masses,
    sample_masses_from_params,
    individual_mass_nll,
)
# Paper-calibrated environment-dependent IMF (v0.3)
from .environment import (
    # Data class
    BirthEnvironment,
    # Main API
    env_to_imf_params,
    # Density functions
    compute_r_half,
    compute_rho_ecl,
    compute_rho_cl,
    compute_log_rho_cl_6,
    # x functions
    x_jerabkova_generalized,
    x_jerabkova_rho,
    x_hat_marks_plane,
    # α₃ functions
    alpha3_jerabkova_generalized,
    alpha3_jerabkova_mecl,
    alpha3_jerabkova_rho,
    alpha3_marks_plane,
    alpha3_marks_table3,
    # Low-mass slopes
    lowmass_slopes_metallicity,
    # Coefficients
    JERABKOVA_COEFFICIENTS,
    MARKS_COEFFICIENTS,
    MARKS_TABLE3_COEFFICIENTS,
    DEFAULT_SFE,
)

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

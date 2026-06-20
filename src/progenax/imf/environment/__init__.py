"""Environment-dependent IMF: Marks+2012 and Jerabkova+2018 prescriptions.

Split out of the former monolithic ``environment.py`` (500-LOC file limit). The
public API is unchanged — every symbol below remains importable from
``progenax.imf`` and ``progenax.imf.environment``.
"""

from .birth_environment import BirthEnvironment
from .coefficients import (
    DEFAULT_SFE,
    JERABKOVA_COEFFICIENTS,
    MARKS_COEFFICIENTS,
    MARKS_TABLE3_COEFFICIENTS,
)
from .density import (
    compute_log_rho_cl_6,
    compute_r_half,
    compute_rho_cl,
    compute_rho_ecl,
)
from .mapping import (
    alpha3_jerabkova_generalized,
    alpha3_jerabkova_mecl,
    alpha3_jerabkova_rho,
    alpha3_marks_plane,
    alpha3_marks_table3,
    env_to_imf_params,
    lowmass_slopes_metallicity,
    x_hat_marks_plane,
    x_jerabkova_generalized,
    x_jerabkova_rho,
)

__all__ = [
    "BirthEnvironment",
    "env_to_imf_params",
    "compute_r_half",
    "compute_rho_ecl",
    "compute_rho_cl",
    "compute_log_rho_cl_6",
    "x_jerabkova_generalized",
    "x_jerabkova_rho",
    "x_hat_marks_plane",
    "alpha3_jerabkova_generalized",
    "alpha3_jerabkova_mecl",
    "alpha3_jerabkova_rho",
    "alpha3_marks_plane",
    "alpha3_marks_table3",
    "lowmass_slopes_metallicity",
    "JERABKOVA_COEFFICIENTS",
    "MARKS_COEFFICIENTS",
    "MARKS_TABLE3_COEFFICIENTS",
    "DEFAULT_SFE",
]

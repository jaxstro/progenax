"""Initial Mass Functions (IMFs) for stellar population synthesis."""

from .base import BaseIMF, _ppf_newton
from .truncated import TruncatedIMF
from .power_law import (
    PowerLawIMF,
    prepare_imf_samples,
    estimate_N_max_for_M_total,
    estimate_pool_size,
)
from .smooth import Maschberger, TaperedPowerLaw, Schechter
from .chabrier import ChabrierIMF
from .environment import (
    GasEnvironment,
    jeans_mass,
    characteristic_mass_from_jeans,
    alpha_bounded,
    bonnor_ebert_mass,
    alpha_marks2012,
    alpha_jerabkova2018,
    alpha_from_sfr,
    EnvironmentIMF,
    CustomEnvironmentIMF,
    is_top_heavy,
    massive_star_fraction,
)

__all__ = [
    "BaseIMF",
    "_ppf_newton",
    "TruncatedIMF",
    "PowerLawIMF",
    "prepare_imf_samples",
    "estimate_N_max_for_M_total",
    "estimate_pool_size",
    "Maschberger",
    "TaperedPowerLaw",
    "Schechter",
    "ChabrierIMF",
    # Environment-conditioned IMFs
    "GasEnvironment",
    "jeans_mass",
    "characteristic_mass_from_jeans",
    "alpha_bounded",
    "bonnor_ebert_mass",
    "alpha_marks2012",
    "alpha_jerabkova2018",
    "alpha_from_sfr",
    "EnvironmentIMF",
    "CustomEnvironmentIMF",
    "is_top_heavy",
    "massive_star_fraction",
]

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
]

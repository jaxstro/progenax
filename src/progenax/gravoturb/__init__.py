"""Gravoturbulent cloud → cluster physics for Progenax.

Implements differentiable gravoturbulent star formation theory:
- BM19: Piecewise lognormal+powerlaw PDF, dense gas fraction (Burkhart & Mocz 2019)
- PN11: Classical critical density framework (Padoan-Nordlund 2011)
- BM19 PDF: CDF remap sampling for 3D density fields
- PP20: Magnification factor for SFR boost (Parmentier & Pfalzner 2020)

Choosing a Model
----------------
**BM19** (recommended default):
    Fewer parameters, powerlaw tail included, s_t derived from physics.

**PN11** (alternative):
    When you need magnetic support (phi_x) or surface density dependence.

Example::

    # BM19 (default)
    from progenax.gravoturb import bm19_pipeline
    result = bm19_pipeline(mach=10.0, alpha=2.0)

    # PN11 (when you have surface density)
    from progenax.gravoturb import pn11_pipeline
    result = pn11_pipeline(mach=10.0, Sigma=100.0)

References
----------
- Burkhart, B. & Mocz, P. 2019, ApJ, 879, 129 (BM19)
- Parmentier, G. & Pfalzner, S. 2020, ApJ, 903, 56 (PP20)
- Federrath, C. & Klessen, R. S. 2012, ApJ, 761, 156 (FK12)
- Padoan, P. & Nordlund, A. 2011, ApJ, 730, 40 (PN11)
"""

# BM19 model (gravoturbulent PDF framework) - DEFAULT
from progenax.gravoturb.bm19_model import (
    BM19Result,
    sigma_s_squared,
    transition_density,
    f_dense_lognormal_limit,
    f_dense_bm19_full,
    power_spectrum_slope,
    bm19_pipeline,
)

# PN11 model (classical critical density framework) - ALTERNATIVE
from progenax.gravoturb.pn11_model import (
    PN11Result,
    s_crit_pn11,
    f_dense_pn11,
    alpha_vir_from_sigma,
    pn11_pipeline,
)

# BM19 PDF sampling (CDF remap for 3D fields)
from progenax.gravoturb.bm19_pdf import (
    bm19_volume_pdf,
    build_bm19_cdf_table,
    bm19_icdf,
    gaussian_to_bm19,
    validate_bm19_field,
)

# PP20 magnification (SFR boost from density structure)
from progenax.gravoturb.pp20_magnification import (
    magnification_factor,
    magnification_factor_with_core,
    zeta_fdf_direct,
    sfr_per_dense_gas,
)

__all__ = [
    # BM19 model (default)
    "BM19Result",
    "sigma_s_squared",
    "transition_density",
    "f_dense_lognormal_limit",
    "f_dense_bm19_full",
    "power_spectrum_slope",
    "bm19_pipeline",
    # PN11 model (alternative)
    "PN11Result",
    "s_crit_pn11",
    "f_dense_pn11",
    "alpha_vir_from_sigma",
    "pn11_pipeline",
    # BM19 PDF sampling
    "bm19_volume_pdf",
    "build_bm19_cdf_table",
    "bm19_icdf",
    "gaussian_to_bm19",
    "validate_bm19_field",
    # PP20 magnification
    "magnification_factor",
    "magnification_factor_with_core",
    "zeta_fdf_direct",
    "sfr_per_dense_gas",
]

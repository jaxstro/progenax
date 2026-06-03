"""Binary star mass functions: mass-ratio distributions, binary-fraction models,
and the composite ``BinaryIMF``.

This subpackage was split out of the former monolithic ``binary.py`` to satisfy
the 500-LOC file limit. The public API is unchanged — every symbol below remains
importable from ``progenax.imf`` and ``progenax.imf.binary``.
"""

from .mass_ratio import (
    MassRatioProtocol,
    FlatMassRatio,
    PowerLawMassRatio,
    TwinPeakedMassRatio,
)
from .moe_di_stefano import MoeDiStefano2017
from .binary_fraction import (
    ConstantBinaryFraction,
    MassDependentBinaryFraction,
)
from .imf import (
    BinaryIMF,
    BinaryFractionCallable,
    MassRatioSamplerCallable,
)

__all__ = [
    "MassRatioProtocol",
    "FlatMassRatio",
    "PowerLawMassRatio",
    "TwinPeakedMassRatio",
    "MoeDiStefano2017",
    "ConstantBinaryFraction",
    "MassDependentBinaryFraction",
    "BinaryIMF",
]

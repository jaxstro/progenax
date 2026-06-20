"""Binary star mass functions: mass-ratio distributions, binary-fraction models,
and the composite ``BinaryIMF``.

This subpackage was split out of the former monolithic ``binary.py`` to satisfy
the 500-LOC file limit. The public API is unchanged — every symbol below remains
importable from ``progenax.imf`` and ``progenax.imf.binary``.
"""

from .binary_fraction import (
    ConstantBinaryFraction,
    MassDependentBinaryFraction,
)
from .imf import (
    BinaryFractionCallable,
    BinaryIMF,
    MassRatioSamplerCallable,
)
from .mass_ratio import (
    FlatMassRatio,
    MassRatioProtocol,
    PowerLawMassRatio,
    TwinPeakedMassRatio,
)
from .moe_di_stefano import (
    MoeDiStefano2017,
    MoeDiStefano2017Full,
    MoeJointOrbit,
    MoePeriod,
)

__all__ = [
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
    "BinaryFractionCallable",
    "MassRatioSamplerCallable",
]

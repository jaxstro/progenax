"""Every symbol the docs promise must import from the package root (audit R7).

IGIMF and EnvironmentIMF are deliberately ABSENT: they never existed
(the environment-dependent IMF is the functional BirthEnvironment +
env_to_imf_params API); the docs are being fixed to stop advertising them.
"""

import pytest

DOCUMENTED_ROOT_SYMBOLS = [
    "PowerLawIMF",
    "ChabrierIMF",
    "Maschberger",
    "TruncatedIMF",
    "BinaryIMF",
    "FlatMassRatio",
    "PowerLawMassRatio",
    "TwinPeakedMassRatio",
    "MoeDiStefano2017",
    "MoeDiStefano2017Full",
    "MoePeriod",
    "MoeJointOrbit",
    "ConstantBinaryFraction",
    "MassDependentBinaryFraction",
    "UniformSphereProfile",  # audit L1: only profile missing from the root
]


@pytest.mark.parametrize("name", DOCUMENTED_ROOT_SYMBOLS)
def test_symbol_importable_from_root(name):
    import progenax

    assert hasattr(progenax, name), f"progenax.{name} promised by docs, not exported"


def test_phantom_classes_stay_dead():
    import progenax

    for phantom in ("IGIMF", "EnvironmentIMF"):
        assert not hasattr(progenax, phantom)

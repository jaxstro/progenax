"""Bit-identical equivalence gate for the Batch-4b binaries/ refactor.

The 4b refactor only RELOCATES code (split population.py into
period/eccentricity/orientation/mass_dependent, clean orbital_state.py, typed
NamedTuple returns, Protocols) and adjusts API signatures (Moe arg order,
pdf/cdf/ppf). It must NOT change any sampled value. These references were
captured from the post-4a code (commit 17666ac) at PRNGKey(0); every sampler
must reproduce them EXACTLY (bit-identical) after the refactor.
"""

import jax
import jax.numpy as jnp

import progenax  # noqa: F401  (enables float64 at import)
from progenax.binaries import (
    LogUniformPeriod,
    LogNormalPeriod,
    SanaOBPeriod,
    ThermalEccentricity,
    UniformEccentricity,
    LogisticThermalEccentricity,
    RadialBinaryFraction,
    sample_isotropic_orientations,
    MassDependentBinaryConfig,
    sample_mass_dependent_orbits,
)

# Frozen references (post-4a, PRNGKey(0), first 4 values).
REFS = {
    "loguniform": [2226.675310829189, 53.74816439619553, 52792434.40146059, 39446.1180250208],
    "lognormal": [21211.094517228317, 988.641220853479, 948417037.5690745, 170624.43632659054],
    "sana": [14.824895762165548, 4.6942636124085935, 2099.5153502671246, 45.98166331281892],
    "sana_neg1": [6.897204858829157, 3.23879853845318, 1637.4307245036646, 17.0021256298785],
    "thermal": [0.6404137843560446, 0.4604249939554343, 0.9726826635870146, 0.7503785532025334],
    "uniform": [0.3766114005447478, 0.19466590914496024, 0.8687893150070978, 0.5170504803547542],
    "moe": [0.04419236825544376, 0.16293176290038544, 0.7799522037083727, 0.7260963733245235],
    "radial": [1.0, 0.75, 0.55, 0.525],
    "iso_inc": [0.3484707345497839, 1.9494138738695883, 1.2427514499632522, 0.18588202222277103],
    "iso_M": [4.715963377161148, 3.013709286238115, 6.144430612738175, 1.2164354418312062],
    "mdep_P": [318.83184588874474, 8407529.043206416, 2.47554539654342, 1187.7021089208833],
    "mdep_e": [0.6294349587055099, 0.23771526232277707, 0.036228316321676286, 0.6665456601344019],
}


def _assert_eq(name, arr):
    ref = jnp.asarray(REFS[name])
    got = jnp.ravel(arr)[:4]
    # The 4b refactor only relocates code, so samplers must reproduce the frozen
    # post-4a values. We assert equality to a 1e-12 floor rather than bit-exact:
    # a real regression shifts values by >>1e-12, while last-ULP reduction-order
    # drift between JAX/XLA versions (e.g. 0.7.0 vs 0.10.1, ~1e-17) is numerical
    # noise, not a code change. atol=1e-12 is a noise floor, NOT a physics tolerance.
    assert jnp.allclose(got, ref, rtol=0.0, atol=1e-12), (
        f"{name}: {list(map(float, got))} != {REFS[name]}"
    )


def test_period_samplers_bit_identical():
    k = jax.random.PRNGKey(0)
    _assert_eq("loguniform", LogUniformPeriod().sample(k, 4))
    _assert_eq("lognormal", LogNormalPeriod().sample(k, 4))
    # Pin the original [0.3, 3.5] range (the 4b-relocation reference). The default
    # log_P_min was changed 0.3 -> 0.15 in 4c-b (Sana 2012 Fig.2); that is tested
    # separately in TestSanaOBPeriodDistribution::test_default_range_is_sana_2012.
    _assert_eq("sana", SanaOBPeriod(log_P_min=0.3).sample(k, 4))
    _assert_eq("sana_neg1", SanaOBPeriod(power=-1.0, log_P_min=0.3).sample(k, 4))


def test_eccentricity_samplers_bit_identical():
    k = jax.random.PRNGKey(0)
    _assert_eq("thermal", ThermalEccentricity().sample(k, 4))
    _assert_eq("uniform", UniformEccentricity().sample(k, 4))
    # The 4a heuristic MoeEccentricity was renamed to LogisticThermalEccentricity
    # in 4c (values unchanged); the "moe" reference pins that bit-identical rename.
    _assert_eq("moe", LogisticThermalEccentricity().sample(k, jnp.array([5.0, 50.0, 500.0, 5000.0])))


def test_orientation_radial_massdep_bit_identical():
    k = jax.random.PRNGKey(0)
    _assert_eq("radial", RadialBinaryFraction().compute(jnp.array([0.1, 1.0, 5.0, 10.0])))
    inc, _Omega, _omega, M = sample_isotropic_orientations(k, 4)
    _assert_eq("iso_inc", inc)
    _assert_eq("iso_M", M)
    cfg = MassDependentBinaryConfig(
        m_break=8.0,
        low_mass_period=LogNormalPeriod(),
        high_mass_period=SanaOBPeriod(log_P_min=0.3),  # original range (relocation pin)
        low_mass_eccentricity=ThermalEccentricity(),
        high_mass_eccentricity=LogisticThermalEccentricity(),
    )
    P, e = sample_mass_dependent_orbits(jnp.array([1.0, 5.0, 10.0, 20.0]), cfg, k)
    _assert_eq("mdep_P", P)
    _assert_eq("mdep_e", e)

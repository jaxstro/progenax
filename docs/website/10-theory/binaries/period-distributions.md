---
title: Binary period distributions
description: Implemented period-distribution families in progenax — log-uniform, log-normal, Sana+2012 OB, and mass-dependent routing helpers.
---

# Binary period distributions

The orbital period distribution $f(P)$ is one of the binary-population
ingredients used alongside the mass-ratio and eccentricity
distributions. progenax currently exposes three direct period samplers
and one mass-dependent routing helper.

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers choosing a binary period distribution; no prior binary-statistics literature assumed.
**Prerequisites:** [Kepler orbital elements](kepler-elements.md) (period $P$ is the element these distributions sample; $a$ follows from Kepler III).
**You'll get:** the three implemented period families (log-uniform, log-normal, Sana OB), when each applies, the mass-dependent routing helper, and the honest note on period–mass-ratio coupling.
:::

```{list-table}
:header-rows: 1

* - Family
  - Class/function
  - Best for
* - Log-uniform (Öpik)
  - `LogUniformPeriod`
  - Baseline population-synthesis comparisons
* - Log-normal
  - `LogNormalPeriod`
  - Solar-type primaries
* - {cite:t}`Sana2012` OB-type
  - `SanaOBPeriod`
  - Massive-star primaries
* - Mass-dependent routing
  - `MassDependentBinaryConfig`, `sample_mass_dependent_orbits`
  - Route low- and high-mass primaries to different period/eccentricity prescriptions
```

There is no public `Moe17Period` class in this checkout. The full
Moe & Di Stefano mass-ratio and multiplicity machinery lives in
`progenax.imf.binary`, especially `MoeDiStefano2017` and `BinaryIMF`.

## Log-uniform

`LogUniformPeriod` samples uniformly in $\log_{10} P$:

```python
from progenax.binaries import LogUniformPeriod

period_dist = LogUniformPeriod(log_P_min=0.0, log_P_max=8.0)
periods_days = period_dist.sample(key, 10000)
```

The returned values are periods in days, not $\log P$.

## Log-normal

`LogNormalPeriod` samples a normal distribution in $\log_{10} P$ and
returns periods in days:

```python
from progenax.binaries import LogNormalPeriod

solar = LogNormalPeriod(mu_log_P=4.8, sigma_log_P=2.3)
periods_days = solar.sample(key, 10000)
```

The defaults follow the broad solar-type binary period distribution.

## Sana OB periods

`SanaOBPeriod` samples the short-period-biased massive-star
distribution from {cite:t}`Sana2012`:

```python
from progenax.binaries import SanaOBPeriod

ob = SanaOBPeriod(log_P_min=0.3, log_P_max=3.5, power=-0.55)
periods_days = ob.sample(key, 10000)
```

This distribution is appropriate for O/B-type primaries, not for a
generic low-mass field population.

## Mass-dependent routing

For a mixed population, use `sample_mass_dependent_orbits` to route
low- and high-mass primaries to different period and eccentricity
models:

```python
from progenax.binaries import (
    LogNormalPeriod,
    MassDependentBinaryConfig,
    MoeEccentricity,
    SanaOBPeriod,
    ThermalEccentricity,
    sample_mass_dependent_orbits,
)

config = MassDependentBinaryConfig(
    m_break=8.0,
    low_mass_period=LogNormalPeriod(),
    high_mass_period=SanaOBPeriod(),
    low_mass_eccentricity=ThermalEccentricity(),
    high_mass_eccentricity=MoeEccentricity(),
)

periods_days, eccentricities = sample_mass_dependent_orbits(
    primary_masses, config, key
)
```

## Period-mass-ratio coupling

The {cite:t}`MoeDiStefano2017` joint distribution is non-separable: mass-ratio
statistics depend on primary mass and period, and eccentricity depends
on period. progenax currently exposes these pieces separately rather
than as one public all-parameters sampler.

```python
from progenax.imf import BinaryIMF, MoeDiStefano2017, PowerLawIMF

binary_imf = BinaryIMF(
    primary_imf=PowerLawIMF.kroupa(),
    q_distribution=MoeDiStefano2017(),
)
m1, m2, is_binary = binary_imf.sample_systems(key, 1000)
```

## Domain of validity

1. **Period range.** The implemented samplers encode finite period
   ranges or broad log-normal support; compact binaries and extremely
   wide marginally bound systems require extra physics.
2. **Population dependence.** `LogNormalPeriod` and `SanaOBPeriod` are
   calibrated for different primary-mass regimes. Use the routing helper
   for mixed populations.
3. **IC-time only.** Dynamical encounters and stellar evolution can
   alter periods after birth; those belong in downstream evolution.

## Implementation, validation & references

- **In code:** `src/progenax/binaries/period.py` (`LogUniformPeriod`,
  `LogNormalPeriod`, `SanaOBPeriod`) and
  `src/progenax/binaries/mass_dependent.py`
  (`MassDependentBinaryConfig`, `sample_mass_dependent_orbits`); the
  full Moe & Di Stefano machinery lives in `src/progenax/imf/binary/`.
  See the [binaries API](../../30-api/binaries.md).
- **Validated in:** [binary-aware recovery](../../50-validation/binary-imf.md).
- **Primary sources:** {cite:t}`Sana2012` provides the OB-period anchor;
  the broader mass-dependent binary-statistics reference is
  {cite:t}`MoeDiStefano2017`. Full notes in the
  [bibliography](../../99-bibliography/per-paper/sana-2012.md); see also
  [](../imfs/multiplicity-statistics.md) and
  [](../imfs/mass-ratio-distributions.md).

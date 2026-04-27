---
title: Eccentricity distributions
description: Three eccentricity distribution families — thermal f(e) = 2e, uniform f(e) = 1, and Moe & Di Stefano (2017) period-dependent — with their physical meaning, sampling, and tidal-circularisation regime.
---

# Eccentricity distributions

The orbital eccentricity $e$ is the third free parameter (alongside
$P$ and $q$) in the {cite:t}`Moe2017` joint binary-population
calibration. Unlike $P$ and $q$, the eccentricity distribution is
**period-dependent**: short-period binaries are tidally circularised
(small $e$); long-period binaries retain whatever eccentricity they
formed with (typically thermal). progenax implements three families:

```{list-table}
:header-rows: 1
:widths: 22 22 56

* - Distribution
  - $f(e)$
  - Physical meaning
* - **Thermal**
  - $f(e) = 2e$
  - Long-period binaries with random angular-momentum vectors; classical Heggie (1975) prediction
* - **Uniform**
  - $f(e) = 1$
  - Short-period binaries that have lost angular momentum but not yet circularised
* - **{cite:t}`Moe2017`**
  - Period-dependent
  - Empirical: thermal at long $P$, smoothly transitioning to uniform/circular at short $P$
```

This chapter derives each form, sets out the period-dependent
transitions, and documents progenax's sampling implementation.

## Thermal: $f(e) = 2e$

The thermal eccentricity distribution dates to {cite:t}`Heggie1975`
and represents the steady-state distribution of a population of
binaries with isotropic angular-momentum vectors:

```{math}
:label: thermal
f(e) \;=\; 2e,\qquad e \in [0, 1].
```

Physically, the thermal $f(e)$ arises because a binary's specific
angular momentum scales as $L \propto \sqrt{1 - e^2}$, and an
isotropic distribution in $L$ produces $f(e) \propto e$. The factor
of 2 normalises so that $\int_0^1 f(e)\,\mathrm{d}e = 1$.

The mean and median:

```{math}
\langle e \rangle \;=\; 2/3 \approx 0.67,\qquad
e_{\mathrm{med}} \;=\; 1/\sqrt{2} \approx 0.71.
```

Most thermal binaries are highly eccentric. progenax's
`ThermalEccentricity()` samples via inverse-CDF:

```python
u = jax.random.uniform(key, (N,))
e = jnp.sqrt(u)            # Inverse of F(e) = e^2
```

Differentiable analytically.

## Uniform: $f(e) = 1$

The uniform distribution $f(e) = 1$ for $e \in [0, 1]$ is the
maximum-entropy distribution in the absence of any prior information
about the angular-momentum distribution. It is the *correct* choice
for binaries whose formation channel does not preserve a thermal
distribution — e.g. binaries formed via core fragmentation in
turbulent clouds, where the angular momentum is set by the local flow
and is not isotropic.

```{math}
:label: uniform-ecc
f(e) \;=\; 1,\qquad e \in [0, 1].
```

Sampling: $e = u$ for $u \sim \mathcal{U}(0, 1)$. Mean and median both
$0.5$.

`UniformEccentricity()` is the default progenax option for
short-period binaries that have not yet been processed by tidal
evolution.

## Moe & Di Stefano (2017): period-dependent transition

{cite:t}`Moe2017` find that the eccentricity distribution depends
strongly on orbital period:

```{list-table} {cite:t}`Moe2017` eccentricity-period regimes.
:header-rows: 1

* - Period range
  - Eccentricity distribution
  - Physical mechanism
* - $P \le 4$ d
  - $f(e) = \delta(e)$
  - Tidal circularisation; $e$ rapidly damped by stellar tides
* - $4$ d $< P \le 100$ d
  - Smoothly interpolating
  - Partial circularisation; transition from thermal to circular
* - $P > 100$ d
  - Thermal $f(e) = 2e$
  - Far enough that tides do not act on Hubble timescales
```

progenax's `MoeEccentricity()` parameterises the period dependence as

```{math}
:label: moe-ecc
f(e \mid P) \;=\; (1 - w(P))\,\delta(e) \;+\; w(P)\,2e
```

where $w(P)$ is a smooth blending function $w(P \le 4\,\mathrm{d}) = 0$,
$w(P \ge 100\,\mathrm{d}) = 1$, with a smooth cubic-spline transition
in between. The parameterisation is chosen so $f(e \mid P)$ is
continuous in $P$ — important for HMC inference where $P$ might
itself be a free parameter.

## Sampling implementation

```python
from progenax.binaries import (
    ThermalEccentricity,
    UniformEccentricity,
    MoeEccentricity,
)

# Thermal (long-period or "no information" choice)
thermal = ThermalEccentricity()
e_thermal = thermal.sample(key, 10000)

# Uniform (short-period unprocessed)
uniform = UniformEccentricity()
e_uniform = uniform.sample(key, 10000)

# Moe17 period-dependent
moe = MoeEccentricity()
e_moe = moe.sample(periods_days, key)   # Uses the per-binary periods
```

The `MoeEccentricity` sampler accepts a `log_P` array because the
distribution is period-conditional. The implemented sampler accepts
periods in days, with `P_circ` and `P_thermal` controlling the smooth
transition from near-circular to thermal.

## Tidal circularisation in detail

Tidal circularisation operates on a timescale {cite:p}`Hurley2002`

```{math}
:label: tidal-timescale
\tau_{\mathrm{circ}} \;\sim\; \biggl(\frac{a}{R_\star}\biggr)^{\!8} \cdot \frac{1}{q\,(1 + q)}
```

with $R_\star$ the primary's radius. The strong $a^8$ dependence makes
circularisation effective only for very short orbits — typically
$P \le 4$ d for solar-type primaries on the main sequence. For
massive stars (larger radii) the circularisation cutoff extends to
slightly longer periods.

progenax's default `MoeEccentricity()` uses a $P \le 4$ d circular
cutoff matching {cite:t}`Moe2017`. Surveys targeting evolved stars
(red giants with $R_\star \gg R_\odot$) should use the longer cutoff
appropriate to the evolutionary phase — `MoeEccentricity(p_circ_d=12.0)`
overrides the default.

## Joint $f(P, e, q, M_1)$ when consistency matters

For population studies that need mass-dependent period/eccentricity
prescriptions, progenax provides `MassDependentBinaryConfig` and
`sample_mass_dependent_orbits`:

```python
import jax.numpy as jnp
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

primary_masses = jnp.array([1.0, 5.0, 10.0, 20.0])
periods, eccentricities = sample_mass_dependent_orbits(
    primary_masses, config, key
)
```

This samples periods and eccentricities. The fully joint
Moe & Di Stefano mass-ratio and multiplicity model lives in the IMF
binary module, not in this eccentricity sampler.

## Domain of validity

1. **Bound orbits only** ($e < 1$). Marginally-bound or hyperbolic
   encounters are not represented — they are not "binaries" in the
   sense progenax models.
2. **Single-event tidal cutoff** — the circular cutoff is set by the
   primary's *current* radius, ignoring stellar evolution. Stars that
   evolve through giant phases briefly have $R_\star \gg R_\odot$,
   which extends the cutoff temporarily. progenax's IC-time treatment
   uses ZAMS radii; subsequent evolution should be handled by the
   integrator (gravax) and stellar-evolution code (startrax, planned).
3. **Metallicity assumption** — the {cite:t}`Moe2017` calibration is
   solar-metallicity. Tidal-circularisation efficiency varies with
   internal stellar structure, which is metallicity-dependent. For
   metal-poor populations expect the cutoff period to shift slightly.
4. **No magnetic-braking effects** — magnetic braking circularises
   short-period binaries on timescales comparable to $\tau_{\mathrm{circ}}$.
   The progenax default lumps both effects into a single empirical
   cutoff, but for science targeting magnetic activity (CV-progenitor
   binaries, RS CVn analogues) the effects should be modelled
   separately.

## References

The thermal distribution is {cite:t}`Heggie1975`; the period-
dependent treatment is {cite:t}`Moe2017`; the tidal-circularisation
timescale follows {cite:t}`Hurley2002`. For population-synthesis
context see {cite:t}`Sana2012`.

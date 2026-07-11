---
title: Eccentricity distributions
description: Three eccentricity distribution families — thermal f(e) = 2e, uniform f(e) = 1, and Moe & Di Stefano (2017) period-dependent — with their physical meaning, sampling, and tidal-circularisation regime.
---

# Eccentricity distributions

The orbital eccentricity $e$ is the third free parameter (alongside
$P$ and $q$) in the {cite:t}`MoeDiStefano2017` joint binary-population
calibration. Unlike $P$ and $q$, the eccentricity distribution is
**period-dependent**: short-period binaries are tidally circularised
(small $e$); long-period binaries retain whatever eccentricity they
formed with (typically thermal). progenax implements three families:

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers choosing a binary eccentricity distribution and learning why it is period-dependent; no prior binary-statistics literature assumed.
**Prerequisites:** [Kepler orbital elements](kepler-elements.md) (eccentricity $e$ is the orbit-shape element) and [period distributions](period-distributions.md) (the Moe form is period-dependent).
**You'll get:** the thermal, uniform, and Moe & Di Stefano period-dependent $f(e)$ laws, their sampling, the Roche-lobe ceiling, and how tidal circularisation enters.
:::

```{list-table}
:header-rows: 1

* - Distribution
  - $f(e)$
  - Physical meaning
* - **Thermal**
  - $f(e) = 2e$
  - Long-period binaries with random angular-momentum vectors; classical Heggie (1975) prediction
* - **Uniform**
  - $f(e) = 1$
  - Short-period binaries that have lost angular momentum but not yet circularised
* - **{cite:t}`MoeDiStefano2017`**
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

[↗ model card](#card-thermal)

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

[↗ model card](#card-uniform-ecc)

Sampling: $e = u$ for $u \sim \mathcal{U}(0, 1)$. Mean and median both
$0.5$.

`UniformEccentricity()` is the default progenax option for
short-period binaries that have not yet been processed by tidal
evolution.

## Moe & Di Stefano (2017): the period- and mass-dependent power law

{cite:t}`MoeDiStefano2017` find (their §9.2, Fig. 36) that the
eccentricity distribution is a **power law** $p(e) \propto e^{\eta}$ on
$0 \le e \le e_{\max}(P)$, with the slope $\eta$ depending on *both*
orbital period and primary mass — not the $\delta$-plus-thermal blend
that older qualitative pictures suggest. The qualitative
{cite:t}`DuquennoyMayor1991`-style picture (short-period orbits tidally
circularised, long-period orbits thermal) is a useful intuition for the
*trend*, but progenax implements Moe's quantitative law:

```{list-table} Qualitative period picture (Duquennoy & Mayor 1991), for intuition only — progenax samples the quantitative Moe & Di Stefano $e^{\eta}$ law below.
:header-rows: 1

* - Period range
  - Eccentricity behaviour
  - Physical mechanism
* - Short ($P \lesssim$ few d)
  - Near-circular ($e \approx 0$)
  - Tidal circularisation; $e$ rapidly damped by stellar tides
* - Intermediate
  - Smoothly rising $\langle e\rangle$
  - Partial circularisation
* - Long ($P$ large)
  - Approaching thermal $f(e) = 2e$
  - Tides do not act on Hubble timescales
```

The slope follows {cite:t}`MoeDiStefano2017` Eqs. 17–18 (verified
against their Fig. 36):

```{math}
:label: moe-eta-ecc
\eta(M_1, P) =
\begin{cases}
0.6 - \dfrac{0.7}{\log_{10} P - 0.5}, & 0.8 < M_1 < 3\,\Msun\ \text{(Eq. 17, late-type)} \\[1ex]
0.9 - \dfrac{0.2}{\log_{10} P - 0.5}, & M_1 > 7\,\Msun\ \text{(Eq. 18, early-type)}
\end{cases}
```

[↗ model card](#card-moe-eta-ecc)

with linear interpolation in $M_1$ across $3$–$7\,\Msun$. Here $\eta = 0$
is uniform ($\langle e\rangle = 0.5$) and $\eta = 1$ is thermal
($\langle e\rangle = 2/3$); short-period massive binaries are driven to
$\eta < 0$ (circularising), and $\eta \le -1$ (very short $P$,
$e^{\eta}$ non-normalisable) returns $e \approx 0$ by construction. The
upper limit is the period-dependent Roche-lobe ceiling (their Eq. 3),

```{math}
:label: moe-emax-ecc
e_{\max}(P) \;=\; 1 - \Bigl(\frac{P}{2\,\mathrm{d}}\Bigr)^{-2/3} \quad (P > 2\ \mathrm{d}),
```

[↗ model card](#card-moe-emax-ecc)

so the components do not overflow their Roche lobes at periapsis (e.g.
$e_{\max}(10\,\mathrm{d}) \approx 0.66$, $e_{\max}(100\,\mathrm{d})
\approx 0.93$); $P \le 2$ d circularises. Sampling uses the inverse CDF
$e = e_{\max}(P)\,u^{1/(\eta+1)}$, which is continuous in $P$ and $M_1$ —
important for HMC inference where they may be free parameters.

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

# Moe17 period- and mass-dependent
moe = MoeEccentricity()
e_moe = moe.sample(key, periods_days, primary_masses)  # masses REQUIRED
```

The `MoeEccentricity` sampler signature is `sample(key, periods,
masses)` — **both** the per-binary periods (in days) and the primary
masses are required, because the slope $\eta(\log P, M_1)$
({eq}`moe-eta-ecc`) depends on both. The module's only field is `e_max`
(the long-$P$ numerical ceiling, default 0.99); the physical cap is the
period-dependent Roche relation {eq}`moe-emax-ecc`.

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

progenax's `MoeEccentricity` handles short-period circularisation
*intrinsically*: where the Roche ceiling {eq}`moe-emax-ecc` and the
$\eta \le -1$ branch drive $e \to 0$ (at $P \le 2$ d the ceiling is
exactly 0). There is **no** separate circular-cutoff knob — the module's
only field is `e_max`, the long-$P$ numerical ceiling. Surveys
targeting evolved stars (red giants with $R_\star \gg R_\odot$) need a
longer effective cutoff than the main-sequence relation provides; that
would require a stellar-evolution-aware Roche radius (a future
`startrax` coupling), not a constructor argument here.

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
3. **Metallicity assumption** — the {cite:t}`MoeDiStefano2017` calibration is
   solar-metallicity. Tidal-circularisation efficiency varies with
   internal stellar structure, which is metallicity-dependent. For
   metal-poor populations expect the cutoff period to shift slightly.
4. **No magnetic-braking effects** — magnetic braking circularises
   short-period binaries on timescales comparable to $\tau_{\mathrm{circ}}$.
   The progenax default lumps both effects into a single empirical
   cutoff, but for science targeting magnetic activity (CV-progenitor
   binaries, RS CVn analogues) the effects should be modelled
   separately.

## Implementation, validation & references

- **In code:** `src/progenax/binaries/eccentricity.py`
  (`ThermalEccentricity`, `UniformEccentricity`, `MoeEccentricity`),
  with mixed-population routing in
  `src/progenax/binaries/mass_dependent.py`. See the
  [binaries API](../../30-api/binaries.md).
- **Validated in:** [binary-aware recovery](../../50-validation/binary-imf.md).
- **Primary sources:** the thermal distribution is {cite:t}`Heggie1975`;
  the period-dependent power law is {cite:t}`MoeDiStefano2017` (Eqs.
  17–18, verified against their Fig. 36); the tidal-circularisation
  timescale follows {cite:t}`Hurley2002`. Full notes in the
  [bibliography](../../99-bibliography/per-paper/moe-distefano-2017.md);
  for population-synthesis context see {cite:t}`Sana2012`.

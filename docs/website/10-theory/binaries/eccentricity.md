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
formed with. The empirical distribution depends on primary mass as well
as period; long-period solar-type binaries need not be thermal. progenax
implements the three families below, plus a qualitative
`LogisticThermalEccentricity` blend.

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
  - A statistical phase-space law; isotropic directions alone do not determine $f(e)$
* - **Uniform**
  - $f(e) = 1$
  - A uniform baseline on the chosen eccentricity interval
* - **{cite:t}`MoeDiStefano2017`**
  - Period-dependent
  - Empirical mass/period-dependent power law, with a circular point-mass branch
```

This chapter derives each form, sets out the period-dependent
transitions, and documents progenax's sampling implementation.

## Thermal: $f(e) = 2e$

The thermal eccentricity distribution dates to {cite:t}`Heggie1975`
and is a statistical model for dynamically processed binaries:

```{math}
:label: thermal
f(e) \;=\; 2e,\qquad e \in [0, 1].
```

[↗ model card](#card-thermal)

At fixed semi-major axis and masses, the specific angular momentum
satisfies $h^2=G(m_1+m_2)a(1-e^2)$. A distribution uniform in $h^2$
therefore gives $f(e)=2e$ by its change-of-variables Jacobian. This
assumption concerns angular-momentum magnitude; isotropy of its direction
alone does not imply a thermal eccentricity law. The factor of two
normalizes the ideal interval $[0,1]$.

The mean and median:

```{math}
\langle e \rangle \;=\; 2/3 \approx 0.67,\qquad
e_{\mathrm{med}} \;=\; 1/\sqrt{2} \approx 0.71.
```

On the implemented interval $[0,e_{\max}]$, the normalized density is
$2e/e_{\max}^2$, the mean is $2e_{\max}/3$ and the median is
$e_{\max}/\sqrt2$. `ThermalEccentricity()` defaults to $e_{\max}=0.99$
and samples via inverse-CDF:

```python
u = jax.random.uniform(key, (N,))
e = thermal.e_max * jnp.sqrt(u)  # F(e) = (e/e_max)^2
```

The interior quantile is analytic. Endpoints and parameter/support
boundaries require their own derivative convention.

## Uniform: $f(e) = 1$

A uniform eccentricity law is a baseline choice, not a consequence of
an unspecified formation channel. It maximizes entropy relative to
the measure $\mathrm{d}e$ on a fixed interval; another parameterization
changes that statement.

```{math}
:label: uniform-ecc
f(e) \;=\; 1,\qquad e \in [0, 1].
```

[↗ model card](#card-uniform-ecc)

`UniformEccentricity(e_max=c)` instead normalizes on $[0,c]$, with
density $1/c$, quantile $e=cu$, and mean and median $c/2$. Its default
is $c=0.9$, with `e_min=0`. A nonzero lower bound gives density
$1/(e_{\max}-e_{\min})$ and quantile
$e_{\min}+u(e_{\max}-e_{\min})$. Choosing it for a physical population requires an explicit
model assumption; there is no universal short-period default law.

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
  - Early-type populations approach near-thermal; solar-type populations can remain flatter
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

with linear interpolation in $M_1$ across $3$–$7\,\Msun$. For a cap
$c=e_{\max}(P)$ and $\eta>-1$, the normalized density is
$(\eta+1)e^\eta/c^{\eta+1}$ and mean is $c(\eta+1)/(\eta+2)$.
Thus $\eta=0$ is uniform and $\eta=1$ is thermal on that interval;
short-period massive binaries are driven to
$\eta < 0$ (circularising), and $\eta \le -1$ (very short $P$,
$e^{\eta}$ non-normalisable) returns $e \approx 0$ by construction. The
upper limit is the period-dependent Roche-lobe ceiling (their Eq. 3),

```{math}
:label: moe-emax-ecc
e_{\max}(P) \;=\; 1 - \Bigl(\frac{P}{2\,\mathrm{d}}\Bigr)^{-2/3} \quad (P > 2\ \mathrm{d}),
```

[↗ model card](#card-moe-emax-ecc)

used by the paper for main-sequence binary populations (e.g.
$e_{\max}(10\,\mathrm{d}) \approx 0.66$, $e_{\max}(100\,\mathrm{d})
\approx 0.93$); $P \le 2$ d has zero cap. The implementation additionally clips the cap
to its `e_max` field, floors $\log_{10}(P/\mathrm{day})-0.5$ at $10^{-3}$
in the slope denominator, and assigns $e=0$ when $\eta+1\le10^{-6}$.
These are explicit numerical extensions around the short-period regime,
where the source fit is not well defined. On the active branch the quantile
is $e=c u^{1/(\eta+1)}$. The circular branch is a point mass at zero;
`pdf` returns only the continuous part, zero on that branch. Clipping,
branch thresholds and inverse-angle degeneracies exclude a universal
smoothness claim across the whole parameter domain.

## Sampling implementation

```python
from progenax.binaries import (
    ThermalEccentricity,
    UniformEccentricity,
    MoeEccentricity,
)

# Thermal statistical model on [0, 0.99]
thermal = ThermalEccentricity()
e_thermal = thermal.sample(key, 10000)

# Uniform baseline on [0, 0.9]
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
\tau_{\mathrm{circ}} \;\propto\; \biggl(\frac{a}{R_\star}\biggr)^{\!8} \cdot \frac{1}{q\,(1 + q)}
```

with $R_\star$ the primary's radius, holding the stellar tidal coefficient
and other structure factors fixed. The omitted prefactor carries time
units; this scaling is not a complete timescale formula. Radiative and
convective damping require different stellar prescriptions. The strong
$a^8$ dependence in the quoted equilibrium-tide scaling makes
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
2. **Population ceiling, not individual contact rejection.** The
   `MoeEccentricity` cap depends on period, not supplied stellar radii,
   masses or mass ratio. It follows the source's main-sequence population
   prescription; it does not prove that each realized system avoids
   contact or Roche-lobe overflow. Use component radii and periapsis
   diagnostics for a particular catalogue. Evolved-star tides and
   radius changes require a separate stellar-evolution prescription.
3. **Metallicity assumption** — the {cite:t}`MoeDiStefano2017` calibration is
   solar-metallicity. Tidal-circularisation efficiency varies with
   internal stellar structure, which is metallicity-dependent. For
   metal-poor populations expect the cutoff period to shift slightly.
4. **No magnetic-braking evolution.** Magnetic braking changes spin
   angular momentum and can couple to an orbit through tides.
   This sampler does not evolve either mechanism. Applications targeting
   magnetic activity or evolved binaries require an explicit evolution model.

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

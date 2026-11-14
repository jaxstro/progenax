---
title: Environment-dependent IMFs
description: Marks+2012 high-mass IMF variation as a function of cloud-core density and metallicity, and the Jeřábková+2018 density relation for the cluster-scale high-mass slope (the per-cluster input to the IGIMF framework). progenax implements the cluster-scale mapping; the galaxy-wide IGIMF integral is background theory, not an implemented sampler.
---

# Environment-dependent IMFs

```{seealso}
This chapter covers the **environment-dependent** branch of IMF theory:
how the high-mass slope $\alpha_3$ and the low-mass slopes
$\alpha_{1,2}$ vary with cloud-core density and metallicity, and how
those variations feed the galaxy-wide IMF (gwIMF) framework — background
theory here, since progenax implements the cluster-scale input, not the
gwIMF integral. For the
**universal** IMF baseline (Salpeter, Kroupa, Chabrier, Maschberger),
see [](classic.md). For binary-induced biases on inferred $\alpha$,
see [](binary.md).
```

The canonical IMF is treated as universal in most population-synthesis
work — a single $\xi(m \mid \alpha)$ applied to every star-forming
environment. This is observationally inadequate at the extremes:
ultra-compact dwarfs and massive globular cluster progenitors require
*top-heavy* IMFs ($\alpha_3 < 2.3$) to reproduce their stellar
populations and integrated colours, while metal-poor populations show
hints of low-mass-suppressed IMFs {cite:p}`Marks2012,Jerabkova2018`.
progenax implements the {cite:t}`Marks2012` cluster-scale variation and
the {cite:t}`Jerabkova2018` density relation for the high-mass slope (the
cluster-IMF input to the IGIMF framework) as piecewise differentiable models.
The galaxy-wide IGIMF *integral* itself is background theory here, not an
implemented sampler — see the scope note below.

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers learning how the IMF varies with cloud-core density and metallicity, and what progenax actually implements (the cluster-scale mapping, not the galaxy-wide integral); no prior IGIMF literature assumed.
**Prerequisites:** the [classical IMFs](classic.md) (the universal $(\alpha_1, \alpha_2, \alpha_3)$ baseline this varies).
**You'll get:** the Marks+2012 high-mass-slope relations, the Fundamental-Plane $\alpha_3(\rho_{\mathrm{cl}}, [\mathrm{Fe/H}])$ fit (with the corrected $-0.87$ threshold), where the IGIMF framework fits as background theory, and the differentiable `env_to_imf_params` API.
:::

## Two physical drivers of IMF variation

Here $\xi(m)=\mathrm dN/\mathrm dm$ is the number of stars per unit mass,
$m$ is stellar mass in $\Msun$, and the slopes $\alpha_i$ are dimensionless.
The cloud-core density $\rho_{\mathrm{cl}}$ is in $\Msun\,\mathrm{pc}^{-3}$;
$[\mathrm{Fe/H}]$ is the base-10 logarithm of the iron-to-hydrogen abundance
relative to solar, in dex. The embedded stellar mass is $M_{\mathrm{ecl}}$,
and the star-formation efficiency $\varepsilon=M_{\mathrm{ecl}}/M_{\mathrm{cl}}$
is a fraction. $M_{\mathrm{cl}}$ denotes the total birth-cloud mass, including
the material converted into stars, rather than the residual gas mass.

Both papers attribute IMF variation to two environmental parameters:

```{list-table}
:header-rows: 1

* - Driver
  - Mechanism
* - **Cloud-core density** $\rho_{\mathrm{cl}}$
  - Higher density → more frequent protostellar collisions, higher accretion rates, cosmic-ray heating in starbursts. Net effect: top-heavy IMF in dense regimes.
* - **Metallicity** [Fe/H]
  - Lower metallicity → less efficient cooling → higher Jeans mass → fragmentation favours more massive stars. Net effect: top-heavy IMF in metal-poor regimes.
```

The two effects are correlated: starbursts that produce dense cores are
typically metal-enriched, so cancelling rather than reinforcing the
effects. {cite:t}`Marks2012` resolves the degeneracy via the
*Fundamental Plane* parameterisation in §[](#fundamental-plane).

## Notation: 3-segment vs 4-segment IMF

```{note}
{cite:t}`Marks2012` and {cite:t}`Jerabkova2018` use a **3-segment**
labelling that omits the brown-dwarf segment, while progenax uses the
**4-segment** form internally. The table below connects the two conventions.
```

The progenax 4-segment IMF:

```{math}
:label: imf-4-segment
\xi(m) \;=\; k\,a_i\,m^{-\alpha_i}
\quad\text{for}\quad m_i \le m < m_{i+1},
\quad i = 0, 1, 2, 3
```

```{list-table}
:header-rows: 1

* - Segment
  - Mass range $(\Msun)$
  - Canonical $\alpha$
  - Marks+12 label
* - 0
  - $0.01 \le m < 0.08$
  - 0.3 (fixed)
  - (not used)
* - 1
  - $0.08 \le m < 0.50$
  - 1.3
  - $\alpha_1$
* - 2
  - $0.50 \le m < 1.00$
  - 2.3
  - $\alpha_2$
* - 3
  - $1.00 \le m \le m_{\max}$
  - 2.3 (Salpeter)
  - $\alpha_3$
```

The bounds are $(m_0,m_1,m_2,m_3,m_4)=(0.01,0.08,0.5,1,m_{\max})$ in
$\Msun$, with the final upper endpoint included. Continuity coefficients
$a_i$ are computed from $a_0 = 1$ via
$a_i = a_{i-1}\,m_i^{\alpha_i - \alpha_{i-1}}$ for $i=1,2,3$.
$k$ sets the total number normalization; the sampling API instead normalizes
the corresponding probability density to integrate to one. progenax's
environment code currently exposes `BirthEnvironment`,
`env_to_imf_params`, and the `alpha3_*` helper functions; it does not
export an `EnvironmentIMF` class. The brown-dwarf segment $\alpha_0$ is held fixed at
$0.3$ regardless of environment, reflecting the lack of strong
observational constraints on sub-stellar IMF variation.

The returned `IMFParams` use a fixed upper-mass cutoff of $150\,\Msun$.
A caller can choose other bounds, including through `TruncatedIMF`;
this mapping does not compute a Weidner-Kroupa $m_{\max}(M_{\mathrm{ecl}})$ relation.

## Marks+2012: high-mass slope variation

{cite:t}`Marks2012` derives empirical relations between the high-mass
slope $\alpha_3$ and four candidate environmental parameters $\lambda$:

```{math}
:label: alpha3-lambda
\alpha_3(\lambda) \;=\;
\begin{cases}
  p_\lambda\,\lambda + q_\lambda, & \lambda\;\gtrless\;\lambda_{\mathrm{lim}} \\
  2.3, & \text{otherwise}
\end{cases}
```

{cite:t}`Marks2012` Table 3 — linear fit coefficients (page 2251).

```{list-table}
:header-rows: 1

* - Parameter $\lambda$
  - $p_\lambda$
  - $q_\lambda$
  - $\lambda_{\mathrm{lim}}$
  - Top-heavy when
* - $\log_{10}(M_{\mathrm{cl}}/10^6\,\Msun)$
  - $-0.94$
  - $+2.14$
  - $0.68$
  - $\lambda > \lambda_{\mathrm{lim}}$
* - $\log_{10}(M_{\mathrm{ecl}}/10^6\,\Msun)$
  - $-0.77$
  - $+1.59$
  - $0.27$
  - $\lambda > \lambda_{\mathrm{lim}}$
* - $\log_{10}(\rho_{\mathrm{cl}}/10^6\,\Msun\,\mathrm{pc}^{-3})$
  - $-0.43$
  - $+1.86$
  - $0.095$
  - $\lambda > \lambda_{\mathrm{lim}}$
* - [Fe/H]
  - $+0.66$
  - $+2.63$
  - $-0.5$
  - $\lambda < \lambda_{\mathrm{lim}}$
```

Mass and density become top-heavy *above* their thresholds (intuition:
denser, more massive clouds form more massive stars). Metallicity is
top-heavy *below* its threshold (intuition: metal-poor clouds fragment
into more massive stars).

### Verification on Marks+12 Table 4

Setting [Fe/H] $= -2.0$ and applying the [Fe/H] coefficients:

```{math}
\alpha_3 \;=\; 0.66 \cdot (-2.0) + 2.63 \;=\; 1.31
```

Marks+12 Table 4 reads $\alpha_3 = 1.31$ at [Fe/H] $= -2.0$ — exact
agreement for this high-mass-slope entry. The
[validation page](../../50-validation/environment-imf.md) distinguishes the
selected low-mass Table 4 anchors and the implementation's test criteria.

## Low-mass slope variation (tentative)

{cite:t}`Marks2012` extends the metallicity dependence to the low-mass
slopes following {cite:t}`Kroupa2001`:

```{math}
:label: alpha-low
\alpha_{1,2}([\mathrm{Fe/H}]) \;=\; \alpha_{1,2}^{\mathrm{c}} + \Delta\alpha\cdot[\mathrm{Fe/H}]
```

with $\Delta\alpha \approx 0.5$ and canonical values
$\alpha_1^{\mathrm{c}} = 1.3$, $\alpha_2^{\mathrm{c}} = 2.3$.

```{warning}
**Low-mass variation is observationally tentative.** The
{cite:t}`Kroupa2001` $\Delta\alpha$ relation was fit to open clusters
with [Fe/H] $\gtrsim -0.5$. Applying it to globular cluster populations
([Fe/H] $\le -1$) is an extrapolation of unknown reliability —
{cite:t}`Marks2012` themselves describe Eq. {eq}`alpha-low` as
"illustrative." progenax exposes low-mass variation as an opt-in flag
`include_lowmass_variation=False`. With `clamp_domain=True`, the mapping
clips [Fe/H] to $[-2.5,+0.5]$. The low-mass floors $\alpha_1\ge0$ and
$\alpha_2\ge0.5$ remain even with `clamp_domain=False`.
These numerical guards do not establish empirical validity over that interval.
```

## The Fundamental Plane: $\alpha_3(\rho_{\mathrm{cl}}, [\mathrm{Fe/H}])$

(fundamental-plane)=

The single-parameter fits in {eq}`alpha3-lambda` capture density and
metallicity effects separately. {cite:t}`Marks2012` combine them into a
2D "Fundamental Plane" via a coordinate rotation:

```{math}
:label: fundamental-x
\hat x(\vartheta) \;=\; \cos\vartheta\cdot[\mathrm{Fe/H}]
                       + \sin\vartheta\cdot\log_{10}\!\left(\frac{\rho_{\mathrm{cl}}}{10^6\,\Msun\,\mathrm{pc}^{-3}}\right)
```

The optimal rotation angle $\vartheta = 98°$ gives
$\cos\vartheta \approx -0.139$ and $\sin\vartheta \approx +0.990$,
i.e. the rotated coordinate is dominated by $\log\rho_{\mathrm{cl}}$
with a weak negative metallicity contribution. The fit reads

```{include} ../../15-model-reference/_environment-marks-fit.md
```

[↗ model card](#card-fundamental-fit)

```{important}
**The adopted threshold is $\hat x\ge-0.87$**, following the
{cite:t}`Marks2014` erratum; source history is recorded in the
[Marks et al. (2012) per-paper note](../../99-bibliography/per-paper/marks-2012.md).
With the retained coefficients, the line evaluates to $2.292564$ at this
threshold, so the default hard branch has a jump of $0.007436$ from $2.3$.
`alpha3_marks_plane` clips its output to $[0.5,2.3]$.
Optional `smooth=True` blends the branches with a hyperbolic tangent;
it changes the mapping near the threshold and does not remove the output clipping.
```

Writing $R_\rho=\log_{10}(\rho_{\mathrm{cl}}/(10^6\,\Msun\,\mathrm{pc}^{-3}))$,
the paper also prints the following expanded form ({cite:t}`Marks2012` Eq. 15):

```{include} ../../15-model-reference/_environment-marks-eq15.md
```

[↗ model card](#card-alpha3-marks-eq15)

valid when $\hat x \ge -0.87$, with $\alpha_3 = 2.3$ otherwise.
This is the paper transcription. The kernel evaluates {eq}`fundamental-fit`
using the rounded rotation $\hat x=-0.139[\mathrm{Fe/H}]+0.990\log_{10}
(\rho_{\mathrm{cl}}/(10^6\,\Msun\,\mathrm{pc}^{-3}))$; its active, unclipped
branch expands instead to coefficients $0.0566008$ and $-0.403128$.
The printed Eq. 15 and the implemented rounded rotation are therefore
close expressions, rather than algebraically identical ones. The discrepancy
includes a coordinate normalization: the paper describes the Fig. 6 abscissa
as $x'/\sin98^\circ$. Applying the slope to that coordinate gives metallicity
coefficient $0.057228\ldots$ and density coefficient $-0.4072$, close to the
printed Eq. 15. Applying it to the unnormalized Eq. 13 coordinate instead
gives $0.056671\ldots$ and $-0.403237\ldots$, before the kernel's additional
rounding. The retained kernel follows the unnormalized coordinate; the
printed expanded expression is not an exact oracle for that convention.

## Jeřábková+2018: the IGIMF framework

```{admonition} Scope — progenax implements the cluster-scale input, not the galaxy-wide integral
:class: important
progenax provides the **per-cluster** IMF $\xi_{\mathrm{cl}}(m \mid M_{\mathrm{ecl}}, [\mathrm{Fe/H}])$ —
the integrand of {eq}`igimf` — via `env_to_imf_params`. It does **not** perform the galaxy-wide
ECMF integration {eq}`igimf` (no ECMF sampler, no SFR$\to M_{\mathrm{ecl,max}}$ integration, no
`gwimf` object). The IGIMF material in this section is **background theory** showing what the
cluster-scale mapping feeds into.
```

The cluster-scale IMF variation in {cite:t}`Marks2012` propagates to
*galaxy-wide* IMFs via the Integrated Galaxy-wide IMF (IGIMF) framework
of {cite:t}`Jerabkova2018`. Stars in a galaxy do not form in a single
cluster — they form in a population of embedded clusters (ECMF) over
some star-formation timescale $\delta t \sim 10$ Myr. The galaxy-wide
IMF is

```{math}
:label: igimf
\xi_{\mathrm{IGIMF}}(m \mid \mathrm{SFR}, [\mathrm{Fe/H}])
\;=\;\int_{M_{\mathrm{ecl,min}}}^{M_{\mathrm{ecl,max}}(\mathrm{SFR})}
   \xi_{\mathrm{cl}}\!\bigl(m \mid M_{\mathrm{ecl}}, [\mathrm{Fe/H}]\bigr)\;
   \xi_{\mathrm{ECMF}}(M_{\mathrm{ecl}} \mid \mathrm{SFR})
   \,\mathrm{d}M_{\mathrm{ecl}}
```

where $\xi_{\mathrm{cl}}(m \mid M_{\mathrm{ecl}}, [\mathrm{Fe/H}])$ is
the {cite:t}`Marks2012` cluster IMF and
$\xi_{\mathrm{ECMF}}$ is the embedded-cluster mass function. The
upper integration limit $M_{\mathrm{ecl,max}}(\mathrm{SFR})$ is the
mass of the most massive cluster a galaxy of star-formation rate SFR
can produce — a "Weidner-Kroupa" cap that scales with SFR.

The qualitative IGIMF predictions:

- **Low SFR ($\sim 10^{-3}\,\Msun/\mathrm{yr}$)**: galaxy populated by
  small clusters, each with bottom-heavy IMF. gwIMF is *bottom-heavy*
  (steeper than Salpeter at high masses).
- **High SFR ($\gtrsim 10\,\Msun/\mathrm{yr}$)**: galaxy populated by
  many massive clusters with top-heavy IMFs. gwIMF is *top-heavy*.
- **Metal-poor**: shifts the entire predicted gwIMF toward top-heavy at
  fixed SFR.

This SFR-and-metallicity-dependent gwIMF has consequences for chemical
evolution, supernova rates, and the ionising-photon budget at high
redshift — see {cite:t}`Jerabkova2018` Fig. 5–7 for the predicted
trends.

### Cluster density and mass conversion

(cluster-density-conversion)=

Write $L_M=\log_{10}(M_{\mathrm{ecl}}/(10^6\,\Msun))$ and
$R_\rho=\log_{10}(\rho_{\mathrm{cl}}/(10^6\,\Msun\,\mathrm{pc}^{-3}))$.
The nominal {cite:t}`MarksKroupa2012` radius law is
$r_h/\mathrm{pc}=0.1(M_{\mathrm{ecl}}/\Msun)^{0.13}$.
The stellar half-mass density is
$\rho_{\mathrm{ecl}}=3M_{\mathrm{ecl}}/(8\pi r_h^3)$, since half the
stellar mass lies within $r_h$. The adopted cloud conversion uses
$\rho_{\mathrm{cl}}=\rho_{\mathrm{ecl}}/\varepsilon$ in the same reference
volume; it does not specify a separate gas profile. Substitution gives
$R_\rho=0.61(L_M+6)+\log_{10}[3/(8\pi\,0.1^3)]-\log_{10}\varepsilon-6$.
Jeřábková Eq. 7 then uses $\hat x=-0.14[\mathrm{Fe/H}]+0.99R_\rho$.
The retained mass helper evaluates

```{include} ../../15-model-reference/_environment-mass-coordinate.md
```

Here $0.6039=0.99\times0.61$ exactly, while $0.2161$ reflects intermediate
rounding in the adopted conversion. Substitution without that rounding
gives $0.2161837806\ldots$. The mass and computed-density coordinates
therefore differ by a constant $-8.3781\times10^{-5}$; their local first
derivatives agree before guards, though hard-switch locations can differ.
The `jerabkova_mecl` route fixes $\varepsilon=0.33$ and ignores any supplied
SFE or density override. [Equation, kernel and test links](#card-environment_imf).

```{admonition} Printed Jeřábková mass formula and adopted conversion
:class: note
{cite:t}`Jerabkova2018` Eq. 9 prints
$\hat x=-0.14[\mathrm{Fe/H}]+0.6L_M+2.83$. This is preserved as a
[separate source expression](#card-environment-jerabkova-printed-mass).
Changing the density denominator from $8\pi$ to $4\pi$ adds
$0.99\log_{10}2=0.2980197\ldots$ to the mass-coordinate intercept,
giving $0.5142035\ldots$ at SFE $0.33$, rather than $2.83$.
The printed mass expression is thus not recovered by a density-factor
change alone. progenax retains the nominal half-mass conversion and its
rounded intercept; it does not implement a conjectured correction of Eq. 9.
```

## Input guards and domain of validity

The published fits have limited empirical coverage. The implementation's
numerical guards are listed separately below; they are not calibrated
validity ranges. The [validation page](../../50-validation/environment-imf.md)
records the named anchors and derivative checks. Wider empirical domains
remain unqualified.

```{list-table}
:header-rows: 1

* - Parameter
  - `env_to_imf_params` guard (`clamp_domain=True`)
  - Scope
* - [Fe/H]
  - $[-2.5, +0.5]$
  - Input clipped; low-mass extrapolation especially uncertain below $-0.5$
* - $M_{\mathrm{ecl}}$
  - $\log_{10}(M_{\mathrm{ecl}}/\Msun)\in[3,8]$
  - Logarithmic mass input clipped
* - $\rho_{\mathrm{cl}}$
  - No clipping of an explicit density override
  - Otherwise computed from mass, radius and SFE
* - SFE $\varepsilon$
  - No clipping or validation
  - Caller supplies a physically meaningful fraction
```

`BirthEnvironment` and the individual helpers do not provide a complete
physical-domain validator. Positive mass and density and a meaningful SFE
remain caller responsibilities. `clamp_domain=False` disables the mapping's
metallicity and mass-input clips; the slope helpers retain their output
clips and low-mass floors. The public API is `env_to_imf_params()` plus the
`alpha3_*` / `x_*` helpers.

## Implementation in progenax

The [IMF-sampling tutorial](../../00-getting-started/imf-sampling.md#environment-dependent-slopes)
constructs all nine supported mappings and retains the seed-0, 10000-draw
Marks-plane example at $M_{\rm ecl}=10^6\,\Msun$ and $[\mathrm{Fe/H}]=-1.5$.
It passes all four slopes and three breaks to `PowerLawIMF`, using the
returned $0.01$–$150\,\Msun$ bounds. The environment mass selects slopes;
a fixed-count draw does not constrain the catalogue's total mass.

Within active branch interiors, gradients flow from `IMFParams` back to
`BirthEnvironment`. The default Fundamental Plane uses `jnp.where` with the
hard threshold in {eq}`fundamental-fit`; its derivative is not defined at the
jump, and saturated branches have zero local slope. Set `smooth_alpha3=True`
in `env_to_imf_params` to choose the tanh-blended mapping near the threshold;
the individual `alpha3_*` helpers use `smooth=True`.
Output clips still introduce boundaries, and mapping flags should remain
static under JIT. These local derivative contracts do not qualify recovery
of all birth parameters from a mass catalogue.

See [](../../30-api/imf.md) for the full API and
[](../../50-validation/environment-imf.md) for the published-table and
implementation derivative checks.

## Connection to other chapters

The environment IMF feeds into:

- [](binary.md) — binary-aware likelihood, where the IMF $\alpha$ is
  inferred from observed system masses. Misspecifying the
  *environment* (using a universal IMF for a metal-poor cluster) adds
  bias on top of the binary contamination bias. They compose: a naive
  single-star, universal-IMF fit to a metal-poor binary-rich cluster
  returns an $\alpha$ that is biased by both effects.
- [](../gravoturbulence/bm19.md) — the BM19 dense-gas SFR framework
  predicts cloud/ECMF-scale properties that the IGIMF folds into a
  galaxy-wide IMF. progenax implements the cluster-scale link (cloud
  properties $\to \alpha_3$); the galaxy-wide ECMF integration is
  background theory, not an implemented end-to-end chain.

## Implementation, validation & references

- **In code:** the environment package is `src/progenax/imf/environment/`
  — `birth_environment.py` (`BirthEnvironment`), `mapping.py`
  (`env_to_imf_params`, the `alpha3_*` / `x_*` helpers), `coefficients.py`,
  and `density.py`. There is **no** `EnvironmentIMF` class; the public API
  is functional. See the [IMF API](../../30-api/imf.md).
- **Validated in:** [environment IMF](../../50-validation/environment-imf.md)
  — named published-table anchors, the retained $-0.87$ threshold,
  and bounded derivative checks, each with its recorded criterion.
- **Primary sources:** the cluster-scale variation follows
  {cite:t}`Marks2012` (with the {cite:t}`Marks2014` erratum); the IGIMF
  aggregation follows {cite:t}`Jerabkova2018`; the {cite:t}`Kroupa2001`
  broken-power-law baseline anchors the canonical $(\alpha_1, \alpha_2,
  \alpha_3) = (1.3, 2.3, 2.3)$ reference. Full notes in the
  [Marks et al. (2012)](../../99-bibliography/per-paper/marks-2012.md)
  per-paper page.

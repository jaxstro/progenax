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
cluster-IMF input to the IGIMF framework) as fully differentiable models.
The galaxy-wide IGIMF *integral* itself is background theory here, not an
implemented sampler — see the scope note below.

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers learning how the IMF varies with cloud-core density and metallicity, and what progenax actually implements (the cluster-scale mapping, not the galaxy-wide integral); no prior IGIMF literature assumed.
**Prerequisites:** the [classical IMFs](classic.md) (the universal $(\alpha_1, \alpha_2, \alpha_3)$ baseline this varies).
**You'll get:** the Marks+2012 high-mass-slope relations, the Fundamental-Plane $\alpha_3(\rho_{\mathrm{cl}}, [\mathrm{Fe/H}])$ fit (with the corrected $-0.87$ threshold), where the IGIMF framework fits as background theory, and the differentiable `env_to_imf_params` API.
:::

## Two physical drivers of IMF variation

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

```{warning}
{cite:t}`Marks2012` and {cite:t}`Jerabkova2018` use a **3-segment**
labelling that omits the brown-dwarf segment, while progenax uses the
**4-segment** form internally. Misalignment of indices between the two
conventions has produced subtle bugs in past implementations.
```

The progenax 4-segment IMF:

```{math}
:label: imf-4-segment
\xi(m) \;=\; k\,a_i\,m^{-\alpha_i}
\quad\text{for}\quad m_{i-1} < m \le m_i,
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

Continuity coefficients $a_i$ are computed from $a_0 = 1$ via
$a_i = a_{i-1}\,m_i^{\alpha_i - \alpha_{i-1}}$. progenax's
environment code currently exposes `BirthEnvironment`,
`env_to_imf_params`, and the `alpha3_*` helper functions; it does not
export an `EnvironmentIMF` class. The brown-dwarf segment $\alpha_0$ is held fixed at
$0.3$ regardless of environment, reflecting the lack of strong
observational constraints on sub-stellar IMF variation.

The upper-mass cutoff is $m_{\max,\star} \approx 150\,\Msun$, with an
optional cluster-mass-dependent $m_{\max}$ via the Weidner-Kroupa
relation when the cluster IMF (rather than individual stars) is being
sampled.

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

```{list-table} {cite:t}`Marks2012` Table 3 — linear fit coefficients (page 2251).
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
agreement. progenax's regression suite locks every Marks+12 Table 4
entry to within $10^{-3}$.

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
`include_lowmass_variation: bool = False`. When using it, clamp
[Fe/H] to $[-2.5, +0.5]$ to avoid unphysical slopes (e.g.
$\alpha_1 < 0$ at [Fe/H] $= -3$).
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

```{math}
:label: fundamental-fit
\alpha_3(\hat x) \;=\;
\begin{cases}
  -0.4072\,\hat x + 1.9383, & \hat x \ge -0.87 \\
  2.3, & \hat x < -0.87
\end{cases}
```

```{warning}
**The threshold is $\hat x \ge -0.87$ (negative).** The *printed* {cite:t}`Marks2012`
Eq. 14/15 (p. 2252) give "$\hat x \ge +0.87$", but this is a missing-minus-sign typo
corrected by the {cite:t}`Marks2014` erratum (MNRAS **442**, 3315; see the
[Marks et al. (2012) per-paper note](../../99-bibliography/per-paper/marks-2012.md)).
With the corrected $-0.87$, the fitted line
meets the canonical $\alpha_3 = 2.3$ *continuously* at the knee ($\hat x \approx -0.888$).
The uncorrected $+0.87$ is not merely shifted — it is **discontinuous**, spuriously pinning
$\alpha_3 = 2.3$ out to $\hat x = +0.87$ and then dropping to $\alpha_3 \approx 1.58$.
progenax's `alpha3_marks_plane` helper encodes the corrected $-0.87$ threshold; see
[](../../50-validation/environment-imf.md) for the erratum-vs-typo comparison and
`tests/unit/imf/` for coverage.
```

Substituting the rotation back yields the expanded form ({cite:t}`Marks2012`
Eq. 15):

```{math}
:label: alpha3-marks-eq15
\alpha_3(\log\rho_{\mathrm{cl}}, [\mathrm{Fe/H}])
\;=\; 0.0572\,[\mathrm{Fe/H}] - 0.4072\,\log_{10}\!\left(\frac{\rho_{\mathrm{cl}}}{10^6\,\Msun\,\mathrm{pc}^{-3}}\right) + 1.9383
```

valid when $\hat x \ge -0.87$, with $\alpha_3 = 2.3$ otherwise.

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

```{admonition} Density convention departure from Jeřábková+2018
:class: note
{cite:t}`Jerabkova2018` Eq. 9 prints a coefficient of 2.83 in the
mass-based $\hat x$ formula. Under the $8\pi$ half-mass density
convention used by {cite:t}`Marks2012` (which reproduces their Table 1
exactly), the consistent coefficient is **0.2161**. progenax adopts the
internally-consistent {cite:t}`Marks2012` convention; the 2.83 in
Jeřábková+2018 Eq. 9 likely reflects a different density convention or
a typo. progenax also exposes the star-formation efficiency $\varepsilon$
explicitly so the cluster-mass scaling is auditable. See `env_to_imf_params`
and the `alpha3_jerabkova_*` helpers (`progenax.imf.environment.mapping`)
for the conversion.
```

## Domain of validity

The fits in this chapter are derived from samples with limited
parameter coverage. Extrapolation outside these ranges yields predictions
of unknown reliability:

```{list-table}
:header-rows: 1

* - Parameter
  - Valid range
  - Notes
* - [Fe/H]
  - $[-2.5, +0.5]$
  - Low-mass slope extrapolation especially uncertain below $-0.5$
* - $M_{\mathrm{ecl}}$
  - $[10^4, 10^7]\,\Msun$
  - GC/UCD regime; smaller clusters scatter
* - $\rho_{\mathrm{cl}}$
  - $[10^4, 10^8]\,\Msun\,\mathrm{pc}^{-3}$
  - Dense embedded-cluster regime
* - SFE $\varepsilon$
  - $[0.1, 0.7]$
  - Below 0.1 clusters unbind; above 0.7 is extreme
```

The current implementation exposes differentiable helper functions for
these relations. There is **no** `EnvironmentIMF` class; the public API
is the functional `env_to_imf_params()` (which takes a `clamp_domain`
flag, default `True`, to clip inputs to the calibrated ranges above)
plus the `alpha3_*` / `x_*` helpers.

## Implementation in progenax

```python
from progenax.imf import BirthEnvironment, env_to_imf_params, PowerLawIMF

env = BirthEnvironment.from_cluster_mass(
    M_ecl=1e6,
    FeH=-1.5,
)
params = env_to_imf_params(env, model="marks_plane")

imf = PowerLawIMF(
    exponents=[params.alpha0, params.alpha1, params.alpha2, params.alpha3],
    breakpoints=[params.m_break0, params.m_break1, params.m_break2],
    m_min=params.m_min,
    m_max=params.m_max,
)
masses = imf.sample(key, 10000)
```

The `env_to_imf_params` mapping is differentiable in $\rho_{\mathrm{cl}}$,
[Fe/H], and $M_{\mathrm{ecl}}$ (and the SFE $\varepsilon$); gradients flow
from `IMFParams` back to `BirthEnvironment`. The Fundamental Plane threshold {eq}`fundamental-fit` uses
`jax.nn.sigmoid` rather than `jnp.where` to keep gradients flowing
through the threshold region — this matters when fitting $\alpha_3$
directly to data near the boundary $\hat x = -0.87$.

See [](../../30-api/imf.md) for the full API and
[](../../50-validation/imf-statistics.md) for the regression suite that
locks {cite:t}`Marks2012` Table 4 + Eq. 15 numerical values.

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
  — the regression suite that locks Marks+12 Table 4 + Eq. 15 to within
  $10^{-3}$ and pins the corrected $-0.87$ Fundamental-Plane threshold.
- **Primary sources:** the cluster-scale variation follows
  {cite:t}`Marks2012` (with the {cite:t}`Marks2014` erratum); the IGIMF
  aggregation follows {cite:t}`Jerabkova2018`; the {cite:t}`Kroupa2001`
  broken-power-law baseline anchors the canonical $(\alpha_1, \alpha_2,
  \alpha_3) = (1.3, 2.3, 2.3)$ reference. Full notes in the
  [Marks et al. (2012)](../../99-bibliography/per-paper/marks-2012.md)
  per-paper page.

---
title: Binary-aware IMF recovery
description: How unresolved binaries bias inferred IMF slopes, and how progenax marginalises over them with the Moe & Di Stefano (2017) statistics.
---

# Binary-aware IMF recovery

The stellar initial mass function (IMF) governs chemical enrichment,
supernova rates, and the integrated light of galaxies. Measuring its
shape — and whether it varies with environment — is one of the central
problems in stellar astrophysics {cite:p}`Salpeter1955,Kroupa2001,Chabrier2003,Marks2012,Jerabkova2018`.
Most stars do not form alone: roughly 50% of solar-type stars and >90%
of O-type stars have at least one bound companion {cite:p}`Sana2012,MoeDiStefano2017`.
When those companions are unresolved, the distribution of observed
*system* masses differs from the primary-mass distribution. The resulting
slope bias depends on the IMF, binary statistics, support and selection.

This chapter is the single source of truth for the **binary-aware IMF
recovery** calculation built from the `progenax.imf.BinaryIMF` sampler and exercised
in `scripts/validate_binaries.py` (fast MLE) and
`validation/imf/validate_binary_aware_recovery.py` (historical NumPyro NUTS).
The likelihood is script-owned rather than a public `BinaryIMF` method.
This chapter derives the system mass
function under a period-averaged reduction of the {cite:t}`MoeDiStefano2017` statistics, formulates a
likelihood that marginalises over latent binary status and mass ratio, and
explains why a misspecified single-star fit can become "**confidently
wrong**" as its uncertainty shrinks. The retained MLE examples and
historical posterior diagnostics belong to the
[validation page](../../50-validation/binary-imf.md).

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers learning why unresolved binaries bias inferred IMF slopes and how progenax accounts for them; no prior binary-statistics literature assumed.
**Prerequisites:** the [classical IMFs](classic.md) (the single-star $\xi(m \mid \alpha)$ backbone).
**You'll get:** the system mass function under the Moe & Di Stefano (2017) statistics, the marginalisation likelihood, and the "confidently wrong" regime where the naive single-star fit excludes the truth at large $N$.
:::

```{figure} ../figures/binary_phase_space.webp
:label: fig-binary-phase-space
:width: 100%

Binaries in $(r, v_r)$ phase space, and why they lie to your dispersion
(seed 21, $4\times10^4$ systems each; symlog wings beyond
$|v_r| = 11$ pc/Myr). **Left column (resolved):** plum points are
**envelope violators** — stars above the local single-star
$\pm v_{\rm esc}$ that cannot be bound singles: 2,873 (EFF + Moe, top) and
1,689 (King + $f_b = 0.15$, bottom) contaminants reaching hundreds of
pc/Myr, inflating $\sigma_{1\mathrm{d}}$ to 8.4 (singles: 2.2) and 7.9
(singles: 1.5). **Right column (unresolved):** each binary photometrically
blended (COM position, Tout-ZAMS flux-weighted velocity) — the violator
clouds all but vanish and the population *looks* clean, yet the measured
dispersion stays silently inflated (2.93 vs 2.20; 2.41 vs 1.53) — and the
few blend hexes still leaking past the envelope are near-equal-luminosity
pairs that keep half their orbital velocity. This figure shows velocity
contamination; the IMF-recovery calculation below instead uses system-mass
addition. These are distinct observation operators.
Regenerate: `python -m laboratory.icviz --only binary-phase-space`.
```

## The Maschberger (2013) functional form

progenax uses the {cite:t}`Maschberger2013` "L3 distribution" as the
single-star IMF backbone:

```{math}
:label: maschberger
f(m) \;\propto\; \left(\frac{m}{\mu}\right)^{-\alpha}
                \left[\,1 + \left(\frac{m}{\mu}\right)^{1-\alpha}\right]^{-\beta}
```

with default parameters $\alpha = 2.3$ (Salpeter high-mass slope), $\beta
= 1.4$, $\mu = 0.2\,\Msun$. The high-mass limit recovers $f \propto
m^{-\alpha}$; the second factor produces the observed turnover below
$\sim 0.3\,\Msun$. Crucially, the cumulative distribution has a closed-form
inverse, enabling exact analytical sampling — no numerical root-finding,
compatible with fixed-shape JAX execution and smooth-parameter derivatives
on its valid domain. See
[](../../30-api/imf.md) for the implementation in
`progenax.imf.Maschberger`.

```{note}
**Single-star vs. system IMF.** The single-star IMF $\xi(m \mid \alpha)$
describes the birth mass distribution of *individual stars*. The system
mass function describes the distribution of *observed system masses*,
including unresolved multiples. {cite:t}`Maschberger2013` Table 1 lists
distinct canonical parameter values for the two — they are not the same
function. progenax uses $\xi(m \mid \alpha)$ throughout for the
*primary-star* IMF; the system mass function is derived from it via the
binary contamination model below.
```

## The binary contamination problem

For an unresolved binary system, the observer measures the system mass

```{math}
:label: msys
m_{\mathrm{sys}} \;=\;
\begin{cases}
  m_1 & \text{single, with probability } 1 - f_b(m_1) \\
  m_1(1+q) & \text{binary, with probability } f_b(m_1)
\end{cases}
```

where $q = m_2/m_1 \in [q_{\min}, 1]$ and $f_b(m_1)$ is the mass-dependent
binary fraction. Binaries transfer probability mass from intermediate to
high system masses — a primary $m_1 = 5\,\Msun$ with $q = 0.5$ appears as
a single $7.5\,\Msun$ system. This changes the system-mass distribution,
but does not imply a shallower slope in every regime. For a scale-free
primary density $\xi(m)=C m^{-\alpha}$, mass-independent binary probability
$f_b$ and mass-ratio law $g(q)$, away from support boundaries,

```{math}
p_{\mathrm{sys}}(M)=C M^{-\alpha}
\left[(1-f_b)+f_b\int_{q_{\min}}^1(1+q)^{\alpha-1}g(q)\,\mathrm{d}q\right].
```

Here $C$ is the primary-density coefficient; the bracket is independent of
$M$, so the local slope is unchanged. Turnover, finite mass bounds,
mass-dependent binary statistics and observational selection can produce
a slope bias, as in the retained recovery example.

The naive practice of fitting a single-star IMF to the observed system
masses — likelihood $\mathcal{L}_{\mathrm{naive}}(\alpha) = \prod_i
\xi(m_{\mathrm{sys},i} \mid \alpha)$ — can converge as $N\to\infty$ to a
value of $\alpha$ displaced from the generating primary slope. {cite:t}`Kroupa2001` quantified the bias at
$\Delta\alpha \sim 0.05$–$0.10$ depending on mass range and binary
fraction. Section [](#confidently-wrong) below explains why model bias
can persist as statistical uncertainty shrinks. The transition sample
size depends on the IMF, binary statistics, observable and selection.

## The Moe & Di Stefano (2017) joint binary statistics

{cite:t}`MoeDiStefano2017` compiled the most comprehensive census of binary star
properties to date, combining spectroscopy, eclipsing binaries, long-baseline
interferometry, adaptive optics, and common proper motion surveys. After
correcting each sample for its selection function, they derived the
*intrinsic* joint distribution $f(M_1, q, P, e)$ of primary mass, mass
ratio, orbital period, and eccentricity. Two findings shape progenax's
implementation:

1. **Binary properties are not universal.** Binary fraction, mass-ratio
   distribution, and period distribution all vary systematically with
   $M_1$. Treating them as constants introduces correlated errors.

2. **The joint distribution is not separable:** $f(M_1, q, P, e) \neq
   f(M_1)\,f(q)\,f(P)\,f(e)$. The mass-ratio distribution depends on both
   $M_1$ and $P$; eccentricity depends on $P$; companion frequency
   depends on both.

### Binary fraction $f_b(M_1)$

The mass-dependent binary fraction $f_b(M_1) = 1 - \mathcal{F}_{n=0}$ —
the multiplicity *fraction* ($\le 1$, the probability of $\ge 1$
companion) — is tabulated, with its full per-mass values and provenance
({cite:t}`MoeDiStefano2017` Table 13 single-star row for
$M_1 \ge 0.8\,\Msun$, M-dwarf surveys below), in the canonical
[multiplicity-statistics chapter](multiplicity-statistics.md#mass-dependent-binary-fraction).
The trend is steep: O-type stars are almost always in multiples ($f_b
\approx 0.9$), while M-dwarfs are mostly single ($f_b \approx 0.2$–0.3) —
a factor-of-4 dynamic range that is the leading-order driver of the
binary-IMF bias.

This $f_b$ is the multiplicity *fraction*, **not** the companion
*frequency* $f_{\mathrm{mult}}$, which exceeds 1.0 ($\sim 2.1$ per
O-type primary, {cite:t}`MoeDiStefano2017` Table 13) once hierarchical
triples are counted — see [Limitations](#binary-imf-limitations) below
for the single-companion approximation progenax adopts.

### Mass-ratio distribution $g(q \mid M_1)$

For a primary of mass $M_1$, progenax's reduced `MoeDiStefano2017` model
approximates the source statistics with a power law plus a narrow
Gaussian component near $q=1$:

```{math}
:label: gq
p(q \mid M_1) \;=\; (1 - f_{\mathrm{twin}})\,\frac{q^{\gamma(M_1)}}{Z_{\mathrm{pl}}}
                  \;+\; f_{\mathrm{twin}}\,
                  \frac{\mathcal{N}(q \mid 1,\, \sigma_{\mathrm{twin}})}{Z_{\mathrm{twin}}},
\qquad q\in[q_{\min},1].
```

where $Z_{\mathrm{pl}} = \int_{q_{\min}}^{1} q^\gamma\,\mathrm{d}q$ and
$Z_{\mathrm{twin}}=\int_{q_{\min}}^1
\mathcal{N}(q\mid1,\sigma_{\mathrm{twin}})\,\mathrm{d}q$ normalizes
the truncated twin Gaussian, and $\sigma_{\mathrm{twin}}=0.03$ is
its dimensionless width. The density is zero outside this support.
progenax uses a period-averaged single-slope
$\gamma(M_1)$ — a reduction of {cite:t}`MoeDiStefano2017` Table 13's period-dependent
$\gamma_{\mathrm{smallq}}/\gamma_{\mathrm{largeq}}$ (averaged over $\log P$) — appropriate
for total mass-function analyses, but losing the small-$q$ structure that matters
for surveys sensitive to specific period ranges (see
[Limitations](#binary-imf-limitations)). The per-mass $\gamma(M_1)$ and
$f_{\mathrm{twin}}(M_1)$ tables live in the canonical
[](mass-ratio-distributions.md).

Qualitatively, the slope trend reverses at high masses: massive stars
preferentially have *low-q* companions ($\gamma < 0$), while low-mass
stars prefer near-equal-mass companions ($\gamma > 0$). The twin
fraction $f_{\mathrm{twin}}$ peaks at solar-type primaries
($f_{\mathrm{twin}} = 0.10$ for $0.8$–$1.2\,\Msun$) and falls to 0.03
for $M_1 > 3.5\,\Msun$.

## The system mass function

Given the forward generative model — draw $m_1$ from the IMF, decide
binary or not, add $m_2$ if binary — the probability density of the
observed system mass $M$ is

```{math}
:label: psys
p_{\mathrm{sys}}(M) \;=\;
\underbrace{\xi(M)\,(1 - f_b(M))}_{\text{single-star contribution}}
\;+\;
\underbrace{\int_{M/2}^{\;M/(1+q_{\min})}
   \frac{\xi(m_1)\, f_b(m_1)\, g\!\left(\tfrac{M}{m_1}-1 \,\big|\, m_1\right)}{m_1}\,\mathrm{d}m_1}_{\text{binary contribution}}.
```

The binary term integrates over every primary mass $m_1$ that could
produce the observed system mass $M$, weighted by the IMF, the binary
fraction, and the mass-ratio distribution. The integration limits come
from $q_{\min} \le q \le 1$: $q \ge q_{\min}$ implies $m_1 \le
M/(1+q_{\min})$; $q \le 1$ implies $m_1 \ge M/2$. The Jacobian $1/m_1$
arises from the change of variables $q = M/m_1 - 1$.
Here $\xi$ is a normalized primary-mass probability density per solar
mass, zero outside $[m_{\min},m_{\max}]$; the actual integration bounds
intersect this interval with the displayed candidate-primary bounds.

The scripts evaluate this piecewise-smooth integral with 128-point
Gauss-Legendre quadrature. Its accuracy needs an independent convergence
comparison; the node count does not provide a universal error bound.

## The binary-aware likelihood

For an observed system mass $M$, let $m_1=M/(1+q)$ be the candidate
primary mass at each mass-ratio value $q$:

```{math}
:label: likelihood
p(M \mid \alpha) \;=\;
\underbrace{[1 - f_b(M)]\,\xi(M \mid \alpha)}_{\text{single}}
\;+\;
\underbrace{\int_{q_{\min}}^{1}
  \frac{\xi(m_1\mid\alpha)\,f_b(m_1)\,g(q\mid m_1)}{1+q}
  \,\mathrm{d}q}_{\text{binary}}.
```

The binary fraction is evaluated at each candidate primary mass **inside**
the integral. For a constant fraction it can be factored out; a population
average cannot replace the conditional weight for a mass-dependent model.

For $N$ observed systems and 128 quadrature points, each likelihood
evaluation carries $N\times128$ candidate primaries. At $N=30{,}000$
this is $3.84\times10^6$ candidates, not a measured execution time or
NUTS chain cost. See [](binary-aware-likelihood.md) for the numerical
calculation and its current limits.

## Confidently wrong: the naive likelihood at large N

(confidently-wrong)=

The naive likelihood $\mathcal{L}_{\mathrm{naive}}(\alpha)=\prod_i
\xi(M_i\mid\alpha)$ omits binaries from the data-generating model.
For an identifiable, correctly specified likelihood with suitable
regularity, uncertainty decreases approximately as $N^{-1/2}$ around
the true parameter. A misspecified fit can instead concentrate around
a different value. Its interval narrows while the model bias persists.

For the retained mass-addition example, the naive slope is lower than
the generating slope: $\Delta\alpha=\hat\alpha-\alpha_{\mathrm{true}}<0$.
The [validation owner](../../50-validation/binary-imf.md) holds the
MLE figures and their observed-curvature $1\sigma$ intervals, plus the
separate historical posterior diagnostics. These are bounded examples,
not a universal transition at a particular survey size. Marginalizing
binaries accounts for the selected generative model; finite-sample bias,
quadrature error and misspecified binary statistics still require checks.

## Observation operators: what does the telescope measure?

The likelihood {eq}`likelihood` uses the dynamical-mass operator
$m_{\mathrm{sys}} = m_1 + m_2$. This describes a total mass when a dynamical analysis constrains it;
an SB2 radial-velocity orbit alone also depends on inclination.
Photometric surveys instead measure combined flux, requiring a
different observation model.

For a main-sequence luminosity scaling $L \propto m^s$ with $s \approx
3$–4, the inferred photometric mass of an unresolved binary is

```{math}
:label: mphot
m_{\mathrm{phot}} \;\approx\; m_1 \,(1 + q^s)^{1/s}
```

Because $s \gtrsim 3$, the secondary contributes appreciably to the flux
only when $q \to 1$:

```{list-table} Photometric vs. dynamical mass for $s = 3.5$.
:header-rows: 1

* - $q$
  - $m_1 + m_2$
  - $m_{\mathrm{phot}}$ ($s=3.5$)
  - Comment
* - 1.0
  - $2.0\,m_1$
  - $1.22\,m_1$
  - 0.75 mag brighter
* - 0.5
  - $1.5\,m_1$
  - $1.0245\,m_1$
  - Companion barely changes flux
* - 0.3
  - $1.3\,m_1$
  - $1.0042\,m_1$
  - Negligible photometric effect
```

For this fixed main-sequence power-law luminosity model with $s>1$,
$m_1 \le m_{\mathrm{phot}} \le m_1+m_2$: the per-system mass shift is
smaller than under mass addition. This ordering does not establish an
upper bound on inferred slope bias or a universal sample-size transition;
those depend on the likelihood, support and survey selection. Full
LSST-realistic inference replaces $m_1 + m_2$ with isochrone-mediated
flux addition; see the
[](../../40-howto/add-binary-population.md) recipe for the planned
extension.

(binary-imf-limitations)=
## Limitations

```{admonition} Observation operator
:class: note
Mass addition and photometric flux addition define different observables.
The simple luminosity power law bounds the per-system mass shift,
not the inferred slope bias. Survey inference requires its own forward model.
```

```{admonition} Single-companion approximation
:class: note
We model only singles and binaries. ~10% of solar-type systems are
triples, rising to >50% for O-type primaries
{cite:p}`Sana2012,MoeDiStefano2017`. For O-type stars {cite:t}`MoeDiStefano2017` report
companion frequency $\sim 2.1$ per primary; our single-companion model
compresses this into one effective binary, *underestimating* the
high-mass distortion.
```

```{admonition} Period-averaged mass-ratio distribution
:class: note
progenax uses period-averaged $\gamma$ and $f_{\mathrm{twin}}$ reduced from
{cite:t}`MoeDiStefano2017` Table 13. Surveys sensitive to specific period ranges
(spectroscopic = short-period, visual = wide) need the full
period-conditional $g(q \mid M_1, P)$.
```

```{admonition} Resolved-vs-unresolved selection function
:class: note
Whether a binary is resolved depends on projected separation (period,
phase, orientation and distance), instrument PSF, crowding and contrast
ratio. Treating every binary as unresolved changes the selection model;
the sign and size of any slope bias require the actual survey operator.
```

```{admonition} Primary IMF vs. all-stars IMF
:class: note
progenax draws *primaries* from the IMF and generates secondaries
conditionally; the resulting $\{m_1, m_2\}$ distribution does not equal
$\xi(m)$ in general (twin excess biases secondaries toward the primary
mass). Always be explicit which definition is in use.
```

```{admonition} Metallicity dependence
:class: note
{cite:t}`MoeDiStefano2017` are calibrated primarily on solar-metallicity samples.
Low-metallicity environments may have higher binary fractions
{cite:p}`Moe2019`. progenax's "Low-Z" and "Starburst" environment
presets currently use the same Moe+17 statistics as Solar — likely an
underestimate of the binary contamination.
```

## Implementation in progenax

The pipeline lives in three modules:

- `progenax.imf.BinaryIMF` — the {cite:t}`MoeDiStefano2017`-conditioned
  forward sampler. Fixed-draw continuous paths can be differentiated;
  hard binary assignment is discrete, and `sample_all_masses` uses a
  variable-length selection that is not directly JIT-compatible.
- `progenax.binaries.MoeEccentricity` and friends — the
  period/eccentricity models used by the companion-composition layer (see
  [](../binaries/period-distributions.md)).
- `scripts/validate_binaries.py` — the fast (MLE, no-MCMC) figure suite that
  regenerates the "$N$ vs. CI vs. bias" result; `validation/imf/validate_binary_aware_recovery.py`
  is the historical NumPyro NUTS cross-check. New recovery qualification
  uses BlackJAX after an explicit likelihood/convergence design.

See [](../../30-api/imf.md) for full API signatures and
[](../../40-howto/add-binary-population.md) for a recipe building a
binary-corrected mock catalog end-to-end.

## Implementation, validation & references

- **In code:** `src/progenax/imf/binary/imf.py` (`BinaryIMF` — the
  forward model and sampling helpers), with the joint statistics in
  `src/progenax/imf/binary/moe_di_stefano.py` and the single-star
  backbone in `src/progenax/imf/smooth.py` (`Maschberger`) — see the
  [IMF API](../../30-api/imf.md). The recovery figures are regenerated by
  `scripts/validate_binaries.py` (fast MLE) and
  `validation/imf/validate_binary_aware_recovery.py` (historical NumPyro NUTS).
- **Validated in:** [binary-aware recovery](../../50-validation/binary-imf.md)
  — retained MLE figures and historical two-chain diagnostics, with
  quadrature accuracy and repeated-catalogue coverage still open.
- **Primary sources:** the single-star backbone is {cite:t}`Maschberger2013`;
  the binary-contamination statistics are
  {cite:t}`Sana2012,MoeDiStefano2017,Moe2019`; the "confidently wrong"
  framing is original to progenax. Full notes in the
  [bibliography](../../99-bibliography/per-paper/moe-distefano-2017.md).
  For the environment-dependent IMF that consumes $\alpha$ from this
  chapter, see [](environment.md).

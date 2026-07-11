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
When those companions are unresolved, the IMF inferred from observed
*system* masses is systematically biased toward shallower slopes.

This chapter is the single source of truth for the **binary-aware IMF
recovery** pipeline implemented in `progenax.imf.BinaryIMF` and exercised
in `scripts/validate_binaries.py` (fast MLE) and
`validation/imf/validate_binary_aware_recovery.py` (full numpyro NUTS). It derives the system mass
function under the {cite:t}`MoeDiStefano2017` joint binary statistics, formulates a
likelihood that marginalises over latent binary status and mass ratio, and
quantifies the regime — sample size $N \gtrsim 10^4$ — where the naive
single-star likelihood becomes "**confidently wrong**": its 95% credible
interval shrinks below the bias and excludes the true slope.

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers learning why unresolved binaries bias inferred IMF slopes and how progenax removes the bias; no prior binary-statistics literature assumed.
**Prerequisites:** the [classical IMFs](classic.md) (the single-star $\xi(m \mid \alpha)$ backbone).
**You'll get:** the system mass function under the Moe & Di Stefano (2017) statistics, the marginalisation likelihood, and the "confidently wrong" regime where the naive single-star fit excludes the truth at large $N$.
:::

```{figure} ../figures/binary_phase_space.webp
:label: fig-binary-phase-space
:width: 100%

Binaries in $(r, v_r)$ phase space, and why they lie to your dispersion
(seed 21, $4\times10^4$ systems each; symlog velocity axis). **Left column
(resolved):** binary orbital motion (coral) punctures the single-star escape
envelope by up to *two orders of magnitude* — the classic high-velocity
contaminants. Measured $\sigma_{1\mathrm{d}}$: 8.4 vs a singles-only 2.2
(EFF + Moe, top) and 7.9 vs 1.5 (King + $f_b = 0.15$, bottom).
**Right column (unresolved):** each binary photometrically blended (COM
position, Tout-ZAMS flux-weighted velocity) — the punctures vanish, the
population *looks* bound and clean, but the measured dispersion is still
silently inflated (2.93 vs 2.20; 2.41 vs 1.53). That hidden pedestal is
exactly the mechanism behind the binary-blind mass bias demonstrated on this
page. Regenerate: `python -m laboratory.icviz --only binary-phase-space`.
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
fully `jax.lax.scan`-compatible, fully differentiable. See
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
a single $7.5\,\Msun$ system. Consequently the observed system mass
function is *shallower* than the true single-star IMF.

The naive practice of fitting a single-star IMF to the observed system
masses — likelihood $\mathcal{L}_{\mathrm{naive}}(\alpha) = \prod_i
\xi(m_{\mathrm{sys},i} \mid \alpha)$ — converges as $N\to\infty$ to the
*wrong* value of $\alpha$. {cite:t}`Kroupa2001` quantified the bias at
$\Delta\alpha \sim 0.05$–$0.10$ depending on mass range and binary
fraction. Section [](#confidently-wrong) below shows that the bias does
not shrink with $N$, and at LSST-scale samples ($N \sim 10^{4}$–$10^{5}$)
the naive 95% credible interval becomes narrower than the bias and
*excludes the truth*.

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

For a primary of mass $M_1$, {cite:t}`MoeDiStefano2017` parameterise $g(q \mid M_1)$
as a power law plus a narrow "twin excess" peak near $q = 1$:

```{math}
:label: gq
p(q \mid M_1) \;=\; (1 - f_{\mathrm{twin}})\,\frac{q^{\gamma(M_1)}}{Z_{\mathrm{pl}}}
                  \;+\; f_{\mathrm{twin}}\,\mathcal{N}(q \mid 1,\, \sigma_{\mathrm{twin}})
```

where $Z_{\mathrm{pl}} = \int_{q_{\min}}^{1} q^\gamma\,\mathrm{d}q$ and
$\sigma_{\mathrm{twin}} \approx 0.03$. progenax uses a period-averaged single-slope
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

This integral has no closed form because $f_b(m_1)$ and $g(q \mid m_1)$
are piecewise functions of $M_1$. progenax evaluates it via 128-point
Gauss-Legendre quadrature, which gives ~14 digits of accuracy for smooth
integrands.

## The binary-aware likelihood

For an observed system mass $M$:

```{math}
:label: likelihood
p(M \mid \alpha) \;=\;
\underbrace{(1 - \bar{f}_b)\,\xi(M \mid \alpha)}_{\text{single}}
\;+\;
\underbrace{\bar{f}_b \int_{q_{\min}}^{1}
  \frac{\xi\!\left(\tfrac{M}{1+q} \,\big|\, \alpha\right)}{1+q}\,
  g\!\left(q \,\big|\, \tfrac{M}{1+q}\right)\,\mathrm{d}q}_{\text{binary}}
```

where $\bar{f}_b$ is evaluated at the candidate primary mass $m_1 =
M/(1+q)$ inside the integral. The integrand is smooth in $q$, so
Gauss-Legendre quadrature on $[q_{\min}, 1]$ converges fast.

Computational cost. For $N$ observed systems and 128 quadrature points,
each likelihood evaluation requires $N \times 128$ IMF + mass-ratio
evaluations. At $N = 30{,}000$ (LSST-scale) this is $\sim 3.8 \times 10^6$
evaluations per NUTS step, $\sim 6 \times 10^9$ flops per chain. On a
MacBook Pro CPU one chain takes $\sim 35$ minutes; on an A100 the same
$N \times 128$ tile is embarrassingly parallel via `jax.vmap`, bringing
the per-chain time down to $\sim 30$ s — an estimated $\sim 70\times$
speedup (see [](binary-aware-likelihood.md) for the per-step breakdown).

## Confidently wrong: the naive likelihood at large N

(confidently-wrong)=

The naive likelihood $\mathcal{L}_{\mathrm{naive}}(\alpha) = \prod_i \xi
(M_i \mid \alpha)$ is a **misspecified** model: the data-generating
process includes binaries; the likelihood does not. As $N\to\infty$ both
naive and binary-aware posteriors shrink as $\sigma \propto 1/\sqrt{N}$
(Bernstein–von Mises), but they shrink around *different* values:

- The binary-aware posterior shrinks around $\alpha_{\mathrm{true}}$.
- The naive posterior shrinks around $\alpha_{\mathrm{true}} + \Delta\alpha$,
  where $|\Delta\alpha| \sim 0.05$–$0.10$ is set by the binary statistics
  and is independent of $N$.

The 95% credible interval shrinks; the bias does not. At some $N$ the CI
becomes narrower than the bias, and the posterior *excludes the true
value*:

```{list-table} progenax binary-aware validation results, Salpeter $\alpha = 2.35$.
:header-rows: 1

* - $N$
  - Naive 95% CI width
  - Naive $|\Delta\alpha|$
  - Status
* - 500
  - 0.28
  - 0.045
  - CI contains truth
* - 1{,}000
  - 0.20
  - 0.035
  - CI contains truth
* - 3{,}000
  - 0.12
  - 0.057
  - CI contains truth (barely)
* - 10{,}000
  - 0.06
  - **0.082**
  - **CI excludes truth**
* - 30{,}000
  - 0.035
  - **0.098**
  - **Confidently wrong**
```

This regime — narrow posterior centred on the wrong answer — is the
central danger of ignoring binaries with large datasets. progenax's
binary-aware likelihood eliminates the bias by construction; the
${\sim}128\times$ cost of the inner $q$-quadrature is the price.

## Observation operators: what does the telescope measure?

The likelihood {eq}`likelihood` uses the dynamical-mass operator
$m_{\mathrm{sys}} = m_1 + m_2$. This is exact for eclipsing binaries,
SB2 spectroscopic binaries, and astrometric binaries — observations that
recover the *true* system mass. It overstates the distortion for
*photometric* surveys, where the telescope measures combined flux rather
than combined mass.

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
  - $1.04\,m_1$
  - Companion barely changes flux
* - 0.3
  - $1.3\,m_1$
  - $1.01\,m_1$
  - Negligible photometric effect
```

The mass-addition operator $m_1 + m_2$ therefore gives an *upper bound*
on the binary distortion. Photometric masses are much less affected,
especially at moderate $q$ — but the bias does not vanish, and the naive
model still becomes confidently wrong at sufficiently large $N$. Full
LSST-realistic inference replaces $m_1 + m_2$ with isochrone-mediated
flux addition; see the
[](../../40-howto/add-binary-population.md) recipe for the planned
extension.

(binary-imf-limitations)=
## Limitations

```{admonition} Mass addition is an upper bound
:class: note
Section above. Real photometric surveys produce smaller distortion;
LSST-realistic inference requires isochrone-mediated flux addition.
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
Whether a binary is resolved depends on angular separation (period,
distance), instrument PSF, crowding, and contrast ratio. Treating all
binaries as unresolved overestimates the distortion in nearby/sparse
fields and underestimates it in distant/crowded fields.
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

- `progenax.imf.BinaryIMF` — the {cite:t}`MoeDiStefano2017`-conditioned forward
  model and likelihood. Differentiable via `jax.grad`, JIT-compatible,
  vectorisable via `jax.vmap`.
- `progenax.binaries.MoeEccentricity` and friends — the joint binary
  statistics tables used by `BinaryIMF` (see
  [](../binaries/period-distributions.md)).
- `scripts/validate_binaries.py` — the fast (MLE, no-MCMC) figure suite that
  regenerates the "$N$ vs. CI vs. bias" result; `validation/imf/validate_binary_aware_recovery.py`
  is the full numpyro NUTS cross-check.

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
  `validation/imf/validate_binary_aware_recovery.py` (full numpyro NUTS).
- **Validated in:** [binary-aware recovery](../../50-validation/binary-imf.md)
  — the regression suite that demonstrates unbiased $\alpha$ recovery at
  $N = 30{,}000$ where the naive fit is confidently wrong.
- **Primary sources:** the single-star backbone is {cite:t}`Maschberger2013`;
  the binary-contamination statistics are
  {cite:t}`Sana2012,MoeDiStefano2017,Moe2019`; the "confidently wrong"
  framing is original to progenax. Full notes in the
  [bibliography](../../99-bibliography/per-paper/moe-distefano-2017.md).
  For the environment-dependent IMF that consumes $\alpha$ from this
  chapter, see [](environment.md).

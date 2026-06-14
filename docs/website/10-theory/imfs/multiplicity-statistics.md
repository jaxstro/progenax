---
title: Multiplicity statistics (Moe & Di Stefano 2017)
description: The joint binary-population statistics f(M₁, q, P, e) from Moe & Di Stefano 2017 — mass-dependent binary fractions, three period regimes, terminology pitfalls, and the survey-selection corrections that produced the calibration.
---

# Multiplicity statistics: the {cite:t}`MoeDiStefano2017` joint distribution

```{seealso}
This chapter covers the *empirical* binary statistics that progenax's
[](binary.md) framework consumes. For the likelihood that
*marginalises* over them, see [](binary-aware-likelihood.md). For how
the choice of *observation operator* (mass-add vs photometric)
controls the size of the bias, see [](observation-operators.md).
```

A practical binary-aware IMF requires knowing *the joint distribution*
of binary parameters: the probability that a primary of mass $M_1$ has
a companion at all, and conditional on having one, the joint
distribution of mass ratio $q$, orbital period $P$, and eccentricity
$e$. {cite:t}`MoeDiStefano2017` compiled the most comprehensive such census
to date, combining spectroscopic, eclipsing, long-baseline
interferometric, adaptive-optics, and common-proper-motion samples,
each carefully corrected for its own selection function. The result
is the calibration progenax depends on.

This chapter unpacks two of the most consequential findings from that
work — that binary properties are not universal, and that the joint
distribution is not separable — then catalogues the per-mass and
per-period numbers progenax stores as backing data.

## Two properties that complicate the naive picture

**Property 1: binary properties vary systematically with primary mass.**
Lower-mass stars and higher-mass stars have qualitatively different
binary populations. Treating "the binary fraction" as a constant
$f_b \approx 0.5$ (a common shortcut in earlier work) is correct on
average for solar-type primaries but wrong by factors of $\sim 2$ at
the mass-spectrum extremes.

**Property 2: the joint distribution is not separable.**

```{math}
:label: not-separable
f(M_1, q, P, e) \;\neq\; f(M_1)\,f(q)\,f(P)\,f(e)
```

The mass-ratio distribution depends on both $M_1$ *and* $P$;
eccentricity depends on $P$; companion frequency depends on both
$M_1$ and $P$. Treating these as independent factors introduces
correlated errors that are subtle but systematic — a common pitfall
in older population-synthesis work.

progenax respects this non-separability by storing the joint
$f(M_1, q, P, e)$ as a 4D table sampled per the
{cite:t}`MoeDiStefano2017` Tables 10–13 piecewise fits.

## Terminology: companion frequency vs binary fraction

```{warning}
{cite:t}`MoeDiStefano2017` are careful about three closely-related quantities;
many earlier papers conflate them. progenax's docstrings follow the
Moe & Di Stefano vocabulary exactly.
```

```{list-table}
:header-rows: 1

* - Quantity
  - Definition
  - Range
* - **Companion frequency** $f_{\log P;\,q > q_{\min}}(M_1, P)$
  - Mean number of companions per primary, per decade of $\log P$, with $q$ above some threshold
  - $\ge 0$, can exceed 1
* - **Multiplicity frequency** $f_{\mathrm{mult};\,q > 0.1}(M_1)$
  - Companion frequency integrated over all periods (still per primary, not per system)
  - $\ge 0$, can exceed 1 (= 2.1 for O-stars)
* - **Binary fraction** (multiplicity fraction)
  - Probability that a primary has *at least one* companion
  - $\le 1$ by definition
```

For O-type primaries, $f_{\mathrm{mult}} \sim 2.1$ — the typical
O-star has 2+ companions. The "binary fraction" in the sense of
"probability of at least one companion" is then close to 1, while
the *companion frequency* exceeds 2.

In simplified models (like progenax's `BinaryIMF`) that allow at most
one companion per primary, the "binary fraction" $f_b(m_1)$ is treated
as a probability — i.e. it is bounded above by 1. This is a deliberate
simplification; the implications for inferred IMF slopes are
documented at [](binary.md#binary-imf-limitations).

## Mass-dependent binary fraction

```{list-table} Multiplicity fraction $f_b = 1 - \mathcal{F}_{n=0}$ ($\le 1$; **not** the companion *frequency* $f_{\mathrm{mult}}$): {cite:t}`MoeDiStefano2017` Table 13 single-star row for $M_1 \ge 0.8\,\Msun$, M-dwarf surveys below.
:header-rows: 1

* - Primary mass
  - $f_b$
  - Stellar type
* - $M_1 < 0.1\,\Msun$
  - 0.22
  - Very low mass / brown dwarfs
* - $0.1$–$0.5\,\Msun$
  - 0.26
  - M-dwarfs
* - $0.5$–$1.0\,\Msun$
  - 0.44
  - K/G-dwarfs
* - $1.0$–$2.0\,\Msun$
  - 0.50
  - F/A-stars
* - $2.0$–$5.0\,\Msun$
  - 0.60
  - B-stars
* - $5$–$10\,\Msun$
  - 0.80
  - Early B
* - $M_1 > 10\,\Msun$
  - 0.90
  - O-stars
```

The trend is steep: M-dwarfs are mostly single, O-stars are almost
always in multiples. The factor-of-4 dynamic range in $f_b$ across
the mass spectrum is the leading-order reason that single-IMF
inferences from binary-rich populations are biased — see
[](binary.md) for the inference-side consequences.

## Three orbital-period regimes

{cite:t}`MoeDiStefano2017` identify three qualitatively distinct period regimes:

```{list-table}
:header-rows: 1

* - Regime
  - Period range
  - Properties
* - **Short**
  - $P \lesssim 20$ d
  - Tidally circularised orbits ($e \lesssim 0.4$). Modest mass ratios ($\langle q\rangle \approx 0.5$). Small twin excess.
* - **Intermediate**
  - $\log_{10}(P/\mathrm{d}) \approx 3.5$ ($a \approx 10$ AU)
  - Peak companion frequency. Mass ratios weighted toward small values ($q \approx 0.2$–$0.3$). Thermal eccentricity distribution $f(e) = 2e$.
* - **Long**
  - $\log_{10}(P/\mathrm{d}) \approx 5.5$–$7.5$ ($a \approx 200$–$5000$ AU)
  - Outer tertiary components in hierarchical triples. Mass-ratio distribution nearly consistent with random pairings drawn from the IMF.
```

The three regimes have distinct mass-ratio distributions, which is
the source of the period-conditional structure in $g(q \mid M_1, P)$.
progenax's default `BinaryIMF` uses *period-averaged* mass-ratio
parameters reduced from {cite:t}`MoeDiStefano2017` Table 13; surveys sensitive to a
specific period range (spectroscopic = short-period, visual = wide)
need a full period-conditional likelihood layer. That layer is
described conceptually in [](mass-ratio-distributions.md), but the
current `BinaryIMF` does not export a `with_period_conditional()`
constructor.

## How the calibration was derived

{cite:t}`MoeDiStefano2017` analysed dozens of binary samples, each spanning a
narrow interval of $M_1$ and $P$. For each sample they:

1. Identified the relevant **selection function** of the survey
   technique (spectroscopic surveys are biased toward large $q$ and
   short $P$; visual surveys are biased toward wide separations).
2. **Corrected** for incompleteness using the known selection
   function of each technique.
3. Fit the **intrinsic** mass-ratio distribution as a power-law plus
   twin excess (see [](mass-ratio-distributions.md)).
4. Identified a $\sim 30\%$ contamination rate from white-dwarf
   companions masquerading as main-sequence binaries in spectroscopic
   samples, and corrected for it.

The resulting joint $f(M_1, q, P, e)$ is the closest thing the field
has to a "ground truth" binary-population calibration. progenax
stores it as backing data in `progenax.imf._moe17_tables` and exposes
the high-level statistics through `BinaryIMF`'s API.

## Solar-mass calibration and the twin excess

For solar-type primaries ($0.8$–$1.2\,\Msun$), the twin fraction
$f_{\mathrm{twin}} = 0.10$ is the highest in the mass spectrum. The
twin excess is a *narrow* peak in $g(q)$ near $q = 1$ with width
$\sigma_{\mathrm{twin}} \approx 0.03$, sitting on top of the
power-law $q^\gamma$ background. Solar-type primaries therefore have
two distinct populations of binaries: the bulk power-law and a
$\sim 10\%$ excess of near-equal-mass twins. The relative weights
shift across the mass spectrum:

```{list-table}
:header-rows: 1

* - Primary mass
  - $f_{\mathrm{twin}}$
  - Behaviour
* - $M_1 < 0.8\,\Msun$
  - 0.05
  - Modest twin excess
* - $0.8$–$1.2\,\Msun$
  - **0.10**
  - Peak — solar-type strongest twin signature
* - $1.2$–$3.5\,\Msun$
  - 0.08
  - Slightly lower
* - $M_1 > 3.5\,\Msun$
  - 0.03
  - Massive stars rarely twin
```

The full $g(q \mid M_1)$ form combining the power law and twin excess
is documented in [](mass-ratio-distributions.md).

## Higher-order multiples

progenax's default `BinaryIMF` model caps at one companion per
primary. Real populations include triples, quadruples, and higher:
$\sim 10\%$ of solar-type systems are triples, rising to $> 50\%$ for
O-type {cite:p}`MoeDiStefano2017,Sana2012`. The single-companion approximation
*underestimates* the high-mass distortion (a triple with masses
$m_1 + m_2 + m_3$ inflates the system mass more than the binary
$m_1 + m_2$). For O-star-dominated populations this is a substantial
effect; for solar-type-dominated populations it is a small correction.

The architectural extension to triples is straightforward — replace
the single $f_b(m_1)$ with a Poisson-companion model — but is
deferred to a future progenax version.

## References

{cite:t}`MoeDiStefano2017` is the comprehensive joint-distribution paper.
{cite:t}`Sana2012` is the foundational massive-star multiplicity
work. {cite:t}`Moe2019` extends to metallicity dependence. The
period-distribution side of the calibration is documented at
[](../binaries/period-distributions.md); the mass-ratio side at
[](mass-ratio-distributions.md).

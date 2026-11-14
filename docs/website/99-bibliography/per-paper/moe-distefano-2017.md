---
title: Moe & Di Stefano (2017)
description: "Annotated reference for M. Moe & R. Di Stefano — the period (P) / mass-ratio (q) interrelation of binaries (Table 13 multiplicity statistics used by progenax)."
---

# Moe & Di Stefano (2017)

```{admonition} Mind Your Ps and Qs: The interrelation between period (P) and mass-ratio (Q) distributions of binary stars
:class: note

**Authors.** Maxwell Moe, Rosanne Di Stefano (Steward Observatory; Harvard-Smithsonian CfA).

**Reference.** *The Astrophysical Journal Supplement Series* **230**, 15 (55 pp., 2017).

**DOI.** [10.3847/1538-4365/aa6fb6](https://doi.org/10.3847/1538-4365/aa6fb6) ·
**arXiv.** [1606.05347](https://arxiv.org/abs/1606.05347) ·
**ADS.** [2017ApJS..230...15M](https://ui.adsabs.harvard.edu/abs/2017ApJS..230...15M)
```

## Abstract (paraphrased)

Compiles ~30 surveys of early-type (and re-analyses solar-type) main-sequence binaries across
spectroscopy, eclipses, interferometry, AO, and common proper motion, correcting each for its
selection effects. Measures the intrinsic joint distribution $f(M_1,q,P,e)$ and shows it is
**not separable**: at short periods binaries have small $e$, modest $q$, and a small twin excess;
at intermediate $\log P\approx3.5$ the companion frequency peaks with $q$ weighted to small values;
at long periods companions approach random IMF pairings. The corrected statistics are fit with
mathematical functions for use in binary population synthesis.

## Multiplicity statistics — Table 13 (verified, p. 52)

Table 13 gives, per primary-mass bin (solar 0.8–1.2, A/late-B 2–5, mid-B 5–9, early-B 9–16,
O-type >16 $M_\odot$):

| statistic | solar | A/late-B | mid-B | early-B | O |
|-----------|-------|----------|-------|---------|---|
| single-star fraction $\mathcal F_{n=0}$ | 0.60 | 0.41 | 0.24 | 0.16 | 0.06 |
| binary-star fraction $\mathcal F_{n=1}$ | 0.30 | 0.37 | 0.36 | 0.32 | 0.21 |
| total multiplicity freq. $f_{\rm mult}$ | 0.50 | 0.84 | 1.3 | 1.6 | 2.1 |
| $\gamma_{\rm largeq}(\log P=1)$ | −0.5 | −0.5 | −0.5 | −0.5 | −0.5 |
| $\mathcal F_{\rm twin}(\log P=1)$ | 0.30 | 0.22 | 0.17 | 0.14 | 0.08 |

The mass-ratio distribution is a **three-parameter, period-dependent** form (Table 1, Eq. 2):
a small-$q$ slope $\gamma_{\rm smallq}$ ($0.1<q<0.3$), a large-$q$ slope $\gamma_{\rm largeq}$
($0.3<q<1.0$), and a twin excess $\mathcal F_{\rm twin}$ ($q>0.95$) — all functions of $M_1$
**and** $\log P$ ($\gamma_{\rm largeq}$ steepens from $-0.5$ at $\log P=1$ to $-2.0$ at long $P$;
$\mathcal F_{\rm twin}$ falls to $<0.03$ beyond $\log P\gtrsim3$). The multiplicity *frequency*
$f_{\rm mult}>1$ for massive stars because O/B stars are commonly triples/quadruples.

## Eccentricity distribution — §9.2 (verified, p. 38, Fig. 36)

The eccentricity follows a power law $p(e) \propto e^{\eta}$ on $0 \le e \le e_{\max}(P)$ (their
Fig. 36), with the slope $\eta$ a function of orbital period **and** primary mass. The upper limit
is the period-dependent Roche-lobe ceiling (their Eq. 3, p. 38),

```{math}
:label: moe-emax
e_{\max}(P) = 1 - \left(\frac{P}{2\,\mathrm{d}}\right)^{-2/3} \quad (P > 2\ \mathrm{d}),
```

which the source adopts for its main-sequence population to limit periapsis
Roche filling (e.g.
$e_{\max}(10\,\mathrm{d})\approx0.66$, $e_{\max}(100\,\mathrm{d})\approx0.93$); $P \le 2$ d circularizes.
On a cap $c$, the normalizable power law has mean
$c(\eta+1)/(\eta+2)$. Thus $\eta=0$ and 1 give uniform and thermal laws,
with means $c/2$ and $2c/3$, respectively. The analytic $\eta$ fits are

```{math}
:label: moe-eta
\eta(M_1, P) =
\begin{cases}
0.6 - \dfrac{0.7}{\log P - 0.5}, & 0.8 < M_1 < 3\,M_\odot\ \text{(Eq. 17, late-type)} \\[1ex]
0.9 - \dfrac{0.2}{\log P - 0.5}, & M_1 > 7\,M_\odot\ \text{(Eq. 18, early-type)}
\end{cases}
```

with linear interpolation in $M_1$ for $3 \le M_1 \le 7\,M_\odot$ (Eqs. 17–18 valid
$0.5 < \log P < 6$/$5$). Late-type binaries approach $\eta\approx0.5$ over the fitted long-period interval; early-type
intermediate-period binaries reach $\eta \approx 0.8$ (near-thermal); short periods circularize
($\eta$ "not well defined" for $\log P \lesssim 1$). Sana et al. (2012) likewise find
short-period O-star binaries are eccentricity-poor ($\eta < 0$); their precise slope is
tabulated in the supplementary Table S3 (paywalled, not in the held main report).

## Use in progenax

- [](../../10-theory/imfs/mass-ratio-distributions.md), [](../../10-theory/imfs/binary.md).
- `progenax.imf.MoeDiStefano2017` — a period-independent single-slope
  Gaussian-mixture reduction informed by the source. Its coefficients are
  stored directly, without computing a weighted period integral of the full
  model. Adequacy for a selected mass-function analysis requires a comparison;
  the retained recovery scripts use this reduced model.
- `progenax.imf.MoeDiStefano2017Full` / `MoePeriod` / `MoeJointOrbit` (Batch 4i) — the
  **faithful** two-slope, period-dependent $q\,|\,(M_1,P)$ from Table 13 (bilinear
  interpolation, grid inverse-CDF) and the joint $(P,q,e)$ interrelation sampler.
- `progenax.binaries.MoeCompanions` (Batch 4k) — wires `MoeJointOrbit` + Moe's own
  $f_b(M_1)$ into `build_binary_cluster` as a `CompanionModel`; the same $q$ sets
  $m_2 = q\,m_1$, so the P–q correlation is **self-consistent** in the secondary masses.
- `progenax.imf.MassDependentBinaryFraction` — a coarse step approximation
  to the multiplicity probability, informed by Table 13 and lower-mass surveys.
  The source probabilities include solar $1-0.60=0.40$ and O-type
  $1-0.06=0.94$; the code uses 0.44 at 0.5–1$\,M_\odot$ and 0.90 above 10$\,M_\odot$.
  These stored bins are not a demonstrated rebinning of Table 13.
- `progenax.binaries.MoeEccentricity` — **faithful** implementation of the $p(e)\propto e^{\eta}$
  law with $\eta(\log P, M_1)$ from {eq}`moe-eta` (Eqs. 17–18) on $[0, e_{\max}(P)]$ with the
  period-dependent population ceiling {eq}`moe-emax` (Eq.3); samples via
  $e=c u^{1/(\eta+1)}$ on the active branch. The numerical extension assigns
  a circular atom for $\eta+1\le10^{-6}$ and floors the slope denominator
  at $10^{-3}$; the continuous PDF does not encode the atom. A fixed
  ceiling (`e_max`, default 0.99) caps the long-period limit. The cap does
  not inspect individual radii or guarantee contact avoidance in a catalogue.
- `progenax.binaries.LogisticThermalEccentricity` — a smooth circular→thermal **heuristic** (a
  logistic blend toward $f(e)=2e$); not Moe's $\eta(P)$ law (see [](duquennoy-mayor-1991.md)).

## Notes

The source statistics enter at two levels: a reduced period-independent
Gaussian mixture and the conditional grid model composed by `MoeJointOrbit`
and `MoeCompanions`. Both companion routes generate at most one companion
per primary. Table interpolation, grid inverse-CDF accuracy, empirical
extrapolation and population-gradient precision require their own checks;
a completed implementation record is not general physical qualification.

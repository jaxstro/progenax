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

which guarantees the components do not fill their Roche lobes at periapsis (e.g.
$e_{\max}(10\,\mathrm{d})\approx0.66$, $e_{\max}(100\,\mathrm{d})\approx0.93$); $P \le 2$ d circularizes.
$\eta = 0$ is uniform ($\langle e\rangle = 0.5$); $\eta = 1$ is thermal $f(e)=2e$
($\langle e\rangle = 2/3$). The analytic $\eta$ fits are

```{math}
:label: moe-eta
\eta(M_1, P) =
\begin{cases}
0.6 - \dfrac{0.7}{\log P - 0.5}, & 0.8 < M_1 < 3\,M_\odot\ \text{(Eq. 17, late-type)} \\[1ex]
0.9 - \dfrac{0.2}{\log P - 0.5}, & M_1 > 7\,M_\odot\ \text{(Eq. 18, early-type)}
\end{cases}
```

with linear interpolation in $M_1$ for $3 \le M_1 \le 7\,M_\odot$ (Eqs. 17–18 valid
$0.5 < \log P < 6$/$5$). Late-type binaries asymptote to $\eta \approx 0.5$ at long $P$; early-type
intermediate-period binaries reach $\eta \approx 0.8$ (near-thermal); short periods circularize
($\eta$ "not well defined" for $\log P \lesssim 1$). Sana et al. (2012) measure
$\eta = -0.4 \pm 0.2$ for short-period O-stars.

## Use in progenax

- [](../../10-theory/imfs/mass-ratio-distributions.md), [](../../10-theory/imfs/binary.md).
- `progenax.imf.MoeDiStefano2017` — a **period-averaged single-slope** reduction of the
  $\gamma_{\rm smallq}/\gamma_{\rm largeq}+\mathcal F_{\rm twin}$ model (captures the trend; the
  faithful two-slope, period-dependent model is a tracked enhancement).
- `progenax.imf.MassDependentBinaryFraction` — the **multiplicity fraction** $1-\mathcal F_{n=0}$
  from Table 13 (e.g. solar $1-0.60=0.40$, O-type $1-0.06=0.94$), with $<0.8\,M_\odot$ bins from
  M-dwarf surveys.
- `progenax.binaries.MoeEccentricity` — **faithful** implementation of the $p(e)\propto e^{\eta}$
  law with $\eta(\log P, M_1)$ from {eq}`moe-eta` (Eqs. 17–18) on $[0, e_{\max}(P)]$ with the
  period-dependent Roche ceiling {eq}`moe-emax` (Eq. 3); samples via inverse-CDF
  $e = e_{\max}(P)\,u^{1/(\eta+1)}$, with $\eta \le -1$ (very short $P$) → circular. A fixed
  numerical ceiling (`e_max`, default 0.99) caps the long-$P$ limit where Eq. 3 → 1.
- `progenax.binaries.LogisticThermalEccentricity` — a smooth circular→thermal **heuristic** (a
  logistic blend toward $f(e)=2e$); not Moe's $\eta(P)$ law (see [](duquennoy-mayor-1991.md)).

## Notes

**The most-referenced paper across progenax binary modelling.** The central result — that $P$,
$q$, and $e$ are interrelated — is only partially captured by the current period-averaged
`MoeDiStefano2017`; a faithful $(P,q)$-joint sampler is tracked in
`docs/notes/2026-06-04-moe-twoslope-q-distribution-ticket.md`.

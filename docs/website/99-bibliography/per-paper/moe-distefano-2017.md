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

## Use in progenax

- [](../../10-theory/imfs/mass-ratio-distributions.md), [](../../10-theory/imfs/binary.md).
- `progenax.imf.MoeDiStefano2017` — a **period-averaged single-slope** reduction of the
  $\gamma_{\rm smallq}/\gamma_{\rm largeq}+\mathcal F_{\rm twin}$ model (captures the trend; the
  faithful two-slope, period-dependent model is a tracked enhancement).
- `progenax.imf.MassDependentBinaryFraction` — the **multiplicity fraction** $1-\mathcal F_{n=0}$
  from Table 13 (e.g. solar $1-0.60=0.40$, O-type $1-0.06=0.94$), with $<0.8\,M_\odot$ bins from
  M-dwarf surveys.
- `progenax.binaries.MoeEccentricity` — the period-dependent eccentricity slope $\eta$.

## Notes

**The most-referenced paper across progenax binary modelling.** The central result — that $P$,
$q$, and $e$ are interrelated — is only partially captured by the current period-averaged
`MoeDiStefano2017`; a faithful $(P,q)$-joint sampler is tracked in
`docs/notes/2026-06-04-moe-twoslope-q-distribution-ticket.md`.

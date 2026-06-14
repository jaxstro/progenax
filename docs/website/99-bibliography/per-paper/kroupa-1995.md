---
title: Kroupa (1995a)
description: Annotated reference for Kroupa's inverse dynamical population synthesis — the IMF-consistent birth binary-star population.
---

# Kroupa (1995a)

```{admonition} Inverse dynamical population synthesis and star formation
:class: note

**Author.** Pavel Kroupa (Astronomisches Rechen-Institut, Heidelberg).

**Reference.** *Monthly Notices of the Royal Astronomical Society* **277**, 1491–1506 (1995).

**ADS.** [1995MNRAS.277.1491K](https://ui.adsabs.harvard.edu/abs/1995MNRAS.277.1491K)
```

## Abstract (paraphrased)

All stars are taken to form in binary systems; the observed Galactic-field population
(~50–60% binaries) then emerges from the dynamical evolution of binary-rich embedded
clusters. Kroupa derives the **birth** orbital-parameter distributions by *inverse
dynamical population synthesis*: assume an initial period distribution **flat in
$\log_{10}P$ over $3 < \log_{10}P \le 7.5$** (P in days), an uncorrelated birth mass-ratio
distribution, and the three-segment IMF; distribute $N_{\rm bin}=200$ binaries in clusters
of half-mass radius ~0.08–2.5 pc (tightly clustered → isolated); and follow the
subsequent evolution by direct $N$-body integration. The evolved population reproduces the
observed field binaries (e.g. the $\log_{10}P > 4$ period distribution) for clusters of
$R_{0.5}\approx 0.8$ pc.

## Use in progenax

- `progenax.imf.binary` (`imf.py`) — cited as the IMF-consistent **birth binary-population**
  framework (period distribution flat in $\log_{10}P$), alongside Moe & Di Stefano (2017).

```{note}
**Citation-page correction (2026-06 provenance audit).** `imf.py` previously cited "MNRAS 277,
**1507**" (the companion Paper II, *"The dynamical properties of stellar systems in the Galactic
disc"*); this was corrected to **1491** — the held paper, and the one whose content matches
"IMF-consistent binary populations" (Paper I, this note).
```

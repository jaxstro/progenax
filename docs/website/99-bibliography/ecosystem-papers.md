---
title: Ecosystem papers
description: Reference papers stored locally in `docs/core-papers/` — code methodology PDFs, McLuster, COMPAS, BoOST, and other resources informing progenax's design but not directly cited.
---
# Ecosystem papers

This page indexes the **PDF** reference material under
`docs/core-papers/`. These are papers that informed progenax's
design choices but that may not be directly cited in any single
chapter — comparison codes, methodology references, surveys.

For the formally-cited papers (with their per-paper detail pages),
see [](per-paper/index.md). For the full bibliography, see
[](bibliography.md).

## Code methodology papers

```{list-table}
:header-rows: 1

* - PDF
  - Topic
* - `Allison_MassSegregation_2009.pdf`
  - {cite:t}`Allison2009` MST-based mass-segregation diagnostic
* - `Baumgardt_MassSegregation_2008.pdf`
  - {cite:t}`Baumgardt2008` energy-ranked primordial segregation
* - `Subr_MassSegregation_2008.pdf`
  - Šubr+ alternative interparticle-energy construction (not implemented in progenax)
* - `Goodwin_fractal_substructure_2004.pdf`
  - {cite:t}`Goodwin2004` recursive-tree fractal IC
* - `McLusterManual.pdf`, `McLuster_Methods_2011.pdf`
  - {cite:t}`Kuepper2011` McLuster code — progenax's primary cross-validation reference
```

## Population synthesis codes

```{list-table}
:header-rows: 1

* - PDF
  - Topic
* - `COMPAS-methods-01.pdf`, `COMPAS-methods-02.pdf`
  - COMPAS binary population synthesis methodology — comparable framework to progenax+startrax (planned)
* - `BoOST-2022.pdf`
  - Bonn Optimised Stellar Tracks — relevant to stellar evolution (stellax planned)
```

## IMF reference papers

```{list-table}
:header-rows: 1

* - PDF
  - Topic
* - `Marks-IMF-mnras-2012.pdf`
  - {cite:t}`Marks2012` cluster-scale IMF variation (incl. the Fundamental Plane)
* - `Jerabkova-IMF-aa-2018.pdf`
  - {cite:t}`Jerabkova2018` IGIMF framework
```

## Binary statistics

```{list-table}
:header-rows: 1

* - PDF
  - Topic
* - `Moe_2019_ApJ_875_61.pdf`
  - {cite:t}`Moe2019` metallicity dependence of close-binary fraction
* - `Sana-HM-binaries-2025.pdf`
  - Sana high-mass-binary follow-up (post-2012 work)
```

## How these are used

The PDFs themselves are *reference assets* — read to understand the
methodology, not directly cited line-by-line. Progenax's chapters
cite these papers via the BibTeX entries in `references.bib`; the
chapters explain the relevant physics in their own words rather than
quoting verbatim.

For published work that *is* cited (and gets its own detail page),
see [](per-paper/index.md).

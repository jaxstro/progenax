---
title: Marks & Kroupa (2012)
description: Annotated reference for M. Marks & P. Kroupa — Inverse dynamical population synthesis (the r_h–M_ecl relation and the 8π half-mass density definition).
---

# Marks & Kroupa (2012)

```{admonition} Inverse dynamical population synthesis. Constraining the initial conditions of young stellar clusters by studying their binary populations
:class: note

**Authors.** M. Marks, P. Kroupa (Argelander-Institut für Astronomie, Bonn).

**Reference.** *Astronomy & Astrophysics* **543**, A8 (2012). Received 2011 October 10,
accepted 2012 April 11.

**DOI.** [10.1051/0004-6361/201118231](https://doi.org/10.1051/0004-6361/201118231) ·
**ADS.** [2012A&A...543A...8M](https://ui.adsabs.harvard.edu/abs/2012A%26A...543A...8M)
```

## Abstract (paraphrased)

Applies *inverse dynamical population synthesis* — evolving a universal initial binary
population in clusters of varying density and matching the result to observed multiplicity
fractions — to eight young regions (Taurus, ρ Oph, Chamaeleon, Orion/ONC, IC 348, Upper Sco A,
Praesepe, Pleiades). Constrains each region's birth stellar mass $M_{\rm ecl}$ and half-mass
radius $r_h$. Identifies a **stellar-mass–half-mass-radius relation** for cluster-forming cloud
clumps and shows that environment-dependent dynamical evolution shapes present-day binary
populations.

## Key relations used by progenax (verified against the paper)

**Half-mass radius–mass relation (Abstract; Sect. 4).**

```{math}
:label: mk2012-rh
\frac{r_h}{\rm pc} = 0.1\left(\frac{M_{\rm ecl}}{M_\odot}\right)^{0.13\pm0.04}.
```

The slope $0.13\pm0.04$ and the proportionality are stated in the abstract; the prefactor
$0.1$ is the calibrated value (also quoted verbatim by [](jerabkova-2018.md), Eq. 8).

**Half-mass density (§2.1, p. 2).** The dynamical evolution of a cluster's binary population
is driven by the **8π half-mass density**

```{math}
:label: mk2012-rho
\rho_{\rm ecl} = \frac{3\,M_{\rm ecl}}{8\pi\,r_h^3},
```

i.e. clusters of the same $\rho_{\rm ecl}$ are *dynamically equivalent*. This is the
authoritative definition behind the density used in the Marks (2012) / Jeřábková (2018)
$\alpha_3$–$\rho$ relations.

## Use in progenax

- `progenax.imf.environment.compute_r_half` — implements {eq}`mk2012-rh`.
- `progenax.imf.environment.compute_rho_ecl` — implements {eq}`mk2012-rho` (8π half-mass density).
- Underpins the mass-based IGIMF path ([](jerabkova-2018.md), [](marks-2012.md)).

## Notes

The **8π** definition {eq}`mk2012-rho` is the reason progenax does **not** use Jeřábková
(2018) Eq. 8's "4π" form: Eq. 8 is internally inconsistent with Jeřábková's own
$\log_{10}\rho_{\rm ecl}=0.61\log_{10}M_{\rm ecl}+2.08$ relation (which is 8π). progenax's
`compute_rho_ecl` reproduces Marks (2012) Table 1 densities exactly. The companion paper
**Marks & Kroupa (2011)**, MNRAS 417, 1702 provides the binary-population dynamical operator
(period / mass-ratio distributions) used elsewhere in progenax's binary modelling.

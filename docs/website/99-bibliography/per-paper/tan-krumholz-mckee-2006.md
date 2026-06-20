---
title: Tan, Krumholz & McKee (2006)
description: Annotated reference for J. C. Tan et al. — the equilibrium / single-mean-density star-cluster-formation framework that uses the free-fall star-formation kernel without integrating over a density PDF.
---

# Tan, Krumholz & McKee (2006)

```{admonition} Equilibrium star cluster formation
:class: note

**Authors.** J. C. Tan, M. R. Krumholz, C. F. McKee.

**Reference.** *The Astrophysical Journal Letters* **641**, L121–L124 (2006).

**DOI.** [10.1086/504150](https://doi.org/10.1086/504150) ·
**ADS.** [2006ApJ...641L.121T](https://ui.adsabs.harvard.edu/abs/2006ApJ...641L.121T)
```

```{warning}
**UNVERIFIED — no PDF held.** This paper's PDF is not held in the progenax reference set, so the
characterisation below has **not** been checked against the published article. It records the
role the paper plays in the gravoturbulence chapter (a representative "single-mean-density"
framework), not a verified transcription. Verify against the published ApJL before relying on
any specific quantity.
```

## Abstract (paraphrased)

Argues that massive star clusters form in approximate dynamical equilibrium over several
free-fall times, rather than in a single rapid free-fall collapse, with star formation
proceeding at a low rate per free-fall time. In this picture the star-formation rate is set by
applying a free-fall collapse kernel to the cloud's characteristic (mean) density, integrated
over the cluster-formation timescale.

## Role in progenax (attribution — UNVERIFIED)

progenax's gravoturbulence chapter cites this work as an **earlier "single-mean-density"
framework**: it applies the same free-fall star-formation kernel ($\dot\rho_\star \propto
\rho/t_{\rm ff} \propto \rho^{3/2}$) as the modern PDF-based treatments, **but evaluated at a
single mean density** rather than integrated over a full density probability distribution.
progenax uses it as the conceptual baseline against which the BM19 / Burkhart PDF-integrated
magnification factor $\zeta$ is the improvement: integrating the same kernel over the
lognormal+power-law density PDF (rather than a single mean) is what produces the geometric SFR
boost.

## Use in progenax

- [](../../10-theory/gravoturbulence/index.md) — the equilibrium cluster-formation framework as
  context.
- [](../../10-theory/gravoturbulence/magnification-factor.md) — cited as a single-mean-density
  baseline for the $\zeta$ magnification factor and the α↔p correspondence discussion.
- [](../../10-theory/gravoturbulence/density-pdf-and-fdf.md) — noted as the earlier framework
  that uses the same kernel without integrating over a density PDF.

## Notes

The gravoturbulence subsystem is the **experimental, repo-only** `gravoturb_fdf` package (not
in the released wheel). Until the PDF is held and verified, treat this note as a role/context
summary rather than a confirmed transcription.

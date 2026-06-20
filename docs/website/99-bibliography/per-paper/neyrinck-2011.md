---
title: Neyrinck, Szapudi & Szalay (2011)
description: Annotated reference for Neyrinck, Szapudi & Szalay — Gaussianizing the discrete galaxy density field, the 1pt+2pt sufficiency of a monotone-mapped Gaussian field, and how the Fisher gain depends on grid resolution.
---

# Neyrinck, Szapudi & Szalay (2011)

```{admonition} Rejuvenating power spectra. II. The Gaussianized galaxy density field
:class: note

**Authors.** Mark C. Neyrinck, István Szapudi, Alexander S. Szalay

**Reference.** *The Astrophysical Journal* **731, 116** (2011).

**DOI.** [10.1088/0004-637X/731/2/116](https://doi.org/10.1088/0004-637X/731/2/116)

**Verified.** Abstract and §1 checked against the held PDF (2026-06). Two statements progenax leans
on: a Gaussian-mapped field has **vanishing higher-order correlations** (1pt+2pt sufficiency), and
the **Fisher gain peaks at a grid resolution set by the sampling level**.
```

## The big idea

Paper II carries the log-density Gaussianization of [](neyrinck-2009.md) from the continuous matter
field to the **discrete, shot-noise-limited galaxy field**. A Gaussianizing transform (one that makes
the one-point distribution Gaussian) again reduces power-spectrum non-linearity and increases the
Fisher information, in real and redshift space — with the redshift-space gain smaller because
peculiar velocities partly Gaussianize the field already.

## Two results progenax uses

**1-point + 2-point sufficiency.** The paper states the assumption cleanly: *"in the approximation
that a non-Gaussian field is a non-linear transformation of a Gaussian field, PDF Gaussianization
will produce a Gaussian field, vanishing all higher-order correlations."* That is **exactly the
`gravoturb_fdf` generative model** — a Gaussian random field passed through a monotone marginal map
(the copula). For such a field the one-point PDF plus the two-point function are *sufficient
statistics*; the higher-order correlations carry no independent information. This is the formal
licence for restricting the [](../../10-theory/gravoturbulence/inference.md) likelihood to 1pt + 2pt: it is not a
lossy shortcut but the complete description of the model, and therefore the most honest use of it.

**The Fisher gain peaks at a finite resolution.** The cumulative Fisher information recovered by
Gaussianizing "peaks at a particular grid resolution [that] depends on the sampling level." Finer
than that, shot noise dominates and adds nothing; coarser, the small-scale information is averaged
away. This is the cosmology precedent for the **resolution / $N_{\rm eff}$ forecast** in
[](../../10-theory/gravoturbulence/inference.md): there is an optimal map resolution for a given tracer density, and
pushing past it does not help.

## Use in progenax

- **Justifies the 1pt+2pt inference design** — the GRF-through-monotone-map field is fully captured
  by its one- and two-point statistics, so the phase-randomness of the model is not a
  misspecification for this likelihood.
- **Frames the resolution trade-off** behind the survey-design forecast (how many independent
  resolution elements a map needs).

## Notes

- The genuine filamentary phase coherence that a Gaussian-mapped field *cannot* reproduce is the
  signal reserved for the (deferred) 3-point null test — the one place the 1pt+2pt sufficiency
  argument is expected to break for real clouds.
- Companion to [](neyrinck-2009.md); optimal-transform theory in [](carron-szapudi-2013.md);
  discrete sufficiency in [](carron-szapudi-2014.md).

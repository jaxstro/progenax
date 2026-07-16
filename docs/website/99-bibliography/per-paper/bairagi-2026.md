---
title: Bairagi & Wandelt (2026)
description: Annotated reference for Anirban Bairagi & Benjamin Wandelt — neural simulation-based field-level inference (PatchNet), the SBI alternative that the gravoturb physics-direct layer deliberately contrasts with.
---

# Bairagi & Wandelt (2026)

```{admonition} PatchNet: A hierarchical approach for neural field-level inference from Quijote simulations
:class: note

**Authors.** Anirban Bairagi, Benjamin D. Wandelt

**Reference.** *Journal of Cosmology and Astroparticle Physics* **2026(03), 028**
(arXiv:2509.03165).

**DOI.** [10.1088/1475-7516/2026/03/028](https://doi.org/10.1088/1475-7516/2026/03/028)

**Verified.** Abstract checked against the held PDF (2026-06). Cited as the **contrast**: the neural,
simulation-based field-level inference that the `gravoturb` physics-direct approach is the
counterpoint to.
```

## The big idea

How much cosmological information lives in the non-linear density field, and how do you extract it
without an analytic likelihood? Bairagi & Wandelt take the **simulation-based inference (SBI)** route:
learn the posterior directly from forward-modelled simulations with a neural network, sidestepping the
need for a tractable likelihood. Their PatchNet is a *hierarchical* architecture that fuses
small-scale information from field sub-volumes ("patches") with large-scale summaries (power spectrum
and bispectrum), which is efficient in both compute and training data and avoids the memory cost of
full-field training. On the Quijote dark-matter suite it enhances the Fisher information over
analytic summaries and matches a very different method (wavelet statistics), suggesting it captures
"most of the information content of the dark matter density field at the resolution of
$\sim 7.8\,\mathrm{Mpc}/h$."

## Why it is the contrast for progenax

SBI is the *default* modern answer to a non-differentiable stochastic simulator — train a network on
many runs. The [](../../10-theory/gravoturbulence/inference.md) layer makes the opposite bet: keep the physics
**direct** and **differentiable** by predicting the summary statistic analytically (Gaussianization
two-point + counts-in-cells + the peaks-over-threshold tail) and differentiating *that*, rather than
learning a black-box mapping. The two approaches are complementary:

- **SBI (this paper)** captures non-Gaussian, phase-coherent information a Gaussian-mapped model
  cannot, at the cost of interpretability and a large simulation budget.
- **Physics-direct (progenax)** keeps every parameter physically meaningful and the gradients exact,
  at the cost of restricting to the 1pt + 2pt statistics the model represents faithfully.

The shared theme is the **information-vs-resolution** question: Bairagi & Wandelt quantify the
information content at a given grid resolution; the FDF forecast quantifies how many independent tail
resolution elements a gas map needs to measure the density-PDF slope $\alpha$.

## Use in progenax

- Cited in [](../../10-theory/gravoturbulence/inference.md) as the SBI alternative the physics-direct layer
  deliberately avoids, and as a pointer for where a neural cross-check (or a future hybrid) would
  add the genuinely phase-coherent, higher-order information the 1pt+2pt model omits.

## Notes

- Recent (2026) and cosmology-focused (Euclid/DESI/Rubin); the relevance to star clusters is by
  analogy — the FDF reframes cluster substructure as a galaxy-clustering-style inference problem.

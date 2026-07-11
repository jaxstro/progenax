---
title: Parravano, McKee & Hollenbach (2011)
description: Annotated reference for A. Parravano, C. F. McKee & D. J. Hollenbach — An IMF for individual stars in galactic disks (the Smoothed Two-Power Law behind TaperedPowerLaw).
---

# Parravano, McKee & Hollenbach (2011)

```{admonition} An Initial Mass Function for Individual Stars in Galactic Disks. I.
:class: note

**Authors.** Antonio Parravano, Christopher F. McKee, David J. Hollenbach.

**Reference.** *The Astrophysical Journal* **726**, 27 (2011 January 1; 20 pp).

**DOI.** [10.1088/0004-637X/726/1/27](https://doi.org/10.1088/0004-637X/726/1/27) ·
**ADS.** [2011ApJ...726...27P](https://ui.adsabs.harvard.edu/abs/2011ApJ...726...27P)
```

## Abstract (paraphrased)

Assuming the disk IMF is **universal and simple**, the authors adopt a single smooth
analytic form and constrain its parameters with integral observables (star counts,
luminosities). The form approaches a power law at both ends — low-mass
($\psi \propto m^{\gamma}$) and high-mass ($\psi \propto m^{-\Gamma}$) — joined smoothly,
which they term the **Smoothed Two-Power Law (STPL)**.

## What progenax uses (PDF-verified 2026-07-11)

- **Eq. 1 (p. 2) — the STPL form.**

  ```{math}
  \psi(m) \;\propto\; m^{-\Gamma}\,\bigl\{1 - \exp\!\bigl[-(m/m_{\rm ch})^{\gamma+\Gamma}\bigr]\bigr\},
  ```

  which is exactly progenax's `TaperedPowerLaw` PDF
  $f(m) \propto m^{-\alpha}\,(1 - e^{-(m/m_{\rm peak})^{\beta}})$ **up to the slope
  convention**: PMH11 write $\psi$ per unit $\log m$ (their Salpeter is $\Gamma = 1.35$),
  progenax's `alpha` is the linear-mass slope ($\alpha = \Gamma + 1$), and PMH11's taper
  exponent $\gamma + \Gamma$ is the code's `beta`. This convention mapping is a formal
  `deviations` entry on the
  [tapered_powerlaw model card](../../15-model-reference/imfs.md).
- **Historical attribution (p. 2).** The functional form was *first proposed by
  Paresce & De Marchi (2000)* for globular-cluster present-day mass functions; PMH11
  adopted and calibrated it as a disk IMF (with Hollenbach et al. 2005 an earlier STPL
  step). Maschberger (2013) lists it as his comparison Eq. 12 — the route by which
  progenax originally cited it.

## Connections in progenax

- `progenax.imf.smooth.TaperedPowerLaw` — implements Eq. 1 in the linear-mass convention.
- [Maschberger (2013)](../../10-theory/imfs/classic.md) — the house-default smooth IMF; the
  STPL is the closest sibling *without* a closed-form quantile (Newton ppf), which is why
  Maschberger remains the production default.

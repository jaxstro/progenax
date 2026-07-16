---
title: Heyer et al. (2009)
description: Annotated reference for Heyer, Krawczyk, Duval & Jackson — re-examining Larson's relations and the surface-density dependence of the cloud structure function.
---

# Heyer et al. (2009)

```{admonition} Re-examining Larson's scaling relationships in Galactic molecular clouds
:class: note

**Authors.** M. Heyer, C. Krawczyk, J. Duval, J. M. Jackson

**Reference.** *The Astrophysical Journal* **699, 1092** (2009).

**DOI.** [10.1088/0004-637X/699/2/1092](https://doi.org/10.1088/0004-637X/699/2/1092) ·
**ADS.** 2009ApJ...699.1092H

**Verified.** Abstract, §1–2, and the structure-function / virial results checked against
the held PDF (2026-06).
```

## The big idea

Larson (1981) proposed three scaling relations for molecular clouds: a size–linewidth law
$\sigma_v\propto L^{0.38}$, virial equilibrium $2\sigma_v^2 L/GM\sim1$, and an inverse
density–size law $n\propto L^{-1.1}$. Together these imply that **all clouds share the same
surface density** — a cornerstone assumption of many cloud/star-formation models. Heyer+2009
re-test this with higher-quality, optically-thinner $^{13}$CO data (the BU-FCRAO Galactic
Ring Survey) on the Solomon+1987 (SRBY) cloud sample, and find it **fails**: the
size–linewidth *coefficient* is not universal but increases with surface density. Clouds are
still close to self-gravitating equilibrium — but across a *range* of surface densities, not
a single one.

## Core result — the surface-density dependence

Define the **structure-function coefficient**

$$
v_0 \equiv \frac{\sigma_v}{R^{1/2}} \quad [\mathrm{km\,s^{-1}\,pc^{-1/2}}].
$$

Larson's relations require $v_0=$ const. Heyer+2009 instead find it scales with the cloud
mass surface density $\Sigma$:

$$
\boxed{\;v_0 \propto \Sigma^{1/2}\;}
$$

exactly as expected for clouds in **self-gravitational (virial) equilibrium**. To see why,
write the virial parameter for a uniform sphere of mass $M=\pi R^2\Sigma$:

$$
\alpha_\mathrm{vir} \equiv \frac{5\sigma_v^2 R}{GM}
= \frac{5\,(v_0^2 R)\,R}{G\,\pi R^2\Sigma}
= \frac{5\,v_0^2}{\pi G\,\Sigma}.
$$

If $v_0^2\propto\Sigma$ (the Heyer relation), then $\alpha_\mathrm{vir}\approx$ **constant**
($\sim1$–2) across clouds — they self-regulate to virial equilibrium *at every* surface
density. Only if one (wrongly) holds $v_0$ fixed at Larson's constant does one get
$\alpha_\mathrm{vir}\propto\Sigma^{-1}$.

## Numbers worth pinning

- **LTE masses** (from $^{13}$CO) are typically **~5× smaller** than SRBY virial masses.
- **Median mass surface density** of the Heyer sample: $\Sigma\approx 42\,M_\odot\,\mathrm{pc}^{-2}$.
- SRBY had assumed a *constant* $\Sigma(\mathrm{H_2})\approx 200\,M_\odot\,\mathrm{pc}^{-2}$
  (corrected from the 170 quoted by SRBY for an updated galactocentric radius).
- Larson (1981): $\sigma_v\propto L^{0.38}$; Solomon+1987 (SRBY): $\sigma_v\propto R^{0.5}$
  (steeper) — the convention progenax's `larson_sigma_v` default ($\alpha=0.5$) follows.

## Use in progenax

- The **correct grounding** for any $\sigma_v(R,\Sigma)$ or $\alpha_\mathrm{vir}(\Sigma)$
  relation in the PN11 alternative path
  ([collapse_threshold.py](../../../../src/experimental/gravoturb/theory/collapse_threshold.py)), which currently
  takes $\alpha_\mathrm{vir}$ as an explicit input.
- Confirms the Larson/Solomon size–linewidth exponents used by
  [cluster/turbulence.py](../../../../src/progenax/cluster/turbulence.py) `larson_sigma_v`.

## Notes

- **Citation correction.** The legacy `cluster/gravoturbulent.py` attributed an
  $\alpha_\mathrm{vir}\propto\Sigma^{-1}$ relation (with an unsourced $\Sigma_0=85$) to
  "Heyer & Dame (2015)". That is **wrong on three counts**: (i) the relevant paper is
  Heyer+**2009** (Heyer & Dame 2015 is a separate ARA&A review); (ii) Heyer+2009 *refutes*
  the constant-$v_0$ assumption that gives $\alpha_\mathrm{vir}\propto\Sigma^{-1}$, finding
  $v_0\propto\Sigma^{1/2}\Rightarrow\alpha_\mathrm{vir}\approx$ const; (iii) $\Sigma_0=85$
  matches none of the paper's values (42 sample median; 200 SRBY constant). That legacy
  module is being removed in the clean-room pass.
- $\alpha_\mathrm{vir}\approx1$ is marginally bound/virial; $\alpha_\mathrm{vir}\gg1$ is
  unbound (turbulence-dominated). This is the $\alpha_\mathrm{vir}$ that enters the PN11
  critical density ([](padoan-nordlund-2011.md)).
- Caveats: $^{13}$CO LTE masses assume constant abundance and can be uncertain in cloud
  envelopes; cloud boundaries/areas carry definitional scatter.

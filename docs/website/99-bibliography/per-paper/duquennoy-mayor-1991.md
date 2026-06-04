---
title: Duquennoy & Mayor (1991)
description: Annotated reference for A. Duquennoy & M. Mayor — the solar-type binary period (log-normal) and eccentricity (three-period) distributions used by progenax.
---

# Duquennoy & Mayor (1991)

```{admonition} Multiplicity among solar-type stars in the solar neighbourhood. II. Distribution of the orbital elements in an unbiased sample
:class: note

**Authors.** A. Duquennoy, M. Mayor (Geneva Observatory).

**Reference.** *Astronomy & Astrophysics* **248**, 485–524 (1991).

**ADS.** [1991A&A...248..485D](https://ui.adsabs.harvard.edu/abs/1991A%26A...248..485D)
```

## Abstract (paraphrased)

A CORAVEL radial-velocity survey of a complete, volume-limited sample of 164 G-dwarf
(F7–G9 IV/V/VI) primaries from the Gliese catalogue, combined with visual and
common-proper-motion data, yields the present-day distributions of orbital period,
eccentricity and mass ratio in an unbiased sample. The orbital-period distribution is unimodal
and remarkably well approximated by a Gaussian in $\log P$ with a median period of 180 yr.
Tight binaries ($P < 11$ d) are all tidally circularized; intermediate periods follow a
bell-shaped eccentricity distribution; wide binaries ($P > 1000$ d) tend toward the thermal
$f(e) = 2e$. This is the foundational dataset for solar-type binary statistics.

## Verified facts

**Period distribution (§7.3, p. 514, Fig. 7).** The orbital periods follow

```{math}
:label: dm91-period
f(\log P) = C \exp\!\left[-\frac{(\log P - \overline{\log P})^2}{2\,\sigma_{\log P}^2}\right],
\qquad \overline{\log P} = 4.8,\quad \sigma_{\log P} = 2.3,\quad P\ \text{in days}
```

(median period $\simeq 180$ yr $= 10^{4.8}$ d).

**Eccentricity distribution (§6.1, §7.2).** Three period regimes:

- $P < 10$ d: tidally circularized, $e \approx 0$. The circularization period is
  $P_{\rm circ} \approx 11.6$ d (last circularized binary HD 13974 at 10.0 d; first eccentric
  HD 17433 at 13.2 d).
- $10 < P < 1000$ d: bell-shaped, mean $\bar e = 0.31 \pm 0.04$.
- $P > 1000$ d: tends toward the thermal $f(e) = 2e$ — the distribution expected if it is a
  function of energy only (**Ambartsumian 1937**).

## Use in progenax

- `progenax.binaries.LogNormalPeriod` — its defaults $\mu_{\log P} = 4.8$, $\sigma_{\log P} = 2.3$
  (days) are DM91 §7.3 verbatim ({eq}`dm91-period`). See [](raghavan-2010.md) for the modern
  revision (median ~300 yr, $\sigma \approx 2.28$).
- `progenax.binaries.LogisticThermalEccentricity` — the smooth circular→thermal heuristic is
  motivated by DM91's three-period model; defaults $P_{\rm circ}=10$ d ($\approx$ DM91 11.6 d) and
  $P_{\rm thermal}=1000$ d (DM91 wide-thermal onset).
- `progenax.binaries.MassDependentBinaryConfig` — DM91 is the low-mass (solar-type) prescription.

## Notes

DM91 is the canonical solar-type multiplicity reference. Its wide-binary $f(e)=2e$ traces to
**Ambartsumian (1937)** (energy-only distribution) — the same thermal law in
`progenax.binaries.ThermalEccentricity` (see [](heggie-1975.md)). The faithful Moe & Di Stefano
($e^\eta$) eccentricity model is in [](moe-distefano-2017.md).

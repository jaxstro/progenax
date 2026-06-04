---
title: Heggie (1975)
description: Annotated reference for D. C. Heggie — binary evolution in stellar dynamics; the thermal f(e)=2e equilibrium eccentricity distribution and Heggie's law.
---

# Heggie (1975)

```{admonition} Binary evolution in stellar dynamics
:class: note

**Authors.** Douglas C. Heggie (Institute of Astronomy, Cambridge).

**Reference.** *Monthly Notices of the Royal Astronomical Society* **173**, 729–787 (1975).

**DOI.** [10.1093/mnras/173.3.729](https://doi.org/10.1093/mnras/173.3.729) ·
**ADS.** [1975MNRAS.173..729H](https://ui.adsabs.harvard.edu/abs/1975MNRAS.173..729H)
```

## Abstract (paraphrased)

A theoretical treatment of the behaviour of binaries in $N$-body systems. The paper begins by
testing possible "equilibrium" distributions for binaries against the results of computational
experiments, then analyses the dynamics of encounters between binaries and other cluster members
using an impulsive approximation. Pairs with low binding energies (soft binaries — much less bound
than the average kinetic energy of single stars) tend to be disrupted by encounters, while
energetic (hard) pairs tend to become still more energetic at a rate approximately independent of
their binding energy. This is the origin of **"Heggie's law": hard binaries harden and soft
binaries soften.**

## Verified facts (Summary, p. 729)

- The equilibrium / thermal eccentricity distribution for dynamically formed binaries is
  $f(e) = 2e$ on $0 \le e < 1$, with $\langle e\rangle = 2/3$, arising from energy equipartition
  (a function of energy only; cf. Ambartsumian 1937; Jeans 1919).
- Heggie's law: in close encounters, hard binaries (binding energy $\gg$ mean cluster kinetic
  energy) tend to harden; soft binaries tend to be disrupted.

## Use in progenax

- `progenax.binaries.ThermalEccentricity` — implements $f(e) = 2e$ (CDF $F(e)=e^2$, PPF
  $e=\sqrt{u}$), the thermal/equilibrium distribution discussed here. Cited together with
  Ambartsumian (1937) and Jeans (1919).
- `progenax.binaries.MoeEccentricity` / `LogisticThermalEccentricity` — the thermal law is the
  $\eta = 1$ (long-period) limit toward which both eccentricity models tend.

## Notes

The thermal $f(e)=2e$ law is textbook and triple-sourced (Jeans 1919; Ambartsumian 1937;
Heggie 1975). Heggie (1975) provides the dynamical-encounter justification; the energy-only
derivation is Ambartsumian's (cited by [](duquennoy-mayor-1991.md), which confirms wide solar-type
binaries approach $f(e)=2e$).

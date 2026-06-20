---
title: Raghavan et al. (2010)
description: Annotated reference for D. Raghavan et al. — the modern volume-limited solar-type multiplicity census (updates DM91; no radial binary-fraction law).
---

# Raghavan et al. (2010)

```{admonition} A Survey of Stellar Families: Multiplicity of Solar-type Stars
:class: note

**Authors.** D. Raghavan, H. A. McAlister, T. J. Henry, D. W. Latham, G. W. Marcy,
B. D. Mason, D. R. Gies, R. J. White, T. A. ten Brummelaar.

**Reference.** *The Astrophysical Journal Supplement Series* **190**, 1–42 (2010).

**DOI.** [10.1088/0067-0049/190/1/1](https://doi.org/10.1088/0067-0049/190/1/1) ·
**ADS.** [2010ApJS..190....1R](https://ui.adsabs.harvard.edu/abs/2010ApJS..190....1R)
```

## Abstract (paraphrased)

A comprehensive assessment of companions to a volume-limited sample of 454 solar-type
(F6–K3) stars within 25 pc, drawn from *Hipparcos* and combining long-baseline and speckle
interferometry, proper-motion companions, radial-velocity monitoring, and literature data. The
observed fractions of single : double : triple : higher-order systems are 56 : 33 : 8 : 3 % (54 :
34 : 9 : 3 % counting confirmed companions), implying the **majority (54%) of solar-type stars are
single** — in contrast to earlier estimates. The orbital-period distribution is unimodal and
roughly log-normal with a **median of about 300 years**; the period–eccentricity relation shows
the expected circularization for $P < 12$ d followed by a **roughly flat** eccentricity
distribution; the mass-ratio distribution favours near-equal pairs. This is the modern update to
{cite:t}`DuquennoyMayor1991`.

## Verified facts (abstract, Table 1, §2)

- **Volume-limited field census**, 454 solar-type primaries within 25 pc — characterizes
  multiplicity vs colour ($B-V$, i.e. primary mass), period, mass ratio, and eccentricity, **not**
  a spatial / radial binary-fraction profile.
- **Period distribution:** log-normal, **median ~300 yr** ($\log_{10}(P/\mathrm{d}) \approx 5.0$),
  $\sigma_{\log P} \approx 2.28$ — broader and longer-period than DM91's 180 yr / $\mu=4.8$.
- **Eccentricity:** circularized below $P \approx 12$ d, then a roughly **flat** distribution for
  wider pairs (differs from DM91's tendency toward thermal $f(e)=2e$ at the widest separations).

## Use in progenax

- `progenax.binaries.LogNormalPeriod` — the docstring notes Raghavan (2010) as the modern revision
  of the DM91 defaults ($\mu_{\log P}=4.8$, $\sigma=2.3$). See [](duquennoy-mayor-1991.md).
- `progenax.binaries.RadialBinaryFraction` — cited (with Sana 2012, Moe & Di Stefano 2017) only to
  **motivate** that the binary fraction varies with primary mass / environment. Raghavan is a field
  census and provides **no** radial $f_b(r)$ law; the implemented form is phenomenological.

## Notes

Raghavan et al. (2010) supersedes DM91 as the most complete solar-type multiplicity census, but
progenax keeps the DM91 period defaults for continuity (both are documented). The eccentricity
difference (DM91 wide-thermal vs Raghavan flat) reflects sample and bias differences and is noted
where the eccentricity samplers cite their provenance.

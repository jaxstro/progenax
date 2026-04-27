---
title: Maschberger (2013)
description: Annotated reference for T. Maschberger — On the function describing the stellar initial mass function.
---

# Maschberger (2013)

```{admonition} On the function describing the stellar initial mass function
:class: note

**Authors.** T. Maschberger

**Reference.** *Monthly Notices of the Royal Astronomical Society* **429, 1725** (2013).

**DOI.** [10.1093/mnras/sts479](https://doi.org/10.1093/mnras/sts479)
```

## Abstract (paraphrased)

Introduces the L3 three-parameter smooth IMF $\xi(m) \propto (m/\mu)^{-\alpha}\,[1 + (m/\mu)^{1-\alpha}]^{-\beta}$ with closed-form CDF and analytical inverse-CDF. Reproduces both Salpeter ($m \gg \mu$) and the observed low-mass turnover ($m \ll \mu$) with a single smooth function.

## Use in progenax

- [](../../10-theory/imfs/classic.md) — progenax production default; closed-form inverse CDF
- [](../../10-theory/imfs/binary.md) — single-star backbone for binary-aware IMF
- `progenax.imf.Maschberger` — primary implementation

## Notes

**progenax default IMF.** The closed-form inverse-CDF makes it the only canonical IMF that samples in $\mathcal{O}(1)$ per particle without Newton iterations.

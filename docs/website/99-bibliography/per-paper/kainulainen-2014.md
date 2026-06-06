---
title: Kainulainen, Federrath & Henning (2014)
description: "Annotated reference for J. Kainulainen et al. — Unfolding the laws of star formation: The density distribution of molecular clouds."
---

# Kainulainen, Federrath & Henning (2014)

```{admonition} Unfolding the laws of star formation: The density distribution of molecular clouds
:class: note

**Authors.** J. Kainulainen, C. Federrath, T. Henning

**Reference.** *Science* **344, 183** (2014).

**DOI.** [10.1126/science.1248724](https://doi.org/10.1126/science.1248724)

**Source note.** The Kainulainen+2014 PDF is **not held locally** (`docs/core-papers/`),
so the values below are **cross-referenced from the PP20 and BM19 PDFs** (which were
verified against their held PDFs), **not** read from the Kainulainen primary source.
```

## What we use it for (cross-referenced via PP20 / BM19)

Kainulainen et al. (2014) analyse the **volume-density PDF** of a sample of nearby
molecular clouds, building a 3-D dense-gas estimate from column-density maps via
hierarchical prolate spheroids. Two numbers flow into the gravoturbulence chain:

- **Radial-profile slope $p \approx 1.67$.** Parmentier & Pasquali (2020, §3) quote the
  Kainulainen sample mean as $p\approx1.67$ (range $1<p<2.2$); BM19 (§3.2) likewise use
  $p=1.67$ (with $p\equiv\kappa=3/\alpha$). This is the canonical observational anchor for
  the PP20 magnification factor: $\zeta(1.67)\approx1.79$.
- **Dense-gas threshold $s_\mathrm{th}\approx4.2$.** PP20 report the Kainulainen averaged
  threshold number density $n_\mathrm{th}=n_0 e^{s_\mathrm{th}}\approx6700\,\mathrm{cm^{-3}}$
  ($n_0\approx100$), i.e. $s_\mathrm{th}\approx4.2$; BM19 (§3.2) use the Kainulainen YSO-count
  critical density $s_\mathrm{crit}\approx4$ as an observational anchor for the PDF
  transition density $s_t$.

## Use in progenax

- [](../../10-theory/gravoturbulence/pp20.md) — anchors $\zeta(p = 1.67) \approx 1.79$ (AC3 spot value).
- [](../../50-validation/gravoturbulent-pp20.md) — the $\zeta(1.67)$ regression anchor.
- [](burkhart-mocz-2019.md) — Kainulainen $s_\mathrm{crit}\approx4$ as the observational check on $s_t$.

## Notes

- **Unverified primary source.** To fully verify, add `Kainulainen_2014_Science_344_183.pdf`
  to `docs/core-papers/` and confirm $p\approx1.67$ and $s_\mathrm{th}\approx4.2$ directly.
  Until then these are second-hand (PP20/BM19) values.

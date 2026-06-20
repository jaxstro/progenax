---
title: Kritsuk, Norman & Wagner (2011)
description: Annotated reference for A. G. Kritsuk et al. — the density power-law tail of self-gravitating turbulent clouds and the α↔p mapping between the density-PDF tail slope and the radial-profile slope used in the gravoturbulence chapter.
---

# Kritsuk, Norman & Wagner (2011)

```{admonition} On the density distribution in star-forming interstellar clouds
:class: note

**Authors.** A. G. Kritsuk, M. L. Norman, R. Wagner.

**Reference.** *The Astrophysical Journal Letters* **727**, L20 (2011).

**DOI.** [10.1088/2041-8205/727/1/L20](https://doi.org/10.1088/2041-8205/727/1/L20) ·
**ADS.** [2011ApJ...727L..20K](https://ui.adsabs.harvard.edu/abs/2011ApJ...727L..20K)
```

```{warning}
**UNVERIFIED — no PDF held.** This paper's PDF is not held in the progenax reference set, so the
equation-level claims below (in particular the $p = 3/\alpha$ mapping) are recorded as
progenax's *attribution* and have **not** been checked cell-by-cell against the published
article. They reflect the role the paper plays in the gravoturbulence chapter, not a verified
transcription. Verify against the published ApJL before relying on any specific coefficient.
```

## Abstract (paraphrased)

Studies the volume-weighted density probability distribution function (PDF) of supersonic,
self-gravitating turbulent clouds using high-resolution simulations. Beyond the well-known
lognormal core (from non-self-gravitating supersonic turbulence), the onset of self-gravity
produces a **power-law tail** at high density, corresponding to gas in approximately isothermal
gravitational collapse. The paper connects this power-law density PDF tail to the radial
density structure of collapsing regions.

## Role in progenax (attribution — UNVERIFIED)

progenax's gravoturbulence chapter attributes to this work the **α↔p mapping** between the
high-density power-law tail slope of the density PDF, $\alpha$, and the radial-profile slope
$p$ of a collapsing region $\rho \propto r^{-p}$:

```{math}
:label: kritsuk-alpha-p
p = \frac{3}{\alpha},
```

which follows (under spherical symmetry and a power-law volume↔density correspondence) from
relating the amount of gas above a density threshold to the radial profile that produces it.
The Burkhart-Mocz canonical window $\alpha \in [1.5, 3.0]$ maps to $p \in [1.0, 2.0]$
(marginally collapsing → singular isothermal). This mapping is the hand-off between the BM19
forward chain and the magnification-factor calculation.

## Use in progenax

- [](../../10-theory/gravoturbulence/density-pdf-and-fdf.md) — the power-law tail and the α↔p
  mapping {eq}`kritsuk-alpha-p`.
- [](../../10-theory/gravoturbulence/bm19.md) — step 5 of the forward chain applies the mapping.
- [](../../10-theory/gravoturbulence/magnification-factor.md) — consumes the resulting $p$ to
  compute the geometric SFR boost $\zeta(p)$.
- `gravoturb_fdf` (experimental) — `pdf_slope_to_radial` implements $p = 3/\alpha$;
  `tests/experimental/unit/test_bm19.py` verifies it.

## Notes

The gravoturbulence subsystem is the **experimental, repo-only** `gravoturb_fdf` package (not
in the released wheel). The α↔p mapping is also discussed alongside
[](federrath-klessen-2012.md). Until the PDF is held and verified, treat the specific
$p = 3/\alpha$ form as progenax's working attribution rather than a confirmed transcription.

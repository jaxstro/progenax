---
title: Chabrier (2003)
description: Annotated reference for G. Chabrier — Galactic stellar and substellar initial mass function (lognormal + power-law; single-object vs system IMF).
---

# Chabrier (2003)

```{admonition} Galactic stellar and substellar initial mass function
:class: note

**Author.** Gilles Chabrier (École Normale Supérieure de Lyon, CRAL).

**Reference.** *Publications of the Astronomical Society of the Pacific* **115**,
763–795 (2003). Received & accepted 2003 March 31.

**DOI.** [10.1086/376392](https://doi.org/10.1086/376392) ·
**ADS.** [2003PASP..115..763C](https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C)
```

## Abstract (paraphrased)

Reviews present-day and initial mass functions across Galactic components and into the
substellar regime. The disk IMF is well described by a **lognormal** form below
$\sim 1\,M_\odot$ joined continuously to a **power-law** tail above. Chabrier reports two
distinct parameterisations — one for **single objects** (individual stars, binaries
resolved) and one for **stellar systems** (unresolved) — which differ in characteristic
mass and width. The single-object form extends smoothly into the brown-dwarf regime.

## Definitions and the two disk parameterisations (verified, §1.2 + Table 1)

Chabrier uses both the logarithmic and linear mass functions (Eqs. 1–2):

```{math}
:label: chabrier-defs
\xi(\log m) = \frac{dn}{d\log m},
\qquad
\xi(m) = \frac{dn}{dm} = \frac{1}{m\ln 10}\,\xi(\log m).
```

The original Salpeter slope is stated as $x = 1.35,\ \alpha = 2.35$ (p. 765).

**Single-object disk IMF — Table 1 (p. 769).** For $m \le 1\,M_\odot$ a lognormal,
for $m > 1\,M_\odot$ a power law $\xi(\log m) = A\,m^{-x}$:

```{math}
:label: chabrier-single
\xi(\log m)_{m\le 1} = 0.158 \,
\exp\!\left[-\frac{(\log m - \log 0.079)^2}{2\,(0.69)^2}\right],
\qquad x = 1.3 \pm 0.3 \ \Rightarrow\ \alpha = x+1 = 2.3 \ (dN/dm).
```

**System IMF — Eq. 18 (p. 770).** The unresolved-system MF has a *different*
characteristic mass and width:

```{math}
:label: chabrier-system
\xi(\log m)_{m\le 1} = 0.086 \,
\exp\!\left[-\frac{(\log m - \log 0.22)^2}{2\,(0.57)^2}\right].
```

```{admonition} Which form does progenax implement?
:class: important
`progenax.imf.ChabrierIMF` implements the **single-object disk IMF**
({eq}`chabrier-single`): $m_c = 0.079$ (rounded to 0.08), $\sigma = 0.69$,
$A_\mathrm{ln} = 0.158$, and a high-mass slope $\alpha = 2.3$ (Chabrier's Table 1
$x = 1.3$, *not* the system values $m_c \approx 0.22$, $\sigma \approx 0.57$). The
lognormal and power-law are joined **continuously** at $m_\mathrm{trans} = 1\,M_\odot$
by setting $A_\mathrm{pl} = \xi_\mathrm{ln}(m_\mathrm{trans})\,m_\mathrm{trans}^{\alpha}$
(value-continuous; only the slope has a kink). The original Salpeter $\alpha = 2.35$ is
available separately via `PowerLawIMF.salpeter()`.
```

## Use in progenax

- [](../../10-theory/imfs/classic.md) — the lognormal + power-law disk IMF.
- `progenax.imf.ChabrierIMF` — single-object disk IMF, sampled via a Newton inverse-CDF.

## Notes

The single-object vs system distinction is the key subtlety: the system IMF
({eq}`chabrier-system`) is broader-peaked because unresolved binaries shift the apparent
characteristic mass upward. For sampling *individual* stars in an N-body IC (progenax's
use case), the single-object form is the correct choice. For HMC inference where analytic
invertibility matters, prefer [](maschberger-2013.md).

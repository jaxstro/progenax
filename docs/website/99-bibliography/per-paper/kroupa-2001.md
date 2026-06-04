---
title: Kroupa (2001)
description: Annotated reference for P. Kroupa — On the variation of the initial mass function (the canonical multi-segment power-law IMF).
---

# Kroupa (2001)

```{admonition} On the variation of the initial mass function
:class: note

**Author.** Pavel Kroupa (Institut für Theoretische Physik und Astrophysik der
Universität Kiel).

**Reference.** *Monthly Notices of the Royal Astronomical Society* **322**, 231–246
(2001). Accepted 2000 September 12.

**DOI.** [10.1046/j.1365-8711.2001.04022.x](https://doi.org/10.1046/j.1365-8711.2001.04022.x) ·
**ADS.** [2001MNRAS.322..231K](https://ui.adsabs.harvard.edu/abs/2001MNRAS.322..231K)
```

## Abstract (paraphrased)

Defines an average Galactic-field IMF as a multi-segment power law, with changes in the
power-law index at only two masses ($\sim 0.5\,M_\odot$ and $\sim 0.08\,M_\odot$).
Quantifies how Poisson noise, unresolved binaries, and dynamical evolution introduce
*apparent* scatter in measured power-law indices, and argues that no convincing evidence
for a *variable* IMF exists once these are accounted for. The resulting canonical
broken power law is the standard IMF for star-cluster modelling.

## The canonical IMF (verified against the paper, §2.2)

Kroupa writes the IMF as a piecewise power law (Eqs. 1–2, p. 234)

```{math}
:label: kroupa-imf
\xi(m) \propto m^{-\alpha_i},
```

with, for *single* stars, $\xi(m)\,dm$ the number in $[m, m+dm]$ (Eq. 2):

| segment | slope $\alpha_i$ | mass range $[M_\odot]$ |
|---------|------------------|------------------------|
| $\alpha_0$ | $0.3 \pm 0.7$ | $0.01 \le m < 0.08$ |
| $\alpha_1$ | $1.3 \pm 0.5$ | $0.08 \le m < 0.50$ |
| $\alpha_2$ | $2.3 \pm 0.3$ | $0.50 \le m < 1.00$ |
| $\alpha_3$ | $2.3 \pm 0.7$ | $1.00 \le m$ |

The breaks are at the hydrogen-burning limit ($0.08\,M_\odot$) and at $0.5\,M_\odot$.
Kroupa quotes the original Salpeter value as $\alpha = 2.35$ and adopts the rounded
$\alpha = 2.3 \pm 0.3$ for the high-mass slope (p. 234). The mean stellar mass of this
IMF is $\langle m\rangle = 0.36\,M_\odot$ over $0.01$–$1\,M_\odot$ (p. 235).

Because $\alpha_2 = \alpha_3 = 2.3$, the two highest segments can be merged into a single
$\ge 0.5\,M_\odot$ segment with no change to the IMF.

## Use in progenax

- [](../../10-theory/imfs/classic.md) — the canonical multi-segment broken power law.
- [](../../10-theory/imfs/environment.md) — the 4-segment basis for the
  environment-dependent (IGIMF/Marks) IMF.
- `progenax.imf.PowerLawIMF.kroupa()` — three-segment form
  ($\alpha=[0.3,1.3,2.3]$, breaks $[0.08,0.5]$), the exact merge of Kroupa's
  $\alpha_2=\alpha_3=2.3$.
- `progenax.imf.IMFParams.kroupa()` — explicit four-segment form
  ($\alpha=[0.3,1.3,2.3,2.3]$, breaks $[0.08,0.5,1.0]$), kept separate for
  gradient-based inference of the high-mass slope $\alpha_3$.

## Notes

The canonical IMF in cluster modelling. Maschberger (2013)'s smooth $L_3$ form
approximates it with a single analytic expression. The quoted per-segment
uncertainties ($\approx 99\%$ confidence for $m \ge 0.5\,M_\odot$) set the scale below
which a measured IMF variation is consistent with a universal IMF.

---
title: Salpeter (1955)
description: Annotated reference for E. E. Salpeter — The luminosity function and stellar evolution (the original stellar IMF power law).
---

# Salpeter (1955)

```{admonition} The luminosity function and stellar evolution
:class: note

**Author.** Edwin E. Salpeter (Australian National University; Cornell University).

**Reference.** *The Astrophysical Journal* **121**, 161–167 (1955). Received 1954 July 29.

**DOI.** [10.1086/145971](https://doi.org/10.1086/145971) ·
**ADS.** [1955ApJ...121..161S](https://ui.adsabs.harvard.edu/abs/1955ApJ...121..161S)
```

## Abstract (paraphrased)

Derives the birth-mass distribution of solar-neighbourhood stars by combining the
observed present-day luminosity function with main-sequence evolution timescales.
Salpeter defines the **"original mass function"** $\xi(\mathfrak{M})$ — the number of
stars created per unit *logarithmic* mass interval per unit time — and finds it is a
smooth power law over the fitted range. This single power law is the origin of "the
Salpeter slope," still used as the high-mass slope of essentially all later IMF
parameterisations.

## The original mass function (verified against the paper, §III–IV)

Salpeter defines the original mass function by (Eq. 2, p. 164)

```{math}
:label: salpeter-omf
dN = \xi(\mathfrak{M})\, d(\log_{10}\mathfrak{M})\, \frac{dt}{T_0},
```

and gives the central result (Eq. 5, p. 165)

```{math}
:label: salpeter-slope
\xi(\mathfrak{M}) \approx 0.03\left(\frac{\mathfrak{M}}{\mathfrak{M}_\odot}\right)^{-1.35},
\qquad -0.4 \le \log_{10}(\mathfrak{M}/\mathfrak{M}_\odot) \le +1.0,
```

i.e. the fit is calibrated over roughly $0.4 \lesssim \mathfrak{M}/\mathfrak{M}_\odot
\lesssim 10$.

```{admonition} Convention: −1.35 (per d log m) vs −2.35 (per dm)
:class: important
Salpeter's exponent is **−1.35** because {eq}`salpeter-slope` is written per unit
*logarithmic* mass interval, $\xi(\mathfrak{M})\,d\log\mathfrak{M}$. The number per unit
*linear* mass interval carries one extra power of $m$ (since
$d\log m = dm/(m\ln 10)$):

```{math}
:label: salpeter-linear
\frac{dN}{dm} \propto m^{-2.35}.
```

progenax adopts the **linear** convention $dN/dm \propto m^{-\alpha}$, so the
canonical Salpeter value is **$\alpha = 2.35$**. (Kroupa 2001 and Chabrier 2003 both
state Salpeter as $\alpha = 2.35$ in this convention and then *adopt* the rounded
$\alpha = 2.3$ for their own high-mass tails.)
```

Salpeter explicitly flags that the steeper drop above $\sim 10\,\mathfrak{M}_\odot$
"is not yet clear … whether [it] is a real effect" (p. 165), so the −1.35 slope is best
supported over the fitted $\sim 0.4$–$10\,\mathfrak{M}_\odot$ window.

## Use in progenax

- [](../../10-theory/imfs/classic.md) — Salpeter as the high-mass slope of all IMFs.
- `progenax.imf.PowerLawIMF.salpeter()` — single-segment $\alpha = 2.35$ implementation
  (the only place progenax uses the *original* 2.35; the Chabrier/Maschberger/Kroupa
  classes use the modern rounded $\alpha = 2.3$).

## Notes

Observationally well supported above $\sim 1\,\mathfrak{M}_\odot$. Below that, the real
IMF turns over (lognormal/turnover forms — [](chabrier-2003.md), [](kroupa-2001.md),
[](maschberger-2013.md)). The −1.35 ↔ −2.35 convention is the single most common
source of confusion when comparing IMF papers; always check whether a quoted slope is
per $d\log m$ or per $dm$.

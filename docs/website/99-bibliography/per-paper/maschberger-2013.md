---
title: Maschberger (2013)
description: Annotated reference for T. Maschberger — On the function describing the stellar initial mass function (the L3 IMF with closed-form CDF and quantile).
---

# Maschberger (2013)

```{admonition} On the function describing the stellar initial mass function
:class: note

**Author.** Thomas Maschberger (UJF-Grenoble / CNRS, IPAG).

**Reference.** *Monthly Notices of the Royal Astronomical Society* **429**, 1725–1733
(2013). Accepted 2012 November 20.

**DOI.** [10.1093/mnras/sts479](https://doi.org/10.1093/mnras/sts479) ·
**ADS.** [2013MNRAS.429.1725M](https://ui.adsabs.harvard.edu/abs/2013MNRAS.429.1725M)
```

## Abstract (paraphrased)

Proposes the **$L_3$ IMF**, a three-parameter functional form that is a heavy-tailed
approximation to the lognormal: a low-mass power law and a high-mass power law joined
smoothly. The standard Kroupa (2001) and Chabrier (2003) IMFs are essentially
indistinguishable from it. Its decisive practical advantage is that the **cumulative
distribution function and its inverse (the quantile / mass-generating function) are
closed-form**, so sampling needs no special functions and no Newton iteration.

## The $L_3$ functional form (verified against the paper, Table 1)

With the auxiliary function $G(m)$ (Table 1, Eq. 1),

```{math}
:label: maschberger-G
G(m) = \left(1 + \left(\tfrac{m}{\mu}\right)^{1-\alpha}\right)^{1-\beta},
```

the pdf, CDF, and quantile are (Table 1, Eqs. 2–4)

```{math}
:label: maschberger-pdf
p_{L_3}(m) = A\left(\tfrac{m}{\mu}\right)^{-\alpha}
   \left(1 + \left(\tfrac{m}{\mu}\right)^{1-\alpha}\right)^{-\beta},
\qquad
A = \frac{(1-\alpha)(1-\beta)}{\mu}\,\frac{1}{G(m_u)-G(m_l)},
```

```{math}
:label: maschberger-cdf
P_{L_3}(m) = \frac{G(m)-G(m_l)}{G(m_u)-G(m_l)},
\qquad
m(u) = \mu\left[\Big(u\,[G(m_u)-G(m_l)] + G(m_l)\Big)^{\frac{1}{1-\beta}} - 1\right]^{\frac{1}{1-\alpha}}.
```

The **canonical single-star parameters** (Table 1; system/binary values in parentheses):

| parameter | value | meaning |
|-----------|-------|---------|
| $\alpha$ | $2.3\ (2.3)$ | high-mass exponent |
| $\beta$ | $1.4\ (2.0)$ | low-mass turnover |
| $\mu$ | $0.2\ (0.2)\,M_\odot$ | scale parameter |
| $m_l$ | $0.01\,M_\odot$ | lower limit (normalization) |
| $m_u$ | $150\,M_\odot$ | upper limit (normalization) |

The effective low-mass exponent is $\gamma = \alpha + \beta(1-\alpha) = 0.48$
(single-star; Table 1, Eq. 5), i.e. $p(m)\propto m^{-0.48}$ as $m\to 0$. The limits
$m_l, m_u$ "are only needed for the normalization" (Table 1 caption). Maschberger states
the Salpeter exponent as $\alpha = +2.35$ in the linear convention (§2.1).

## Use in progenax

- [](../../10-theory/imfs/classic.md) — progenax production default; closed-form inverse CDF.
- [](../../10-theory/imfs/binary.md) — single-star backbone for the binary-aware IMF.
- `progenax.imf.Maschberger` — uses the analytic primitive
  $P(m) = \tfrac{\mu}{(1-\alpha)(1-\beta)}\,G(m)$ ({eq}`maschberger-G`) and the
  closed-form quantile {eq}`maschberger-cdf` (no Newton, no error function).
  progenax defaults to $m_\mathrm{max} = 300\,M_\odot$ (vs the paper's $m_u = 150$);
  since the limit only sets the normalization, this is a convention choice.

## Notes

**progenax default IMF.** The closed-form inverse CDF makes it the only canonical IMF
that samples in $\mathcal{O}(1)$ per particle without Newton iterations — important for
fast, differentiable IC generation. The infamous "peak" of the $L_3$ pdf is at the scale
parameter $\mu$, which is *not* where the two power laws cross (Table 1, Eq. 11).

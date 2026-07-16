---
title: Coles & Jones (1991)
description: Annotated reference for Peter Coles & Bernard Jones — the lognormal random field as a model for the cosmic density, and the origin of the Gaussianization machinery.
---

# Coles & Jones (1991)

```{admonition} A lognormal model for the cosmological mass distribution
:class: note

**Authors.** Peter Coles, Bernard Jones

**Reference.** *Monthly Notices of the Royal Astronomical Society* **248, 1–13** (1991).

**DOI.** [10.1093/mnras/248.1.1](https://doi.org/10.1093/mnras/248.1.1)

**Verified.** Summary, §2–3 (definitions, the multiplicative central-limit argument, Eqs. 1–7)
and §5 (correlations of all orders) checked against the held PDF (2026-06). This is the
**founding paper** for two pillars of `gravoturb`: *why* a turbulent density field is
lognormal, and *how* one Gaussian covariance fixes the statistics at all orders.
```

## The big idea

A Gaussian random field can never model the *density* of a self-gravitating medium: a Gaussian
assigns finite probability to $\rho < 0$. Coles & Jones fix this with the **lognormal (LN) random
field** — take a Gaussian field $X(\mathbf{r})$ and exponentiate it,

```{math}
:label: cj-exp
Y(\mathbf{r}) \;=\; \exp\!\big[X(\mathbf{r})\big],
```

so $Y = \rho > 0$ everywhere, the field is fully specified statistically (like a Gaussian, by one
covariance function), yet it becomes arbitrarily close to Gaussian at early times / small variance.
The one-point PDF is the familiar lognormal

```{math}
:label: cj-pdf
f_1(y)\,\mathrm{d}y \;=\; \frac{1}{\sigma\sqrt{2\pi}}\,
\exp\!\left[-\frac{(\log y - \mu)^2}{2\sigma^2}\right]\frac{\mathrm{d}y}{y}.
```

## Why turbulent density is lognormal (the multiplicative CLT)

The deepest part of the paper is its §3.2 argument for *why* this form should arise physically.
The ordinary central-limit theorem says a **sum** of many independent influences tends to a
Gaussian. Coles & Jones build the non-linear analogue: if instead the field is a **product** of
many independent multiplicative influences,

```{math}
:label: cj-mclt
Y \;=\; \prod_{i=1}^{n} X_i
\quad\Longrightarrow\quad
\log Y \;=\; \sum_i \log X_i \;\to\; \mathcal{N}(\mu,\sigma^2),
```

then $\log Y$ is Gaussian, i.e. $Y$ is lognormal. In their words, *"the lognormal is a paradigm for
non-linear noise just as the Gaussian is for linear noise."* In a turbulent cloud each passing shock
multiplies the local density by a random factor; the accumulated product is lognormal. This is the
microphysical origin of the **lognormal body** of the [](burkhart-mocz-2019.md) density PDF, and of
the hierarchical-fragmentation lognormal mass functions of Zinnecker (1984) the paper cites.

## One covariance fixes all orders — the Gaussianization seed

Because $Y$ is a fixed monotone transform of a Gaussian, *every* statistic of $Y$ follows from the
single Gaussian covariance $\xi_X(r)$. For the pure lognormal, §5 gives the exact transformed
two-point function

```{math}
:label: cj-lognormal-xi
1 + \xi_Y(r) \;=\; \exp\!\big[\xi_X(r)\big],
```

and analogous closed forms at all higher orders. This is the **prototype of "Gaussianization"**: the
statistics of a non-Gaussian-but-monotone-mapped field are computable analytically from the
underlying Gaussian correlation. Szapudi & Pan ([](szapudi-pan-2004.md)) generalize {eq}`cj-lognormal-xi`
to an arbitrary monotone map via a Hermite expansion; that generalization is exactly the
[`gaussianized_xi`](../../../../src/experimental/gravoturb/theory/log_correlations.py) series used
to predict $\xi_s(r)$ in progenax.

Coles & Jones also flag a subtlety progenax inherits: the LN field is **not** fully specified by its
moments (the moment problem is indeterminate for the lognormal), so moment-based descriptors are a
poor way to test for non-Gaussianity — a point sharpened later by the Neyrinck/Carron information
analyses ([](neyrinck-2009.md), [](carron-szapudi-2013.md)).

## Use in progenax

- **Theoretical root of the FDF.** The fractal density field is a Gaussian $g$ with $P(k)\propto
  k^{-\beta}$ exponentiated/copula-mapped to a target marginal — precisely the Coles & Jones
  construction {eq}`cj-exp`, generalized from a *pure* lognormal to the BM19 *lognormal + power-law*
  marginal via the rank copula
  ([gaussian_field.py](../../../../src/experimental/gravoturb/realization/gaussian_field.py)).
- **The lognormal body.** {eq}`cj-mclt` is the physical justification for the lognormal core of the
  [](../../10-theory/gravoturbulence/density-pdf-and-fdf.md) / [](burkhart-mocz-2019.md) PDF.
- **The Gaussianization series.** {eq}`cj-lognormal-xi` is the $n=1$, pure-exp special case of the
  general Hermite series in [](../../10-theory/gravoturbulence/inference.md); the log-density two-point $\xi_s$ is
  computed this way, with $\beta$ entering through the Gaussian correlation $\rho_g(r)$.

## Notes

- The lognormal is a **model**, not exact for fully evolved fields — Coles & Jones themselves note
  it produces some spurious skewness. BM19 corrects the high-density end with a self-gravity
  power-law tail; progenax imposes that fuller marginal through the copula.
- §7's Monte-Carlo recipe for point patterns with correlations of all orders (1D/2D/3D) is the
  conceptual ancestor of the FDF star-sampling step
  ([placement.py](../../../../src/experimental/gravoturb/realization/placement.py)).

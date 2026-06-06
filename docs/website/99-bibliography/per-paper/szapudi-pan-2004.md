---
title: Szapudi & Pan (2004)
description: Annotated reference for István Szapudi & Jun Pan — the locally-Poisson counts-in-cells likelihood and the skewed-lognormal (Hermite) Gaussianization that underpin the gravoturb_fdf CIC and 2-point machinery.
---

# Szapudi & Pan (2004)

```{admonition} On recovering the nonlinear bias function from counts-in-cells measurements
:class: note

**Authors.** István Szapudi, Jun Pan

**Reference.** *The Astrophysical Journal* **602, 26–37** (2004).

**DOI.** [10.1086/380920](https://doi.org/10.1086/380920)

**Verified.** Abstract, §1–2 (the locally-Poisson CIC relation Eq. 3, the SLN3 Hermite model Eq. 6,
the Poisson CIC likelihood Eq. 8) and Table 1 checked against the held PDF (2026-06). This paper is
the **direct lineage of two pipeline pieces**: the compound-Poisson count likelihood and the
Hermite Gaussianization of the log-density.
```

## The big idea

The galaxy density is observed as **discrete counts in cells**, not as a continuous field. Szapudi &
Pan connect the two with a *locally-Poisson* model: a cell of mean count $\langle N\rangle$ whose
underlying continuous overdensity is $\delta$ contains $N$ galaxies with probability

```{math}
:label: sp-cic
P_N \;=\; \int_{-1}^{\infty} p(\delta)\,
\frac{[\langle N\rangle(1+\delta)]^{N}\, e^{-\langle N\rangle (1+\delta)}}{N!}\,\mathrm{d}\delta .
```

The discrete count distribution is a **Poisson average of the continuous density PDF** $p(\delta)$.
Their goal is to invert this — recover $p(\delta)$ (and hence galaxy bias) from measured $P_N$ — but
for progenax the forward direction of {eq}`sp-cic` *is* the model: it is exactly the compound-Poisson
[`count_distribution`](../../../../src/experimental/gravoturb_fdf/theory/cic.py) that predicts
counts-in-cells from the BM19 density PDF.

## Core results

**Counts-in-cells likelihood (Eq. 8).** Fitting density-PDF parameters to a measured count histogram
$\tilde P_N$ over $M$ cells uses the Poisson likelihood

```{math}
:label: sp-like
\mathcal{L} \;=\; \prod_N \frac{(M P_N)^{M\tilde P_N}\, e^{-M P_N}}{(M\tilde P_N)!},
```

minimised over the PDF parameters. This is precisely the structure of progenax's
[`count_loglike`](../../../../src/experimental/gravoturb_fdf/inference/likelihood.py) — a 1-point
count likelihood whose high-$N$ tail constrains the density PDF.

**Skewed-lognormal Gaussianization (SLN3, Eq. 6).** To model the continuous $p(\delta)$ they expand
the *log-density* $\Phi = \log\rho - \langle\log\rho\rangle$ in Hermite polynomials about a Gaussian:

```{math}
:label: sp-sln3
p_3(\delta)\,\mathrm{d}\delta = \left[1 + \tfrac{1}{3!}T_3\sigma_\Phi H_3(\nu)
+ \tfrac{1}{4!}T_4\sigma_\Phi^2 H_4(\nu)
+ \tfrac{10}{6!}T_3^2\sigma_\Phi^2 H_6(\nu)\right] G(\nu)\,\mathrm{d}\nu,
```

with $\nu = \Phi/\sigma_\Phi$, $G$ a unit Gaussian, $H_m$ the Hermite polynomials, and $T_3, T_4$ the
renormalised skewness and kurtosis. They show explicitly that the **Gaussian–Edgeworth expansion of
the linear density fails** in the strongly non-linear regime (the tail), which is exactly why the
analysis is done in *log* density — the same reasoning progenax uses to carry the 2-point as
$\xi_s$, not the tail-divergent linear $\langle\rho\rho\rangle$.

## Use in progenax

- **The CIC count model.** {eq}`sp-cic` is the compound-Poisson
  [`count_distribution`](../../../../src/experimental/gravoturb_fdf/theory/cic.py); {eq}`sp-like` is
  [`count_loglike`](../../../../src/experimental/gravoturb_fdf/inference/likelihood.py). These give
  the stellar counts-in-cells block of the [](../../10-theory/gravoturbulence/differentiable-inference.md) likelihood.
- **The Hermite Gaussianization.** {eq}`sp-sln3` is the parametric ancestor of the
  [`gaussianized_xi`](../../../../src/experimental/gravoturb_fdf/theory/gaussianization.py) series:
  both expand the log-density transform in Hermite polynomials. progenax differs in two ways — it
  uses the *exact* BM19 copula map (not a 3-term skewed-lognormal truncation), and it propagates the
  Hermite coefficients into the **two-point** function $\xi_s(r) = \sum_n (c_n^2/n!)\,\rho_g(r)^n$
  rather than only the one-point PDF.

## Notes

- Szapudi & Pan build directly on the lognormal field of [](coles-jones-1991.md) and on Sigad,
  Branchini & Dekel (2000) for the cumulative-CDF bias relation. The Gaussianization information
  argument is taken further by Neyrinck ([](neyrinck-2009.md), [](neyrinck-2011.md)) and Carron &
  Szapudi ([](carron-szapudi-2013.md), [](carron-szapudi-2014.md)).
- The "local Poisson" assumption — counts are Poisson given the local density — is the same
  assumption behind the clean inhomogeneous-Poisson star sampler
  ([sampling.py](../../../../src/experimental/gravoturb_fdf/field/sampling.py)); an *incorrect*
  with-replacement multinomial sampler is what produced the spurious $\alpha$-signal diagnosed in
  the [](../../10-theory/gravoturbulence/differentiable-inference.md) work.

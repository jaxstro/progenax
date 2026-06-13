---
title: The scientific throughline
description: "The unifying idea behind the science demos: a differentiable forward model turns cluster inference into information geometry, telling you BEFORE you observe what is measurable, what is degenerate, and what will bias you. The valuable results are the degeneracies and systematics, not the point estimates — organized here into a measurable / degenerate / biased map of cluster observables."
---

# The scientific throughline

Every demo in this section is the **same inverse problem**: *given a present-day
snapshot of a star cluster, which birth and structural parameters can you actually
recover, and what are the fundamental limits?* That is the inverse of what progenax
does in the forward direction (parameters → initial conditions), run through one
[physics-direct differentiable inference engine](index.md#the-shared-method-physics-direct-differentiable-inference):
forward model → analytic predicted statistic → Gaussian/Poisson likelihood → MLE +
**Fisher information**.

The deeper point is not "we can recover $X$." It is that **a differentiable forward
model turns inference into information geometry.** Because every initial condition is
`jax.grad`-able, the Fisher matrix is computable, and it tells you — *before you
observe* — what is measurable, what is degenerate, and what will bias you. And the
recurring discovery across the demos is that **the astrophysics lives in the
degeneracies and systematics, not the point estimates.** Most of the headline
results are not "we recovered $X$" but "$X$ is degenerate with $Y$," or "$X$ is
biased if you assume the wrong $Z$."

The demos sort onto three axes.

## Measurable — clean recovery, then the catch

These establish that the machinery recovers truth, and then find what the point
estimate hides.

```{list-table}
:header-rows: 1

* - Demo
  - Recovered
  - The catch
* - [IMF + equipartition](imf-equipartition.md) (B2)
  - $(\alpha,\,\delta,\,W_0)$ from masses **and** kinematics
  - kinematics alone give a mild $\alpha$–$\delta$ degeneracy; the **mass channel pins $\alpha$ $4.7\times$ tighter** — multi-channel inference is what makes it well-posed
* - [Halo + core](halo-core.md) (B3)
  - $(t,\,r_a,\,r_h)$ for a two-family cluster
  - the three are *near-orthogonal* (Fisher correlations $\sim$0) — a best case, by contrast with B11 below
* - [King concentration](king-concentration.md) (B11)
  - $(W_0,\,r_c)$ from **star counts alone**
  - $\rho(W_0, r_c) = -0.91$: a more-concentrated/smaller-core model mimics the same counts. Tight marginals **hide a strong covariance** — counts need kinematics to break it
* - [Anisotropy](anisotropy.md) (B6)
  - the Osipkov–Merritt $r_a$ from $\beta(r)$
  - fit the OM form to a *Michie* cluster and you get a **$12.9\times$ $\chi^2$ inflation and a biased $r_a$** — "the cluster has anisotropy radius $X$" is meaningless without naming the DF family
```

## Degenerate — the Fisher matrix is rank-deficient

The strongest results: the inverse problem is *fundamentally* (not statistically)
ill-posed, and the differentiable Fisher exposes it exactly.

[**Birth-environment archaeology**](birth-environment.md) (B5) is the deepest. The
environment-dependent IMF {cite:p}`Marks2012,Jerabkova2018` maps
$(\,[\mathrm{Fe/H}],\,\log M_{\rm ecl},\,\mathrm{sfe})\to\alpha_3$, the high-mass
slope. $\alpha_3$ is *trivially* measurable (a $\sim$40$\sigma$ top-heavy detection
at $N=10^4$), but the map is **three-to-one**, so the environment-space Fisher is
**exactly rank 1** — two machine-precision-zero eigenvalues. A continuum of birth
environments produces the *identical* mass function. You cannot read a cluster's
birth metallicity, mass, or star-formation efficiency off its IMF slope without an
external constraint on two of the three. This is a caution for any claim that
infers birth conditions from a present-day mass function.

[**Rotation + projection**](rotation.md) (B8) is the same rank-deficiency from
*geometry*: a cluster rotating at rate $\omega$ viewed at inclination $i$ has a mean
line-of-sight velocity $\propto \omega\sin i$, so $\langle v_{\rm los}\rangle$
constrains only the **product** — the $(\omega, i)$ Fisher is rank 1. Cluster
rotation amplitudes from line-of-sight velocities are lower limits, degenerate with
inclination; breaking it needs the projected flattening, which velocities cannot
supply.

## Biased — wrong-physics systematics

[**Unresolved-binary mass function**](binary-mass-function.md) (B4) is the headline.
Unresolved binaries blend photometrically (their summed light inflates the inferred
mass), distorting the observed mass function. A controlled experiment — shuffle the
mass ratio $q$ against the period $P$ to preserve both marginals and change *only*
the correlation — isolates the Moe & Di Stefano {cite:p}`MoeDiStefano2017` $P$–$q$
coupling. **Ignoring it biases the binary fraction low by $\sim$5% ($3.6\sigma$ at
survey scale)**, because the blended (close) subset is more equal-mass under the
real coupling than an independent model assumes. It is a *systematic* — fixed
fractional bias whose significance grows as $\sqrt N$. Binary-fraction measurements
from the photometric binary sequence that assume an independent $q$-distribution are
biased; you need the realistic $P$–$q$–$e$ covariance.

The [wrong-IMF curve](imf-equipartition.md) (B2d) and the OM-vs-Michie misfit
([B6](anisotropy.md)) are the same lesson in other channels: assume the wrong
generative physics and the recovered parameter shifts in a quantifiable,
detectable way.

## Honest diagnostics — what the observables actually carry

- [**Binary energy budget**](binary-energy-budget.md) (B9): the internal binary
  binding energy is a *separate reservoir* that dwarfs the cluster potential
  (12–1900×) and is environment-dependent — the seed of binary-burning / core-collapse
  arrest. The naive resolved virial ratio is contaminated by orbital phase and is
  **not** the cluster's $Q$; virialize binary ICs on the system centres-of-mass.
- [**Differentiable diagnostics**](diff-diagnostics.md) (B10): the JAX surrogates for
  the CW04 substructure $Q$ {cite:p}`Cartwright2004` and the Allison mass-segregation
  ratio {cite:p}`Allison2009` track their non-differentiable MST counterparts — so
  "substructure" and "segregation" can enter a *gradient-based* loss they never could
  before.
- [**Tidal radius**](tidal-radius.md) (B7): $93\%$ of the truncation-radius
  information lives in the few outermost (count-limited) bins, so the Poisson
  treatment is essential — and a cluster's *orbit* (Galactocentric distance, via the
  Jacobi radius {cite:p}`King1962`) is encoded in its faint outer star counts.
- [**Cross-engine agreement**](cross-engine.md) (B1): two independent engines build
  the *same* cluster — the consistency check the recoveries rest on.

## What this is, and what it is not

```{warning}
**These are clean mocks.** No measurement error, selection, incompleteness, or PSF
(except B8's single projection effect). So the *degeneracies* (B5, B8, B11) are
**floors** — fundamental, and they survive contact with real data — while the
*measurable* numbers (e.g. $\alpha_3$ at $40\sigma$) are optimistic ceilings that
realistic observations only degrade. That asymmetry is what makes the degeneracy
results the durable, transferable ones.

**progenax is initial-conditions only.** Every result here is a *birth-state*
inference. Real clusters are dynamically processed (binary burning, segregation
growth, tidal stripping), which *adds* degeneracy on top of these; the evolved-cluster
version needs an N-body forward chain.

**Some forward models use stand-ins** (Tout+1996 ZAMS as a photometry proxy, a
point-mass Galaxy for the Jacobi step, a fixed EFF slope in B7). These do not change
the qualitative information geometry — the degeneracy/bias structure is robust — but
the exact numbers are illustrative, not predictions for a specific survey.
```

## The implication

The reason to build *differentiable* cluster initial conditions is not faster point
estimation. It is that **the Fisher information of a differentiable forward model is a
forecasting instrument**: for a given observable and sample size, it tells you what a
cluster survey can and cannot claim. These demos are a first map of that information
content for star-cluster observables — and the most valuable entries on the map are
the ones that say *"this is degenerate"* (birth environment $\not\leftrightarrow$ IMF
slope; rotation $\not\leftrightarrow$ inclination) and *"this is biased if you assume
the wrong physics"* (binary fraction under the wrong $P$–$q$ coupling; anisotropy under
the wrong DF). Those are cautions an observer can carry to real data.

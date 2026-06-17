---
title: Optimal experimental design — designing the observation before you take it
subtitle: Where to spend telescope time, derived from a differentiable forward model rather than from heritage
description: "Landing page for progenax's optimal-experimental-design (OED) section. OED optimises the OBSERVATION itself, not just the analysis afterwards: because a differentiable forward model can compute d(information)/d(observing strategy), the precision you will achieve becomes a known function of the design -- before any photon is collected. This page frames the telescope-time hook, maps the full space of OED problems progenax can pose (design levers, science targets, optimality criteria, robustness), and indexes the worked examples: the anisotropy radius (where to put proper motions) and the dynamical mass (how deep to survey)."
---

# Optimal experimental design

Every other demo on this site answers the same kind of question: *given data, can you
recover a parameter?* This section asks the question that comes **before any data
exist**. You have been awarded a fixed amount of telescope time — call it a budget of
$N$ stars you can afford to measure. Where on the sky should you point, how faint
should you go, and *which* measurement (a radial velocity? a proper motion?) should
you make on each star, so that you learn the most about the one number you actually
care about?

That is **optimal experimental design** (OED): you optimise the *observation itself*,
not just the analysis you do afterwards. It is rare in astronomy — almost no simulator
can compute $\partial(\text{information})/\partial(\text{observing strategy})$ — and it
is exactly what a *differentiable* forward model makes possible. The precision you will
achieve is a *known function of the observing strategy*, computable **before you
collect a single photon**, through the Fisher information. So you can treat the
observing strategy as a free variable and maximise the information you expect to
extract. The output is not just "a good plan"; it is a *quantified* plan: "this
allocation reaches the same precision with $3.7\times$ fewer stars," or "this is the
exact depth at which more telescope time stops helping."

:::{note} **Start here**
The shared formalism — Fisher information, the additive design-linear backbone, the
c/D/A criteria, and the sky-projection geometry — is built once on
[the OED formalism page](background.md). Read it first if you want the machinery; each
worked example `{ref}`s it rather than re-deriving it.
:::

## The worked examples

```{list-table}
:header-rows: 1
:label: tbl-oed-examples

* - Example
  - Design lever
  - Target
  - Headline
* - [Anisotropy — where the proper motions belong](anisotropy.md)
  - radial × channel (RV ↔ PM) allocation
  - anisotropy radius $r_a$
  - PMs to the outskirts; $3.66\times$ fewer stars at equal precision on $r_a$
* - [Dynamical mass — how deep to survey](dynamical-mass.md)
  - survey **depth** (limiting magnitude $m_{\rm lim}$)
  - dynamical mass $M$
  - an **interior** optimal depth ($m_{\rm lim}\approx13.3$); deeper *or* shallower is worse
```

## What OED can do with progenax

The examples above are two instances of a general capability: **every differentiable
`progenax` forward model, paired with a likelihood, is a Fisher matrix
{eq}`oed-fisher-gaussian` — and therefore an OED problem.** Because the Fisher is
[additive](background.md#sec-oed-additive), you assemble the design Fisher for *any* of
these by computing per-star (or per-bin) blocks once and optimising weights. Below,
**[done]** is demonstrated in this section, **[B#]** means an existing demo already
provides the differentiable forward model, and **[enabled]** means the machinery
supports it but it is not yet built. We mark the line honestly between *demonstrated*
and *possible*.

### By design space — *what you optimise*

```{list-table}
:header-rows: 1
:label: tbl-oed-designspace

* - Design lever
  - What it allocates
  - Status
* - **Channel allocation** (RV $\leftrightarrow$ PM)
  - kinematic measurement type per star
  - **[done — the anisotropy example]**
* - **Radial / spatial allocation**
  - where on the sky to observe
  - **[done]**; generalises to number-count surveys (King/EFF $W_0$, $r_c$, **$r_t$** from binned counts via the Poisson channel) **[B11, B7]**
* - **Survey depth / magnitude limit**
  - how faint to go (ZAMS $L$ $\to$ detectability), trading supply vs noise
  - **[done — the dynamical-mass example]**
* - **Epochs / cadence / cost**
  - astrometric epochs (PM precision $\propto$ epochs), spectroscopic cadence, under a cost model
  - **[enabled — Stage 3]**
* - **Multi-channel fusion**
  - the optimal *mix of instruments* — Fisher sums across photometry + kinematics + counts
  - **[enabled]**
```

### By science target — *what parameter you measure*

```{list-table}
:header-rows: 1
:label: tbl-oed-targets

* - Target
  - Forward model
  - Status
* - Anisotropy $r_a$ / mass $M$ / $r_h$ (kinematics)
  - `project_dispersion` (B&M82)
  - **[done]** — $r_a$ and $M$ are the two examples here
* - IMF slope $\alpha$ / environment $\alpha_3$
  - IMF likelihood + mass function
  - **[B5]** — design which mass range / how many stars pin the high-mass slope
* - Binary fraction $f_b$ + Moe $P$–$q$–$e$
  - binary-inflated dispersion / mass function
  - **[B12, B4]** — allocate RV epochs + photometry to constrain binarity
* - Concentration, tidal radius $r_t$ $\to$ Jacobi $R_{\rm gal}$
  - count profile + tidal truncation
  - **[B7]** — outer-bin allocation; 93% of $r_t$ info is in the outskirts
* - Rotation $\omega\sin i$
  - rotating projected-kinematics model
  - **[B8]** — break the rank-1 $(\omega,i)$ degeneracy with multiple channels
* - Multi-population (halo+core, mass segregation)
  - `MultiComponentCluster` (Engine A/B)
  - **[enabled]** — where to observe to *separate* species
```

### By optimality criterion — *what "best" means*

**c / $D_s$** (single target or a subset of targets) — **[done, both headlines]**.
**D** (all parameters) and **A** (average) — **[done, as the contrast]**. **E**
(worst-constrained eigendirection), **I/G** (prediction-oriented) — same $F=\sum n\,M$
machinery, **[enabled]**. The standout is **T-optimality (model discrimination)**:
`progenax` carries *multiple* differentiable forward models for the same observable —
OM-Jeans vs the exact Michie second moment (`df_moment_dispersion`), Engine-A vs
Engine-B, OM vs a native $\beta(r)$. The [anisotropy recovery demo (B6)](../anisotropy.md)
already shows OM and Michie *diverge in the outskirts*; a T-optimal design **maximises
that divergence**, telling you where to observe to *distinguish the models*, not merely
fit one. **[enabled — high value]**.

### By robustness and adaptivity

- **Bayesian OED** — expected information averaged over a prior on the nuisances; the
  priors are already in the Fisher. **[enabled]**
- **Robust / maximin OED** — optimise the *worst case* over model or nuisance
  uncertainty; the honest answer to the model-dependence caveat. **[enabled]**
- **Sequential / adaptive OED** — re-optimise as data arrive (greedy or batch); the
  differentiable Fisher is the per-step ingredient. **[enabled, not built]**

The throughline: `progenax` is unusual in being a **fully differentiable
astrophysical IC-and-observable stack**, so
$\partial(\text{information})/\partial(\text{observing strategy})$ is computable for
essentially any cluster-science parameter — which turns "how should we observe?" from a
heritage decision into a gradient-ascent problem across all of the above.

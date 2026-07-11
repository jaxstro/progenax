---
title: Optimal experimental design — designing the observation before you take it
subtitle: Where to spend telescope time, derived from a differentiable forward model rather than from heritage
description: "Landing page for progenax's optimal-experimental-design (OED) section. OED optimises the OBSERVATION itself, not just the analysis afterwards: because a differentiable forward model can compute d(information)/d(observing strategy), the precision you will achieve becomes a known function of the design -- before any photon is collected. This page frames the telescope-time hook and indexes the worked examples: the anisotropy radius (where to put proper motions), the dynamical mass (how deep to survey), the concentration (proper motions to the core), and a robustness design (binaries biasing a mass estimate). The designs were prototyped to demonstrate the capability; the OED tooling is planned for a separate package and is not part of v0.1.0."
---

# Optimal experimental design

:::{warning} Deprecation — this OED demo is migrating to informax
The OED tooling has been ported to the dedicated inference-design package
**informax** (developed alongside progenax). This page and its
`scripts/` harness are retained temporarily while the port is being
verified, and will be removed from progenax afterwards. The demo's
dedicated test suite was retired in 2026-07 (the release gate no longer
exercises it), so treat the scripts here as frozen reference copies.
:::


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
allocation reaches the same precision with several-fold fewer stars," or "this is the
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
* - [Concentration — where the proper motions belong, redux](concentration.md)
  - radial × channel (RV ↔ PM) allocation
  - concentration $W_0$
  - PMs to the **core** (the mirror image of anisotropy); $\approx3.4\times$ fewer stars at equal precision on $W_0$
* - [Robustness — when binaries lie to your mass](binary-robustness.md)
  - binary-aware vs binary-**blind** model (radial RV allocation)
  - dynamical mass $M$ under misspecification
  - a binary-blind design biases $M$ by **$+184\%$** with **$41\times$** false confidence; the binary-aware design removes it
```

## What OED can do with progenax

The worked examples above are instances of one general capability: **every
differentiable `progenax` forward model, paired with a likelihood, is a Fisher matrix
{eq}`oed-fisher-gaussian` — and therefore an OED problem.** Because the Fisher is
[additive](background.md#sec-oed-additive), the design Fisher for any cluster-science
parameter is assembled by computing per-star (or per-bin) blocks once and optimising the
weights. The same machinery extends to other design levers (survey depth, observing
epochs and cadence, multi-instrument fusion), other targets (the IMF slope, binary
fraction, tidal radius, rotation, multi-population separation), and other optimality
criteria (D/A for all parameters, model-discrimination designs across the multiple
forward models `progenax` carries for the same observable). Since the forward stack is
differentiable end to end, $\partial(\text{information})/\partial(\text{observing
strategy})$ is computable for essentially any of these — which turns "how should we
observe?" from a heritage decision into a gradient-ascent problem.

The four designs above were prototyped to demonstrate this; the OED tooling itself is
planned for a separate package and is not part of progenax v0.1.0.

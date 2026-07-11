---
title: Getting started
description: Onboarding path for new progenax users — installation, first cluster IC in 5 minutes, gradient demo, IMF sampling, and the glossary.
---
# Getting started

This section is the **onboarding path** for new progenax users. The
progression is short and concrete: install the package, build your
first Plummer sphere, take a gradient through it, sample an IMF,
and along the way pick up the glossary terms you'll see throughout
the rest of the docs.

## Reading order

```{list-table}
:header-rows: 1

* - Page
  - What you'll do
* - [](science-capabilities.md)
  - Survey what progenax can model (the model inventory with validity regimes)
* - [](installation.md)
  - Install progenax via UV (preferred) or pip; verify with the smoke test
* - [](first-plummer-sphere.md)
  - Build a 1000-particle Plummer cluster from scratch in 5 minutes
* - [](differentiable-ic.md)
  - Take `jax.grad` through the IC builder; see HMC inference in action
* - [](imf-sampling.md)
  - Sample masses from Salpeter, Kroupa, Chabrier, Maschberger, and visualise
* - [](glossary.md)
  - Definitions of every term used in the rest of the docs
* - [](whats-new.md)
  - Release-style changelog
```

## Who this section is for

- **Grad students new to differentiable astrophysics.** Read in order.
- **Researchers familiar with N-body codes (NBODY6, COSMIC, McLuster).**
  Skim [](installation.md), then jump to
  [](../20-architecture/units-policy.md) and
  [](../20-architecture/q-virial-convention.md) — those are the
  convention differences most likely to bite.
- **Returning users.** [](whats-new.md) is the changelog; the rest
  of this section is for first-time users.

## What progenax is

progenax generates **initial conditions** (ICs) for N-body simulations
of star clusters, stellar populations, and binary systems. Every IC
is fully differentiable through `jax.grad`, JIT-compilable through
`jax.jit`, and vectorisable through `jax.vmap` — the four operations
that make HMC inference of cluster parameters tractable.

The package is part of the [jaxstro
ecosystem](https://github.com/jaxstro): progenax for
ICs, gravax for N-body integration, fluxax for photometry, with
stellax (stellar evolution) and startrax (binary population synthesis)
planned.

## What progenax is *not*

progenax does **not** evolve simulations forward in time — that is
gravax's job. The handoff is documented in
[](../40-howto/interface-with-gravax.md). progenax also does not
compute mock observations from final snapshots — that is fluxax
(`fluxax.render`; the old `gravax.render` monolith is legacy).

## Next step

Open [](installation.md) and run the smoke test. The rest of this
section assumes a working install.

---
title: progenax
description: Differentiable initial conditions for N-body simulations
---

# progenax

> Differentiable initial conditions for N-body simulations of star clusters,
> stellar populations, and binary systems, implemented in JAX.

`progenax` is part of the [jaxstro](https://github.com/jaxstro)
ecosystem. It constructs particle masses, positions, and velocities from specified
models and random keys. Differentiation depends on the sampling path and parameter;
individual methods document their smooth regions, numerical safeguards, and gradient
checks. Reproducing a realization requires its configuration and software environment
as well as its random key.

Gravoturbulent /
fractal-density-field substructure is an **experimental** follow-up-paper
feature in the repo-only `gravoturb` package (not in the released wheel).

This site connects theory, numerical methods, architecture, API usage, and validation.
Start with a tutorial to build a cluster, then follow its model and implementation
references to understand the assumptions and checks.

```{figure} 10-theory/figures/phase_space_hexbin.webp
:width: 100%

Sampled radial phase space for the models shown. Escape-speed envelopes mark the
velocity support of the corresponding potentials; remaining inside an envelope is
one diagnostic, not a complete equilibrium test. See the
[velocity-DF theory chapter](10-theory/velocity-dfs/index.md).
```

## Find your reading path

::::{grid} 1 1 3 3

:::{card} 🚀 Get started
:link: 00-getting-started/index.md

**For students new to differentiable astrophysics or JAX.** A guided
narrative path: install, build your first Plummer sphere in 5 minutes,
take a gradient through it, sample an IMF, and understand the glossary
along the way.
:::

:::{card} 📚 Theory & methods
:link: 10-theory/index.md

**For researchers.** Physical models, their assumptions, and numerical methods,
with links to sources, implementations, and validation evidence.

:::

:::{card} 🔧 API reference
:link: 30-api/index.md

**For daily use.** The API reference documents public symbols; theory pages explain
their physical conventions and validation pages record the checks performed.
Search the [full symbol index](30-api/full-symbol-index.md) or browse by module.

:::

::::

## Other entry points

- **[Architecture & design](20-architecture/index.md)** — why progenax is
  shaped the way it is. JAX-native philosophy, three-brick state pattern,
  protocol-based composition, units policy, virial-Q convention.
- **[How-to recipes](40-howto/index.md)** — task-oriented snippets for
  common workflows.
- **[Validation](50-validation/index.md)** — what the test suite asserts,
  and a gallery of physics-validation plots.
- **[Development log](90-development-log/index.md)** — historical decisions
  and notable bug fixes.
- **[Bibliography](99-bibliography/bibliography.md)** — full reference list.

## What progenax is *not*

`progenax` generates initial conditions. It does **not** integrate them
forward in time — for that you want
[gravax](https://github.com/jaxstro) (N-body integrators)
or [stellax](https://github.com/jaxstro) (stellar evolution,
planned). The "Interface with gravax" recipe in
[How-to](40-howto/interface-with-gravax.md) shows the handoff.

## Citing progenax

If you use `progenax` in published work, please cite the package and the
papers underlying any models you use. The
[bibliography](99-bibliography/bibliography.md) page lists the canonical
references for every model in the package.

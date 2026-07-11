---
title: progenax
description: Differentiable initial conditions for N-body simulations
---

# progenax

> Differentiable initial conditions for N-body simulations of star clusters,
> stellar populations, and binary systems — JAX-native, gradient-friendly,
> physically rigorous.

`progenax` is part of the [jaxstro](https://github.com/jaxstro)
ecosystem. Every initial condition you generate — Plummer or King spatial
profiles, isotropic or anisotropic velocity distributions, IMF samples, and
binary orbits — is differentiable through `jax.grad`, vectorisable through
`jax.vmap`, and JIT-compilable through `jax.jit`. Gravoturbulent /
fractal-density-field substructure is an **experimental** follow-up-paper
feature in the repo-only `gravoturb_fdf` package (not in the released wheel).

This site is the **single source of truth** for the package: theory,
architecture, API, tutorials, and history. Whether you arrived to learn
how to build your first cluster, to dig into the (experimental) gravoturbulent
freefall-density factor formalism, to understand why we chose a protocol-based composition
pattern over inheritance, or to look up a specific function signature —
there's a path for you.


```{figure} 10-theory/figures/phase_space_hexbin.webp
:width: 100%

Every model on this site is sampled, verified, and drawn: $4\times10^5$-star
equilibria on the $(r, v_r)$ phase-space plane, with the analytic escape
envelope computed three independent ways and no star crossing it. From the
[velocity-DF theory chapter](10-theory/velocity-dfs/index.md); every figure
regenerates from `laboratory/icviz`.
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

**For researchers wanting full depth.** Derivations from first principles
for every model in the package — spatial profiles, velocity DFs, IMFs,
binaries, tidal physics, gravoturbulence — each anchored on the original
papers via {cite}-rendered citations.
:::

:::{card} 🔧 API reference
:link: 30-api/index.md

**For daily use.** Auto-generated reference for every public symbol,
regenerated on each build so it can never drift from the source. Search
the [full symbol index](30-api/full-symbol-index.md) or browse by module.
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

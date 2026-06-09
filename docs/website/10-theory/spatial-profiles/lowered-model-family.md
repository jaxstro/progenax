---
title: "Roadmap: the differentiable lowered-model family"
description: "Planned progenax generalization of King/Wilson/Woolley into one differentiable, JAX-native lowered-model family (multi-mass + anisotropy), following the Gieles & Zocchi (2015) formalism — implemented natively, not as an external dependency."
---
# Roadmap: the differentiable lowered-model family

:::{admonition} Status — planned (post-MVP)
:class: note
This page describes **future** progenax work. The released package ships four
validated spatial profiles — Plummer, [King](king.md), [EFF](eff.md), and the
anisotropic [Michie](../velocity-dfs/michie-king.md) model. The unified
lowered-model family below is **not yet implemented**; it is the natural next
extension once the single-model profiles are locked for release.
:::

## The idea

The classical truncated-isothermal models — Woolley (no energy truncation),
{cite:t}`King1966` (a linear lowered Maxwellian), and Wilson (a quadratic
lowered form) — are members of **one** family distinguished by how sharply the
distribution function is truncated near the escape energy. {cite:t}`Gieles2015`
made this explicit with a single continuous **truncation parameter** $g$: their
LIMEPY DF reads

```{math}
:label: limepy-df
f(E) \;\propto\; e^{-E/\sigma^2}\,
\gamma\!\Big(g,\; \tfrac{E_{\rm cut}-E}{\sigma^2}\Big),
```

with $g=0$ recovering the Woolley cutoff, $g=1$ the King model, and $g=2$ the
Wilson model, interpolating smoothly in between. The same framework extends
naturally to **multiple mass components** (each mass group its own
$\sigma_j$, sharing one potential) and to **radial anisotropy** (an
Osipkov–Merritt / Michie term).

## Why progenax will reimplement it (rather than depend on LIMEPY)

{cite:t}`Gieles2015` is the reference *formalism*, and the published `limepy`
code is the standard numpy/scipy implementation. progenax does **not** wrap or
depend on it, for one decisive reason: **`limepy` is not differentiable.** The
entire progenax thesis is JAX-native, end-to-end-differentiable initial
conditions, so that structural parameters can be **inferred** from data by
gradient descent or HMC (see [](../../20-architecture/differentiability.md)).

A differentiable lowered-model family would let a single continuous parameter
vector — $(g, W_0, r_c, M, \{\sigma_j\}, r_a)$ — be fit jointly to an observed
cluster, with $g$ itself a *fitted* quantity that selects the truncation sharpness
the data prefer (King-vs-Wilson as a posterior, not a modelling choice). That is
only possible if $\partial(\text{model})/\partial g$ flows through the Poisson
solve — exactly what progenax's `diffrax`-based King ODE already demonstrates for
$\partial/\partial W_0$.

## Planned scope

```{list-table}
:header-rows: 1

* - Capability
  - Status today
  - Planned
* - King ($g=1$), single mass
  - ✅ released ([King](king.md))
  - —
* - Continuous $g$ (Woolley $\to$ King $\to$ Wilson)
  - ✗
  - generalize the dimensionless Poisson ODE in $g$; differentiable in $g$
* - Multi-mass equilibrium (per-group $\sigma_j$)
  - ✗ (only superposition via [two-component](../populations/two-component.md))
  - one self-consistent potential, mass-dependent truncation
* - Radial anisotropy (Osipkov–Merritt / Michie)
  - ✅ for Plummer/EFF/[Michie](../velocity-dfs/michie-king.md)
  - fold into the unified DF
* - Differentiable structural inference
  - ✅ $(W_0, r_c, M)$ for King
  - extend to $(g, \{\sigma_j\}, r_a)$
```

## Relationship to the other roadmap item

This is independent of, but complementary to, the deferred **differentiable
tidal radius** $\partial r_t/\partial W_0$
([](../../20-architecture/differentiability.md#roadmap-differentiable-rt)). A
unified family makes $r_t$ a function of $(g, W_0)$; the implicit-function-theorem
treatment of the $\psi=0$ crossing carries over unchanged.

## References

{cite:t}`Gieles2015` (LIMEPY) is the lowered-model-family formalism; the
single-model members are {cite:t}`King1966` (and Woolley 1954 / Wilson 1975 for
the $g=0,2$ endpoints). The per-paper note is at
[](../../99-bibliography/per-paper/king-1966.md).

---
title: Two-component populations (superseded)
description: "The TwoComponentConfig / generate_two_component_cluster API was retired in the 2026-06 unified redesign; two-component capability now lives in MultiComponentCluster (Engines A and B), which fixes the original API's shared-potential physics bug."
---
# Two-component populations (superseded)

```{admonition} Superseded — two-component capability now lives in `MultiComponentCluster`
:class: important
The `TwoComponentConfig` / `generate_two_component_cluster` API this page used
to validate was **deleted** in the 2026-06 unified multi-component redesign
(together with its unit suite `tests/unit/test_populations.py`). It carried a
real physics bug: each sub-population's velocity DF was fed the **full**
cluster mass (an isolated superposition, never a shared potential), with no
COM/virial finalisation — silent for equal masses, non-self-consistent for
unequal-mass populations. Its replacement solves exactly that problem: every
component is drawn from its own DF in **one shared self-consistent
potential**.
```

## Where the capability went

```{list-table}
:header-rows: 1

* - Need
  - Use now
  - Validation page
* - Two (or more) populations in one self-consistent potential, DF-defined
    (lowered-isothermal/LIMEPY family; GC 1G/2G, halo+core, mass segregation
    via $\delta$)
  - `MultiComponentCluster.from_components` / `.from_mass_segregation` /
    `.from_imf` (**Engine A**)
  - [](multimass-equilibrium.md)
* - Two (or more) populations with *prescribed densities* (Plummer/EFF/King
    mixes) in one shared potential, DFs by Eddington inversion
  - `MultiComponentCluster.from_density_profiles` (**Engine B**)
  - [](engine-b-eddington.md)
```

## The two-component check that survives

`scripts/validate_cluster_ic.py` §2 (rewired 2026-06-09 onto
`MultiComponentCluster`, re-run 2026-06-10, **ALL PASS**) exercises the
canonical two-population case — a cold ($w=0.7$) and a hot ($w=1.0$)
population in one shared $W_0=7$ potential:

```{list-table}
:header-rows: 1

* - Property
  - Expected
  - Measured
  - Gate
* - Radial separation (cold concentrated)
  - median $r_{\rm cold} <$ median $r_{\rm hot}$
  - $1.50$ pc $< 8.36$ pc
  - strict inequality
* - Per-component virial $Q_j$ (exact-quadrature oracle)
  - $[0.5, 0.5]$
  - $[0.5000, 0.5000]$
  - $|Q_j - 0.5| < 5\times10^{-3}$
```

:::{figure} figures/cluster_ic_two_component.png
:label: val-two-component
:width: 80%
:align: center

`MultiComponentCluster.from_components`: radial distributions of a cold
($w=0.7$, concentrated, median $r = 1.50$ pc) and a hot ($w=1.0$, extended,
median $r = 8.36$ pc) population sampled from ONE shared self-consistent
potential, each in detailed equilibrium ($Q_j = [0.5, 0.5]$ by the
exact-quadrature oracle). From `scripts/validate_cluster_ic.py` §2
(2026-06-10 run).
:::

```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_cluster_ic.py
```

## Supersession history

- **Pre-2026-06**: `generate_two_component_cluster` superposed two
  independently generated populations; unit-tested only, flagged on this page
  as "unit-backed, not a dedicated validation suite". The earlier figure here
  showed the configured radial separation (median $r_A = 0.52 < r_B = 2.06$
  pc) but each population's DF saw the full cluster mass.
- **2026-06-09 (Phase 1c of the unified redesign)**: the module
  (`src/progenax/populations.py`) was deleted outright — no
  backwards-compatibility shim — after an adversarial review identified the
  shared-potential bug; `validate_cluster_ic.py` §2 was rewired to
  `MultiComponentCluster` and the figure above regenerated from the new API.

## References

Theory at [](../10-theory/populations/two-component.md). The exact
self-consistent multi-component DF family is {cite:t}`Gieles2015` (LIMEPY);
progenax's own differentiable implementation is validated at
[](multimass-equilibrium.md) and [](engine-b-eddington.md).

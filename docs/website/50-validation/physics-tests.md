---
title: Cross-cutting physics tests
description: Physics properties that span multiple modules — virial-Q recovery across profile/DF combinations, energy conservation under joint modifiers, the units-policy and mass-first API contracts — and where each is asserted in the unit/integration suites.
---
# Cross-cutting physics tests

Some physics properties are not specific to one module — they are
*contracts across* modules: that any profile pairs with any DF and
still virializes, that the units policy holds end-to-end, that
gradients flow through the whole pipeline. progenax does **not** carry
a dedicated `tests/validation/test_cross_cutting.py`; instead these
cross-cutting properties are covered **indirectly** by the unit and
integration suites listed below. This page documents *which* property
each suite enforces, so the cross-cutting coverage is auditable rather
than assumed.

## What is covered, and where

```{list-table}
:header-rows: 1

* - Cross-cutting property
  - Asserted in
* - Virial $Q = T/|V|$ recovery for matched profile/DF pairs (and
    predictable departure for mismatched pairs)
  - `tests/unit/test_builders.py`,
    `tests/unit/builders/test_cluster_builders.py` (plus the per-model
    virial rows on the module pages — [](plummer-equilibrium.md),
    [](king-profile.md), [](eff-profile.md))
* - Energy conservation through builder → modifier chains (tidal
    truncation, mass segregation)
  - `tests/unit/test_tidal.py`, `tests/integration/test_end_to_end.py`
* - Units-policy contract — no implicit `G`/units defaults leak into
    core APIs; explicit $G$ threads unchanged through the pipeline
  - `tests/integration/test_units_through_pipeline.py`
* - Differentiability across the full pipeline — finite `jax.grad`
    at every layer
  - `tests/integration/test_jax_compatibility.py` (and the per-entry-point
    autodiff-vs-FD registry on [](differentiability-audit.md))
* - Mass-first API contract — all builders accept `masses` first
    (exercised by every builder call site)
  - `tests/unit/test_builders.py`,
    `tests/unit/builders/test_cluster_builders.py`
```

## The profile × DF virial contract

The defining cross-cutting check is that **any** spatial profile pairs
with **any** velocity DF (protocol-based composition), and a *matched*
pair sits at virial equilibrium with no external rescale. The expected
behaviour the builder/cluster suites assert:

- **Matched pairs** (Plummer⊗Plummer at the same $r_h$, King⊗King at
  the same $W_0$, EFF⊗EFF) recover $Q = T/|V| \approx 0.5$ to within
  finite-$N$ Poisson noise ($\sim 5\times10^{-3}$ at $N = 10^4$) — the
  unscaled DF is a true equilibrium.
- **Mismatched pairs** (e.g. Plummer positions at $r_h=1$ with Plummer
  velocities scaled to $r_h=2$) land *predictably away* from $0.5$ —
  the suite checks the departure is in the expected direction, which is
  what makes the matched-pair result meaningful rather than imposed.

The per-model virial rows on the module validation pages
([](plummer-equilibrium.md), [](king-profile.md), [](eff-profile.md))
carry the *measured* numbers and tolerances; this page records that the
*composition* across profile/DF combinations is exercised, not just the
diagonal matched cases.

## How to run

```bash
pytest tests/unit/test_builders.py -v
pytest tests/integration/test_units_through_pipeline.py -v
pytest tests/integration/test_jax_compatibility.py -v
pytest tests/integration/test_end_to_end.py -v
```

## References

The per-module measured values live on the per-module validation pages
([](plummer-equilibrium.md), [](king-profile.md), …); the three-tier
methodology and tolerance conventions are at [](methodology.md); the
end-to-end gradient registry is at [](differentiability-audit.md). This
page maps the *interaction* contracts onto the suites that enforce them.

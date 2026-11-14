---
title: Cross-cutting physics tests
description: Physics properties that span multiple modules — virial-Q recovery across profile/DF combinations, energy conservation under joint modifiers, the units-policy and mass-first API contracts — and where each is asserted in the unit/integration suites.
---
# Cross-cutting physics tests

Some physics properties are not specific to one module — they are
*contracts across* modules: that any profile pairs with any DF and
can be passed through the same builder interface, that selected workflows
retain their unit conventions, and that selected gradient paths are checked. progenax does **not** carry
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
* - Array/energy bookkeeping through builder → modifier chains (tidal
    cuts change masses and energy; they are not an energy-conserving evolution)
  - `tests/unit/test_tidal.py`, `tests/integration/test_end_to_end.py`
* - Units-policy contract — no implicit `G`/units defaults leak into
    core APIs; explicit $G$ threads unchanged through the pipeline
  - `tests/integration/test_units_through_pipeline.py`
* - Selected composed gradient paths — tracing/finite-value checks are
    distinguished from numerical derivative agreement
  - `tests/integration/test_jax_compatibility.py` (and the per-entry-point
    autodiff-vs-FD registry on [](differentiability-audit.md))
* - Low-level mass-array API contract — high-level IMF/binary/budget
    constructors have their own signatures
  - `tests/unit/test_builders.py`,
    `tests/unit/builders/test_cluster_builders.py`
```

## The profile × DF virial contract

The defining cross-cutting check is that **any** spatial profile pairs
with **any** velocity DF (protocol-based composition), and a *matched*
continuum pair has a compatible density, potential and DF. An unscaled
finite realization need not have exactly $Q=0.5$, and a virial ratio alone
does not establish stationarity. How the suite
covers this:

- **Matched pairs** (Plummer⊗Plummer at the same $r_h$, King⊗King at
  the same $W_0$, EFF⊗EFF) recover $Q = T/|V| \approx 0.5$ to within
  finite-sample and numerical error specified by the model-specific checks.
  Hard-edged EFF sampling requires its documented boundary qualification. The matched-pair $Q$ is asserted in
  the **validation-tier physics suites** (`test_plummer_physics.py`,
  `test_king_physics.py`, `test_eff_physics.py`); the builder/cluster
  suites assert the *composition* (that any profile pairs with any DF and
  the matched DF is constructed).
- **Mismatched pairs** (e.g. Plummer positions at $r_h=1$ with Plummer
  velocities built for $r_h=2$) are *expected* to depart from $0.5$,
  because the velocities no longer balance the potential they sit in.
  This physical asymmetry is what makes the matched-pair equilibrium a
  real result rather than an imposed one. (This is the rationale for the
  matched-pair checks, not a separately-asserted directional test.)

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

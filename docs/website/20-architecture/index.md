---
title: Architecture & design
description: Why progenax is shaped the way it is — JAX-native patterns, the three-brick state model, protocol-based composition, and the conventions that thread through every module.
---

# Architecture & design

This section documents *why* progenax is shaped the way it is. Where
the [theory section](../10-theory/index.md) covers what the package
*computes*, this section covers what the package *is* as software:
the JAX-native patterns, the protocol-based composition, the units
policy, the conventions every contributor needs to know, and the
historical decisions that shaped the current API.

## Map of the section

```{list-table}
:header-rows: 1

* - Chapter
  - Scope
* - [](jax-native-philosophy.md)
  - Why no `numpy` or `scipy` in core code, why `jax.lax.scan` instead of Python loops, why `equinox` modules for state
* - [](three-brick-state.md)
  - The State / SystemParams / ParticleSystem decomposition that makes differentiable simulations possible
* - [](protocols.md)
  - `SpatialProfile`, `VelocityDF`, `IMFProtocol` — runtime-checkable protocols enabling mix-and-match composition
* - [](differentiability.md)
  - Patterns that preserve gradient flow; antipatterns that break it
* - [](units-policy.md)
  - Explicit `units` / `G` parameters, `DEFAULT_UNITS` per package, no global context managers
* - [](q-virial-convention.md)
  - Why $Q_{\mathrm{vir}} = T/|V|$ (= 0.5 at equilibrium) — and not the factor-of-2 alternatives
* - [](jax-native-substructure-q.md)
  - The {cite:t}`Cartwright2004` substructure Q parameter: kNN approximation + scipy reference
* - [](contributor-guide.md)
  - How to add a new spatial profile, IMF, integrator, or validation suite to progenax
* - [](ic-redesign-history.md)
  - The 2026-02-12 IC redesign that produced the current protocol-based architecture
```

## What this section is *not*

These chapters describe *implementation choices* — the patterns that
let progenax be JAX-native, differentiable, and composable. They do
*not* describe the underlying physics (that is the theory section)
or the per-symbol API (that is the [](../30-api/index.md) reference).
A chapter belongs in this section if its content is invariant under
swapping the implemented physics: "use `jax.lax.scan` not while-loops"
is true for any progenax module, while "the King profile satisfies
$2T + V = 0$ at equilibrium" is specific to King.

## Reading paths

For new contributors: read [](jax-native-philosophy.md) →
[](differentiability.md) → [](protocols.md) → [](contributor-guide.md)
in that order. The first two establish the constraints; the third
shows the composition pattern; the fourth walks through adding new
modules.

For users coming from McLuster or NBODY6: read [](units-policy.md) and
[](q-virial-convention.md) first — these are the convention-mismatches
most likely to bite.

For paper-writers wanting to cite progenax's design choices:
[](three-brick-state.md) and [](ic-redesign-history.md) are the
chapters most worth referencing.

## How architecture and theory interact

Some progenax decisions sit on the boundary between architecture and
physics. The convention is:

- **Pure architecture** — patterns invariant under physics swap. e.g.
  "use Equinox modules for state" applies whether you are sampling
  Plummer, King, or EFF. Lives here.
- **Pure physics** — derivations and parameter values that depend
  only on the underlying model. Lives in the theory section.
- **Coupled** — the soft-mask sigmoid for the BM19 transition
  density: it is a physics choice (using a soft threshold instead of
  hard) but motivated by a JAX constraint (autodiff needs smooth
  thresholds). Lives in [](differentiability.md) here, with the
  scientific consequences in [](../10-theory/gravoturbulence/direct-3d-zeta.md).

When in doubt about which section a topic belongs in, the test is:
"would this discussion still be true if we ported progenax to PyTorch
or NumPy?" If yes → theory. If no → architecture.

## References

The JAX programming model is documented at [the JAX
docs](https://jax.readthedocs.io). Equinox is {cite:t}`Equinox`. The
broader JAX ecosystem progenax sits in (jaxstro, gravax, stellax) is
described in the [jaxstro README](https://github.com/drannarosen/jaxstro-dev).

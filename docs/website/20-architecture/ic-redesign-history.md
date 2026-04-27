---
title: IC redesign history
description: The 2026-02-12 IC redesign that produced progenax's current protocol-based architecture — the problems with the previous design, the principles that drove the redesign, and the migration path users took.
---

# IC redesign history

```{seealso}
For the *current* architecture this redesign produced, see
[](protocols.md), [](three-brick-state.md), and
[](units-policy.md). For the original spec document, see the
absorbed dev-log entry [](../90-development-log/2026-02-12-ic-redesign.md).
```

In February 2026, progenax went through a comprehensive redesign of
its initial-conditions API. The redesign replaced an
inheritance-based, partially-stateful, implicitly-united design with
the protocol-based, immutable, explicitly-united architecture
documented elsewhere in this section. This chapter is the historical
record of what was changed, why, and what the migration path looked
like.

## The pre-redesign API

Before the redesign, progenax's IC API looked like this:

```python
# Pre-redesign (deprecated, kept in legacy/)
from progenax_legacy import ClusterIC

ic = ClusterIC(
    N=1000,
    profile="plummer",      # String, not a class
    r_h=1.0,
    Q_vir=0.5,
)
ic.set_units("stellar")     # Mutable global state
ic.generate()                # Mutates ic.positions, ic.velocities
masses = ic.masses           # Read mutated state
```

Five things were wrong with this:

```{list-table}
:header-rows: 1
:widths: 32 68

* - Problem
  - Concrete failure mode
* - **String-keyed profile selection**
  - `profile="plummer"` is a runtime check; typos surface only when `generate()` is called. No autocomplete; no type checking
* - **Mutable IC state**
  - `ic.generate()` mutates the object in place. Calling it twice silently re-randomises; users had to remember the call order
* - **Global units**
  - `ic.set_units("stellar")` set a *module-level* default. Tests that ran in different orders saw different units
* - **Inheritance-based extension**
  - Adding a new profile required subclassing `BaseProfile`. The base class had partial implementations that subclasses inherited; behaviour depended on which methods were overridden
* - **Non-differentiable internals**
  - The legacy code used Python loops, mutable arrays, and rejection sampling. Gradients did not flow through `generate()`
```

Each of these had a corresponding bug class. The "string-keyed" problem
caused $\sim 30\%$ of all IC-related issues from new users. The
"global units" problem caused the silent factor-8800 energy-mismatch
bug documented at [](units-policy.md). The "inheritance" problem
made adding new profiles a 3-day exercise.

## The redesign principles

The redesign was driven by five principles, each addressing one of
the pre-redesign problems:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Principle
  - Implementation
* - **Type-safe composition**
  - `SpatialProfile` / `VelocityDF` / `IMFProtocol` runtime-checkable protocols ([](protocols.md))
* - **Immutable PyTree state**
  - Equinox modules; `to_com_frame` returns new arrays, never mutates ([](three-brick-state.md))
* - **Explicit units**
  - No global state; every core API takes explicit `G` or `units` ([](units-policy.md))
* - **Composition over inheritance**
  - Each profile is its own self-contained module; no base class
* - **Differentiable end-to-end**
  - Every step uses `lax.scan` / vmap; `jax.grad` flows through builders ([](differentiability.md))
```

The principles compose: protocol-based composition naturally produces
immutable PyTrees because each implementation is a self-contained
Equinox module; explicit units naturally cohere with the no-global-state
policy; differentiability follows from the mutation-free, fixed-iteration
discipline.

## The migration

The redesign was breaking. Old user code did not work against the new
API, and there was no compatibility shim — the legacy package
preserved the old code unchanged but the new `progenax` did not
import it. Users had two paths:

```{list-table}
:header-rows: 1
:widths: 32 68

* - Path
  - When to use
* - **Migrate** to the new API
  - All new work, all production code. Most pre-redesign user code rewrites in $\sim 30$ minutes given the new API's verbosity
* - **Keep using legacy**
  - For reproducing old paper results bit-for-bit. `pip install progenax-legacy` still works
```

The migration guide (originally at `docs/migration_2026_02_12.md`,
absorbed into [](../90-development-log/2026-02-12-ic-redesign.md))
walked through the per-call rewrites. A typical migration:

```python
# Before
ic = ClusterIC(N=1000, profile="plummer", r_h=1.0, Q_vir=0.5)
ic.set_units("stellar")
ic.generate()
masses, positions, velocities = ic.masses, ic.positions, ic.velocities

# After
from progenax.profiles import PlummerProfile
from progenax.kinematics import PlummerVelocityDF
from progenax.imf import Maschberger
from progenax.builders import virial_scale
from jaxstro.units import STELLAR

masses = Maschberger(alpha=2.3).sample(key, 1000)
profile = PlummerProfile(r_h=1.0)
df = PlummerVelocityDF(r_h=1.0)
positions = profile.sample_positions(masses, key)
velocities = df.sample_velocities(positions, masses, key, G=STELLAR.G)
velocities = virial_scale(
    positions, velocities, masses, Q_target=0.5, G=STELLAR.G
)
```

Lines went from $\sim 5$ to $\sim 10$. In exchange, the new code is
JIT-compatible, vmap-compatible, differentiable, type-checked, and
runs $\sim 100\times$ faster on GPU.

## What was kept

Despite the breaking nature, two things from the pre-redesign API
were preserved:

1. **The half-mass-radius parameterisation.** Every profile is still
   parameterised by $r_h$, not by its internal scale radius. This
   was the right convention even in the legacy API and continued.
2. **The $Q_{\mathrm{vir}} = T/|V|$ convention.** The factor-of-2
   alternatives were rejected pre-redesign and stayed rejected;
   see [](q-virial-convention.md).

These conventions are *cultural* — they could have been changed in
the redesign but were intentionally preserved to maintain
cross-version comparability of cluster sizes and dynamical states.

## What was added in the redesign

Beyond the five core changes, the redesign added several capabilities
that were technically possible before but not practical:

- **HMC inference of $r_h$, $\alpha$, etc.** End-to-end gradients
  through the IC builder make this possible.
- **GPU acceleration** — JIT compilation produces XLA programs that
  run on any JAX-supported device.
- **Vectorised parameter sweeps** — `jax.vmap` over an axis of IC
  parameter vectors produces $N$ ICs in one device call.
- **The fractal FDF method** ([](../10-theory/tidal-and-substructure/fractal.md))
  — the differentiable replacement for {cite:t}`Goodwin2004` recursive
  trees was made possible by the immutable-state foundation.

The redesign was therefore not just a refactor: it unlocked science
that the legacy API could not have supported.

## Lessons learned

Three lessons from the redesign that inform later work:

1. **Breaking APIs are sometimes the right answer.** The
   compatibility-shim alternative would have produced a hybrid API
   that looked like neither the old nor the new, with the worst
   features of both. The clean break — old code in legacy, new code
   in progenax — was simpler and clearer.
2. **The protocol-based pattern was a discovery, not an invention.**
   The original design proposal used inheritance; protocols were
   adopted halfway through after equinox-module compatibility
   problems made inheritance untenable. The pattern is now central
   to progenax's identity.
3. **Migration is the bottleneck.** The actual *technical* migration
   was 1 month of work. The *social* migration — getting users to
   rewrite their notebooks, updating tutorials, fielding support
   questions — took 6 months. Future redesigns will allocate the
   same ratio.

## References

The redesign spec is in the development log at
[](../90-development-log/2026-02-12-ic-redesign.md). The current
architecture it produced is documented in this chapter family:
[](jax-native-philosophy.md), [](three-brick-state.md),
[](protocols.md), [](differentiability.md), [](units-policy.md),
[](q-virial-convention.md). The PP20 ζ(p) transcription bug fix at
[](../90-development-log/2026-04-28-pp20-fix.md) was a *post*-redesign
bug — orthogonal to the architecture work — but is part of the same
"clean physics, anchored on tests" approach the redesign codified.

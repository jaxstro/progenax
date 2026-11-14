---
title: IC redesign history
description: The dated IC redesign specification and its relationship to the current protocol-based API, including retired interfaces and current compatibility limits.
---

# IC redesign history

```{seealso}
For the *current* architecture this redesign produced, see
[](protocols.md), [](three-brick-state.md), and
[](units-policy.md). For the original spec document, see the
absorbed dev-log entry [](../90-development-log/2026-02-12-ic-redesign.md).
```

The February 12, 2026 specification proposed protocol-based composition,
immutable array state, and explicit units for initial-condition generation.
This chapter relates that dated design to the current API. It does not
establish a user-migration timeline or independently revalidate the
specification's scientific and performance claims.

## The pre-redesign API

Before the redesign, progenax's IC API looked like this:

```python
# Historical API sketch; progenax_legacy is not an installed package
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

* - Problem
  - Concrete failure mode
* - **String-keyed profile selection**
  - `profile="plummer"` is a runtime check; typos surface only when `generate()` is called. No autocomplete; no type checking
* - **Mutable IC state**
  - `ic.generate()` mutates the object in place. Calling it twice silently draws a new realization; users had to remember the call order
* - **Global units**
  - `ic.set_units("stellar")` set a *module-level* default. Tests that ran in different orders saw different units
* - **Inheritance-based extension**
  - Adding a new profile required subclassing `BaseProfile`. The base class had partial implementations that subclasses inherited; behavior depended on which methods were overridden
* - **Non-differentiable internals**
  - The legacy code used Python loops, mutable arrays, and rejection sampling. Gradients did not flow through `generate()`
```

The dated specification motivates the redesign with these failure modes.
Current unit conventions belong to [](units-policy.md). This repository
does not provide a measured user-incident rate or profile-development time
for the historical migration.

## The redesign principles

The redesign was driven by five principles, each addressing one of
the pre-redesign problems:

```{list-table}
:header-rows: 1

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
* - **Derivatives on supported paths**
  - Parameter, observable, domain, and randomness conventions are explicit; fixed loops alone do not establish differentiability ([](differentiability.md))
```

The packaged implementations use Equinox modules for array state and
protocols for method contracts. Protocol conformance alone does not
establish PyTree compatibility, differentiability, or equilibrium. Those
properties depend on the implementation and require their own evidence.

## The migration

The historical specification proposed a breaking redesign. Retired
interfaces are not a separate installable legacy package. For current
work, use the supported API; reproducing an earlier implementation
requires its recorded source revision and environment.

```{list-table}
:header-rows: 1

* - Path
  - When to use
* - **Migrate** to the new API
  - New work using the current model definitions, units, and supported parameter domains
* - **Reproduce pre-redesign results**
  - Check out the pre-redesign commit from git history (the redesign is documented in the [development log](../90-development-log/2026-02-12-ic-redesign.md)). There is **no** separate legacy package or install.
```

The migration guide (originally at `docs/migration_2026_02_12.md`,
absorbed into [](../90-development-log/2026-02-12-ic-redesign.md))
describes proposed per-call rewrites. The following sketch contrasts the
retired interface with current explicit construction:

```python
# Before
ic = ClusterIC(N=1000, profile="plummer", r_h=1.0, Q_vir=0.5)
ic.set_units("stellar")
ic.generate()
masses, positions, velocities = ic.masses, ic.positions, ic.velocities

# After
import jax
from progenax.profiles import PlummerProfile
from progenax.kinematics import PlummerVelocityDF
from progenax.imf import Maschberger
from progenax.builders import virial_scale
from jaxstro.units import STELLAR

key = jax.random.PRNGKey(42)
key_mass, key_position, key_velocity = jax.random.split(key, 3)
masses = Maschberger(alpha=2.3).sample(key_mass, 1000)
profile = PlummerProfile(r_h=1.0)
df = PlummerVelocityDF(r_h=1.0)
positions = profile.sample_positions(masses, key_position)
velocities = df.sample_velocities(positions, masses, key_velocity, G=STELLAR.G)
velocities = virial_scale(
    positions, velocities, masses, Q_target=0.5, G=STELLAR.G
)
```

The explicit construction makes the profile, DF, units, and independent
random keys visible. JIT, batching, and derivative support depend on the
chosen path; see [](differentiability.md). No GPU speedup follows from this
example. Measured costs belong to the
[performance page](../50-validation/performance-memory.md).

## What was kept

Two current conventions are important when reading historical examples:

1. **Radius parameters depend on the model.** Plummer takes the half-mass
   radius $r_h$; King takes its scale radius $r_c$ and central potential;
   EFF takes scale radius $a$, outer slope, and truncation radius. They are
   not interchangeable. See the [profile theory](../10-theory/spatial-profiles/index.md).
2. **The current virial ratio is $Q=T/|V|$.** Here $T$ is kinetic energy and
   $V$ is gravitational potential energy. Equilibrium global balance is
   $Q=0.5$. Some archived examples use a different factor-of-two convention;
   convert it explicitly before comparing results. See [](q-virial-convention.md).

## What was added in the redesign

The current implementation provides ingredients for the following workflows,
with separate evidence requirements:

- **Gradient-based inference.** Supported continuous derivatives can enter
  a likelihood; they do not by themselves establish statistical identifiability
  or posterior calibration.
- **Device execution.** JAX can compile supported kernels for its available
  devices. Published CPU timings do not establish unmeasured GPU performance.
- **Parameter sweeps.** Compatible array paths can be batched when static
  shapes and model construction permit; support is path-specific.
- **Historical FDF designs.** The specification discussed turbulent/fractal
  models. Those experimental features are outside the released core; see the
  [substructure scope](../10-theory/tidal-and-substructure/fractal.md).

## Lessons learned

Three lessons from the redesign that inform later work:

1. **Breaking APIs are sometimes the right answer.** The
   compatibility-shim alternative would have produced a hybrid API
   that looked like neither the old nor the new, with the worst
   features of both. A current implementation should make its supported
   interface and reproduction path explicit.
2. **Type compatibility does not establish physical compatibility.**
   Match the density profile, velocity DF, and units. Protocol conformance
   alone cannot establish a self-consistent initial condition.
3. **Migration needs documentation.** Retired interfaces need clear
   replacement examples and versioned reproduction instructions. The dated
   design does not supply evidence for a quantitative migration-duration claim.

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

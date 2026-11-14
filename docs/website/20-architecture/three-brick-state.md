---
title: Three-brick state pattern
description: progenax's State / SystemParams / ParticleSystem decomposition — why separating evolving state from static configuration makes differentiable simulations practical, with examples and gotchas.
---

# The three-brick state pattern

```{important}
Implementation status: **architecture pattern, not current public API**.

The current progenax package exposes IC containers (`ICResult`,
`MultiComponentCluster`) plus array-returning builder utilities. It
does not expose public `State`, `SystemParams`, `ParticleSystem`,
`GravityPolicy`, `SofteningPolicy`, `BinState`, or `ExternalState`
classes — those live downstream in gravax. The names below document
the design direction rather than importable progenax symbols. (The
legacy `ClusterState` container was retired in the 2026-06 unified
multi-component redesign.)
```

A simulated stellar system has two distinct kinds of "state":

1. **Evolving state** — positions, velocities, internal stellar
   properties. These change every timestep.
2. **Static configuration** — gravity model, softening parameters,
   physical units, integration scheme. These do not change during
   the simulation.

This chapter describes one possible separation into three immutable PyTrees.
The current Progenax boundary returns physical IC containers; downstream
integrators own their evolving state and numerical configuration. These
conceptual names do not define a new Progenax API or an AD restriction.

## The three bricks

```{list-table}
:header-rows: 1

* - Brick
  - Contents
  - Role
* - **State**
  - $\mathbf{x}_i$, $\mathbf{v}_i$, $m_i$ (evolving)
  - The *time-dependent* phase-space data. Updated by every integration step
* - **SystemParams**
  - Policies (gravity, softening, integration), units, $G$
  - The *static* configuration. Set at IC time, never changes during evolution
* - **ParticleSystem**
  - State + SystemParams (composite)
  - The user-facing wrapper that exposes both bricks together
```

```python
class State(eqx.Module):
    positions: Float[Array, "N 3"]
    velocities: Float[Array, "N 3"]
    masses: Float[Array, "N"]
    time: Float[Array, ""]

class SystemParams(eqx.Module):
    gravity_policy: GravityPolicy           # e.g. NewtonianPlummer
    softening_policy: SofteningPolicy       # e.g. ConstantSoftening
    units: UnitSystem                        # e.g. STELLAR
    G: Float[Array, ""]

class ParticleSystem(eqx.Module):
    state: State
    params: SystemParams
```

Both `State` and `SystemParams` are Equinox modules, so they are
PyTrees and JAX-traceable. `ParticleSystem` is a composite that
exposes the two bricks together — the user-facing shape that builders
construct and integrators consume.

## Why separate evolving state from static config

The separation matters for *gradient flow*. Consider an HMC chain
that infers cluster parameters $\boldsymbol{\theta}$ — say, the
half-mass radius $r_h$ or the IMF slope $\alpha$ — from a final
snapshot. The forward model is

```{math}
\boldsymbol{\theta}\;\xrightarrow{\text{IC builder}}\;\mathrm{State}_0\;\xrightarrow{\text{integrate}}\;\mathrm{State}_T\;\xrightarrow{\text{observe}}\;\mathcal{O}
```

The gradient $\partial \mathcal{O} / \partial \boldsymbol{\theta}$
can flow through floating-point array leaves in either container. An immutable
parameter can remain fixed during an integration while varying between inference
evaluations. Storage in `SystemParams` does not by itself freeze a derivative.

```{admonition} The static-vs-traced split
:class: note
JAX tracing, differentiation, and compile-time static selection are distinct.
A traced input can be nondifferentiated, and a parameter constant along a
trajectory can still be an inference target. Equinox fields are PyTree leaves
unless declared otherwise. `eqx.field(static=True)` removes a field from JAX
transformations; the name of its containing class has no such effect.

Choose the argument/leaf differentiation set explicitly and check the intended
observable. The [Equinox field contract](https://docs.kidger.site/equinox/api/module/advanced_fields/)
explains why array-valued physical parameters generally should not be made static.

```

## How the bricks interact during evolution

A single integration step is

```python
@jax.jit
def integrate_step(particle_system, dt):
    # Read both bricks
    state = particle_system.state
    params = particle_system.params

    # Compute forces from both
    forces = params.gravity_policy.compute_forces(
        state.positions, state.masses, params.G,
        softening=params.softening_policy.compute_softening(state.positions),
    )

    # Update state (immutable — produces new state)
    new_state = State(
        positions=state.positions + dt * state.velocities,
        velocities=state.velocities + dt * forces / state.masses[:, None],
        masses=state.masses,
        time=state.time + dt,
    )

    # Wrap back into ParticleSystem with same params
    return ParticleSystem(state=new_state, params=params)
```

The pattern is: read state and params, compute new state from both,
return a new `ParticleSystem` with the new state and the *same*
params. SystemParams never gets mutated; the integrator threads it
through the call chain unchanged.

## Builder pattern

ICs are constructed in two steps that map cleanly onto the three bricks. **progenax**
builds the *physical state* — an `ICResult` (positions, velocities, masses, stellar radii)
— and **gravax** assembles the `ParticleSystem` (`State` + `SystemParams`) from it:

```python
from progenax import build_plummer_cluster   # -> ICResult (pure physical state)
from gravax import ParticleSystem
from jaxstro.units import STELLAR

# 1. progenax builder: parameters + PRNG key -> ICResult (the State data).
ic = build_plummer_cluster(n=1000, r_h=1.0, key=key)   # differentiable in r_h

# 2. gravax assembles the bricks: State from the ICResult, SystemParams from the
#    units + its numerical configuration. Verify the adapter and policies in
#    the downstream package; this conceptual example does not qualify them.
system = ParticleSystem.from_ic(ic, units=STELLAR)
```

The Progenax builder's supported derivatives are documented by parameter,
observable, and sampling path. A discrete unit-system choice or policy type is
configuration; continuous numerical parameters within a policy can be traced
when the downstream implementation supports it. No complete downstream
inference contract follows from this container diagram.

## Gotchas and rules

### Rule 1: Preserve the physical-state boundary

When code outside progenax (a custom integrator, a renderer) consumes
a `ParticleSystem`, it should treat `state` as the input data and
`params` as the configuration. Reading either is fine; writing
either requires producing a *new* `ParticleSystem` rather than
mutating in place.

### Rule 2: Separate lifetime from differentiation

A softening length may be a floating-point array leaf inside a policy/configuration
container. It need not move into evolving state to support a derivative. Avoid
marking that leaf static when it is an inference target; verify the actual
force law and transformation contract before interpreting the derivative.

### Rule 3: Keep ownership explicit

Keep evolving coordinates and fixed policies in their established owners.
Whether a field is traced or static is a separate implementation choice;
putting a value in a PyTree alone implies no measured compilation penalty.

### Rule 4: Use the actual public boundary

Progenax builders return IC containers such as `ICResult`,
`CatalogedBinaryClusterIC`, or masked `ResolvedBinaries`. Downstream adapters
may construct a `ParticleSystem`; consult that package's public contract.
Do not infer importable Progenax types from this conceptual diagram.

## When to add a fourth brick

A different decomposition can be useful when a concrete downstream owner
requires additional state. Two possible examples are:

1. **Per-mass-bin state** for multi-mass clusters where each
   bin needs its own integration substep. A hypothetical fourth brick
   `BinState` would carry per-bin auxiliary state. (On the IC side,
   per-component structure is already handled without a new brick:
   `MultiComponentCluster.sample_cluster` labels each star's
   generating component via `ICResult.component_id`.)
2. **External-field state** for clusters in time-varying galactic
   potentials. The galactic potential is technically `SystemParams`
   but its time evolution is `State`; the fourth brick `ExternalState`
   carries the time-varying piece.

Adding a fourth brick is a coordinated change across builders,
integrators, and renderers — it is not done lightly. The
three-brick pattern is the default.

## References

The state-vs-config separation is a standard pattern in differentiable
programming; see {cite:t}`Equinox` for the Equinox-Module idiom that
makes it ergonomic. progenax's specific naming follows {cite:t}`JAX`'s
PyTree convention. The pattern is consistent with Diffrax's
"diffeq state" / "diffeq solver state" split.

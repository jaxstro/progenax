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

Mixing the two into a single mutable object is the most common
N-body-code structure (NBODY6, COSMIC, McLuster all use it). progenax
deliberately does *not*: it separates them into three Equinox modules
called the **three bricks**, each immutable, each a PyTree. This
separation is the keystone of differentiable simulation. This chapter
documents the three bricks, why they exist, and the rules for using
them correctly.

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
@jax.tree_util.register_pytree_node_class
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
flows through **State** but not through **SystemParams**. Things in
SystemParams are *not* the inference target; freezing them in a
separate brick clarifies the gradient graph.

```{admonition} The static-vs-traced split
:class: note
JAX has a notion of "traced" (gradient-flowing) vs "static" (compile-
time-fixed) values. progenax uses the brick separation to map cleanly
to it: anything in `State` is traced; anything in `SystemParams` is
typically static (passed via `static_argnums` or stored as
non-PyTree fields).

Equinox supports this via the `eqx.field(static=True)` annotation,
which marks a field as compile-time constant. progenax uses this for
e.g. integer dimensions, integration-step counts, and PRNG-key
buffers that should not flow gradients.
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

ICs are constructed by *builder* functions that produce a
`ParticleSystem` from a parameter vector and a PRNG key:

```python
@jax.jit
def build_plummer_cluster(N, r_h, alpha, key, *, units=STELLAR):
    # Sample masses, positions, velocities
    masses = sample_masses(N, key, alpha=alpha)
    positions = sample_positions(masses, key, r_h=r_h)
    velocities = sample_velocities(positions, masses, key, r_h=r_h, G=units.G)

    # Construct State
    state = State(
        positions=positions, velocities=velocities,
        masses=masses, time=jnp.zeros(()),
    )

    # Construct SystemParams (static config)
    params = SystemParams(
        gravity_policy=NewtonianPlummer(),
        softening_policy=ConstantSoftening(eps=0.05 * r_h),
        units=units,
        G=units.G,
    )

    return ParticleSystem(state=state, params=params)
```

Builders are JIT-compatible and differentiable in `r_h` and `alpha` —
the parameters that flow through `state`. They are *not* differentiable
in `units`, `gravity_policy`, or other `SystemParams` fields, which is
the right behaviour: those are configuration choices, not inference
targets.

## Gotchas and rules

### Rule 1: `State` is owned by progenax (and downstream)

When code outside progenax (a custom integrator, a renderer) consumes
a `ParticleSystem`, it should treat `state` as the input data and
`params` as the configuration. Reading either is fine; writing
either requires producing a *new* `ParticleSystem` rather than
mutating in place.

### Rule 2: Don't put traced data in `SystemParams`

If you want to infer the softening-length, do *not* put it in
`SofteningPolicy.eps`. Put it in `State` (or a new dedicated brick).
SystemParams is for compile-time-static values; placing traced
parameters there will retrigger JIT every time the parameter changes.

### Rule 3: Don't put static config in `State`

Conversely, do not put gravity policy, units, or `G` in `State`. State
should contain only the data that *changes* over the integration. A
mistake here doesn't break correctness but bloats the PyTree and
slows JIT compilation.

### Rule 4: `ParticleSystem` is the only public type

User-facing code (builders, integrators, renderers) returns and
accepts `ParticleSystem`. The two bricks are accessed via attribute
lookup (`ps.state.positions`) but should not appear in public function
signatures. This decouples the user-facing API from internal
restructuring of the bricks.

## When to add a fourth brick

The three-brick pattern handles ~95% of progenax's needs. Two
situations call for a fourth:

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

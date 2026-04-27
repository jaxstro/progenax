---
title: Virial Q convention (Q = T/|V|)
description: progenax's virial-ratio convention, why Q = 0.5 corresponds to virial equilibrium, and the contract enforced across builders, kinematics, and validation.
---

# Virial Q convention

```{seealso}
This chapter defines the virial $Q_{\mathrm{vir}} \equiv T/|V|$ used by
`progenax.builders.virial_scale`, every velocity DF, and every
equilibrium validation test. It is *not* the {cite:t}`Cartwright2004`
substructure Q parameter, which is documented in
[](jax-native-substructure-q.md).
```

The virial Q parameter pins the dynamical state of an N-body initial
condition. progenax adopts the convention

```{math}
:label: q-virial
Q_{\mathrm{vir}} \;\equiv\; \frac{T}{|V|}
```

where $T$ is the total kinetic energy of the particles and $V$ is the
total gravitational potential energy. This page documents the
convention, justifies the choice, and lists the call sites.

## Three conventions in the literature

Three definitions appear in the literature; readers should not assume
any value without checking which is in use:

```{list-table}
:header-rows: 1
:widths: 22 20 30 28

* - Convention
  - Equilibrium value
  - Reference
  - Used by
* - $Q = T/|V|$
  - $0.5$
  - {cite:t}`Aarseth1974`
  - **progenax** (this work), McLuster {cite:p}`Kuepper2011`
* - $Q = 2T/|V|$
  - $1.0$
  - Direct virial-theorem statement
  - Some textbook derivations; older N-body papers
* - $\alpha_{\mathrm{vir}} = 2T/|U|_{\mathrm{grav}}$
  - $1.0$
  - {cite:t}`FederrathKlessen2012` for clouds
  - Cloud / GMC literature
```

The {cite:t}`Aarseth1974` $Q = T/|V|$ form is the dominant convention in
star-cluster N-body work and is what every progenax convenience
function expects. The factor-of-2 forms appear when the virial theorem
is quoted directly from $2T + V = 0$.

## Why $Q = 0.5$ is equilibrium

The scalar virial theorem for a gravitationally bound system in steady
state reads

```{math}
:label: virial-theorem
2T + V \;=\; 0
```

Substituting $V = -|V|$ (gravitational potential energy is negative) and
solving for $T/|V|$:

```{math}
T \;=\; \tfrac{1}{2}|V| \quad\Longrightarrow\quad
Q_{\mathrm{vir}} \;=\; \frac{T}{|V|} \;=\; \frac{1}{2}.
```

A system at $Q_{\mathrm{vir}} = 0.5$ is in dynamical equilibrium: the
kinetic energy is exactly the value that balances the gravitational
self-binding. Departures from $0.5$ are physically meaningful:

```{list-table} progenax interpretation of $Q_{\mathrm{vir}}$.
:header-rows: 1
:widths: 16 24 60

* - $Q_{\mathrm{vir}}$
  - State
  - Astrophysical interpretation
* - $< 0.5$
  - **Subvirial** (cold)
  - Collapsing system. Used to seed primordial mass-segregation studies {cite:p}`Allison2009`; cluster contracts on a crossing time before bouncing.
* - $0.5$
  - Virial equilibrium
  - Stable steady-state. Default for production ICs unless an out-of-equilibrium starting condition is needed.
* - $> 0.5$
  - **Supervirial** (hot)
  - Expanding/unbound. Used to model post-gas-expulsion clusters where rapid mass loss leaves stars on hyperbolic-like orbits {cite:p}`Goodwin2004`.
```

The subvirial regime is particularly important for fractal-substructure
work: {cite:t}`Allison2009` showed that cool ($Q_{\mathrm{vir}} \approx
0.3$) clumpy initial conditions produce *dynamical* mass segregation on
$\sim 1$ Myr timescales — much faster than the classical relaxation
time. Reproducing that result therefore requires the ability to seed
$Q_{\mathrm{vir}}$ explicitly, which is what `virial_scale` provides.

## Implementation

Every velocity DF in progenax produces velocities at exactly
$Q_{\mathrm{vir}} = 0.5$ by construction (the equilibrium DFs of
{cite:t}`Plummer1911`, {cite:t}`King1966`, and EFF). To reach a target
$Q_{\mathrm{vir}}^\star$, progenax rescales the velocities by

```{math}
:label: virial-rescaling
v_i \;\to\; v_i \cdot \sqrt{2\,Q_{\mathrm{vir}}^\star}
```

The factor of $2$ converts from the equilibrium $Q = 0.5$ baseline. The
rescaling is implemented in `progenax.builders.virial_scale`:

```python
@jax.jit
def virial_scale(positions, velocities, masses, Q_target=0.5, *, G):
    T = compute_kinetic_energy(velocities, masses)
    V = compute_potential_energy(positions, masses, G=G)
    Q_current = T / jnp.abs(V)
    scale = jnp.sqrt(Q_target / Q_current)
    return positions, velocities * scale, masses
```

Three properties hold by construction:

1. **Energy ratio is enforced exactly** (to floating-point precision).
   Tests in `tests/integration/test_physics_validation.py::test_virial_scaling`
   verify $|Q_{\mathrm{measured}} - Q_{\mathrm{target}}| < 10^{-12}$ for
   $N \le 10^4$.
2. **Spatial structure is preserved.** Only velocities scale; positions
   are untouched. The density profile, half-mass radius, and any
   substructure are unchanged.
3. **Differentiable.** `virial_scale` is JIT-compatible and survives
   `jax.grad`, so it can sit inside a posterior-evaluation chain.

## Call sites and contract

All progenax IC builders accept `Q_target` and pipe it through
`virial_scale`:

```{list-table}
:header-rows: 1

* - Function
  - Default $Q_{\mathrm{target}}$
  - Notes
* - `gravax.ic.plummer_sphere`
  - $0.5$
  - Mass-first API; explicit `Q_target` overrides
* - `progenax.populations.generate_two_component_cluster`
  - $0.5$
  - Both components rescaled together to a single global Q
* - `gravax.ic.sample_kroupa_stochastic` consumers
  - $0.5$
  - IMF + Plummer composition
* - `progenax.builders.virial_scale`
  - explicit
  - Bare rescaling utility
```

```{warning}
**Always use $Q_{\mathrm{target}} = 0.5$ for production initial conditions
unless you specifically want a non-equilibrium system.** The default
across the API is 0.5; passing a different value is a deliberate
out-of-equilibrium choice that should be documented in the calling
script.
```

## Validation

The progenax test suite includes physics-anchored regressions for the
virial Q convention:

- `tests/validation/test_plummer_physics.py::test_virial_ratio` — sample
  $N = 10^4$ Plummer particles, measure Q post-rescaling, assert
  agreement to $\sim 0.5\%$ (statistical from finite-N kinetic-energy
  noise).
- `tests/integration/test_physics_validation.py::test_virial_scaling` —
  parametrise $Q_{\mathrm{target}} \in \{0.1, 0.3, 0.5, 0.7, 1.0\}$,
  assert Q recovered to $10^{-12}$.
- See [](../50-validation/plummer-equilibrium.md) for the rendered
  test results.

## Why not $\alpha_{\mathrm{vir}} = 2T/|U|$?

The cloud-evolution literature uses $\alpha_{\mathrm{vir}} = 2T/|U|$ for
GMC virial parameters, where $|U|$ is the gravitational binding energy
{cite:p}`FederrathKlessen2012`. progenax does *not* mirror that
convention because the IC builders operate on point-mass particle
distributions, not continuous gas. Using the cluster-N-body $Q = T/|V|$
convention keeps consistency with {cite:t}`Aarseth1974`, McLuster
{cite:p}`Kuepper2011`, NBODY6, and the rest of the star-cluster
N-body ecosystem.

When converting between conventions: $\alpha_{\mathrm{vir}} = 2 \cdot
Q_{\mathrm{vir}}$. Equilibrium is $\alpha_{\mathrm{vir}} = 1$ in cloud
notation, $Q_{\mathrm{vir}} = 0.5$ in the progenax notation.

## References

The convention follows {cite:t}`Aarseth1974`'s star-cluster N-body
tradition and matches McLuster {cite:p}`Kuepper2011`. The
out-of-equilibrium use cases are {cite:t}`Allison2009` (subvirial cold
collapse for primordial segregation) and {cite:t}`Goodwin2004`
(supervirial post-gas-expulsion). For the *substructure* Q parameter
that shares the letter, see [](jax-native-substructure-q.md).

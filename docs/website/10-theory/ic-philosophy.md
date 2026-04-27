---
title: What is an initial condition?
description: The core conventions and concepts every progenax IC builds on — virial theorem, half-mass radius, COM frame, units policy, and what "differentiable IC" actually means.
---

# What is an initial condition?

An **initial condition** (IC) for an N-body simulation is a complete
specification of the phase-space coordinates of every particle at
$t = 0$: positions $\{\mathbf{r}_i\}$, velocities $\{\mathbf{v}_i\}$,
and masses $\{m_i\}$ for $i = 1, \ldots, N_\star$. progenax produces
ICs that are simultaneously *physically realistic*, *exactly
reproducible* (given a JAX PRNG key), and *differentiable* with respect
to every continuous control parameter — the radial scale, the IMF
slope, the virial ratio, the substructure strength, and so on.

This chapter establishes the conventions and core concepts that every
subsequent theory chapter assumes. Read it first if you are new to the
package.

## Three things every progenax IC fixes

A progenax IC is built up from three orthogonal ingredients:

```{list-table}
:header-rows: 1
:widths: 22 28 50

* - Component
  - Specifies
  - Implemented by
* - **IMF**
  - The mass spectrum $\xi(m) \equiv \mathrm{d}N/\mathrm{d}m$
  - [](imfs/index.md) — `progenax.imf.PowerLawIMF`, `ChabrierIMF`, `Maschberger`, `BinaryIMF`, `BirthEnvironment`
* - **Spatial profile**
  - The radial density $\rho(r)$
  - [](spatial-profiles/index.md) — `progenax.profiles.PlummerProfile`, `KingProfile`, `EFFProfile`
* - **Velocity DF**
  - The velocity distribution given $(\mathbf{r}, m)$
  - [](velocity-dfs/index.md) — `progenax.kinematics.PlummerVelocityDF`, `KingVelocityDF`, `EFFVelocityDF`
```

Each ingredient satisfies a runtime-checkable protocol — `IMFProtocol`,
`SpatialProfile`, `VelocityDF` — that defines its mathematical
contract. Any IMF can pair with any spatial profile, which can pair
with any velocity DF, so long as physical compatibility is maintained
(e.g. a King profile pairs with a King DF for equilibrium; mixing a
Plummer profile with a King DF produces an out-of-equilibrium starting
state, which can be useful for studying violent relaxation but is not
"the King cluster" in the equilibrium sense).

The composability is documented in
[](../20-architecture/protocols.md); the equilibrium-vs-mismatch
question is documented per-velocity-DF in [](velocity-dfs/index.md).

## The virial theorem and $Q_{\mathrm{vir}} = 0.5$

A bound, gravitationally self-interacting N-body system in steady state
satisfies the *virial theorem*:

```{math}
:label: virial-theorem
2T + V \;=\; 0
```

where $T = \tfrac{1}{2}\sum_i m_i |\mathbf{v}_i|^2$ is the total
kinetic energy and $V = -G\sum_{i<j} m_i m_j / |\mathbf{r}_i - \mathbf{r}_j|$
is the total gravitational potential energy. Solving for the ratio
$Q_{\mathrm{vir}} \equiv T/|V|$ gives $Q_{\mathrm{vir}} = 0.5$ at
equilibrium.

This $Q_{\mathrm{vir}} = T/|V|$ convention is what every progenax
builder, kinematics function, and validation test uses. Three of the
non-equilibrium states that show up in the literature have direct
physical meaning:

```{list-table}
:header-rows: 1
:widths: 18 22 60

* - $Q_{\mathrm{vir}}$
  - State
  - Use case in progenax
* - $\sim 0.3$
  - Subvirial / cold
  - {cite:t}`Allison2009` cool-fractal setup. Cluster contracts on a crossing time and produces *dynamical* mass segregation within $\sim 1$ Myr.
* - $0.5$
  - Virial equilibrium
  - Default for production ICs.
* - $\sim 0.75$
  - Supervirial / hot
  - Post-gas-expulsion clusters {cite:p}`Goodwin2004`. Rapid mass loss leaves stars on hyperbolic-like orbits.
```

The progenax convention, the algorithm that enforces it, and the alternative
$\alpha_{\mathrm{vir}} = 2T/|U|$ form are documented at length in
[](../20-architecture/q-virial-convention.md).

```{warning}
Three conventions for "Q" appear in the literature: $Q = T/|V|$
(equilibrium $0.5$, used here), $Q = 2T/|V|$ (equilibrium $1$), and
$\alpha_{\mathrm{vir}} = 2T/|U|$ (cloud-physics version). progenax uses
the first; do not assume any value without checking which is in use.
And do not confuse this *virial Q* with the *substructure Q* of
{cite:t}`Cartwright2004` — different physical quantity, same letter.
See [](../20-architecture/jax-native-substructure-q.md).
```

## The half-mass radius $r_h$

The natural length scale for a gravitating system is its **half-mass
radius**:

```{math}
:label: r-half
M(< r_h) \;=\; \frac{1}{2}\,M_{\mathrm{total}}
```

i.e. half the mass lies inside $r_h$. progenax parameterises every
spatial profile by $r_h$ rather than by the scale-radius parameter
internal to that profile (which differs between Plummer, King, EFF).
This makes ICs comparable across profile choices: a Plummer cluster
with $r_h = 1$ pc and a King cluster with $r_h = 1$ pc occupy the same
physical extent, even though their internal scale radii differ.

The Plummer scale-radius–to–half-mass-radius relation is

```{math}
:label: plummer-rh-a
a \;=\; r_h\,\sqrt{2^{2/3} - 1} \;\approx\; 0.7664\,r_h.
```

The factor $\sqrt{2^{2/3} - 1}$ recurs in every Plummer-related
derivation; misinverting it produces a 1.7× error in cluster size, and
this exact bug existed in an earlier version of progenax. It is now
locked by `tests/validation/test_plummer_physics.py::test_half_mass_radius`.
See [](spatial-profiles/plummer.md) for the derivation.

For King and EFF profiles, the $r_h$↔scale-radius mapping has no closed
form and is computed via numerical integration of the cumulative
mass profile.

## The centre-of-mass frame

Every progenax IC is returned in the **centre-of-mass (COM) frame**:

```{math}
:label: com-frame
\sum_i m_i \mathbf{r}_i \;=\; \mathbf{0},\quad
\sum_i m_i \mathbf{v}_i \;=\; \mathbf{0}.
```

The transform that achieves this — `progenax.builders.to_com_frame` —
is just a mass-weighted shift of positions and velocities. It is
applied as the last step of every builder; ICs leave the package with
zero net momentum and zero net displacement.

```{note}
**Why this matters for downstream integration.** N-body integrators do
not generally conserve linear momentum to machine precision — leapfrog
is symplectic but not strictly momentum-conserving with floating-point
arithmetic. A small initial COM drift compounds over evolution and can
produce $\mathcal{O}(\sigma_v t)$ centre-of-mass motion at the end of
a long run, displacing the cluster from its expected position. The
COM-frame IC eliminates this entirely.
```

## Units policy: explicit, no globals

progenax adopts a strict explicit-units convention: **no global state,
no implicit unit defaults, no `get_G()`-style context managers in core
APIs.** Every function that consumes a gravitational constant takes
either an explicit `G` value or an explicit `units` argument carrying
one.

Three unit systems appear most often:

```{list-table}
:header-rows: 1

* - System
  - Mass unit
  - Length unit
  - Time unit
  - $G$
* - **STELLAR** (default)
  - $\Msun$
  - pc
  - Myr
  - $\sim 0.00450$
* - **PLANETARY / BINARY**
  - $\Msun$
  - AU
  - yr
  - $\sim 39.478$
* - **CGS**
  - g
  - cm
  - s
  - $6.674 \times 10^{-8}$
```

Convenience wrappers may accept `units=None` and resolve to the
package-level `DEFAULT_UNITS = STELLAR`, but this resolution happens
only at the *API surface*; once inside the call stack, $G$ is an
explicit parameter. The architectural rationale and the dropped
context-manager idiom are documented in [](../20-architecture/units-policy.md).

```{warning}
**Mixing unit systems silently breaks energy conservation.** If you
sample positions/velocities under STELLAR but compute energies under
PLANETARY, the recovered $Q_{\mathrm{vir}}$ will be off by a factor
$G_{\mathrm{stellar}} / G_{\mathrm{planetary}} \approx 8800$. Always
pass the same `units` (or `G`) through the entire pipeline.
```

## What "differentiable IC" actually means

A progenax IC is a function

```{math}
\mathrm{IC}: \boldsymbol{\theta},\,\mathrm{key} \;\mapsto\; \{(\mathbf{r}_i,\mathbf{v}_i,m_i)\}_{i=1}^{N_\star}
```

mapping continuous control parameters $\boldsymbol{\theta} = (r_h,\,
\alpha_{\mathrm{IMF}},\,Q_{\mathrm{vir}},\,\lambda_{\mathrm{seg}},\ldots)$
and a JAX PRNG key to a phase-space realisation. *Differentiable*
means

```{math}
:label: jax-grad-ic
\frac{\partial\,\mathcal{O}\bigl(\mathrm{IC}(\boldsymbol{\theta},\,\mathrm{key})\bigr)}{\partial\,\boldsymbol{\theta}}
```

is well-defined and computable via `jax.grad`, for any observable
$\mathcal{O}$ — energy, kinetic-energy ratio, half-mass radius,
two-point correlation function, the post-evolution snapshot a few
crossing times later, the mock observation through an instrument
forward model. Every step that turns $\boldsymbol{\theta}$ into a
particle realisation must satisfy three constraints:

1. **No `jax.lax.while_loop`.** Convergence-based loops have variable
   step counts that are not differentiable. Use `jax.lax.scan` with a
   *fixed* iteration count instead. Inverse-CDF samplers in progenax
   solve the cumulative integrals to a fixed accuracy in a fixed
   number of Newton or bisection steps.
2. **No mutation, no `array[i] = val`.** Every state update creates a
   new immutable array. progenax uses Equinox modules for stateful
   classes, so all state lives in PyTree leaves.
3. **No discrete-without-smoothing.** Sorts, `argmin`, hard
   thresholds, and `where` selections all produce gradient
   discontinuities. Where they are unavoidable (rank-based radial
   remapping in [](tidal-and-substructure/fractal.md)), gradients flow
   through the *values* being sorted rather than the permutation.
   Where smoothing is possible (the Fundamental Plane threshold in
   [](imfs/environment.md)), progenax substitutes a sigmoid for the
   hard `where`.

The architectural rules are documented at length in
[](../20-architecture/jax-native-philosophy.md) and
[](../20-architecture/differentiability.md). Every theory chapter in
this section flags the steps where differentiability needed special
care.

## Reading order

The theory chapters are roughly self-contained, but a recommended
ordering for a student-style first pass:

1. This page.
2. [](spatial-profiles/plummer.md) — the canonical example, simplest
   closed form.
3. [](velocity-dfs/plummer-dfs.md) — the canonical equilibrium DF.
4. [](imfs/classic.md) — single-star IMFs.
5. [](spatial-profiles/king.md), [](spatial-profiles/eff.md) — the two
   alternative profiles.
6. [](imfs/binary.md) — when and why binaries change everything.
7. [](tidal-and-substructure/fractal.md) — the differentiable
   substructure construction.
8. [](gravoturbulence/index.md) — the SFR-from-cloud-properties chain.

Each later chapter assumes only the conventions from this page plus
the chapter immediately before it.

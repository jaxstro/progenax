---
title: What is an initial condition?
description: The core conventions and concepts every progenax IC builds on — virial theorem, half-mass radius, COM frame, units policy, and what "differentiable IC" actually means.
---

# What is an initial condition?

An **initial condition** (IC) for an N-body simulation is a complete
specification of the phase-space coordinates of every particle at
$t = 0$: positions $\{\mathbf{r}_i\}$, velocities $\{\mathbf{v}_i\}$,
and masses $\{m_i\}$ for $i = 1, \ldots, N_\star$. Progenax constructs these arrays from specified models and random keys.
Reproducing a realization requires the model configuration and software environment
as well as the key. Differentiation depends on the sampling path and parameter;
individual methods document their smooth regions, numerical safeguards, and gradient checks.

This chapter establishes the conventions and core concepts that every
subsequent theory chapter assumes. Read it first if you are new to the
package.

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers starting with progenax; this is the conventions page every other theory chapter builds on, so no prior stellar-dynamics literature is assumed.
**Prerequisites:** none — a good entry point.
**You'll get:** the three IC ingredients (IMF, spatial profile, velocity DF), the $Q_{\mathrm{vir}} = T/|V|$ virial convention, the half-mass radius $r_h$, the COM-frame and explicit-units policies, and what "differentiable IC" means in practice.
:::

## Three things every progenax IC fixes

A progenax IC is built up from three separately configured ingredients:

```{list-table}
:header-rows: 1

* - Component
  - Specifies
  - Implemented by
* - **IMF**
  - The mass spectrum $\xi(m) \equiv \mathrm{d}N/\mathrm{d}m$
  - [](imfs/index.md) — `progenax.imf.PowerLawIMF`, `ChabrierIMF`, `Maschberger`, `BinaryIMF`, `BirthEnvironment`
* - **Spatial profile**
  - The radial density $\rho(r)$
  - [](spatial-profiles/index.md) — `progenax.profiles.PlummerProfile`, `KingProfile`, `MichieProfile`, `EFFProfile`
* - **Velocity DF**
  - The velocity distribution given $(\mathbf{r}, m)$
  - [](velocity-dfs/index.md) — `progenax.kinematics.PlummerVelocityDF`, `KingVelocityDF`, `MichieVelocityDF`, `EFFVelocityDF`
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

For an isolated Newtonian system without additional virial terms, steady
equilibrium implies the virial relation:

```{math}
:label: ic-virial-theorem
2T + V \;=\; 0
```

where $T = \tfrac{1}{2}\sum_i m_i |\mathbf{v}_i|^2$ is the total
kinetic energy and $V = -G\sum_{i<j} m_i m_j / |\mathbf{r}_i - \mathbf{r}_j|$
is the total gravitational potential energy. Solving for the ratio
$Q_{\mathrm{vir}} \equiv T/|V|$ gives $Q_{\mathrm{vir}} = 0.5$ at
equilibrium. An instantaneous value $Q_{\mathrm{vir}}=1/2$ does not by itself
establish a stationary distribution.

This $Q_{\mathrm{vir}} = T/|V|$ convention is what every progenax
builder, kinematics function, and validation test uses. Three of the
non-equilibrium states that show up in the literature have direct
physical meaning:

```{list-table}
:header-rows: 1

* - $Q_{\mathrm{vir}}$
  - State
  - Use case in progenax
* - $\sim 0.3$
  - Subvirial / cold
  - {cite:t}`Allison2009` cool-fractal setup. Cluster contracts on a crossing time and produces *dynamical* mass segregation within $\sim 1$ Myr.
* - $0.5$
  - Virial balance
  - An energy constraint; stationarity requires additional checks.
* - $\sim 0.75$
  - Supervirial / hot
  - Post-gas-expulsion clusters {cite:p}`Goodwin2004`. The global energy ratio alone does not determine individual stellar binding energies.
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

Half the mass lies inside $r_h$. Plummer accepts $r_h$ directly; other profiles
use their own scale and shape parameters. Matching $r_h$ gives models the same
enclosed-half-mass radius, but their central densities, outer envelopes, and
truncation radii can still differ.

The Plummer scale-radius–to–half-mass-radius relation is

```{math}
:label: ic-plummer-rh-a
a \;=\; r_h\,\sqrt{2^{2/3} - 1} \;\approx\; 0.7664\,r_h.
```

The factor $\sqrt{2^{2/3}-1}$ converts the two Plummer length conventions.
The [Plummer chapter](spatial-profiles/plummer.md) derives the relation and links
to checks of its defining condition, $M(<r_h)/M=1/2$.

For King and EFF profiles, the $r_h$↔scale-radius mapping has no closed
form and is computed via numerical integration of the cumulative
mass profile.

## The center-of-mass frame

The low-level `build_spatial_ic` builder shifts its output into the
**center-of-mass (COM) frame**:

```{math}
:label: com-frame
\sum_i m_i \mathbf{r}_i \;=\; \mathbf{0},\quad
\sum_i m_i \mathbf{v}_i \;=\; \mathbf{0}.
```

The transform — `progenax.builders.to_com_frame` — is a mass-weighted
shift of positions and velocities. Direct profile/DF samplers return finite
realizations about their model origin without this shift. Later selections,
tidal cuts, or added velocity fields can change the sampled COM, so check the
final arrays when a downstream integrator requires a particular rest frame.

```{note}
**Why this matters for downstream integration.** N-body integrators do
accumulate floating-point momentum error during integration. An initial
COM velocity $\boldsymbol v_{\rm COM}$ also produces a displacement
$\boldsymbol v_{\rm COM}t$ in an isolated system. Recentering removes that
initial bulk velocity; it does not eliminate subsequent integration error.
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
$G_{\mathrm{planetary}} / G_{\mathrm{stellar}} \approx 8800$
(equivalently $G_{\mathrm{stellar}}/G_{\mathrm{planetary}} \approx
1.1\times10^{-4}$). Always pass the same `units` (or `G`) through the
entire pipeline.
```

## What "differentiable IC" actually means

A progenax IC is a function

```{math}
\mathrm{IC}: \boldsymbol{\theta},\,\mathrm{key} \;\mapsto\; \{(\mathbf{r}_i,\mathbf{v}_i,m_i)\}_{i=1}^{N_\star}
```

mapping continuous control parameters $\boldsymbol{\theta} = (r_h,\,
\alpha_{\mathrm{IMF}},\,Q_{\mathrm{vir}},\,\lambda_{\mathrm{seg}},\ldots)$
and a JAX PRNG key to a phase-space realization. *Differentiable*
means

```{math}
:label: jax-grad-ic
\frac{\partial\,\mathcal{O}\bigl(\mathrm{IC}(\boldsymbol{\theta},\,\mathrm{key})\bigr)}{\partial\,\boldsymbol{\theta}}
```

depends on both the sampling path and the observable $\mathcal{O}$.
An energy moment, an order statistic, a discrete catalogue membership,
and a post-evolution mock observation have different derivative contracts.
The main implementation considerations are:

1. **Control flow and derivative rules.** JAX does not provide ordinary
   reverse-mode differentiation through a dynamic `lax.while_loop`.
   Fixed-length `lax.scan` loops and custom implicit/adjoint rules are
   available alternatives. A fixed iteration count alone does not establish
   convergence or derivative accuracy.
2. **No mutation, no `array[i] = val`.** Every state update creates a
   new immutable array. progenax uses Equinox modules for stateful
   classes, so all state lives in PyTree leaves.
3. **Discrete choices and boundaries.** Sorting is piecewise differentiable
   through the selected values, while changing an ordering or membership can
   introduce a boundary. Hard selections, clipped endpoints and categorical
   labels need explicit treatment. A surrogate derivative is an estimator,
   and smoothing changes the represented law. The environment-dependent IMF
   retains its documented piecewise relation; smoothing is not a universal
   package policy.

The architectural rules are documented at length in
[](../20-architecture/jax-native-philosophy.md) and
[](../20-architecture/differentiability.md). Every theory chapter in
this section flags the steps where differentiability needed special
care.

## Reading order

This page is step 1 of the section's canonical path — the full
seven-step ramp (profiles → velocity DFs → IMFs → binaries → modifiers →
composite populations) lives in one place:
[the theory index's suggested reading order](index.md).

## Implementation, validation & references

- **In code:** the conventions on this page are enforced in
  `src/progenax/builders.py` (`build_spatial_ic`, `to_com_frame`,
  `virial_scale`) and `src/progenax/dynamics/virial.py` (the
  $Q_{\mathrm{vir}} = T/|V|$ energy utilities); units come from
  `jaxstro.units`. See the [builders API](../30-api/builders.md) and the
  [dynamics API](../30-api/dynamics.md).
- **Validated in:** [end-to-end equilibrium & energy checks](../50-validation/physics-tests.md)
  exercise the COM-frame, virial, and units conventions through the full
  IC pipeline; each ingredient's own validation page (Plummer, King, EFF,
  IMF) is linked from the chapters below.
- **Primary sources:** the virial theorem and Eddington/lowered-model
  machinery are standard textbook material; the non-equilibrium states
  cited here are {cite:t}`Allison2009` (subvirial cool-fractal) and
  {cite:t}`Goodwin2004` (supervirial post-gas-expulsion), and the
  substructure-$Q$ caution is {cite:t}`Cartwright2004` — full notes in the
  [bibliography](../99-bibliography/index.md).

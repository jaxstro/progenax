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

For an isolated Newtonian system, $Q_{\mathrm{vir}}=0.5$ gives the
instantaneous scalar energy balance. It is a necessary equilibrium diagnostic,
not sufficient evidence of a stationary DF or dynamical stability. The virial
identity is $\tfrac12\ddot I=2T+V$, with $I=\sum_i m_i r_i^2$; the sign
of $\dot I$ additionally determines whether the system is currently expanding
or contracting. Surface terms and external fields require their own treatment.

```{list-table} progenax interpretation of $Q_{\mathrm{vir}}$.
:header-rows: 1

* - $Q_{\mathrm{vir}}$
  - State
  - Astrophysical interpretation
* - $< 0.5$
  - **Subvirial** (cold)
  - Negative scalar virial acceleration in the isolated Newtonian case; collapse histories also depend on structure and initial flow. {cite:t}`Allison2009` studies a specific cold-clumpy configuration.
* - $0.5$
  - Virial equilibrium
  - Global energy balance; a compatible DF and separate stability checks remain necessary.
* - $> 0.5$
  - **Supervirial** (hot)
  - Positive scalar virial acceleration in the isolated case. $0.5<Q<1$ still gives negative total energy $T+V$; supervirial does not mean unbound.
```

The subvirial regime is particularly important for fractal-substructure
work: {cite:t}`Allison2009` showed that cool ($Q_{\mathrm{vir}} \approx
0.3$) clumpy initial conditions produce *dynamical* mass segregation on
$\sim 1$ Myr timescales — much faster than the classical relaxation
time. Reproducing that result therefore requires the ability to seed
$Q_{\mathrm{vir}}$ explicitly, which is what `virial_scale` provides.

## Implementation

Matched continuum equilibrium DFs satisfy the virial relation under their
stated assumptions. Finite sampled realizations fluctuate, and numerical
approximations such as a sharp EFF edge have additional limits. To reach a
target $Q_{\mathrm{vir}}^\star$, `virial_scale` measures the current ratio
$Q_{\mathrm{current}}$ and rescales velocities by

```{math}
:label: virial-rescaling
v_i \;\to\; v_i \sqrt{\frac{Q_{\mathrm{vir}}^\star}{Q_{\mathrm{current}}}}
```

The simplified factor $\sqrt{2Q_{\mathrm{vir}}^\star}$ applies only if
the input already has $Q_{\mathrm{current}}=0.5$. The implementation measures
the actual finite realization instead:

```python
@jax.jit
def virial_scale(positions, velocities, masses, Q_target, G, softening=0.0):
    T = compute_kinetic_energy(velocities, masses)
    V = compute_potential_energy(positions, masses, G=G, softening=softening)
    Q_current = T / jnp.abs(V)
    scale = jnp.sqrt(Q_target / Q_current)
    return velocities * scale   # only velocities are rescaled
```

Three properties hold by construction:

1. **Energy ratio is enforced exactly** (to floating-point precision).
   This requires nonzero kinetic energy and a valid binding-energy calculation.
   `tests/unit/test_builders.py` checks target recovery and invalid zero-velocity
   inputs; inspect those tests for their exact exposure and criteria.
2. **Spatial structure is preserved.** Only velocities scale; positions
   are untouched. The density profile, half-mass radius, and any
   substructure are unchanged.
3. **Differentiable.** `virial_scale` is JIT-compatible and survives
   `jax.grad`, so it can sit inside a posterior-evaluation chain.

## Call sites and contract

The single-population and binary builders expose `Q`; the rescaling utility
uses the name `Q_target`. Multicomponent sampling instead preserves its modeled
velocity scale without a global rescale:

```{list-table}
:header-rows: 1

* - Function
  - Default $Q_{\mathrm{target}}$
  - Notes
* - `progenax.build_cluster` / `build_spatial_ic`
  - $0.5$
  - `Q=None` skips velocity rescaling
* - `progenax.build_binary_cluster`
  - $0.5$
  - Target applies to system centers of mass, not internal orbital energy
* - `progenax.MultiComponentCluster.sample_cluster`
  - No rescaling parameter
  - Samples the shared-potential model; finite and numerical checks are separate
* - `progenax.builders.virial_scale`
  - explicit
  - Bare rescaling utility
```

```{note}
Use the energy ratio required by the scientific setup. `Q=0.5` sets a global
balance; `Q=None` retains the sampled DF normalization. A rescale cannot repair
an incompatible density–DF pairing, and rotating or tidally modified products
need their own diagnostics. For softened gravity, potential energy and the
force virial need not be the same quantity, so $T/|V|=0.5$ is not generally the
softened-force equilibrium criterion.
```

## Validation

The progenax test suite includes physics-anchored regressions for the
virial Q convention:

- `tests/validation/test_plummer_physics.py::TestPlummerVirialEquilibrium::test_virial_ratio`
  compares sampled, unrescaled kinetic energy with the analytic continuum
  Plummer binding energy; its finite-sample tolerance is not an exact-rescale test.
- `tests/unit/test_builders.py` checks the rescaling utility separately from
  raw-DF sampling, including the zero-kinetic-energy failure path.
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

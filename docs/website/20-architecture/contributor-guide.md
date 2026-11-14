---
title: Contributor guide
description: How to add a new spatial profile, velocity DF, IMF, binary distribution, or validation suite to progenax — the protocol-based composition pattern walked end-to-end with a worked example.
---

# Contributor guide

This chapter is the practical guide to extending progenax. If you
want to add a new spatial profile, velocity DF, IMF, binary
distribution, modifier layer, or validation suite, this is where to
start. The chapter assumes you have read [](jax-native-philosophy.md),
[](differentiability.md), and [](protocols.md) — the *why* of
progenax's structure. This chapter is the *how*.

## Six contribution targets

```{list-table}
:header-rows: 1

* - Target
  - Protocol / file location
  - Worked example
* - **Spatial profile**
  - `SpatialProfile`, `progenax/profiles/`
  - [Adding a Sérsic profile](#sersic-walkthrough)
* - **Velocity DF**
  - `VelocityDF`, `progenax/kinematics/`
  - [Adding an Osipkov-Merritt extension](../10-theory/velocity-dfs/rotation-anisotropy.md)
* - **IMF**
  - `IMFProtocol`, `progenax/imf/`
  - [Adding a custom IMF](#imf-walkthrough)
* - **Binary distribution**
  - `progenax/binaries/`
  - Period / mass-ratio / eccentricity distributions
* - **Modifier layer**
  - `progenax/tidal.py`, `progenax/cluster/`, and `progenax/kinematics/`
  - Tidal truncation, rotation, and primordial mass segregation
* - **Validation suite**
  - `tests/validation/`
  - Per-physics regression tests with quantitative pass/fail criteria
```

The pattern is the same for all six: write an `eqx.Module`, satisfy
the relevant protocol (or follow the modifier-layer convention),
add tests, document.

## Step-by-step: adding a new spatial profile

(sersic-walkthrough)=

A new profile starts with a specified **three-dimensional** density, its
normalization, parameter domain, and boundary conditions. A projected Sérsic
surface-brightness law is not itself a three-dimensional stellar density.
Its projected half-light constant cannot be reused as a spherical half-mass
constant without deriving the projection/deprojection and mass convention.

### Step 1: derive the model and inspect a complete owner

Use `src/progenax/profiles/plummer.py` as a complete implementation example,
with its [theory owner](../10-theory/spatial-profiles/plummer.md) and
[API contract](../30-api/profiles.md#api-profiles-plummerprofile). Derive the
shell probability $4\pi r^2\rho(r)$, cumulative mass, radius convention,
and sampling inverse or numerical scheme before writing the new sampler.
Keep physical assumptions and numerical approximations explicit; a proposed
new scientific model requires maintainer review.

### Step 2: check structure and the actual numerical contract

The existing structural check is executable:

```python
from progenax import PlummerProfile
from progenax.protocols import SpatialProfile

assert isinstance(PlummerProfile(r_h=1.0), SpatialProfile)
```

It only checks required attributes. It does not establish the new model's
normalization, units, sampled distribution, or derivatives. Name the failure
each additional test catches, compare with an independent analytic limit or
reference, and derive its error budget from the measured quantity. For JAX
loops, reuse a compiled program and shared static configurations. A finite
or nonzero gradient alone is not a derivative-accuracy test.

### Step 3: validate the scientific and numerical claims

State the intended output and the smallest discriminating comparison. Distinguish
an algebraic identity, numerical convergence, finite-sample agreement, and
physical qualification. Choose population size, grid resolution, seeds, and
tolerances for that question; do not copy a generic percentage or particle
count from another model. Retain failed and unresolved checks. The
[validation methodology](../50-validation/methodology.md) and the existing
model's tests provide examples of bounded claims.

### Step 4: integrate at the existing owners

After the implementation and its checks exist, export its supported surface
from the owning submodule, add theory and validation in their established
sections, and regenerate the API. Add the new page to `myst.yml` and its
Documentation tracker record. A custom profile/DF pair can use
`build_spatial_ic`; the convenience `build_cluster` dispatcher also needs an
explicit supported matched-DF route. Protocol membership alone does not add
that dispatch or establish physical compatibility.

## Step-by-step: adding a custom IMF

(imf-walkthrough)=

Use the actual [IMF protocol](protocols.md) and a complete owner such as
`src/progenax/imf/maschberger.py`, rather than copying a class with unfinished
methods. The scalar protocol requires mass bounds, normalized `logpdf`, `cdf`,
`ppf`, `sample`, and `mean_mass`. A binary population composition is a separate
interface. Specify whether masses denote stars, primaries, or unresolved
systems, and define support and units before implementing the distribution.

Test normalization and inverse consistency against independent expressions or
quadrature, then check the required derivatives at interior and limiting
parameters. Mixture choices and hard support changes need an explicit
randomness/expectation contract; implementing an Equinox class does not supply it.

## Modifier layers

Modifiers act on existing arrays or IC containers. Inspect the actual owner
and return contract: `tidal.py` owns truncation, `kinematics/rotation.py` owns
streaming overlays, and `cluster/mass_segregation.py` owns the primordial
energy-ordering transform. Fractal generation is excluded experimental work,
not a released-core modifier. A decorator does not make a discrete selection
differentiable; report its derivative or surrogate contract and its physical
consequences separately.

## Validation suite contributions

Validation tests live under `tests/validation/` and have a different
character from unit tests: they verify *physical correctness* against
analytical or published values, with quantitative pass/fail criteria.

Each test needs a named failure and an independent reference. Analytic,
convergence, statistical, and physical checks have different error sources.
Set a scale-aware tolerance with its derivation; arbitrary default percentages
and fixed minimum sample sizes do not qualify a result. Expensive tests must
be marked and justified. Plot measured values with their uncertainty and
retain the configuration, precision, versions, and reference provenance.

## Code-style conventions

Follow `pyproject.toml`, the repository instructions, and the owning module.
Use JAX arrays on core numerical paths, define shapes and units, and provide
usable argument, return, domain, and reference documentation. Keep changes
within the smallest scientific/numerical owner. File length or a count of
fields alone is not a reason to add another abstraction.

## CI: what tests run

Run the local checks appropriate to the change: focused tests first, then the
required integrated gate and source-owned documentation checks. The current
workflow YAML describes configured events, not proof of enabled or successful
remote execution. For a private repository, Anna must explicitly approve the
repository and intended CI scope before enabling, dispatching, rerunning, or
triggering CI through a push, tag, or pull request. A green local run does not
authorize that remote action. Do not infer scientific qualification from a
workflow or documentation build.

## References

The contributor pattern follows the protocol-based composition
documented in [](protocols.md). The differentiability constraints
contributors must respect are at [](differentiability.md). The
broader JAX-native conventions are at [](jax-native-philosophy.md).

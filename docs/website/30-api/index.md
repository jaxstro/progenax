---
title: API reference
description: Auto-generated reference for every public symbol in progenax, regenerated from source on every release.
---

# API reference

This section is **auto-generated** from progenax source. The pages
in this section are produced by `scripts/build_api_reference.py`,
which walks each public submodule's `__all__`, checks package-root exports and extracts
signatures and docstrings via `inspect`, and emits one MyST page per
module plus an alphabetical full-symbol index.

```{seealso}
For physics derivations of the algorithms behind these symbols, see
[](../10-theory/index.md). For design rationale on patterns like
"masses-first" and protocol-based composition, see
[](../20-architecture/index.md).
```

## Browse by module

```{list-table}
:header-rows: 1

* - Module
  - Scope
* - [](profiles.md)
  - Spatial density profiles: `PlummerProfile`, `KingProfile`, `EFFProfile`
* - [](kinematics.md)
  - Velocity DFs and Eddington-inverted samplers
* - [](imf.md)
  - Initial mass functions: power-law, Maschberger, Chabrier, binary-aware, environment-dependent
* - [](binaries.md)
  - Kepler elements, period and eccentricity distributions, joint binary populations
* - [](analytical.md)
  - Analytical IC test cases: two-body Kepler, three-body figure-eight, harmonic oscillator
* - [](builders.md)
  - IC pipeline utilities: `virial_scale`, `to_com_frame`, energy diagnostics
* - [](builders_cluster.md)
  - Convenience cluster builders and named profile wrappers
* - [](stellar.md)
  - Tout ZAMS luminosity, radius, effective temperature, surface gravity and luminosity inverse
* - [](tidal.md)
  - Jacobi-radius computation and tidal truncation
* - [](cluster.md)
  - Multi-component equilibrium clusters: `MultiComponentCluster` (Engines A and B), primordial `energy_sorted_segregation`, turbulence physics
* - [](diagnostics.md)
  - Cluster diagnostics: Λ_MSR, CW04 Q parameter, differentiable segregation observables
* - [](dynamics.md)
  - Virial and energy utilities: kinetic/potential energy, per-group virial ratios
* - [](protocols.md)
  - Runtime-checkable protocols: `SpatialProfile`, `VelocityDF`, `IMFProtocol`
```

(api-progenax-default_units)=
## `progenax.DEFAULT_UNITS`

*value*

`DEFAULT_UNITS` is `jaxstro.units.STELLAR`: mass in solar masses, length in
parsecs and time in Myr. Convenience wrappers resolve `units=None` to this
system; core APIs require explicit `G` or `units`. See the
[units policy](../20-architecture/units-policy.md).

## Browse alphabetically

The [full symbol index](full-symbol-index.md) lists every public
export, public method and documented property across all submodules, sorted
alphabetically, with classification and a link to its contract. Re-exported
names have entries in each owning module; inherited methods link to their
defining class. The top-level
`progenax` package re-exports a curated subset for convenience
(e.g. `from progenax import PlummerProfile`), so `len(progenax.__all__)`
is smaller. Use the index when you know the symbol name but don't
remember which module it lives in.

## How the API reference is built

The `scripts/build_api_reference.py` script:

1. Imports each module in `PUBLIC_MODULES` and walks `__all__`.
2. For each public symbol, extracts:
   - Class / function / protocol classification
   - Signature via `inspect.signature`
   - Cleaned docstring via `inspect.getdoc`
   - Source-file link via `inspect.getsourcefile`
3. Includes public methods, class/static constructors and documented properties.
   Inherited contracts link to their defining class rather than duplicating it.
4. Translates Google/NumPy sections and Sphinx/RST markup to MyST, preserving
   code examples, equations, units and limitations.
5. Emits one MyST page per module and the alphabetical full-symbol index.
   Missing exports or failed contracts make generation fail with their owner.

Introspection and markup conversion use Python stdlib. Registry badges use
the existing optional PyYAML dependency; no Sphinx runtime is needed. To
regenerate after public-API changes:

```bash
cd docs/website
python scripts/build_api_reference.py
```

The script is **idempotent**: re-running with no source changes
produces no diff. The docs workflow checks relevant source, dependency and
registry changes with a non-mutating comparison:

```bash
python scripts/build_api_reference.py --check
```

Run it in the project environment. A stale or missing generated page fails
the check; it does not overwrite files. The workflow also runs focused
generator tests. Local checks do not establish remote CI success.

## What the API reference does *not* contain

- **Tutorials** — see [](../00-getting-started/index.md).
- **Theory derivations** — see [](../10-theory/index.md).
- **How-to recipes** — see [](../40-howto/index.md).
- **Validation criteria** — see [](../50-validation/index.md).

The API reference is *deliberately* minimal: signature + docstring +
source link. For richer pedagogical context on any symbol, follow
the cross-link in its docstring to the relevant theory or how-to
chapter.

## Stable-vs-unstable API

```{list-table}
:header-rows: 1

* - Status
  - Convention
  - Example
* - **Stable** (default)
  - In a shipped public module's `__all__`; validation scope is recorded separately
  - `PlummerProfile`, `MultiComponentCluster`
* - **Experimental**
  - Repo-only packages under `src/experimental/`; not in the released wheel
  - `gravoturb` (turbulent/fractal ICs + inference)
* - **Internal**
  - Underscore-prefixed; not in `__all__`
  - `_grid_density_components`
```

Only stable-API symbols appear in this reference. Experimental and
internal symbols are documented in their own modules' docstrings
but do not show up here.

## Reporting issues

Report a wrong signature or contract against its source owner in
`progenax/<module>/<file>.py`. Report missing member coverage or malformed
markup against `scripts/build_api_reference.py`. Regenerate the reference
after fixing the owning source.

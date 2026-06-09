---
title: API reference
description: Auto-generated reference for every public symbol in progenax, regenerated from source on every release.
---

# API reference

This section is **auto-generated** from progenax source. The pages
in this section are produced by `scripts/build_api_reference.py`,
which walks `progenax.__all__` for each public submodule, extracts
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
* - [](tidal.md)
  - Jacobi-radius computation and tidal truncation
* - [](populations.md)
  - Multi-component populations (two-component clusters)
* - [](protocols.md)
  - Runtime-checkable protocols: `SpatialProfile`, `VelocityDF`, `IMFProtocol`
```

## Browse alphabetically

The [full symbol index](full-symbol-index.md) lists every public
symbol across all submodules, sorted alphabetically, with classification
(class / function / protocol / value) and link to the per-module page.
This counts every symbol in each submodule's `__all__`; the top-level
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
3. Emits one MyST page per module, with anchored sections per symbol.
4. Builds the alphabetical full-symbol index from the same data.

The script uses only Python stdlib (no Griffe / pdoc / sphinx-autoapi
dependency) and runs in $\\sim 5$ seconds. To regenerate after
public-API changes:

```bash
cd docs/website
python scripts/build_api_reference.py
```

The script is **idempotent**: re-running with no source changes
produces no diff. CI verifies this by running the script on every
PR and checking for unstaged changes.

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
  - In `progenax.__all__`; covered by tests
  - `PlummerProfile`, `magnification_factor`
* - **Experimental**
  - In `progenax.experimental.*`; minimal tests
  - Native lowered-model family (planned)
* - **Internal**
  - Underscore-prefixed; not in `__all__`
  - `_f_dense_bm19_full_jit`
```

Only stable-API symbols appear in this reference. Experimental and
internal symbols are documented in their own modules' docstrings
but do not show up here.

## Reporting issues

If you find an API reference page that is wrong (missing symbol,
wrong signature, stale docstring), the cause is almost always a
docstring problem in the source rather than a script problem. Open
an issue against the relevant `progenax/<module>/<file>.py` rather
than against the docs site. Then re-run the build script to update
the reference page.

---
title: Testing architecture
description: progenax's validation backbone — the three-tier test suite, the four frozen-literal registries that ratchet test/model/constant coverage against the public API, the generated dashboard, and the four-part release gate.
---
# Testing architecture

```{seealso}
This page is the **concepts** companion to the machine-generated
[](test-dashboard.md). The dashboard is the live census (per-module
counts, coverage, registry fill); this page explains the design those
numbers come from. For the per-test tolerance rules and the "anchor on
the defining condition" lesson, see [](methodology.md); for the
gradient registry specifically, see [](differentiability-audit.md).
```

progenax's validation backbone has three load-bearing pieces: a
**three-tier test suite** (what runs), **four registries** (a frozen
manifest per concern that ratchets coverage against the public API so a
new untested symbol turns CI red), and a **generated dashboard + release
gate** (the single source of truth, guarded against silent drift). This
page describes each in turn and closes with the workflow for adding a
new model or public symbol.

## The three-tier architecture

The suite separates *mechanical* correctness, *end-to-end* behaviour,
and *physical* correctness. Each tier has a distinct purpose and runs at
a distinct frequency.

```{list-table}
:header-rows: 1

* - Tier
  - Directory
  - Verifies
  - Count
* - **Unit**
  - `tests/unit/`
  - Per-function mechanical correctness (signatures, shapes, JIT-compatibility, gradient finiteness)
  - ~956
* - **Integration**
  - `tests/integration/`
  - End-to-end builder/pipeline behaviour through the user-facing entry points
  - ~43
* - **Validation**
  - `tests/validation/`
  - Quantitative match to analytic or published physics, with explicit pass/fail tolerances
  - ~244
```

The **FAST gate** (the inner loop) excludes `@pytest.mark.slow` tests;
the **FULL gate** (the phase/commit gate) runs everything. The
tolerance conventions for the validation tier — and the rule never to
weaken a tolerance to make a test pass — live in [](methodology.md).

## Tests versus registries

The backbone's central idea is the distinction between a **test** and a
**registry**.

- A **test** checks *behaviour*: it constructs a symbol, exercises it,
  and asserts something about the output (a virial ratio, a density
  shape, a gradient that matches finite differences).
- A **registry** is a hand-curated **frozen-literal manifest** plus a
  **ratchet test**. The ratchet cross-checks the manifest against
  `progenax.__all__` (114 public symbols) and the runtime-checkable
  protocols, so a *new* untested symbol, an *unregistered* model, or an
  *unprovenanced* constant turns CI **red** the moment it lands.

A registry is therefore a **meta-test**: it asserts that a test (or a
citation) *exists*, not that the test is good.

```{important}
Registries assert **existence, not quality**. The API-coverage registry
guarantees every public symbol has *an asserting test*; it does not
judge whether that test is strong. Test quality is the job of
[](methodology.md) (anchor on the defining condition) and of code
review — the registry only guarantees the floor is non-empty.
```

The manifests are **deliberately hand-curated frozen literals**, not
computed from `__all__` or from a coverage report at runtime. This is
the load-bearing design choice: a *derived* manifest re-reads the
current state, so it can never notice a **deletion** — if a model is
removed along with its test, a computed map simply shows one fewer
entry and stays green. A frozen literal, by contrast, still lists the
deleted symbol, so the ratchet fires. Each manifest also records honest
**holes** (an `UNTESTED` / `UNPROVENANCED` / `UNTESTED_MODELS` dict):
an honest hole is correct; a fabricated entry that points at a
non-asserting test is a failure.

## The four registries

Each registry lives under `tests/validation/<name>/manifest.py` (the
frozen literals) with an enforcing `test_*_coverage.py` (the ratchet).
The dashboard's registry block is generated from these manifests.

### API coverage

```{list-table}
:header-rows: 1
:widths: 30 70

* - Ratchets
  - Every `progenax.__all__` symbol has an *asserting* test (one that
    constructs the symbol and asserts on its output — a grep-mention is
    coverage *theater* and does not count).
* - Lives in
  - `tests/validation/api_coverage/manifest.py`
* - Current state
  - **112** symbols → an asserting test, **2** EXEMPT (pure PyTree
    container / unit-system constant), **0** untested. Line-coverage
    floor **90.0%**, currently measured **94.77%**.
```

Each symbol lands in exactly one of `SYMBOL_TESTS` (→ a verified node
id), `EXEMPT` (with a reason), or `UNTESTED` (a real hole — empty
today). The line-coverage floor is ratchet-up-only and is checked
against the committed full-suite `coverage.json`.

### Physics validation

```{list-table}
:header-rows: 1
:widths: 30 70

* - Ratchets
  - Every *model* has at least one enumerated physics invariant mapped
    to an asserting validation test (a virial Q, an ODE boundary
    condition, a concentration vs. King 1966, ...).
* - Lives in
  - `tests/validation/physics_registry/manifest.py`
* - Current state
  - **21** models → physics invariants, **90** non-model symbols
    exempt, **3** non-equilibrium carve-outs, **0** untested models.
```

The three carve-outs (`LIMEPYProfile`, `LIMEPYVelocityDF`,
`UniformSphereProfile`) are documented as reference-parity /
uniform-density models rather than equilibrium-DF models, so their
exclusion from the equilibrium-invariant class is auditable rather than
silent.

```{note} What counts as a "model" — the operational `IS_MODEL`
The ratchet does **not** trust a hand-written model list. It computes
membership: a symbol is a model **iff** it implements `SpatialProfile`,
`VelocityDF`, or `IMFProtocol` (the `runtime_checkable` protocols)
**or** is named `build_*_cluster`. A newly-added profile/DF/IMF or
cluster builder therefore *cannot* silently escape the registry — if it
conforms to a protocol but has no invariant, the ratchet reds CI.
```

### Provenance of constants

```{list-table}
:header-rows: 1
:widths: 30 70

* - Ratchets
  - Every cited physical constant / empirical coefficient carries a
    literature citation (paper + equation/table/page, or CODATA/IAU, or
    a derivable identity).
* - Lives in
  - `tests/validation/provenance_registry/manifest.py`
* - Current state
  - **29** constants → citations, allowlist-scoped to **8** `imf/` and
    `binaries/` modules, **0** unprovenanced.
```

This manifest is a hand-curated **port of `docs/provenance-ledger.md`**
(the 2026-06 provenance audit that read the held PDFs directly), **not**
a regex float-scanner — a regex over `src/` returns thousands of
matches dominated by `1.0`/`0.0`/array fills it cannot distinguish from
a citable coefficient. The new-literal ratchet is scoped to the
allowlisted modules (the densest paper-coefficient files); a new
citable-shaped literal there with no citation reds CI.

### Differentiability

```{list-table}
:header-rows: 1
:widths: 30 70

* - Ratchets
  - Every `progenax.__all__` symbol is `AUDITED` or `EXEMPT_*`, and each
    audited entry point passes an autodiff-vs-finite-difference gradient
    check (gradient integrity = Fisher integrity).
* - Lives in
  - `tests/validation/grad_audit/manifest.py`
* - Current state
  - The pre-existing template for the pattern; every public symbol is
    categorized, with the AD-vs-FD results recorded per case and zero
    unresolved hazards.
```

This is the registry the other three were modelled on; its design
(three independent frozen dicts — coverage targets, symbol categories,
a known-not-FD-consistent allowlist) and its two found-and-fixed
hazards are documented on the [](differentiability-audit.md) page.

## The generated dashboard and the release gate

The [](test-dashboard.md) page is **generated**, never hand-edited. The
generator is `scripts/build_test_dashboard.py`, which:

1. collects the per-module test census via `pytest --collect-only`;
2. reads the committed full-suite `coverage.json`, the registry
   manifests, durations, and validation-script exit codes;
3. assembles a timestamped dict, writes
   `validation/data/test_dashboard.json`, and re-renders the MyST
   matrix page.

```{code} bash
:caption: Regenerate the dashboard JSON + page
uv run python scripts/build_test_dashboard.py --emit --render
```

Because the page is generated, it could silently drift from the code.
The `@pytest.mark.slow` staleness gate
(`tests/validation/test_dashboard_fresh.py`) prevents that: it
regenerates the dashboard in-process and semantic-diffs it against the
committed JSON (and re-renders the page). Edit a test, change a
registry, or move a validate-script, and the gate goes **red** until
the JSON is regenerated and recommitted.

Coverage freshness is **source-based** (a deliberate ratification): the
committed coverage is stale **iff**
`git diff <coverage_provenance.git_sha> HEAD -- src/progenax/` is
non-empty. Adding or removing *tests* never invalidates the coverage of
unchanged *source*, so the ~14-minute full-suite `--cov` re-run is
required only when `src/progenax/` actually changes.

```{important} The four-part release gate
All four conditions must hold to release:

1. **`registries_full`** — all four registries built, full, and with
   zero holes.
2. **`line_cov ≥ 90%`** floor — the committed full-suite coverage clears
   the ratchet floor.
3. **Dashboard fresh** — the staleness gate is green (no drift between
   the committed JSON/page and a fresh regeneration).
4. **FULL suite green** — the complete (not `-m "not slow"`) suite
   passes.
```

## How to add a new model or public symbol

The registries are the **load-bearing friction**: adding a public
symbol means updating the relevant manifest(s), or the ratchet reds CI.
The workflow:

```{list-table}
:header-rows: 1
:widths: 45 55

* - If you add...
  - You must register it in...
* - Any new `progenax.__all__` symbol
  - **API-coverage** — add a `SYMBOL_TESTS` entry pointing at an
    asserting test (or justify an `EXEMPT`).
* - A new model (a `SpatialProfile` / `VelocityDF` / `IMFProtocol`
    conformer, or a `build_*_cluster`)
  - **Physics-validation** — add a `MODEL_INVARIANTS` entry mapping each
    invariant to an asserting validation test. The operational
    `IS_MODEL` ratchet will red until you do.
* - A new literature constant / empirical coefficient (in an allowlisted
    module)
  - **Provenance** — add a `PROVENANCE` row with the paper + eq/table
    citation (or an inline comment / `ALLOWLIST_NON_COEFFICIENT` carve
    for a numerical-method literal).
* - A new differentiable entry point
  - **Differentiability** — categorize it (`AUDITED` with a registry
    case, or `EXEMPT_*`) so the grad audit covers it.
```

After any of these, regenerate the dashboard
(`--emit --render`) and recommit it so the staleness gate stays green.
The point of the friction is that a new symbol cannot ship without a
deliberate, reviewable statement about how it is tested, validated,
sourced, and differentiated.

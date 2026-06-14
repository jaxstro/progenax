# progenax SoTA pre-release validation backbone — design

**Date:** 2026-06-14 · **Status:** design ratified (brainstorm with Anna) · **Branch (when built):**
`feat/test-backbone` off `main` · **Plan:** `docs/plans/2026-06-14-test-validation-backbone-plan.md`

## Motivation

progenax is being prepared for release. Anna's requirement: **one thorough, timestamped, non-drifting
source of truth for all pre-validation checks**, a test suite that is **current / high-value / SoTA**
with **100% coverage** (in the meaningful sense below), and a coherent **documentation** layer
(testing pages + a dev-facing matrix).

Today the "source of truth" is fragmented across artifacts that overlap and **drift**:
`docs/website/50-validation/index.md` (a hand-maintained dashboard that already carried stale counts),
`docs/provenance-ledger.md`, `tests/README.md`, `STATUS.md`, `docs/website/50-validation/physics-tests.md`,
the CLAUDE.md "Definition of Complete", and the one piece that is generated + timestamped + self-policing —
the **grad-audit** (`tests/validation/grad_audit/manifest.py` → `scripts/audit_gradients.py` → committed
JSON → website page → CI ratchet). This design generalizes the grad-audit pattern into a unified backbone.

## The conceptual foundation: tests vs registries

- A **test** is *imperative* and verifies **behavior** ("run this, assert that"). It catches behavioral
  regressions but is **blind to what is missing** — add a public function and forget to test it, and
  nothing fails.
- A **registry** is *declarative* and verifies **coverage completeness**: a `manifest` (the frozen
  "what must exist") + a `ratchet` (a test that cross-checks the manifest against the *live* system and
  goes RED on divergence). The grad-audit is a registry — its manifest cross-checks `progenax.__all__`,
  so an uncategorized new symbol or a deleted case reds CI.

**Tests catch behavior regressions; registries catch coverage regressions.** A registry turns "did we
test everything?" from a hope into a checkable invariant — the SoTA move that lets coverage *not* rot.

## Ratified decisions (brainstorm)

1. **Source of truth = generated + gated** (not hand-maintained). A script introspects the live suite +
   registries + coverage each run and emits a timestamped JSON + a rendered matrix page; a staleness
   test reds CI if the committed artifact drifts. Cannot go stale by construction.
2. **Scope = full backbone**: the 3 new registries + the existing differentiability one, the generated
   dashboard that unions them, AND a profiling-driven suite refactor.
3. **"100% coverage" = three checkable invariants** (not strict 100% line, which is a false god given
   defensive/error branches and JAX-traced dead branches):
   - **API-coverage:** 100% of `progenax.__all__` has ≥1 real test.
   - **Line coverage** (pytest-cov, FULL suite only — a `-m "not slow"` pass understates and would spuriously
     fail) ≥ a high floor (start **90%**, ratchet-up-only via the `LINE_COV_FLOOR` literal), every exclusion an
     explicit `# pragma: no cover` **with a reason**.
   - **Zero registry holes:** all 4 registries full.

## Architecture

Three layers:

1. **Registry layer** (declarative, self-policing coverage) — 4 registries, each = manifest + ratchet.
2. **Generated dashboard** (the single timestamped source of truth) — the union of the registries +
   coverage + durations + PASS/FAIL.
3. **Refactored suite** — every test traces to a registry entry or a physics claim; runtime tiered with
   `@slow`; cross-tier redundancy consolidated.

### The 4 registries

| Registry | Manifest enumerates | Ratchet reds when | Status |
|---|---|---|---|
| **Differentiability** | every public entry × param → measured AD-vs-FD | new `__all__` symbol uncategorized / case deleted | ✅ exists (85 cases) |
| **API-coverage** | every `__all__` symbol → ≥1 test that CONSTRUCTS/CALLS it and ASSERTS on its output (NOT a grep-mention); a full-suite line-coverage floor | a public symbol untested / line-cov < floor | 🆕 |
| **Physics-validation** | every public model (defined operationally: implements `SpatialProfile`/`VelocityDF`/`IMFProtocol` or is a `build_*_cluster`) → its physics invariants (equilibrium Q, density recovery, conservation, closed-form anchors, OR reference-parity for tabulated models like LIMEPY) | a new public model has no physics-validation entry | 🆕 |
| **Provenance-of-constants** | hand-curated allowlist of constant-bearing literals in `src/` (ported from `docs/provenance-ledger.md`, NOT a regex float-scanner — ~2,525 noisy matches) → a cited source | an unprovenanced literal in an allowlisted location ships without a `# provenance:` comment | 🆕 (the 2026-06 audit did this by hand once; make it self-policing) |

Each registry follows the grad-audit's frozen-literal pattern: a hand-curated manifest (NOT computed
from the live system at runtime — a derived manifest can't catch a deletion), an `__all__` cross-check,
and a coverage ratchet, all as pytest tests in a fast dedicated job. Note the grad-audit ALREADY maintains a
full `__all__`→`AUDITED|EXEMPT_*` partition (`SYMBOL_CATEGORY`, 7 EXEMPT categories); API-coverage and
physics-validation are additional partitions over the SAME `__all__`, so each must cross-check its EXEMPT set
against the grad-audit's to keep the partitions from drifting apart.

### The generated dashboard (the source of truth)

`scripts/build_test_dashboard.py` introspects, in one run:
- the live suite (collected test counts per module/tier, pass/fail),
- `pytest-cov` line coverage per `src/` module,
- each of the 4 registries' fill status (covered / holes),
- the validation scripts' PASS/FAIL (`scripts/validate_*.py` exit codes),
- the `@slow` split + the slowest test per module (from `--durations`),

and emits:
- **`validation/data/test_dashboard.json`** — the machine-readable, **timestamped** source of truth
  (committed),
- **`docs/website/50-validation/test-dashboard.md`** — the rendered human matrix (per-module rows ×
  the columns above), built into the website.

A **staleness test** (`tests/validation/test_dashboard_fresh.py`) regenerates the JSON and semantic-diffs
it against the committed artifact (exact on discrete fields, rtol on floats) → CI red on drift. The
release gate is: **all 4 registries full + line-cov ≥ floor + dashboard fresh + FULL suite green.**

### The suite refactor (profiling-grounded)

The FULL gate is **10:32**, dominated by a handful of tests (measured `--durations=45`, 2026-06-14):

| Test | Call time | Action |
|---|---|---|
| `grad_audit/test_json_fresh::test_committed_json_matches_fresh_regeneration` | **429 s** | `@slow` (FULL-only) + consolidate |
| `grad_audit/test_audit_script::test_run_audit_emits_required_keys` | **406 s** | `@slow` + consolidate (both regenerate the SAME JSON ≈835 s of near-duplicate work) |
| `test_find_alpha_ift` (multimass IFT grad/quadrature, ~7 tests) | ≈620 s total | `@slow` (already partly marked) |
| `test_engine_b_physics::test_plummer_halo_eff_core_equilibrium` | 186 s | `@slow` |
| grad-audit `MultiComponentCluster.sample_cluster[EngineA]` cases | ≈370 s | `@slow` (these run the multimass solve under `jax.grad`) |
| `test_limepy_multimass` eigenvalue/differentiable (~4 tests) | ≈255 s | `@slow` |

`@slow`-marking these high-value-but-slow tests (and consolidating the two JSON regenerators) drops the
**FAST inner loop to minutes** with **zero coverage loss**; the FULL gate stays complete. Then:
- **Consolidate cross-tier redundancy** — the King pattern (unit `test_king.py` duplicated validation
  `test_king_physics.py`, already done) repeats for **Plummer / EFF / Michie** (each has a unit *and* a
  validation `test_<p>_physics.py`). Per-file: keep the higher-value validation test, remove the unit
  duplicate, keep unit-UNIQUE guards. Verified scope: Plummer = partial (keep virial-Q / q²-variance
  unit-unique), EFF = clean (remove ~6, keep ~5), Michie = partial (keep the table-routing guards).
  **LIMEPY is NOT symmetric** — it has NO `test_limepy_physics.py`; its validation tier is
  `test_limepy_reference_parity.py` (a reference-parity oracle) + `test_multimass_equilibrium_physics.py`,
  which do not duplicate the unit limepy files. Treat LIMEPY as a separate "inspect, likely no-op" task
  (at most de-dup AMONG the unit files).
- **Kill orphan low-value tests** — any test not tracing to a registry entry, a physics claim, OR a listed
  cross-cutting invariant. **Cross-cutting tests are PROTECTED:** units/G-threading
  (`test_units_through_pipeline.py`), protocol-conformance (`test_protocols.py`), end-to-end energy — these
  trace to a named design requirement, not a per-symbol row, and must NOT be deleted as orphans.

### Documentation layer

- **`docs/website/50-validation/test-dashboard.md`** — the generated matrix (the source of truth).
- **`docs/website/50-validation/testing-architecture.md`** (new) — the 3-tier philosophy + the registry
  concept (tests vs registries) + how the dashboard is generated + the release gate. The conceptual
  "why".
- The existing **per-feature validation pages** (`plummer-equilibrium.md`, `king-profile.md`, …) — the
  physics narrative; keep, link from the dashboard.
- Retire/redirect the stale hand-maintained bits (`50-validation/index.md` becomes a thin pointer to the
  generated dashboard).

## Phasing (multi-session, its own branch)

Each phase is its own batch with local verification + an Anna checkpoint; CI minutes are exhausted →
verify LOCALLY, no PR until the arc closes.

1. **Coverage + dashboard scaffold** — confirm `pytest-cov` (already in `[dev]`, installed) + EXTEND the
   existing `[tool.coverage]` config (keep `source=["progenax"]`), build `build_test_dashboard.py` + the JSON +
   the committed full-suite `coverage.json` (with provenance) + the rendered page + the staleness gate. The
   keystone; immediately useful even before the new registries exist.
2. **API-coverage registry** — manifest of `__all__` → test + the line-cov floor; ratchet. Fastest
   registry; surfaces every untested public symbol.
3. **Suite refactor** — `@slow`-mark the runtime sinks (the table above), consolidate the
   Plummer/EFF/Michie/LIMEPY cross-tier redundancy, fill the API holes Phase 2 surfaced. Re-profile to
   confirm the FAST loop dropped.
4. **Physics-validation registry** — the richest: enumerate each public model's physics invariants;
   ratchet that a new model must register one.
5. **Provenance-of-constants registry** — scan `src/` for numeric literals / fit coefficients; manifest
   each → a citation; ratchet.
6. **Docs + close-out** — testing-architecture page, dashboard rendered into the website (`make build`
   0 warnings), completion doc, STATUS/brain, final FULL gate.

## YAGNI / explicitly out of scope

- Strict 100% line coverage (rejected — false god; use 90% floor + documented pragmas).
- Mutation testing, property-based testing frameworks (future; the registries + line floor are enough
  for release).
- A bespoke web app for the dashboard (it's a generated markdown matrix in the existing MyST site).
- Rewriting the existing per-feature validation pages (they stay; the dashboard links them).

## Success criteria (the release gate)

Release-readiness = **every registry full (0 holes) + line-cov ≥ floor + the dashboard fresh + the FULL
suite green** — all **timestamped** and **regenerable by one command**, with the FAST inner loop back to
minutes.

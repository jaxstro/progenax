# Registry Ratchet Harness Hoist — Design

**Date:** 2026-06-19
**Status:** Ratified (brainstorm with Anna)
**Owner:** Anna Rosen (single HITL)
**Arc:** progenax v0.1.0 pre-release hardening

## Problem

progenax has four test registries — `tests/validation/{grad_audit, api_coverage,
physics_registry, provenance_registry}/` — each built, full, and zero-holes. But each
**re-implements its partition / staleness / anti-theater mechanism inline**, and **there
are no tests of that mechanism**. A bug in the duplicated partition or citation-scanning
logic could silently pass a corrupt registry across all four, defeating the guarantees the
registries exist to provide. The provenance and physics registries carry ~150–200 LOC of
inline AST/tokenize/subprocess machinery (`_scan_module_for_unprovenanced`,
`_cited_comment_lines`, `_cited_docstring_spans`, `_is_citable_shaped`, the `--collect-only`
node-id check) that is generic and untested.

## Prior art

fluxax already solved this for itself: `fluxax/tests/validation/_ratchet.py` (397 LOC,
stdlib-only, content-free) factors the generic mechanism into seven primitives, plus
`test_ratchet_harness.py` that unit-tests the harness itself. fluxax ADR-0010 deliberately
built it **hoist-ready** but deferred the hoist to `jaxstro.testing` (rule-of-three:
startrax/stellax). `jaxstro.testing` **already exists** and already hosts the hoisted
AD-vs-FD grad-audit engine (`jaxstro/src/jaxstro/testing/grad_audit.py`), re-exported by
both progenax and fluxax — so the hoist of the ratchet harness is an *additive extension*
of an established pattern, not greenfield.

## SoTA decision (Task 0)

fluxax's `_ratchet.py` is already near-SoTA: content-free, stdlib-only, with actionable
failure messages and correct handling of the subtle cases (signed-literal folding through
`UnaryOp(USub)`; the *module*-level docstring deliberately excluded from citation
whitelisting so a new uncited module-level coefficient cannot hide; a fail-loud
`resolve_node_ids` net when pytest errors in an unexpected mode). The honest SoTA delta is
at the margins, **not** a rewrite — but the design is chosen on its own merits, with fluxax
as prior art, **not** a binding contract (Anna owns fluxax; she will refactor it onto the
canonical harness afterward, scope permitting more than an import-swap).

**Chosen design:** free-function primitives (not a Registry base class — that over-abstracts
and obscures which symbol/bucket failed; legible test failures win over DRY for test code),
adopted as the canonical `jaxstro.testing.ratchet` API, with these improvements over the
fluxax original:

1. **Ecosystem-general docstring** — drop fluxax/ADR-0010 framing; reference the new hoist ADR.
2. **`str | Path` path arguments** on `scan_module_numeric_literals` / `has_nearby_citation`.
3. **Rigorous self-tests in jaxstro** — `test_ratchet_harness.py` ported into jaxstro's own
   test tier so the mechanism is guaranteed centrally for every consumer.
4. **No speculative API** (YAGNI) — keep the primitives separate/composable; do not add a
   parse-once combined scanner (only a handful of allowlist modules are scanned; the
   fast-loop cost is negligible and composability is worth more).

### Canonical public surface (`jaxstro.testing.ratchet`)

| Primitive | Purpose |
|-----------|---------|
| `assert_partition(all_symbols, *buckets, label)` | buckets exactly partition a universe (coverage + disjointness + no-stale) |
| `assert_no_stale(mapping, universe, label)` | every mapping key still exists in the universe |
| `resolve_node_ids(node_ids, *, rootdir) -> set[str]` | subset of node ids that `pytest --collect-only` resolves (fail-loud) |
| `test_body_has_assert(node_id) -> bool` | the cited test body contains an assert or recognized helper call |
| `ASSERT_HELPERS: tuple[str, ...]` | recognized assert-helper call-prefix allowlist |
| `scan_module_numeric_literals(path, *, trivial, small_int_max)` | citable-shaped numeric literals `(value, lineno)` |
| `has_nearby_citation(path, lineno, *, window=4) -> bool` | a citation sits near a literal / in a cited scoped docstring |

### Hoist boundary

- **Hoist (generic):** the seven primitives above.
- **Stays progenax-local (package policy):** the four manifests; `physics_registry/
  _operational_model_kind()` (progenax protocols + `build_*_cluster` regex); `grad_audit/
  test_json_fresh.py` cross-arch float comparator; the *orchestration* that combines
  primitives into policy (provenance's `_scan_module_for_unprovenanced` walk + carve rules;
  the cross-check-vs-grad-audit divergence reports).

## Packaging

`jaxstro.testing.ratchet` ships in the **base** `src/jaxstro` wheel (mirrors grad_audit) —
**no `[testing]` extra**. The harness is stdlib-only and *shells out* to pytest rather than
importing it, so consumers bring their own pytest (progenax already has it in `[dev]`). This
adds zero new dependency edges (progenax already depends on jaxstro via editable path-source,
ADR-0012).

## Cross-cutting constraints

- **Fast inner loop stays fast** (Anna, non-negotiable): nothing slow in `check.sh` /
  `-m "not slow"`. Harness self-tests are pure-stdlib fixtures (ms). Any full-tree
  characterization or collect-heavy check is `@pytest.mark.slow` / `release_gate.sh` only.
- **CI stays dormant until the repo is public**; the local two-tier gate substitutes.
- **HITL** at every step; cross-repo branches (jaxstro + progenax); merge and push are
  separate words.

## Out of scope (deferred; Anna owns)

jaxstro-lab repo; moving OED `optimal-design/` docs + `scripts/_demo_oed*`; `gravoturb_fdf`
migration; fluxax migration; sdist slimming; PyPI / jaxstro publication; re-architecting the
dormant CI YAML.

## ADRs to record

1. Hoist the ratchet harness to `jaxstro.testing.ratchet` (ecosystem-general, base-wheel,
   no extra; fluxax migration deferred to Anna; supersedes fluxax ADR-0010 in principle).
2. CI cheapness/shard decision deferred to repo-publication; dormant-as-is.

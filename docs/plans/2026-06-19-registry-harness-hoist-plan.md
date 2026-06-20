# Registry Ratchet Harness Hoist — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to
> implement this plan task-by-task (fresh subagent per task + independent
> superpowers:code-reviewer between tasks + final whole-arc review).

**Goal:** Hoist progenax's four registries' duplicated, untested partition/staleness/
anti-theater mechanism into a canonical, self-tested `jaxstro.testing.ratchet`, refactor
progenax onto it, and add a local two-tier gate that substitutes for the dormant CI.

**Architecture:** Additive new module `jaxstro/src/jaxstro/testing/ratchet.py` (stdlib-only,
content-free, base-wheel-shipped, no extra) + jaxstro self-tests; then progenax's four
registry test modules swap their inline helpers for imports from it (behavior pinned by
characterization tests first); then `scripts/{check.sh,release_gate.sh}` + committed
observability stubs.

**Tech Stack:** Python 3.11+, pytest, pytest-xdist, jaxstro/progenax editable path-source
(ADR-0012), `ast`/`tokenize`/`subprocess` stdlib, uv.

**Design doc:** `docs/plans/2026-06-19-registry-harness-hoist-design.md`

**Cross-repo:** jaxstro (Phase 1) then progenax (Phases 2–4). Branch off `main` in EACH
repo. Merge and push are SEPARATE words. Delete a branch only once merged AND pushed.

**Fast-loop rule (non-negotiable):** nothing slow in `-m "not slow"` / `check.sh`. Harness
self-tests are ms-scale stdlib fixtures. Any full-tree characterization is `@pytest.mark.slow`.

---

## PHASE 1 — jaxstro: `jaxstro.testing.ratchet` (additive)

Repo: `~/projects/jaxstro-dev/jaxstro`. Branch: `feat/testing-ratchet-harness`.

**Pre-task:** confirm jaxstro's test layout for `testing/` — find where `grad_audit.py` is
tested (`grep -rl grad_audit jaxstro/tests`) and mirror that location/naming for the new
harness test. Use the gate command `scripts/check.sh`.

### Task 1.1: Create the canonical harness module

**Files:**
- Create: `jaxstro/src/jaxstro/testing/ratchet.py`
- Modify: `jaxstro/src/jaxstro/testing/__init__.py`

**Step 1: Seed from the proven prior art.**
Copy `~/projects/jaxstro-dev/fluxax/tests/validation/_ratchet.py` →
`jaxstro/src/jaxstro/testing/ratchet.py` verbatim (it is already content-free / stdlib-only).

**Step 2: Apply the three SoTA deltas (design doc §SoTA decision):**

(a) Replace the module docstring's fluxax/ADR-0010 framing with the ecosystem-general
version (reference the new hoist ADR number once assigned):

```python
"""Generic ratchet primitives for jaxstro-ecosystem testing registries.

Content-free mechanisms every per-package testing registry (api_coverage, grad_audit,
physics_registry, provenance_registry) reuses: partition/staleness assertions, pytest
node-id resolution, an AST asserts-behavior check, and numeric-literal/citation scanning
for provenance tripwires. They bake in NO package-specific symbol names, paths, or
citations — only generic shapes — so each package layers its own manifests/policy on top.

TEST INFRASTRUCTURE, not core differentiable code: ``ast``/``tokenize``/``subprocess`` and
the rest of the stdlib are appropriate here, and nothing on a JAX path imports it. Ships in
the base wheel (dependency-light, no pytest at import); consumers bring their own pytest.

See ADR-00NN (ratchet-harness hoist).
"""
```

(b) Broaden the two path-taking primitives to accept `str | Path` (the body already wraps
in `Path(...)`, so only the annotations change):

```python
def scan_module_numeric_literals(
    path: str | Path, *, trivial: set[float], small_int_max: int
) -> Iterator[tuple[float, int]]:
    ...
    src = Path(path).read_text()
```
```python
def has_nearby_citation(path: str | Path, lineno: int, *, window: int = 4) -> bool:
    ...
    src = Path(path).read_text()
```
(Rename the internal `rel_path` references to `path` consistently.)

(c) Export the public surface from `jaxstro/src/jaxstro/testing/__init__.py`:

```python
from jaxstro.testing.ratchet import (
    ASSERT_HELPERS,
    assert_no_stale,
    assert_partition,
    has_nearby_citation,
    resolve_node_ids,
    scan_module_numeric_literals,
    test_body_has_assert,
)
```
Add those seven names to `__all__` (keep the existing grad_audit exports).

**Step 3: Commit**

```bash
git add src/jaxstro/testing/ratchet.py src/jaxstro/testing/__init__.py
git commit -m "feat(testing): canonical ratchet harness in jaxstro.testing"
```

### Task 1.2: Self-test the harness (mechanism tests)

**Files:**
- Create: `jaxstro/tests/<mirrors grad_audit test location>/test_ratchet_harness.py`
- Create (if needed): a tiny committed fixture file that the collect-resolution test targets.

**Step 1: Write failing tests** — port fluxax's `tests/validation/test_ratchet_harness.py`
coverage, adapted to the canonical API (`from jaxstro.testing.ratchet import ...`). Required
cases (one assertion each, ms-scale, tmp_path fixtures — NO JAX, NO real-suite scan):

- `assert_partition`: clean partition passes; missing symbol raises; overlap raises; stale
  bucket entry raises.
- `assert_no_stale`: clean passes; stale key raises.
- `test_body_has_assert`: False for no-assert body; True for bare `assert`; True for an
  `ASSERT_HELPERS` call (e.g. `np.testing.assert_allclose`); resolves `file::Class::method`.
- `scan_module_numeric_literals`: yields citable literals only; respects `small_int_max`;
  skips bool + |v|<1e-9; preserves a signed negative literal (`-2.35` via `UnaryOp(USub)`).
- `has_nearby_citation`: True for a comment within `window` above; False without; True for a
  scoped (function/class) docstring citation; **module-level docstring citation does NOT
  whitelist a file-level literal** (the tripwire-defeat regression); function-level
  docstring DOES whitelist its own body.
- `resolve_node_ids`: keeps only real ids; **rejects an import-broken file's id** (fail-loud).

**Step 2:** Run, expect import/collection-driven FAIL first where applicable.
**Step 3:** No new impl needed (module exists); fix any API-name drift surfaced by tests.
**Step 4:** `env -u VIRTUAL_ENV uv run --no-sync pytest tests/<...>/test_ratchet_harness.py -q` → PASS.
**Step 5: Commit**
```bash
git add tests/<...>/test_ratchet_harness.py tests/<...>/_ratchet_fixtures/
git commit -m "test(testing): self-tests for the ratchet harness"
```

### Task 1.3: Phase-1 gate + checkpoint
- Run jaxstro `scripts/check.sh` → green (lint + mypy + tests + wheel-smoke; the new module
  must lint/type-clean and ship in the wheel-smoke import).
- **HITL checkpoint** → on Anna's word: merge → (push on her separate word). Do NOT delete
  the branch until pushed.

---

## PHASE 2 — progenax: characterize, then refactor the 4 registries

Repo: `~/projects/jaxstro-dev/progenax`. Branch: `feat/adopt-ratchet-harness`.
**Depends on Phase 1 merged** (the editable jaxstro must expose `jaxstro.testing.ratchet`).
No `src/progenax/` change in this phase.

Gate per task:
`XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -m "not slow" -n auto`

### Task 2.1: Characterization safety-net (pin current behavior FIRST)

**Files:** Create `tests/validation/test_ratchet_characterization.py`.

Before touching any registry, pin the *current* observable output of the mechanism so the
refactor is provably behavior-preserving:
- The exact set of provenance literal "holes" detected on the allowlist modules (scope to
  `ALLOWLIST_MODULES` only — fast; do NOT scan the whole tree on the fast path).
- The exact partition/staleness pass for each registry (they pass today → assert pass).
- A representative `test_body_has_assert` + `resolve_node_ids` result for a known node id.

Run; expect PASS (characterizing green state). Commit.
If any full-tree characterization is wanted, mark it `@pytest.mark.slow` (release-gate only).

### Task 2.2: Refactor `api_coverage` onto the harness

**Files:** Modify `tests/validation/api_coverage/test_api_coverage.py`.
Read the current file first. Replace the inline partition/staleness/`resolve_node_ids`/
`test_body_has_assert` logic with:
```python
from jaxstro.testing.ratchet import (
    assert_partition, assert_no_stale, resolve_node_ids, test_body_has_assert,
)
```
Map: `assert_partition(set(progenax.__all__), SYMBOL_TESTS, EXEMPT, UNTESTED,
label="api_coverage.partition")`; per-mapping `assert_no_stale(...)`; cited-test resolution
via `resolve_node_ids`; body-assert via `test_body_has_assert`. Keep the cross-check-vs-
grad-audit divergence report LOCAL. Run fast gate + characterization → green. Commit.

### Task 2.3: Refactor `physics_registry` onto the harness

**Files:** Modify `tests/validation/physics_registry/test_physics_coverage.py`.
Swap inline partition/staleness for `assert_partition`/`assert_no_stale`; replace the inline
`--collect-only` subprocess with `resolve_node_ids` (assert all `MODEL_INVARIANTS` node ids
resolve). **Keep** `_operational_model_kind()` and the grad-audit cross-check local. Run
fast gate + characterization → green. Commit.

### Task 2.4: Refactor `provenance_registry` onto the harness

**Files:** Modify `tests/validation/provenance_registry/test_provenance_coverage.py`.
This is the delicate one — read the current file carefully. Replace the inline
`_cited_comment_lines` / `_cited_docstring_spans` / `_is_citable_shaped` / `_has_nearby_citation`
with `scan_module_numeric_literals` + `has_nearby_citation` from the harness. **Keep** the
orchestration LOCAL: the `_scan_module_for_unprovenanced` walk, the `PROVENANCE`/
`ALLOWLIST_NON_COEFFICIENT` carve policy, the value-in-provenance matching. The
characterization test (2.1) must stay byte-identical (same holes, same whitelist outcomes) —
this is the proof the swap preserved the subtle docstring/sign semantics. Run fast gate +
characterization → green. Commit.

### Task 2.5: Refactor `grad_audit` onto the harness

**Files:** Modify `tests/validation/grad_audit/test_manifest_coverage.py`.
Swap inline partition/staleness for `assert_partition`/`assert_no_stale`. **Keep**
`test_json_fresh.py`'s cross-arch float comparator untouched (registry-specific, already
`@slow`). Run fast gate + characterization → green. Commit.

### Task 2.6: Phase-2 full gate + checkpoint
- Confirm all four registries still full / zero-holes (dashboard registry blocks unchanged).
- Run progenax FULL gate (same command without `-m "not slow"`). No `src/` change → coverage
  unaffected; confirm dashboard freshness still green (no `--emit` needed unless a test
  module was added — 2.1 adds one, so regen the dashboard if its module inventory shifts and
  re-stamp; coverage stays byte-identical).
- **HITL checkpoint.**

---

## PHASE 3 — progenax: local two-tier gate + observability stubs

Same branch (or `feat/local-gate`). Templates: `fluxax/scripts/{check.sh,release_gate.sh}`,
`jaxstro/scripts/check.sh`.

### Task 3.1: `scripts/check.sh` (fast gate)
Mirror fluxax's structure, adapted to progenax's XLA-thread-capped `-n auto` invocation:
lint (`ruff check` + `ruff format --check` + `mypy src/progenax`) → `uv lock --check` →
fast suite `-m "not slow" -n auto` (registries + dashboard-freshness + coverage-floor READ,
no re-measure) → integration → wheel-smoke. Run it → green. Commit.

### Task 3.2: `scripts/release_gate.sh` (heavy gate) + observability stubs
FULL `--cov` re-measure (deselect the dashboard-freshness test during regen to avoid
circularity) → regen + stamp `coverage.json` + dashboard via `scripts/build_test_dashboard.py`
→ run floor + freshness tests → assert the 4-part conjunction (registries_full ∧
line_cov≥90 ∧ dashboard_fresh ∧ full_suite_green). **Generate + commit**
`validation/data/{validation_runs.json,durations.json}` (the dashboard reads both;
neither is currently committed → closes the unknown/not-measured gap). Run it → green,
dashboard now shows measured durations/runs. Commit.

### Task 3.3: CI audit (read-only) + checkpoint
Confirm both workflows remain `disabled_manually` and the dormant YAML is internally sound;
make NO trigger/enable changes. **HITL checkpoint.**

---

## PHASE 4 — ADRs + close-out

### Task 4.1: ADRs (via `/adr`, progenax is at ADR-0021+)
- ADR: hoist ratchet harness to `jaxstro.testing.ratchet` (ecosystem-general, base-wheel,
  no extra; fluxax migration deferred to Anna; supersedes fluxax ADR-0010 in principle).
- ADR: CI cheapness/shard decision deferred to repo-publication; dormant-as-is.

### Task 4.2: Optional separable polish (only if Anna says so)
Scrub `/Users/anna/…` + `.claude-work/` refs from `docs/website/90-development-log/`;
soften hard-coded test counts to point at the test dashboard (NOT "see CI" — CI is dormant).

### Task 4.3: Close-out
- Final whole-arc independent code-review (superpowers:code-reviewer).
- Update `STATUS.md` (`next:`/`blocker:`/`due:`).
- `brain "..."` milestone capture.
- Completion note `.claude-work/REGISTRY_HARNESS_HOIST_COMPLETE.md`.
- **HITL** for merge/push on each repo; delete branches only once merged AND pushed.

---

## Verification (whole arc)
- jaxstro `scripts/check.sh` green; harness self-tests pass (ms-scale).
- progenax FAST gate green; all 4 registries full/zero-holes; provenance characterization
  byte-identical pre/post refactor.
- progenax `scripts/release_gate.sh` 4-part conjunction green; dashboard shows measured
  durations/runs; coverage ≥ 90 (byte-identical to pre-arc — no `src/` change).
- Cross-repo: progenax FULL gate green against editable jaxstro carrying the new harness.
- Fast-loop acceptance: `check.sh` wall-clock not measurably worse than today's
  `-m "not slow"` (~4 min).

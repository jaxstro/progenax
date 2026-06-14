# Test/Validation Backbone Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan
> task-by-task. STRICT HITL with Anna — stop at every `=== CHECKPOINT ===` for her approval. TDD
> RED→GREEN→REFACTOR. NEVER weaken a test/tolerance; fix the root cause. CI minutes are exhausted →
> verify LOCALLY, no PR until the arc closes. Commit per verified task; end commit messages with
> `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

**Goal:** Build a generated, timestamped, self-policing single source of truth for all of progenax's
pre-validation checks — a registry layer (4 registries) + a generated dashboard + a profiling-driven
suite refactor — so the project releases with SoTA validation, verification, and documentation.

**Architecture:** Generalize the existing grad-audit pattern (manifest + ratchet + generated JSON +
website page + staleness gate) into four registries (differentiability [exists], API-coverage,
physics-validation, provenance-of-constants). A single generator script unions them with `pytest-cov`
line coverage and `--durations` into a committed, timestamped JSON + a rendered MyST matrix page, gated
by a staleness test. The suite is then refactored: `@slow`-mark the measured runtime sinks, consolidate
cross-tier redundancy, fill the coverage holes the registries surface.

**Tech Stack:** pytest + pytest-xdist + pytest-cov; JAX (the code under test); MyST (mystmd) for the
dashboard page; the existing `tests/validation/grad_audit/` registry as the template.

**Design:** `docs/plans/2026-06-14-test-validation-backbone-design.md` (read it first). Relevant skills:
@superpowers:test-driven-development, @superpowers:verification-before-completion,
@research-workflow:provenance-of-constants, @astro-code-review:testing-strategist,
@reviewing-project-quality, @myst:myst-expert (the dashboard page).

**Pre-flight (verify before relying on these — re-check, do not trust):**
- The grad-audit template lives in `tests/validation/grad_audit/{manifest.py,registry.py,core.py,
  test_manifest_coverage.py,test_json_fresh.py}` + `scripts/audit_gradients.py`. Read these first — every
  new registry mirrors this structure (hand-curated frozen literals + an `__all__` cross-check + a
  ratchet test + a generated JSON + a staleness diff).
- `progenax.__all__` has 114 symbols (verify: `python -c "import progenax; print(len(progenax.__all__))"`).
- The FULL gate is `pytest tests/unit tests/integration tests/validation -q -n auto` (~10:32). The FAST
  gate adds `-m "not slow"`. `slow` is a registered marker (`pyproject.toml`).
- Measured `--durations=45` runtime sinks (2026-06-14, in the design doc): `test_json_fresh` 429s +
  `test_audit_script` 406s (both regenerate the grad-audit JSON), `test_find_alpha_ift` ~620s,
  `test_engine_b_physics::test_plummer_halo_eff_core_equilibrium` 186s, the Engine-A grad-audit cases
  ~370s, `test_limepy_multimass` ~255s.
- `pytest-cov` may NOT be installed (`[dev]` lists pytest/black/isort/flake8/mypy). Task 1.1 checks + adds.

**Env prefix (uv, NOT conda):**
```
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest <args>
```

---

## Phase 0: Branch

### Task 0.1: Create the branch
```bash
git switch -c feat/test-backbone   # off main (303cd2b: build_cluster + website follow-ups)
```
`=== CHECKPOINT 0 → Anna ===`

---

## Phase 1: Coverage + dashboard scaffold (the keystone)

### Task 1.1: Wire pytest-cov

**Files:** Modify `pyproject.toml`.

**Step 1:** Check: `env -u VIRTUAL_ENV uv run --no-sync python -c "import pytest_cov"`. If ImportError,
add `pytest-cov` to the `[dev]` optional-dependencies in `pyproject.toml` and
`env -u VIRTUAL_ENV uv pip install -e ".[dev]"`.

**Step 2:** Add a coverage config to `pyproject.toml`:
```toml
[tool.coverage.run]
source = ["src/progenax"]
branch = true
omit = ["*/experimental/*", "*/__pycache__/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
]
```

**Step 3:** Run a smoke coverage pass on a fast subset:
```
XLA_FLAGS=... env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/builders -q \
  --cov=progenax --cov-report=json:/tmp/cov.json
```
Expected: writes `/tmp/cov.json` with a `totals.percent_covered` field. Confirm it parses.

**Step 4:** Commit (`chore(coverage): wire pytest-cov + coverage config`).

### Task 1.2: Dashboard generator — test inventory (TDD)

**Files:** Create `scripts/build_test_dashboard.py`; Test `tests/validation/test_dashboard_gen.py`.

**Step 1 (RED):** write a test that the generator collects per-module test counts:
```python
from scripts.build_test_dashboard import collect_test_inventory

def test_inventory_has_modules_and_counts():
    inv = collect_test_inventory()  # {module: {"unit": n, "integration": n, "validation": n}}
    assert "builders" in inv
    assert inv["builders"]["unit"] > 0
    assert sum(t for m in inv.values() for t in m.values()) > 1000
```
Run → FAIL (no module). 

**Step 2 (GREEN):** implement `collect_test_inventory()` using `pytest --collect-only -q` (parse the
node ids → module/tier). Reductions: map a node id `tests/unit/builders/test_x.py::...` → module
`builders`, tier `unit`. Use `subprocess` to run collect-only (the generator is a script, not core code,
so subprocess + parsing is acceptable — NOT JAX-native-constrained). Run → PASS. Commit.

### Task 1.3: Dashboard generator — line coverage

**Step 1 (RED):** test that the generator attaches per-module line coverage:
```python
def test_inventory_attaches_line_coverage(tmp_path):
    # given a coverage.json (from a prior --cov run), the generator reads totals + per-file %
    cov = load_coverage("/tmp/cov.json")   # or a fixture
    assert 0 <= cov["total_percent"] <= 100
```
**Step 2 (GREEN):** add `load_coverage(path)` parsing the pytest-cov `coverage.json` (`totals.percent_covered`
+ per-file `summary.percent_covered`, mapped to `src/progenax/<module>`). Commit.

### Task 1.4: Dashboard generator — registry status + durations + validation scripts

**Step 1:** add functions that read:
- **grad-audit** fill: import `tests.validation.grad_audit.manifest` (counts AUDITED/EXEMPT, MUST_AUDIT,
  the committed `validation/data/grad_audit_results.json` hazard count). The OTHER 3 registries don't
  exist yet → return `{"status": "not-built"}` placeholders (filled in Phases 2/4/5).
- **durations**: parse a committed `--durations` artifact (or run a `-m "not slow"` quick pass and
  capture the slowest per module). Keep it cheap.
- **validation scripts**: map each `scripts/validate_*.py` → last exit code (store in the JSON; the
  generator does NOT run them — too slow; a separate `--run-validations` flag does).

**Step 2:** Commit.

### Task 1.5: Emit the JSON + render the website page

**Files:** Create `validation/data/test_dashboard.json` (generated); Create
`docs/website/50-validation/test-dashboard.md`; Modify `docs/website/myst.yml` (nav).

**Step 1:** `build_test_dashboard.py --emit` writes the timestamped JSON:
```json
{"generated_utc": "<passed-in timestamp>", "modules": {...}, "registries": {...},
 "line_coverage": {...}, "gate": {"registries_full": false, "line_cov_floor": 90, ...}}
```
NOTE: `Date.now()`/`datetime.now()` are fine in a plain script (this is NOT a workflow); stamp the UTC
time at generation.

**Step 2:** `--render` reads the JSON and writes `test-dashboard.md` — a MyST `{list-table}` matrix
(per-module rows × {tests unit/int/val, line-cov %, grad-audit fill, validation PASS, @slow count,
slowest test}). Use @myst:myst-expert conventions (colon-fence directives, `{list-table}`). Add the page
to `myst.yml` `project.toc` under `50-validation`.

**Step 3:** `cd docs/website && make build` → 0 warnings, page renders. Commit (script + JSON + page + nav).

### Task 1.6: Staleness gate (the ratchet)

**Files:** Create `tests/validation/test_dashboard_fresh.py`.

**Step 1 (RED):** mirror `grad_audit/test_json_fresh.py` — regenerate the dashboard JSON in-process and
semantic-diff vs the committed `validation/data/test_dashboard.json` (exact on discrete fields, rtol on
floats, IGNORE the `generated_utc` timestamp field). Assert they match. Run → it should PASS (just
committed). Then mutate the committed JSON (e.g. bump a count) and confirm the test REDS — proving teeth.
Revert.

**Step 2:** Commit.

`=== CHECKPOINT 1 → Anna (the keystone: a generated, timestamped, gated dashboard exists) ===`

---

## Phase 2: API-coverage registry

### Task 2.1: The manifest (TDD)

**Files:** Create `tests/validation/api_coverage/__init__.py`, `tests/validation/api_coverage/manifest.py`.

Mirror `grad_audit/manifest.py`'s frozen-literal pattern. `manifest.py` holds:
```python
# Every progenax.__all__ symbol -> the test module(s) that exercise it, OR an EXEMPT reason.
# Hand-curated (NOT computed from coverage at runtime — a derived map can't catch a deletion).
SYMBOL_TESTS: dict[str, str] = {
    "PlummerProfile": "tests/unit/profiles/test_plummer.py + tests/validation/test_plummer_physics.py",
    "build_cluster": "tests/unit/builders/test_cluster_builders.py",
    # ... every __all__ symbol ...
}
EXEMPT: dict[str, str] = {  # legitimately not directly unit-tested (re-exports, pure containers)
    # "SOLAR_SYSTEM_PLANETS": "data constant, exercised via get_planet tests",
}
LINE_COV_FLOOR = 90.0   # ratchet-up-only
```

**Step 1 (RED):** write `tests/validation/api_coverage/test_api_coverage.py`:
```python
import progenax
from tests.validation.api_coverage.manifest import SYMBOL_TESTS, EXEMPT, LINE_COV_FLOOR

def test_every_public_symbol_is_mapped():
    public = set(progenax.__all__)
    mapped = set(SYMBOL_TESTS) | set(EXEMPT)
    assert public - mapped == set(), f"public symbols with NO test mapping: {sorted(public - mapped)}"
    assert mapped - public == set(), f"stale mapping for removed symbols: {sorted(mapped - public)}"
```
Run → FAIL until you populate `SYMBOL_TESTS` for all 114 symbols.

**Step 2 (GREEN):** populate `SYMBOL_TESTS` by, for each `__all__` symbol, finding the test that exercises
it (`grep -rl "<Symbol>" tests/`). Anything genuinely untested → either it IS untested (a real hole to
fill in Task 2.3) or EXEMPT with a reason (Anna approves each EXEMPT). Run → PASS.

**Step 3:** add the line-coverage-floor test:
```python
def test_line_coverage_above_floor():
    cov = load_coverage("validation/data/coverage.json")  # committed by the dashboard run
    assert cov["total_percent"] >= LINE_COV_FLOOR, f"line cov {cov['total_percent']:.1f} < {LINE_COV_FLOOR}"
```
Run a FULL `--cov` pass to generate the committed `coverage.json`; if below 90%, list the under-covered
modules (those become Task 2.3 / Phase 3 holes). Commit.

### Task 2.2: Wire API-coverage into the dashboard

**Files:** Modify `scripts/build_test_dashboard.py` (replace the Phase-1 placeholder), regenerate JSON +
page, re-run the staleness gate.

### Task 2.3: Fill the surfaced API holes

For each `__all__` symbol with no mapping (and not EXEMPT), write a minimal high-value unit test (TDD).
Per-symbol commit. STOP and report to Anna the list of holes before filling (some may indicate dead
exports to remove instead — Anna decides per-item).

`=== CHECKPOINT 2 → Anna (every public symbol mapped; line-cov floor measured + enforced) ===`

---

## Phase 3: Suite refactor (profiling-driven)

### Task 3.1: `@slow`-mark the runtime sinks + consolidate the JSON regenerators

**Files:** Modify `tests/validation/grad_audit/test_json_fresh.py`,
`tests/validation/grad_audit/test_audit_script.py`, and the other sink files (design-doc table).

**Step 1:** Add `@pytest.mark.slow` to `test_json_fresh`, `test_audit_script`, the
`test_find_alpha_ift` grad/quadrature tests not already marked, `test_engine_b_physics::
test_plummer_halo_eff_core_equilibrium`, and the slow `test_limepy_multimass` tests. Confirm each is
genuinely FULL-only (they are artifact-freshness / heavy-grad checks, not fast behavioral checks).

**Step 2 (consolidate):** `test_json_fresh` and `test_audit_script` BOTH regenerate the grad-audit JSON
(~835s combined). Investigate whether one can reuse the other's regeneration (a shared session-scoped
fixture that regenerates ONCE, both assert on it). If so, refactor to a single regeneration. Do NOT
weaken either assertion — only remove the duplicate work.

**Step 3:** Re-profile the FAST gate: `pytest tests/unit tests/integration tests/validation -m "not slow"
-q -n auto --durations=10`. Confirm it dropped to minutes. Commit.

### Tasks 3.2–3.5: Consolidate cross-tier redundancy (Plummer / EFF / Michie / LIMEPY)

For EACH profile, repeat the King-consolidation pattern (already done for King in `9bb1f79`):
1. Inventory the unit-tier physics tests (`tests/unit/profiles/test_<p>.py`,
   `tests/unit/kinematics/test_<p>_df.py`) vs the validation-tier (`tests/validation/test_<p>_physics.py`).
2. Identify unit tests that DUPLICATE a validation test (same physics: density, truncation, isotropy,
   concentration, equilibrium Q).
3. Remove the redundant unit duplicates; KEEP unit-UNIQUE guards (differentiability, boundary/edge, jit,
   audit-specific). Update the module docstring to point physics coverage at the validation tier.
4. Verify: the profile's unit + validation files still pass; 0 warnings; the API-coverage manifest still
   maps every symbol.
5. STOP and report the per-profile removal list to Anna BEFORE deleting (per-item approval — removing
   tests removes coverage). Per-profile commit.

### Task 3.6: Re-profile + update the dashboard

Regenerate the dashboard JSON + page (the `@slow` split + counts changed); re-run the staleness gate.

`=== CHECKPOINT 3 → Anna (FAST loop in minutes; redundancy consolidated; coverage preserved) ===`

---

## Phase 4: Physics-validation registry

### Task 4.1: The manifest

**Files:** Create `tests/validation/physics_registry/manifest.py` + `test_physics_coverage.py`.

Mirror the grad-audit pattern. `MODEL_INVARIANTS` enumerates, for every public *model* (the profiles,
DFs, IMFs, builders, cluster engines in `__all__`), its required physics invariants + the test that
checks each:
```python
MODEL_INVARIANTS: dict[str, list[str]] = {
    "PlummerProfile": ["density closed-form", "inverse-CDF sampling", "scale-radius formula"],
    "PlummerVelocityDF": ["virial Q≈0.5 unscaled", "isotropy", "Beta(3/2,9/2) speed dist"],
    "build_cluster": ["Q≈0.5 all profiles", "bit-identical to build_spatial_ic", "modifier physics"],
    # ...
}
EXEMPT_NON_MODEL = {...}  # non-model symbols (utilities, containers) — no physics invariant required
```
Ratchet: every `__all__` symbol that is a *model* (per a curated `IS_MODEL` set) must have a
`MODEL_INVARIANTS` entry; a new model with no entry reds CI.

**Step 1 (RED):** `test_every_model_has_invariants()` cross-checks `IS_MODEL` ⊆ `MODEL_INVARIANTS`.
**Step 2 (GREEN):** populate from the existing validation tests (each `test_<p>_physics.py` already
encodes these — map them). Commit.

### Task 4.2: Wire into the dashboard + close the holes

Replace the Phase-1 placeholder; any model missing an invariant → write the validation test (TDD) or
EXEMPT with Anna's sign-off. Regenerate + staleness gate.

`=== CHECKPOINT 4 → Anna ===`

---

## Phase 5: Provenance-of-constants registry

### Task 5.1: Scan + manifest

**Files:** Create `tests/validation/provenance_registry/manifest.py` + `test_provenance_coverage.py`.

Per @research-workflow:provenance-of-constants. A scanner enumerates numeric literals / fit coefficients
in `src/progenax/` (grep for float literals, `* M`, exponents, hardcoded constants), and the manifest
maps each to a citation (the source comment / paper / CODATA). The ratchet: every flagged constant must
appear in the manifest with a provenance string; a new unprovenanced literal reds CI.

**Step 1 (RED):** `test_every_flagged_constant_has_provenance()` — the scanner's flagged set ⊆ the
manifest keys.
**Step 2 (GREEN):** the 2026-06 provenance audit (`docs/provenance-ledger.md`) already verified most
constants — port those citations into the manifest. New/unflagged → add. Anna adjudicates any genuinely
unsourced number (the audit found ZERO fabricated values, so expect mostly citation-porting). Commit.

### Task 5.2: Wire into the dashboard

Replace the placeholder; regenerate + staleness gate. Now all 4 registries report fill status.

`=== CHECKPOINT 5 → Anna (all 4 registries live; the dashboard gate is complete) ===`

---

## Phase 6: Docs + close-out

### Task 6.1: Testing-architecture page

**Files:** Create `docs/website/50-validation/testing-architecture.md` (@myst:myst-expert). Content: the
3-tier architecture (unit/integration/validation), the **tests-vs-registries** concept, the 4 registries
+ what each ratchets, how the dashboard is generated + the release gate, and how to add a new
model/symbol (you must register it in each registry). Wire into `myst.yml`.

### Task 6.2: Retire the stale dashboard + redirect

Make `50-validation/index.md` a thin pointer to the generated `test-dashboard.md` (remove the
hand-maintained table that drifts). Update `tests/README.md` to reference the registries.

### Task 6.3: Final gate + close-out

**Step 1:** FULL gate: `pytest tests/unit tests/integration tests/validation -q -n auto` → all green.
**Step 2:** FAST gate: `... -m "not slow"` → confirm minutes.
**Step 3:** `make build` → 0 warnings; the dashboard + testing-architecture pages render.
**Step 4:** Regenerate the dashboard JSON; staleness gate green; confirm the release gate
(`gate.registries_full == true`, `line_cov >= floor`).
**Step 5:** Completion doc `.claude-work/TASK_test_backbone_COMPLETE.md`; update `STATUS.md`; `brain "..."`.

`=== CHECKPOINT 6 → Anna: review, then merge feat/test-backbone → local main on her go (no push) ===`

---

## Verification matrix (Definition of Complete)

| Requirement | Phase | Artifact |
|---|---|---|
| Generated, timestamped source of truth | 1 | `validation/data/test_dashboard.json` + `test-dashboard.md` |
| Staleness gate (can't drift) | 1 | `tests/validation/test_dashboard_fresh.py` |
| 100% API coverage | 2 | `api_coverage/manifest.py` + ratchet |
| Line-cov floor (90%, ratchet-up) | 2 | `test_line_coverage_above_floor` |
| FAST loop in minutes | 3 | re-profiled `--durations` |
| Redundancy consolidated | 3 | Plummer/EFF/Michie/LIMEPY (King already done) |
| Physics-validation registry | 4 | `physics_registry/manifest.py` + ratchet |
| Provenance registry | 5 | `provenance_registry/manifest.py` + ratchet |
| Docs (architecture + dashboard) | 6 | `testing-architecture.md` + the matrix |

## Risks / watch-items
- **pytest-cov + xdist + JAX**: coverage under `-n auto` needs `pytest-cov`'s xdist support; run the
  authoritative `--cov` pass WITHOUT `-n auto` if attribution is flaky (slower but correct), or use
  `coverage combine`.
- **Line coverage of JAX-traced code**: branches inside `jax.lax.cond`/`vmap` may not register as
  "covered" by line tracing. Document such exclusions with `# pragma: no cover` + a reason; do NOT chase
  them with artificial tests (the design rejects strict 100% line for exactly this).
- **Registry manifests are hand-curated frozen literals** — that is intentional (a derived manifest
  can't catch a deletion). Do not "DRY" them into a computed structure.
- **Removing tests removes coverage** — every consolidation/removal in Phases 2.3 + 3.2-3.5 needs Anna's
  per-item approval before it lands.
- **The staleness gate's own runtime**: the dashboard regeneration must be cheap (introspection +
  parsing committed artifacts), NOT a full re-run — else it becomes the next `test_json_fresh` sink.
  Mark it `@slow` if it must run anything heavy.

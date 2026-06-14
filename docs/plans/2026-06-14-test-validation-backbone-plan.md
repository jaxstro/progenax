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
- `pytest-cov` IS already installed (7.1.0) and IS in `[dev]` (`["pytest>=7.0","pytest-cov","pytest-xdist>=3.0"]`).
  A `[tool.coverage]` config ALREADY EXISTS (`[tool.coverage.run] source=["progenax"] branch=true`;
  `[tool.coverage.report] show_missing=true`). Task 1.1 **EXTENDS** it — it does NOT add a second block, and does
  NOT switch `source` to `src/progenax` (the package installs top-level as `progenax`, and a smoke `--cov` run
  confirms per-file keys are `src/progenax/<module>`).
- Many of the design-doc runtime sinks are ALREADY `@pytest.mark.slow` (`test_find_alpha_ift`,
  `test_limepy_multimass`, `test_engine_b_physics`, …); the only DOMINANT unmarked sinks are the two grad-audit
  JSON regenerators (`test_json_fresh` 429s + `test_audit_script` 406s) — Phase 3.1's real new work.
- The committed line-coverage artifact is **`validation/data/coverage.json`**, produced ONLY by a FULL-suite
  `--cov` run (NOT `-m "not slow"`, which understates coverage and would spuriously fail the floor), written by
  the dashboard `--emit`, stamped with a `coverage_provenance` field (`{selector, git_sha}`), and read by the
  Phase-2 floor gate. The staleness gate checks `coverage_provenance.git_sha == HEAD`; it does NOT re-run `--cov`.

**Env prefix (uv, NOT conda):**
```
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest <args>
```

---

## Phase 0: Branch

### Task 0.1: Create the branch
```bash
git switch -c feat/test-backbone   # off main (552b87c: build_cluster + website follow-ups + planning docs + this review)
```
`=== CHECKPOINT 0 → Anna ===`

---

## Phase 1: Coverage + dashboard scaffold (the keystone)

### Task 1.1: Wire pytest-cov

**Files:** Modify `pyproject.toml`.

**Step 1:** Smoke-assert the tooling is present (it IS): `env -u VIRTUAL_ENV uv run --no-sync python -c
"import pytest_cov"`. pytest-cov 7.1.0 + pytest-xdist are already installed and in `[dev]` — there is NO
install/add-to-`[dev]` step.

**Step 2:** **EXTEND the EXISTING `[tool.coverage]` block** (do NOT add a second one; do NOT switch `source`).
Keep `source = ["progenax"]` + `branch = true` + `show_missing = true`; only APPEND the new keys:
```toml
[tool.coverage.run]
source = ["progenax"]          # already present — KEEP; resolves to src/progenax/*.py
branch = true                  # already present
omit = ["*/experimental/*", "*/__pycache__/*"]   # ADD

[tool.coverage.report]
show_missing = true            # already present
exclude_lines = [              # ADD
    "pragma: no cover",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
]
```

**Step 3:** Run a smoke coverage pass to confirm the per-file key format BEFORE writing the parser:
```
XLA_FLAGS=... env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/builders -q \
  --cov=progenax --cov-report=json:/tmp/cov_smoke.json
```
Expected: `totals.percent_covered` present; per-file keys are `src/progenax/<module>` (verified: 78 files).
`/tmp` is fine for THIS smoke only; the COMMITTED artifact (`validation/data/coverage.json`) is written by the
FULL-suite run in Task 1.5 / Task 2.1 — never `/tmp`.

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
so subprocess + parsing is acceptable — NOT JAX-native-constrained). **`build_test_dashboard.py` MUST insert
the repo root into `sys.path` at top-of-file (mirror `scripts/audit_gradients.py`) so `import tests.*` /
`import scripts.*` resolve under direct `python scripts/build_test_dashboard.py` invocation** —
`pythonpath=["src","src/experimental"]` does NOT include the repo root, so the standalone entrypoint
ImportErrors otherwise. Run → PASS. Commit.

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
  the committed `validation/data/grad_audit_results.json` hazard count). Safe because `manifest.py` is a pure
  frozen-literal module with NO pytest-collection side effects — keep every registry manifest side-effect-free
  so the generator can import it. The OTHER 3 registries don't exist yet → return `{"status": "not-built"}`
  placeholders (filled in Phases 2/4/5).
- **durations**: parse a committed `--durations` artifact (or run a `-m "not slow"` quick pass and
  capture the slowest per module). Keep it cheap.
- **validation scripts**: map each `scripts/validate_*.py` → last exit code (store in the JSON; the
  generator does NOT run them — too slow; a separate `--run-validations` flag does).

**Step 2:** Commit.

### Task 1.5: Emit the JSON + render the website page

**Files:** Create `validation/data/test_dashboard.json` (generated); Create `validation/data/coverage.json`
(generated, FULL-suite, with `coverage_provenance`); Create `docs/website/50-validation/test-dashboard.md`;
Modify `docs/website/myst.yml` (nav).

**Step 1:** `build_test_dashboard.py --emit` writes the timestamped JSON:
```json
{"generated_utc": "<passed-in timestamp>", "modules": {...}, "registries": {...},
 "line_coverage": {...}, "gate": {"registries_full": false, "line_cov_floor": 90, ...}}
```
NOTE: `Date.now()`/`datetime.now()` are fine in a plain script (this is NOT a workflow); stamp the UTC
time at generation. Also: `--emit` reads/writes the COMMITTED `validation/data/coverage.json` (the SAME file
the Phase-2 floor gate reads — ONE copy, not two) and records `coverage_provenance: {selector, git_sha}`; the
dashboard's `line_coverage` block is derived from that committed file. The full-suite `--cov` run that produces
`coverage.json` is a documented manual step (slow — gate it), NOT run by the staleness gate.

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
Revert. The gate is **introspection-only** (re-collect node ids + parse committed artifacts); it does NOT
re-run `--cov` or the suite. It also asserts `coverage_provenance.git_sha == HEAD` to catch a stale
`coverage.json` without regenerating it. If any sub-step must run something heavy, mark the test `@slow`.

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

**Step 2 (GREEN):** populate `SYMBOL_TESTS` by, for each `__all__` symbol, naming a test **function** that
CONSTRUCTS/CALLS the symbol and ASSERTS on its output — **NOT** a mere `grep -rl "<Symbol>" tests/` hit (a
substring match is coverage *theater*: e.g. `UniformSphereProfile` appears only as a Q-baseline fixture, and
`BinaryState` matches zero test files). Use grep to LOCATE candidates, then verify each names an asserting
test. Anything genuinely untested → a real hole (Task 2.3) or EXEMPT with a reason (Anna approves each EXEMPT;
mirror the grad-audit `SYMBOL_CATEGORY` EXEMPT taxonomy, and cross-check the two EXEMPT sets agree so the two
`__all__` partitions cannot drift). Run → PASS.

**Step 3:** add the line-coverage-floor test — reading the COMMITTED full-suite artifact and refusing to pass
on a partial one (a `-m "not slow"` pass understates coverage and would spuriously fail the floor):
```python
def test_line_coverage_above_floor():
    cov = load_coverage("validation/data/coverage.json")  # committed by Task 1.5 --emit (FULL suite)
    assert cov["coverage_provenance"]["selector"] == FULL_SELECTOR, (
        "coverage.json is not from the FULL suite — regenerate with the full selector")
    assert cov["total_percent"] >= LINE_COV_FLOOR, f"line cov {cov['total_percent']:.1f} < {LINE_COV_FLOOR}"
```
Generate the committed `coverage.json` with the FULL suite (`tests/unit tests/integration tests/validation`,
NO `-m "not slow"`); if below 90%, list the under-covered modules (those become Task 2.3 / Phase 3 holes). Commit.

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

### Tasks 3.2–3.5: Consolidate cross-tier redundancy (Plummer / EFF / Michie; LIMEPY handled separately, see below)

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

**LIMEPY is NOT symmetric — handle separately.** There is no `tests/validation/test_limepy_physics.py`;
LIMEPY's validation tier is `test_limepy_reference_parity.py` (a reference-parity oracle) +
`test_multimass_equilibrium_physics.py`, which do NOT duplicate the four unit limepy files. The King
consolidation does NOT apply. Demote LIMEPY (Task 3.5) to a separate "inspect, likely no-op" task: at most
de-duplicate AMONG the unit limepy files, with its own acceptance check. Verified per-profile scope (from the
review): **Plummer = partial** (keep virial-Q / q²-variance unit-unique guards), **EFF = clean** (remove ~6,
keep ~5 incl. spatial isotropy + the σ∝√M / σ∝1/√a virial channels + JIT guard), **Michie = partial** (keep
the table-routing / quadrature-oracle guards), **LIMEPY = no-op-likely**.

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
Ratchet: every `__all__` symbol that is a *model* must have a `MODEL_INVARIANTS` entry; a new model with no
entry reds CI. Define `IS_MODEL` **operationally**, not by hand-list intuition (the grad-audit shows hand-lists
go inconsistent — `build_plummer_cluster` is `EXEMPT_HELPER` while `build_king_cluster` is `AUDITED`): a symbol
is a model iff it implements `SpatialProfile`/`VelocityDF`/`IMFProtocol` (the runtime-checkable protocols) OR is
a `build_*_cluster` entry point. For models whose physics is reference-parity rather than equilibrium-Q (LIMEPY,
UniformSphere), EITHER accept "reference parity" / "uniform-density recovery" as a valid invariant class in
`MODEL_INVARIANTS`, OR carve a documented `EXEMPT_NON_EQUILIBRIUM_MODEL` category (mirror the grad-audit EXEMPT
taxonomy) so the exclusion is auditable, not silent. Cross-check `IS_MODEL`/EXEMPT against the grad-audit
`SYMBOL_CATEGORY` so the two `__all__` partitions cannot drift apart.

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

Per @research-workflow:provenance-of-constants. Do **NOT** build a regex float-literal scanner over all of
`src/progenax/` — `grep -rEo '[0-9]+\.[0-9]+'` returns ~2,525 matches dominated by `1.0/0.0/0.5`, array fills,
and exponents a regex cannot distinguish from citable coefficients (a massive false-positive triage trap).
Instead, make the manifest a HAND-CURATED port of the constants `docs/provenance-ledger.md` already verified
(the 2026-06 audit found ZERO fabricated values), each mapped to a citation (source comment / paper / CODATA).
The ratchet is a `# provenance:`-comment-presence check over a hand-curated allowlist of constant-bearing
files/lines (e.g. `limepy_tables.py`, `mapping.py`, `moe_di_stefano.py`); a new unprovenanced literal in an
allowlisted location reds CI. A genuinely new unsourced number elsewhere is added to the allowlist with Anna's
sign-off.

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
- **pytest-cov + xdist + JAX — VERIFIED BENIGN (2026-06-14):** a serial-vs-`-n auto` smoke on `tests/unit`
  subsets gave bit-identical line attribution (pytest-cov combines xdist workers natively), and `branch=true`
  did NOT flag `jax.lax.cond`/`vmap` branches as partial (no false-RED to chase with pragmas). Keep the FULL
  `--cov` run as the authoritative source anyway; no `coverage combine` step is needed.
- **`validation/data/coverage.json` provenance (the floor's teeth):** ONE committed file, produced ONLY by the
  FULL suite (NOT `-m "not slow"`), stamped `coverage_provenance: {selector, git_sha}`. The floor gate xfails if
  `selector != FULL_SELECTOR`, and the staleness gate asserts `git_sha == HEAD` WITHOUT re-running `--cov`.
  Without this, the floor is ungated/gameable (hand-edit a high number → green forever).
- **Line coverage of JAX-traced code**: document any genuine exclusion with `# pragma: no cover` + a reason; do
  NOT chase coverage with artificial tests (the design rejects strict 100% line for exactly this).
- **Registry manifests are hand-curated frozen literals** — intentional (a derived manifest can't catch a
  deletion). Do not "DRY" them into a computed structure.
- **Cross-cutting tests are PROTECTED from the "kill orphans" rule:** units/G-threading
  (`test_units_through_pipeline.py`), protocol-conformance (`test_protocols.py`), and end-to-end energy tests
  trace to a named design requirement, NOT a per-symbol registry row — do NOT delete them as "orphans."
- **Removing tests removes coverage** — every consolidation/removal in Phases 2.3 + 3.2-3.5 needs Anna's
  per-item approval before it lands.
- **Re-commit cadence:** Phases 2/3/4/5 each change registry-fill / `@slow` / per-module counts, so each MUST
  regenerate + re-commit `test_dashboard.json` (+ `coverage.json` when coverage changes) and re-run the
  staleness gate as an EXPLICIT task step — else the committed dashboard is stale the moment the phase lands.
- **The staleness gate's own runtime**: the dashboard regeneration must be cheap (introspection + parsing
  committed artifacts), NOT a full re-run — else it becomes the next `test_json_fresh` sink. Mark `@slow` if heavy.

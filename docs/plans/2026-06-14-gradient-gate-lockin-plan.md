# Gradient-gate lock-in + completion — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement
> this plan task-by-task (fresh subagent per task, code-review between). Strict TDD: RED→GREEN,
> never weaken a test. Measured-first frozen gates (≥3 seeds, measured band) for every new physics
> case. JAX-native only (`jnp`, `lax.scan`/`fori_loop`, never `while_loop`, no numpy/scipy in
> `src`). Units STELLAR, `G` explicit. Branch `feat/gradient-gate-lockin` (already created).
> Commit per task. NO push/merge without Anna's explicit go. Talk to Anna at every CHECKPOINT.

**Goal:** Turn the 66-row differentiability gradient-gate from a one-time audit into a permanent,
self-policing, *complete* CI invariant — a curated manifest + `__all__` cross-check + staleness
guard that fail CI on a reintroduced silent-zero, a registry↔JSON drift, or a new/removed entry
point without a case — and close the deferred D4 Fisher-coverage holes.

**Architecture:** A new `tests/validation/grad_audit/manifest.py` holds three frozen structures
(`MUST_AUDIT` keyed `(id,param)`; complete `SYMBOL_CATEGORY` over `progenax.__all__`;
`PARAM_ALLOWLIST` of carry-forward known-limitations). Three enforcement layers run as pytest
tests (run-gate, staleness comparator, coverage ratchet) in a dedicated fast `gradient-gate` CI
job. Phase A stands the machinery up green against current coverage; Phase B closes each D4 hole
per-target RED→GREEN; Phase C regenerates artifacts and ships ONE PR.

**Tech Stack:** JAX (`jax.grad`, float64), pytest (+ `pytest.param`/parametrize), Equinox,
GitHub Actions, mystmd (website). The grad-audit engine in `tests/validation/grad_audit/core.py`
(`audit_entry_point`, `Case`, `EdgeConfig`) is unchanged; we add around it.

**Reference docs:** design `docs/plans/2026-06-14-gradient-gate-lockin-design.md`; charter
`docs/plans/2026-06-14-gradient-gate-lockin-charter.md`; inventory
`.claude-work/grad-test-inventory.md` (the D4 work-list + KEEP list).

---

## Grounding facts (verify before relying)

- The gate engine: `tests/validation/grad_audit/core.py` — `Case(id, direction, fn, param,
  theta0, reduce, expect, tol, h_rel, eps, edges, hazard_id)`; `audit_entry_point(case)` returns
  `AuditResult` with computed `status ∈ {clean, known-limitation, hazard}`. Reverse-mode `jax.grad`
  only.
- The registry: `tests/validation/grad_audit/registry.py` — `REGISTRY: list[Case]`. 66 JSON rows =
  ~56 `(id,param)` Cases + ~10 edge rows. Reductions in `reductions.py`: `mean_radius`,
  `mean_speed`, `mean_mass`, `identity_sum`.
- The run-gate: `tests/validation/test_grad_audit.py` (parametrized over `REGISTRY` + edges +
  the `test_binned_sigma_mutation_has_teeth` teeth test).
- The script: `scripts/audit_gradients.py` — `run_audit(out_json)` writes
  `validation/data/grad_audit_results.json`; `__main__` exits 1 iff any hazard.
- CI: `.github/workflows/tests.yml` — jobs `lock-check`, `released-core` (matrix incl.
  `validation`), `experimental`, `wheel-smoke`, aggregated by `tests` (`needs:` list). The
  `validation` shard already runs `tests/validation` (so the run-gate already executes in CI).
- `progenax.__all__`: `src/progenax/__init__.py:115-228` (~95 symbols).
- Website: `docs/website/50-validation/differentiability-audit.md` *cites* the JSON + carries prose
  counts. `make build` in `docs/website/` = `myst build --html`.

**Run commands (from repo root):**
```bash
# Fast grad-gate subset (the inner loop for this arc):
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest tests/validation/grad_audit tests/validation/test_grad_audit.py -q
# Regenerate the JSON + check the script exits 0:
env -u VIRTUAL_ENV uv run --no-sync python scripts/audit_gradients.py ; echo "exit=$?"
# FULL released-core gate (phase/commit gate):
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -n auto
```

---

# PHASE A — Green machinery baseline (mergeable checkpoint)

## Task A1: Seed `manifest.py` (the three structures) + a generator helper

**Files:**
- Create: `tests/validation/grad_audit/manifest.py`
- Create: `tests/validation/grad_audit/_gen_manifest_seed.py` (one-shot dev helper, committed)
- Reference: `tests/validation/grad_audit/registry.py`, `src/progenax/__init__.py:115-228`

**Step 1 — Generate the `(id,param)` seed from the registry (dev helper).**
Write `_gen_manifest_seed.py` that imports `REGISTRY` and prints the sorted unique `(id, param)`
pairs and, separately, `sorted(progenax.__all__)`. This is a *seeding aid only* — its output is
transcribed into the frozen literals, never imported at runtime by the ratchet (else the ratchet
is vacuous).

```python
"""One-shot seed generator (committed for reproducibility, NOT imported by the gate).
Prints the (id, param) coverage units in REGISTRY and the full __all__ list, so the
manifest literals can be seeded without transcription error. The manifest is then an
INDEPENDENT frozen literal — deleting a registry case must trip the coverage ratchet."""
import progenax  # noqa: F401 (float64 + __all__)
from tests.validation.grad_audit.registry import REGISTRY

def main():
    pairs = sorted({(c.id, c.param) for c in REGISTRY})
    print(f"# {len(pairs)} (id, param) coverage units:")
    for cid, p in pairs:
        print(f'    ("{cid}", "{p}"),')
    print(f"\n# {len(progenax.__all__)} __all__ symbols:")
    for s in sorted(progenax.__all__):
        print(f'    "{s}": ...,')

if __name__ == "__main__":
    main()
```

Run: `env -u VIRTUAL_ENV uv run --no-sync python -m tests.validation.grad_audit._gen_manifest_seed`
Expected: ~56 `(id,param)` lines + ~95 symbol lines. Capture this output.

**Step 2 — Write `manifest.py` with the three frozen literals.**
Transcribe the Step-1 output into `MUST_AUDIT` (every current `(id,param)`, each with a 1-line
rationale) and `SYMBOL_CATEGORY` (categorize every `__all__` symbol). Categories enum (str
constants): `AUDITED`, `EXEMPT_PROTOCOL`, `EXEMPT_CONTAINER`, `EXEMPT_ANALYTICAL_IC`,
`EXEMPT_NON_FISHER_DIAGNOSTIC`, `EXEMPT_HELPER`, `EXEMPT_COVERED_ELSEWHERE`. `PARAM_ALLOWLIST`
holds the charter carry-forwards.

```python
"""Gradient-coverage source of truth (design Q1/Q2/Q6).

THREE frozen structures, all hand-curated independent literals (NOT computed from REGISTRY
or __all__ at runtime — a derived manifest could not catch a deleted case / a new symbol):

  MUST_AUDIT       : the (id, param) coverage units the registry MUST cover (ratchet target).
  SYMBOL_CATEGORY  : every progenax.__all__ symbol -> AUDITED | EXEMPT_* (cross-check target).
  PARAM_ALLOWLIST  : registry (id, param) that are legitimately NOT FD-consistent (carry-forward).

Seeded by tests/validation/grad_audit/_gen_manifest_seed.py, then frozen. Enforced by
tests/validation/grad_audit/test_manifest_coverage.py. The website coverage section is
generated from here.
"""

# --- Categories -------------------------------------------------------------
AUDITED = "AUDITED"                                  # has >=1 registry case (in MUST_AUDIT)
EXEMPT_PROTOCOL = "EXEMPT_PROTOCOL"                  # a typing Protocol, not an entry point
EXEMPT_CONTAINER = "EXEMPT_CONTAINER"               # dataclass/PyTree container, not a sampler
EXEMPT_ANALYTICAL_IC = "EXEMPT_ANALYTICAL_IC"       # fixed-config analytic IC/test fixture
EXEMPT_NON_FISHER_DIAGNOSTIC = "EXEMPT_NON_FISHER_DIAGNOSTIC"  # diagnostic/energy kernel, own tests
EXEMPT_HELPER = "EXEMPT_HELPER"                      # helper/constant/orientation util
EXEMPT_COVERED_ELSEWHERE = "EXEMPT_COVERED_ELSEWHERE"  # differentiable sampler with a scattered
                                                       # FD test; deferred from THIS arc's registry
                                                       # scope (Tier-4 inventory "future candidate")

# --- MUST_AUDIT: (id, param) -> rationale -----------------------------------
# Seeded from REGISTRY (Phase A = current coverage only; D4 entries appended in Phase B).
MUST_AUDIT: dict[tuple[str, str], str] = {
    ("PlummerProfile.sample_positions", "r_h"): "headline spatial sampler",
    # ... (transcribe ALL ~56 from the generator; one rationale line each) ...
}

# --- SYMBOL_CATEGORY: every __all__ symbol -> category ----------------------
SYMBOL_CATEGORY: dict[str, str] = {
    "PlummerProfile": AUDITED,
    "SpatialProfile": EXEMPT_PROTOCOL,
    "ICResult": EXEMPT_CONTAINER,
    "two_body_kepler": EXEMPT_ANALYTICAL_IC,
    "compute_potential_energy": EXEMPT_NON_FISHER_DIAGNOSTIC,
    "jacobi_radius": EXEMPT_HELPER,
    "PowerLawMassRatio": EXEMPT_COVERED_ELSEWHERE,
    # ... (transcribe ALL ~95; every symbol categorized) ...
}

# --- PARAM_ALLOWLIST: registry (id, param) legitimately not FD-consistent ----
PARAM_ALLOWLIST: dict[tuple[str, str], str] = {
    ("PowerLawIMF.ppf[Salpeter]", "alpha"): "alpha=1.0 edge is a branch-limited removable "
        "singularity (known_blocked); alpha=0.999 is FD-consistent",
    ("PowerLawIMF.mean_mass", "alpha"): "alpha=1.0 Z-denominator branch (known_blocked)",
    ("IMFParams.log_prob_nll", "alpha3"): "alpha3=1.0 branch-limited (known_blocked)",
    ("binned_number_density[data, pinned non-diff]", "r_h"): "frozen-edge count is a sum of "
        "indicators; AD=0 correct-by-design, the N(r) Fisher gradient lives in the model p_k",
    # du-monotonicity: the inverse-CDF samplers' grad wrt the frozen uniform draw u is a data-side
    # property out of the param-channel scope; pinned in TestBoundaryGradients, not a registry case.
}
```

**Step 3 — Commit** (no test yet; the literals are inert data).
```bash
git add tests/validation/grad_audit/manifest.py tests/validation/grad_audit/_gen_manifest_seed.py
git commit -m "feat(grad-gate): seed manifest.py (MUST_AUDIT / SYMBOL_CATEGORY / PARAM_ALLOWLIST)"
```

**>>> CHECKPOINT A1 (Anna):** review `SYMBOL_CATEGORY` — specifically the **AUDITED-vs-
`EXEMPT_COVERED_ELSEWHERE` boundary**. Open scoping question: `LogisticThermalEccentricity`
(sibling of the in-scope Thermal/Uniform eccentricity dists, FD-tested) — AUDITED (add a Phase-B
case) or `EXEMPT_COVERED_ELSEWHERE` (defer)? Same question for any other sampler Anna wants pulled
into scope. Do NOT proceed to A2 until ratified.

---

## Task A2: Coverage ratchet test (Layer 3a) — `MUST_AUDIT ⊆ REGISTRY`

**Files:**
- Create: `tests/validation/grad_audit/test_manifest_coverage.py`
- Reference: `manifest.py`, `registry.py`

**Step 1 — Write the failing test.** First write it to assert a property we can make RED on
demand: every `MUST_AUDIT` key is covered by a registry `(id,param)`.

```python
"""Layer 3 coverage ratchet (design D2/D3): the registry must cover every MUST_AUDIT entry,
and the manifest must categorize every public symbol. A deleted case or a new ungated public
entry point -> RED."""
import progenax  # noqa: F401
import pytest
from tests.validation.grad_audit.manifest import (
    AUDITED, MUST_AUDIT, PARAM_ALLOWLIST, SYMBOL_CATEGORY,
)
from tests.validation.grad_audit.registry import REGISTRY

_REGISTRY_KEYS = {(c.id, c.param) for c in REGISTRY}

def test_every_must_audit_entry_is_covered():
    missing = sorted(k for k in MUST_AUDIT if k not in _REGISTRY_KEYS)
    assert not missing, (
        f"MUST_AUDIT entries with NO registry case (coverage ratchet RED): {missing}. "
        f"Add the Case to registry.py (or, if intentionally removing coverage, remove the "
        f"manifest entry WITH Anna's sign-off)."
    )
```

**Step 2 — Demonstrate RED.** Temporarily append a bogus key to `MUST_AUDIT` in a scratch edit
(or add `("__nonexistent__", "x"): "demo"`), run the test, confirm it FAILS naming the bogus key,
then revert.
Run: `... pytest tests/validation/grad_audit/test_manifest_coverage.py::test_every_must_audit_entry_is_covered -v`
Expected: PASS with the real manifest; FAIL (naming `__nonexistent__`) with the bogus key.

**Step 3 — Commit.**
```bash
git add tests/validation/grad_audit/test_manifest_coverage.py
git commit -m "test(grad-gate): Layer-3 coverage ratchet (MUST_AUDIT subset of REGISTRY)"
```

---

## Task A3: `__all__` cross-check test (Layer 3b) — complete category map

**Files:** Modify `tests/validation/grad_audit/test_manifest_coverage.py`

**Step 1 — Add the two cross-check assertions.**
```python
def test_symbol_category_covers_all_public_symbols_exactly():
    public = set(progenax.__all__)
    mapped = set(SYMBOL_CATEGORY)
    unmapped = sorted(public - mapped)   # NEW public symbol not categorized -> RED
    stale = sorted(mapped - public)      # category for a removed/renamed symbol -> RED
    assert not unmapped, (
        f"public symbols missing from SYMBOL_CATEGORY (categorize each as AUDITED or EXEMPT_*): "
        f"{unmapped}")
    assert not stale, f"SYMBOL_CATEGORY entries no longer in __all__: {stale}"

def test_every_audited_symbol_has_a_registry_case():
    audited = {s for s, cat in SYMBOL_CATEGORY.items() if cat == AUDITED}
    covered_ids = {cid for (cid, _p) in _REGISTRY_KEYS}
    # An AUDITED symbol must own at least one registry id (id may be "Class.method[...]").
    uncovered = sorted(s for s in audited
                       if not any(cid == s or cid.startswith(s + ".") or cid.startswith(s + "[")
                                  for cid in covered_ids))
    assert not uncovered, (
        f"AUDITED symbols with no registry case: {uncovered}. Either add a case or "
        f"re-categorize as EXEMPT_COVERED_ELSEWHERE with Anna's sign-off.")

def test_param_allowlist_entries_are_real_registry_cases():
    # Allowlist must reference cases that actually exist (no stale pins).
    stale = sorted(k for k in PARAM_ALLOWLIST if k not in _REGISTRY_KEYS)
    assert not stale, f"PARAM_ALLOWLIST entries not in REGISTRY: {stale}"
```

> Note: the `id.startswith(s + ".")`/`("[")` rule maps a symbol like `PlummerProfile` to ids
> `PlummerProfile.sample_positions`. Verify the helper matches every AUDITED symbol's id shape
> while writing (e.g. `apply_solid_body_rotation` id == symbol exactly; `MultiComponentCluster`
> id == `MultiComponentCluster.sample_cluster[EngineA]`). Adjust the matcher to whatever the real
> ids need — measured against the actual registry, not assumed.

**Step 2 — Run; expect PASS (machinery is green against current state).**
Run: `... pytest tests/validation/grad_audit/test_manifest_coverage.py -v`
Expected: all PASS. If `test_every_audited_symbol_has_a_registry_case` fails, the AUDITED set in
A1 is wrong (a symbol marked AUDITED has no case) — fix the category, re-checkpoint if it changes
scope.

**Step 3 — Commit.**
```bash
git add tests/validation/grad_audit/test_manifest_coverage.py
git commit -m "test(grad-gate): Layer-3 __all__ cross-check (complete category map + AUDITED has case)"
```

---

## Task A4: Staleness comparator test (Layer 2) + cross-arch rtol calibration

**Files:**
- Create: `tests/validation/grad_audit/test_json_fresh.py`
- Reference: `scripts/audit_gradients.py` (`run_audit`), `validation/data/grad_audit_results.json`

**Step 1 — Write the semantic comparator test.** Regenerate to a tmp path, compare to the
committed JSON: exact on discrete/structural fields, rtol on floats.
```python
"""Layer 2 staleness guard (design D2/Q3): the committed grad_audit_results.json must match a
fresh regeneration. EXACT on the discrete/structural projection (id/direction/param/theta/expect/
tol/finite/status); rtol-tolerant on the floats (ad/fd/ratio/abs_ad) because the committed JSON is
generated on macOS arm64 and CI regenerates on Ubuntu x86 — a literal byte-diff is infeasible."""
import json
from pathlib import Path
import pytest
from scripts.audit_gradients import run_audit, _DEFAULT_JSON  # run_audit(out_json) -> rows

_DISCRETE = ("id", "direction", "param", "expect", "status")
_FLOAT = ("ad", "fd", "ratio", "abs_ad")
_RTOL = 2e-3   # calibrated in Step 3 (measured cross-arch with margin); see header note

def _key(row):
    return (row["id"], row["param"], round(float(row["theta"]), 12))

def test_committed_json_matches_fresh_regeneration(tmp_path):
    committed = json.loads(Path(_DEFAULT_JSON).read_text())
    fresh = run_audit(out_json=tmp_path / "fresh.json")
    cset, fset = {_key(r) for r in committed}, {_key(r) for r in fresh}
    assert cset == fset, (
        f"row-set drift (cases added/removed/retheta'd):\n  only committed: {sorted(cset - fset)}"
        f"\n  only fresh: {sorted(fset - cset)}\n  -> regenerate + recommit the JSON.")
    cby, fby = {_key(r): r for r in committed}, {_key(r): r for r in fresh}
    drift = []
    for k in cby:
        c, f = cby[k], fby[k]
        for field in _DISCRETE:
            if c[field] != f[field]:
                drift.append(f"{k} {field}: committed={c[field]!r} fresh={f[field]!r}")
        if bool(c["finite"]) != bool(f["finite"]):
            drift.append(f"{k} finite: committed={c['finite']} fresh={f['finite']}")
        for field in _FLOAT:
            cv, fv = c[field], f[field]
            if cv in (None,) or fv in (None,):
                continue
            denom = max(abs(cv), abs(fv), 1e-30)
            if abs(cv - fv) / denom > _RTOL:
                drift.append(f"{k} {field}: committed={cv:.6e} fresh={fv:.6e} "
                             f"(reldiff={abs(cv - fv)/denom:.2e} > rtol={_RTOL:.0e})")
    assert not drift, "staleness drift (regenerate + recommit JSON if intended):\n  " + \
        "\n  ".join(drift)
```

**Step 2 — Run locally (same arch as the commit) — expect PASS at near-zero drift.**
Run: `... pytest tests/validation/grad_audit/test_json_fresh.py -v`
Expected: PASS (same-machine regeneration is ~bit-identical; floats reldiff ≪ rtol).

**Step 3 — Calibrate `_RTOL` cross-arch (measured-first).** The committed JSON is arm64; CI is
x86. Measure the real cross-arch float delta before trusting `_RTOL=2e-3`:
- Inspect the noisiest cases (ODE-solve: King/Michie `r_t`, Engine-A/B sample_cluster) by checking
  the CI run's actual `fresh` values vs committed in the first `gradient-gate` job run (Task A5).
- If any well-conditioned case exceeds `2e-3` reldiff purely from arch, widen `_RTOL` to the
  measured max ×3, and record the measured cross-arch deltas in the test header. Do NOT loosen
  blindly — a reldiff that large on a closed-form case is a real drift, not noise.
- Document the chosen `_RTOL` + the measured basis in the header comment.

**Step 4 — Commit.**
```bash
git add tests/validation/grad_audit/test_json_fresh.py
git commit -m "test(grad-gate): Layer-2 staleness comparator (semantic diff, cross-arch rtol)"
```

---

## Task A5: Dedicated `gradient-gate` CI job (Layer 1 formalized + wiring)

**Files:** Modify `.github/workflows/tests.yml`

**Step 1 — Add the job** (after `wheel-smoke`, before the `tests` aggregator), mirroring the
existing checkout/uv-sync pattern:
```yaml
  # Gradient-integrity gate (design D2): the grad-audit run-gate + staleness + coverage ratchet,
  # plus the audit-script exit-0 (0 hazards). A reintroduced silent-zero, a registry<->JSON drift,
  # or a new/removed entry point without a case reds this job. Fast (the grad-audit subset only),
  # runtime-isolated from the slow validation shard, and a clearly-named branch-protection signal.
  gradient-gate:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - name: Checkout progenax
        uses: actions/checkout@v4
        with:
          path: progenax
      - name: Checkout jaxstro (path dependency, sibling layout)
        uses: actions/checkout@v4
        with:
          repository: jaxstro/jaxstro
          path: jaxstro
      - name: Setup uv
        uses: astral-sh/setup-uv@v6
        with:
          python-version: "3.13"
      - name: Sync from lockfile (dev extra)
        working-directory: progenax
        run: uv sync --locked --extra dev
      - name: Run gate + staleness + coverage ratchet
        working-directory: progenax
        run: uv run --no-sync pytest tests/validation/grad_audit tests/validation/test_grad_audit.py -q
      - name: Audit script exits 0 (0 hazards)
        working-directory: progenax
        run: uv run --no-sync python scripts/audit_gradients.py
```

**Step 2 — Wire into the aggregator.** In the `tests` job, add `gradient-gate` to `needs:` and to
the success-check condition:
```yaml
  tests:
    if: always()
    needs: [lock-check, released-core, experimental, wheel-smoke, gradient-gate]
    # ... add: && [ "${{ needs.gradient-gate.result }}" = "success" ] to the if-chain + echo
```

**Step 3 — Validate the YAML + the exact pytest selector locally.**
Run: `... pytest tests/validation/grad_audit tests/validation/test_grad_audit.py -q`
Expected: PASS (the run-gate + the 4 new Layer-2/3 tests). Confirm the selector collects
`test_grad_audit.py`, `test_manifest_coverage.py`, `test_json_fresh.py`.
Run: `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/tests.yml')); print('yaml ok')"`

**Step 4 — Commit.**
```bash
git add .github/workflows/tests.yml
git commit -m "ci(grad-gate): dedicated gradient-gate job (run-gate + staleness + ratchet + exit-0)"
```

**>>> CHECKPOINT A5 (Anna):** Phase A machinery complete and green. Before Phase B: confirm the
full released-core gate is green locally, and review the first CI run of `gradient-gate` to read
the **actual cross-arch float deltas** (feeds the Task A4 `_RTOL` calibration). If A4's rtol needs
widening from the CI evidence, do it now as a follow-up commit. This is the mergeable Phase-A
boundary.

---

## Task A6: Website coverage section (generated from manifest)

**Files:**
- Modify: `docs/website/50-validation/differentiability-audit.md`
- Optional create: `scripts/gen_coverage_table.py` (emits a markdown coverage table from
  `manifest.py`, if Anna wants a table vs. prose)

**Step 1 — Add a "Gradient-coverage manifest" section** documenting `MUST_AUDIT` (the enforced
coverage), the `SYMBOL_CATEGORY` policy (every public symbol categorized; new symbol → CI RED),
and `PARAM_ALLOWLIST` (the carry-forward known-limitations). State that CI fails on a reintroduced
silent-zero, a registry↔JSON drift, or an uncovered/new entry point. Keep the counts in sync with
the JSON (still cited, not embedded).

**Step 2 — Build the site, 0 warnings.**
Run: `cd docs/website && make build`
Expected: `myst build --html` completes; **0 warnings**. Fix any broken xref/anchor.

**Step 3 — Commit.**
```bash
git add docs/website/50-validation/differentiability-audit.md scripts/gen_coverage_table.py
git commit -m "docs(grad-gate): document the gradient-coverage manifest on the validation website"
```

---

# PHASE B — Close the D4 holes (per-target RED→GREEN)

> **Per-target recipe (every B task follows this):**
> 1. Add the target's `(id, param)` to `MUST_AUDIT` in `manifest.py` (+ `SYMBOL_CATEGORY` →
>    AUDITED if a new symbol). **RED step.**
> 2. Run `pytest .../test_manifest_coverage.py::test_every_must_audit_entry_is_covered` →
>    confirm it FAILS naming the new key (the ratchet RED before the case lands — the mission's
>    TDD rule for the ratchet).
> 3. Write the registry closure (JAX-native, `G=STELLAR.G` explicit) + **measure** AD/FD over
>    **≥3 seeds**; record AD, FD, |ratio−1| per seed in the closure's docstring (measured-first).
>    Pick `tol` from the measured band (never weaken to hide a real gap; if AD≉FD investigate the
>    physics, don't loosen).
> 4. Append the `Case(...)` to `REGISTRY` with the measured `tol`.
> 5. Run the run-gate → the new case is `clean`. Run the ratchet → GREEN.
> 6. Regenerate JSON (`python scripts/audit_gradients.py`), run the staleness test → PASS.
> 7. `git add` registry + manifest + JSON; commit `feat(grad-gate): <target> registry case`.
> 8. **Code-review (fresh reviewer subagent)** before the next target.

## Task B1: Engine-A `w_j` — `MultiComponentCluster.from_components`

**Files:** Modify `registry.py`, `manifest.py`; reference
`src/progenax/cluster/multicomponent.py:325-348` (`from_components(alpha_j, w_j, m_j, W0, g,
r_c=1.0, ra_hat_j=None, ...)`), existing `tests/unit/cluster/test_multicomponent.py:93-108`
(`test_differentiable_in_w_j`, finite-only).

**Closure (template — measure to fill numbers):** vary `w_j` (shape-`n_comp` array; audit the
gradient wrt a scalar that scales the vector, or wrt one component — choose the scalar-multiplier
form to match the engine's scalar-`theta` contract). `sample_cluster(..., G=STELLAR.G)`; reduce
`mean_speed` (w_j sets velocity scales). Config from the existing test (`alpha_j=[0.6,0.4]`,
`m_j=[0.5,2.0]`, `W0=7`, `g=1`). Measure AD vs central FD at ≥3 seeds; the categorical-assignment
flip-count must be 0 at `±h` (document it, as the existing Engine-A cases do). Suggested
`id="MultiComponentCluster.from_components[EngineA]"`, `param="w_j"`.

## Task B2: Engine-A `r_a` — `MultiComponentCluster.from_mass_segregation`

**Files:** Modify `registry.py`, `manifest.py`; reference `multicomponent.py:350-370`
(`from_mass_segregation(alpha_j, m_j, W0, g, delta, r_a=None, eta=0.0, ...)`), existing
`test_multicomponent.py:545-562` (`test_aniso_sample_differentiable_in_ra`, finite-only, uses
`xi_max=800`). Vary `r_a` (scalar); reduce `mean_speed` (anisotropy lives in velocities; the
existing test reduces mean squared *radial* velocity — consider matching it or `mean_speed`,
measured). Confirm realizability (r_a above the over-anisotropy bound) at `±h`. Suggested
`id="MultiComponentCluster.from_mass_segregation[EngineA]"`, `param="r_a"`.

## Task B3: `build_binary_cluster` end-to-end

**Files:** Modify `registry.py`, `manifest.py` (add symbol `build_binary_cluster` → AUDITED);
reference `src/progenax/builders.py:398-506`, existing
`tests/integration/test_binary_cluster.py:112-129` (`test_grad_through_r_h`, FD-only, builds with
`IndependentCompanions(..., fbin=0.5)`, `target=Systems(100)`, `compact=False`, reduces mean
radius). Closure: vary `r_h` through both `PlummerProfile(r_h=r_h)` and `PlummerVelocityDF(r_h=
r_h)`; `units=STELLAR`; `compact=False` (fixed-shape `ResolvedBinaries`, grad-safe) or `compact=
True` (measure which is the cleaner FD signal); reduce `mean_radius` over `rb.positions`. The IMF
binning + companion sampling + spatial assembly is the full Fisher path. Measure ≥3 seeds; watch
for is_binary mask-flip discreteness (document flip-count at `±h`). Suggested
`id="build_binary_cluster"`, `param="r_h"`.

## Task B4: Binary period distributions (3 cases)

**Files:** Modify `registry.py`, `manifest.py` (`SanaOBPeriod`, `LogNormalPeriod`,
`LogUniformPeriod` → AUDITED). Reference `src/progenax/binaries/period.py`; scattered tests in
`tests/unit/binaries/test_population.py`:
- `SanaOBPeriod.power` — `id="SanaOBPeriod.sample"`, reduce `mean(log10(sample(key,N)))`; FD-tested
  at `test_sana_power_gradient_matches_finite_difference:516`. `expect="consistent"`.
- `LogNormalPeriod.mu_log_P` — closed-form `d⟨log10 P⟩/dμ = 1` (`test_lognormal_location_gradient_
  is_unity:543`); FD will match → `consistent`.
- `LogUniformPeriod.log_P_max` — closed-form grad `⟨u⟩≈0.5` (`test_loguniform_period_logpmax_
  gradient:679`); `consistent`.

Each: a closure `theta -> sample(...)` reduced to a scalar (the dist samples don't take `G`; no
units needed — these are pure period draws in days). Measure ≥3 seeds; the ppf is a smooth inverse
so central FD is clean. One Case per param; **one commit per param** (3 RED→GREEN cycles), or batch
the 3 into one commit if the reviewer agrees they're homogeneous — Anna's call at the B-kickoff.

## Task B5: Binary eccentricity distributions (3 cases)

**Files:** Modify `registry.py`, `manifest.py` (`ThermalEccentricity`, `UniformEccentricity`,
`MoeEccentricity` → AUDITED). Reference `src/progenax/binaries/eccentricity.py`; scattered tests:
- `ThermalEccentricity.e_max` — closed-form grad `⟨√u⟩≈2/3` (`test_thermal_scale_gradient_equals_
  mean_sqrt_u:554`). `id="ThermalEccentricity.sample"`, reduce `mean(sample(key,N))`.
- `UniformEccentricity.e_max` — closed-form grad `⟨u⟩≈0.5` (`test_uniform_eccentricity_emax_
  gradient:668`).
- `MoeEccentricity.e_max` — FD (`test_emax_gradient_matches_fd:445`); `sample(key, periods,
  masses)` — supply fixed `periods=[1e8]*N`, `masses=[20.0]*N` per the test; reduce `mean`.

`expect="consistent"` for all. ≥3 seeds. Per-param or batched (Anna's call).

## Task B6: `IndependentCompanions.e_max`

**Files:** Modify `registry.py`, `manifest.py` (`IndependentCompanions` → AUDITED). Reference
`src/progenax/binaries/companions.py:74-112`; scattered `tests/unit/binaries/test_companions.py:
114-125` (`test_grad_fd_accurate_eccentricity`, FD, `sample(key, m1=[2.0]*N, G=..., day_in_time_
units=...)`, reduces `mean(el.e)`). The `e_max` threads through
`eccentricity_distribution=ThermalEccentricity(e_max=...)`. Closure varies `e_max`, rebuilds the
`IndependentCompanions` with the new ecc dist, samples, reduces `mean(el.e)`. `G=STELLAR.G`,
`day_in_time_units=86400/STELLAR.time_scale_cgs` (per the registry's `_MOE_DAY`). ≥3 seeds.
Suggested `id="IndependentCompanions.sample"`, `param="e_max"`.

**>>> CHECKPOINT B (Anna):** at B-kickoff confirm (a) the `LogisticThermalEccentricity` scoping
from CHECKPOINT A1, (b) per-param vs batched commits for B4/B5, (c) the exact reductions where the
template offered a choice. After all B tasks: every D4 hole has a measured registry case; the
ratchet, staleness, and run-gate are all green.

---

# PHASE C — Regenerate, document, close out

## Task C1: Final artifact regeneration

**Step 1 — Regenerate JSON + plots.**
Run: `... python scripts/audit_gradients.py --plots`
Expected: exit 0; new JSON (~56+D4 cases) + `validation/plots/grad_audit_*.png` + website figure
copies.

**Step 2 — Update website counts + coverage section** to the new totals; rebuild.
Run: `cd docs/website && make build` → **0 warnings**.

**Step 3 — Full released-core gate, green.**
Run: `XLA_FLAGS="..." env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -n auto`
Expected: all PASS (the new cases + the 4 enforcement tests + the migrated/pointer scattered tests
untouched).

**Step 4 — Commit.**
```bash
git add validation/data/grad_audit_results.json validation/plots docs/website
git commit -m "docs(grad-gate): regenerate JSON + plots + website counts (gate complete)"
```

## Task C2: STATUS.md + close-out

**Step 1 — Update `STATUS.md`** (`next:`/`blocker:`/`due:`): gate is now enforced (3 CI layers) +
complete (D4 holes closed); the new case count; the manifest as source of truth. `brain "…"` a
one-line capture of the lock-in.

**Step 2 — Verify the full Definition-of-Done** (charter success criteria):
- [ ] CI fails on a reintroduced silent-zero (Layer 1), a registry↔JSON drift (Layer 2), a
      manifest/`__all__` entry without a case (Layer 3) — demonstrate each RED locally once.
- [ ] D4 holes closed (Engine-A `w_j`/`r_a`, `build_binary_cluster`, the binary dists).
- [ ] `manifest.py` + `PARAM_ALLOWLIST` committed, documented on the website.
- [ ] Full released-core gate green; `audit_gradients.py` exit 0; `make build` 0 warnings.

**Step 3 — Commit, then >>> CHECKPOINT C (Anna):** present the full green evidence. **Only on
Anna's explicit go**: open ONE PR `feat/gradient-gate-lockin → main`. Do not push/merge before.

---

## Risks / watch-items

- **Cross-arch rtol (A4):** the single calibration most likely to flake CI. Measure from the real
  CI run (A5 checkpoint), don't guess; widen only from evidence, keeping `status` exact.
- **`audit_entry_point` scalar-`theta` contract:** every closure is `theta(scalar) -> array`. For
  vector params (`w_j`, `r_a_j`) use a scalar multiplier/leaf, as the existing Engine-B `r_a` case
  does (`jnp.stack([r_a, inf])`).
- **Discreteness (categorical assignment / is_binary mask / bin edges):** every B case with a
  sampler must document the flip-count at `±h` (0 expected at the chosen baseline), per the
  existing Engine-A/B/Moe cases — measured, not assumed.
- **Subagent suspend (last-arc lesson):** if an implementing subagent suspends mid-task waiting on
  a background gate, verify + commit its uncommitted work yourself before dispatching the next.
- **Never weaken a test:** if a D4 case shows AD≉FD, that is a *finding* (investigate the physics
  / conditioning), not a tol to loosen.

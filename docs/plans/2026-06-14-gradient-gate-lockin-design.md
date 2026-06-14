# Design — Gradient-gate lock-in + completion

> **Status:** design (brainstorm complete, all six open questions resolved + signed off
> 2026-06-13). Descends from `docs/plans/2026-06-14-gradient-gate-lockin-charter.md`.
> Next: `writing-plans` → `subagent-driven-development` on `feat/gradient-gate-lockin`.

## Premise (from the charter)

The 2026-06 differentiability gradient-audit arc (PR #8) built a 66-row AD-vs-FD release
gradient-gate (`tests/validation/grad_audit/`) and fixed two real silent-zero/biased-gradient
hazards. But it **measured** gradient integrity once; nothing **enforces** it. A future commit
can reintroduce a silent-zero, drift the registry away from its committed numbers, or add a new
public sampler with no gradient case — and CI would not notice. Gradient integrity = Fisher
integrity: a silently zeroed/NaN gradient corrupts a Fisher matrix **with no error raised**. This
arc turns the gate into a permanent, self-policing, **complete** invariant.

## Resolved design decisions (the six open questions)

| # | Question | Decision |
|---|----------|----------|
| Q1 | Manifest format + location | **Python module** `tests/validation/grad_audit/manifest.py`, keyed `(id, param)`; ratchet is a pure set op vs `REGISTRY`; no parser/dependency; website table generated from it. |
| Q2 | `__all__` cross-check classifier | **Complete `__all__` → category map** (no inference). Map keys must equal `set(progenax.__all__)`; each symbol is `AUDITED` or an `EXEMPT_*` reason. Zero false positives by construction. |
| Q3 | Deterministic JSON regeneration | **Semantic comparator** over one committed full-float JSON: **exact** on `(id,direction,param,theta,expect,tol,finite,status)`, **rtol-tolerant** on `(ad,fd,ratio,abs_ad)` with rtol measured cross-arch (arm64 commit vs x86 CI). A literal byte-diff is infeasible cross-arch. |
| Q4 | Ordering | **Green machinery first, then per-target RED→GREEN.** Phase A stands up the three CI checks against current 66-row coverage (green, mergeable). Phase B closes each D4 hole as its own RED(manifest entry)→GREEN(measured case) commit. |
| Q5 | CI-lane wiring | **Dedicated fast `gradient-gate` job** (grad-audit pytest subset + audit-script exit-0), wired into the `tests` aggregator. Layers implemented as pytest tests so local + CI enforce identically. |
| Q6 | Manifest seeding / counting | Manifest unit = **`(id, param)` coverage unit, edges excluded** (Michie density = 2 entries; edges are case-internal probes). Seed ~56 from current coverage via **generate-once-then-freeze** as an independent literal (so a deleted case trips the ratchet); D4 entries appended per-target in Phase B. |

## Architecture — `manifest.py` and its three structures

A new module `tests/validation/grad_audit/manifest.py` is the single, reviewable
**gradient-coverage source of truth**. It holds three frozen structures:

1. **`MUST_AUDIT`** — the set of `(id, param)` coverage units the registry must cover, each with
   a short rationale. Seeded ~56 from current coverage by a one-shot generator that introspects
   `REGISTRY`, reviewed, then **frozen as an independent literal**. It is deliberately **not**
   computed from `REGISTRY` at runtime — otherwise the ratchet would be vacuous and could not
   catch a *deleted* case. The ratchet checks the registry *against* this literal.
2. **`SYMBOL_CATEGORY`** — a complete map from every `progenax.__all__` symbol to either `AUDITED`
   (must resolve to ≥1 registry case) or an `EXEMPT_*` reason
   (`PROTOCOL`, `CONTAINER`, `ANALYTICAL_IC`, `NON_FISHER_DIAGNOSTIC`, `HELPER`, …). Seeded once
   over all ~95 public symbols. A newly-added public symbol is *absent from the map* → the
   keys-equality check fails → forces explicit categorization. No inference, no false positives.
3. **`PARAM_ALLOWLIST`** — the charter's carry-forward known-limitations (α=1 IMF branch points,
   the binned-count data side, du-monotonicity), each tied to a registry `known_blocked`/expect
   and a reason. Distinct from `SYMBOL_CATEGORY`: it documents `(id,param)`s that *exist* in the
   registry but are legitimately not FD-consistent.

The website table (`docs/website/50-validation/differentiability-audit.md`) is **generated** from
these structures so the doc cannot drift from the enforced manifest.

## The three enforcement layers (data flow)

**Layer 1 — run the gate (exists; formalized).** `pytest tests/validation/test_grad_audit.py`
asserts every registry case is `clean`/`known-limitation`; `scripts/audit_gradients.py` exits 0
iff 0 hazards. Both run in the new `gradient-gate` job.

**Layer 2 — staleness guard (new pytest).** `test_json_fresh.py` calls `run_audit()` into a tmp
path and **semantic-compares** to the committed `validation/data/grad_audit_results.json`:
exact on the discrete/structural projection, rtol on the floats (rtol measured cross-arch with
margin). Drift → fail, printing the offending rows. Protects both structure and numbers.

**Layer 3 — coverage ratchet + `__all__` map (new pytest).** `test_manifest_coverage.py`
asserts (a) every `MUST_AUDIT` `(id,param)` ∈ `{(c.id, c.param) for c in REGISTRY}`, and
(b) `set(SYMBOL_CATEGORY) == set(progenax.__all__)` with every `AUDITED` symbol resolving to ≥1
covered id. A new ungated public symbol, a removed `MUST_AUDIT` case, or an uncategorized symbol
→ RED.

All three are **pytest tests** so local `pytest tests/validation` enforces them identically to CI
(one enforcement engine, no CI-only path). The dedicated `gradient-gate` CI job runs this subset
plus the audit-script exit-0 for a clearly-named, runtime-isolated branch-protection signal.

## Completion work-list — the D4 Fisher-coverage holes (Phase B)

Each is its own RED→GREEN commit (manifest entry first → measured FD/analytic case), fresh
subagent + code-review between, gates frozen **measured-first** (≥3 seeds, ±3σ band, never a
weakened tol). Exact ids/params are pinned by measurement during Phase B.

- **Engine-A `w_j` and `r_a`** (`MultiComponentCluster.from_components` / `from_mass_segregation`)
  — velocity-scale ratios + anisotropy radius; finite-only smoke today, **no FD anywhere**.
- **`build_binary_cluster` end-to-end** — the full IMF→companion→spatial assembly gradient
  (registry stops at `resolve_binary_components` / `MoeCompanions`).
- **Binary orbital-distribution params** — `SanaOBPeriod`, `LogNormalPeriod`, `LogUniformPeriod`
  (period); `ThermalEccentricity`, `UniformEccentricity`, `MoeEccentricity` (eccentricity);
  `IndependentCompanions.e_max`. FD/analytic-tested in scattered tests today, promoted to registry
  cases.

The scattered tests these promote remain as KEEP/pointer (per the Tier-4 inventory's
KEEP-do-not-touch list); the allowlist (`PARAM_ALLOWLIST`) carries the α=1 / binned-count / du
known-limitations forward unchanged.

## Rollout phases

- **Phase A — green machinery baseline.** Build `manifest.py` (seed `MUST_AUDIT` ~56 +
  `SYMBOL_CATEGORY` over current `__all__` + `PARAM_ALLOWLIST`), the Layer-2/3 pytest tests, the
  staleness comparator, and the `gradient-gate` CI job. Calibrate the cross-arch float rtol
  (measured). Everything green against current coverage — proves the machinery and doesn't
  false-positive. Mergeable checkpoint.
- **Phase B — close the D4 holes, per-target RED→GREEN.** One target per fresh subagent: add the
  `MUST_AUDIT` entry (ratchet RED) → write + measure the FD/analytic case (GREEN) → code-review →
  next. Regenerate + recommit the JSON and the generated website table as cases land.
- **Phase C — documentation + close-out.** Generated website table updated; charter success
  criteria verified; STATUS.md updated; ONE final PR when CI green and Anna approves.

## Testing / Definition of Done

- CI fails on: a reintroduced silent-zero (any case → Layer 1), a registry↔JSON drift
  (Layer 2), OR a `MUST_AUDIT` entry / new public `__all__` symbol without a case (Layer 3).
- The D4 holes are closed (Engine-A `w_j`/`r_a`, `build_binary_cluster`, the binary distributions
  all have FD/analytic-matched registry cases) — the gate is **complete**, not just **enforced**.
- `manifest.py` (+ `PARAM_ALLOWLIST`) committed, reviewable, documented as the gradient-coverage
  source of truth on the validation website.
- Full released-core gate green; `scripts/audit_gradients.py` exit 0; `make build` (website)
  0 warnings.

## Hard rules (carried from the mission)

JAX-native only (`jnp`, `lax.scan`/`fori_loop`, never `while_loop`, no numpy/scipy in `src`);
TDD strict (RED→GREEN, never weaken a test — for the ratchet, the new `MUST_AUDIT` entry makes CI
RED before the case lands); measured-first frozen gates (≥3 seeds, measured ±3σ) for any new
physics case; units STELLAR, `G` explicit never hardcoded; branch `feat/gradient-gate-lockin` off
`main`, commit per task, no push/merge without Anna's explicit go, ONE final PR; verify against
code/primary sources not memory; talk to Anna at every checkpoint.

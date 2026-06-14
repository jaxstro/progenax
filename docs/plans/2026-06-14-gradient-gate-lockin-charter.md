# Charter — Gradient-gate lock-in + completion

> **Status:** charter only (decisions locked in a brainstorm 2026-06-14). The full
> brainstorm → design → plan → implement happens in a **fresh session** against this charter.
> This document records the *what* and *why*; the fresh session decides the *how*.

## Premise

The 2026-06 differentiability gradient-audit arc (merged via PR #8, merge `2164a7f`) built a
66-case AD-vs-FD **release gradient-gate** (`tests/validation/grad_audit/`) and found + fixed two
real silent-zero/biased-gradient hazards. But that audit **measured gradient integrity once** —
nothing currently *enforces* it. A future commit can reintroduce a silent-zero, or add a new public
sampler with no gradient case, and CI would not notice. This charter turns the gate from a one-time
audit into a **permanent, self-policing, complete invariant** — the difference between
"gradient integrity = Fisher integrity, today" and "…forever."

Gradient integrity = Fisher integrity: a silently zeroed/NaN gradient corrupts a Fisher matrix
**with no error raised**. The gate is the only thing standing between that failure mode and a
confidently-wrong forecast.

## Locked decisions (do not relitigate in the fresh session)

### D1 — Focus: lock in + complete the gradient gate
Chosen over the other candidate hardening directions (consolidation/dead-code sweep; packaging /
pip-installability R2; doc↔code provenance hygiene). Those remain valid future arcs; this one is
first because it locks in the value the just-merged arc bought.

### D2 — Strength: full self-policing ratchet (three CI layers)
1. **Run the gate** — `tests/validation/test_grad_audit.py` + `scripts/audit_gradients.py` exit-0
   (0 hazards) on every push, as a first-class CI signal (not just a local gate).
2. **Staleness guard** — CI regenerates `validation/data/grad_audit_results.json` from the registry
   and **fails on any diff** vs. the committed JSON, so the registry and its measured numbers can
   never silently drift apart.
3. **Coverage ratchet** — CI **fails if any "must-audit" entry point lacks a registry case**, so the
   gate stays *complete* as the package grows.

### D3 — Ratchet set: curated manifest + `__all__` cross-check
- A single, reviewable **manifest** lists the public entry points that MUST have a gradient-audit
  case. The ratchet fails if the registry does not cover every manifest entry.
- A second, lighter check flags any **new** public sampler/profile/DF/transform in `progenax.__all__`
  that is absent from BOTH the manifest and the allowlist — so a new entry point cannot ship
  silently ungated. (Adding to the manifest is itself the discipline.)
- Rejected alternatives: automatic protocol-based discovery (elegant but brittle — false positives
  on the many public symbols that aren't differentiable entry points); definition-site markers
  (self-documenting but requires touching `src` across the package).

### D4 — Completion work the manifest forces (the deferred Fisher-coverage holes)
From the Tier-4 inventory's "Coverage GAPS" section — real inference targets currently with only
finite-only smoke or no coverage:
- **Engine-A `w_j` and `r_a`** (`MultiComponentCluster.from_components` / `from_mass_segregation`) —
  velocity-scale ratios + anisotropy radius, finite-only smoke today, no FD anywhere.
- **`build_binary_cluster`** end-to-end — the full IMF → companion → spatial assembly gradient
  (registry stops at `resolve_binary_components` / `MoeCompanions`).
- **Binary orbital-distribution params** — Sana/LogNormal/LogUniform period; Thermal/Uniform/Moe
  eccentricity; `IndependentCompanions` `e_max`. FD/analytic-tested in scattered tests
  (`test_population.py`, `test_companions.py`) but not registry cases.

### Allowlist (legitimately not FD-consistent — carry forward, do not "fix")
- The **α=1 IMF branch points** (`PowerLawIMF.ppf`/`mean_mass`, `IMFParams.log_prob_nll` at α=1) —
  measure-zero branch-limited points, finite and FD-exact at α=1±1e-3.
- The **binned number-count data-side** (`binned_number_density`) — a sum of frozen-edge indicator
  functions; AD=0 is correct-by-design (the N(r) Fisher gradient lives in the model `p_k`).
- **du-monotonicity** of the inverse-CDF samplers — gradient w.r.t. the frozen uniform draw `u`, a
  data-side property out of the param-channel scope (boundary NaN-grad finiteness is pinned in
  `TestBoundaryGradients`).

## Open design questions (the fresh session's first decisions — deliberately NOT decided here)
1. **Manifest format + location** — a Python list/set in the grad_audit package? a data file? keyed
   by `(entry_point_id, param)` to mirror the registry's case ids?
2. **The `__all__` cross-check classifier** — how to decide a public symbol is "a differentiable
   entry point that needs a case" vs. a legitimately-exempt public symbol, without false positives.
3. **Deterministic JSON regeneration** for the staleness diff — the audit must regenerate
   bit-identically (float formatting, ordering, key stability) so the CI diff is signal, not noise.
4. **Ordering** — land the D4 completion cases *before* turning on the ratchet (so CI goes green
   immediately) or *alongside* (ratchet RED first, then green as cases land — the arc's TDD style).
5. **CI-lane wiring** — a dedicated `gradient-gate` job vs. folding the three layers into the
   existing `released-core (validation)` lane; runtime budget (the validation suite is the slow one).
6. **Manifest seeding** — derive the initial manifest from the current 66 registry cases + the D4
   targets, and reconcile the "9 Cases vs 66 rows" counting convention (e.g. Michie density's two
   params).

## Success criteria (for the fresh session's eventual Definition-of-Done)
- CI fails on: a reintroduced silent-zero (any case), a registry↔JSON drift, OR a manifest entry /
  new public entry point without a gradient case.
- The D4 holes are closed (Engine-A `w_j`/`r_a`, `build_binary_cluster`, binary distributions all
  have FD-matched registry cases) — the gate is *complete*, not just *enforced*.
- The manifest + allowlist are committed, reviewable, and documented as the gradient-coverage
  source of truth on the validation website.
- Full released-core gate green; `scripts/audit_gradients.py` exit 0; `make build` 0 warnings.

## Pointers for the fresh session
- **Inventory (the completion checklist):** `.claude-work/grad-test-inventory.md` — its "Coverage
  GAPS" section is the D4 work-list; its KEEP-do-not-touch list constrains the allowlist.
- **The gate:** `tests/validation/grad_audit/{core,registry,reductions,binners}.py`,
  `tests/validation/test_grad_audit.py`, `scripts/audit_gradients.py`,
  `validation/data/grad_audit_results.json`.
- **Living doc:** `docs/website/50-validation/differentiability-audit.md` (66-case status + plots).
- **Arc memory:** `differentiability-gradient-audit-arc.md` (the full Tier 0–4 history + the
  subagent-suspend recovery note).
- **CI:** `.github/workflows/` (the released-core shards + lock-check/wheel-smoke lanes the new
  gradient-gate lane sits beside).

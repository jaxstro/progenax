# SoTA per-module review — method & template

**Purpose:** the deliberate, evidence-backed review run on each progenax module during the
pre-release SoTA-design + validation pass
([agenda](../notes/2026-06-03-pre-release-sota-agenda.md)). Each module review fills in this
template and produces a **gate packet** under `docs/reviews/`. Gates are HITL — Anna approves
at every step.

## Engine (complexity-scaled)

- **Every module:** a manual read-driven review against the 5-dimension checklist below.
- **Physics/numerics-heavy modules** (e.g. `imf`, `binaries`, `kinematics`, `profiles`):
  additionally synthesize the `astro-code-review:*` panel (`scientific-code-reviewer`,
  `jax-code-validator`, `numerical-methods-auditor`, `code-craft-reviewer`).
- **Tiny foundational modules** (e.g. `protocols`, `dynamics`, `tidal`): manual only.

## The 5-dimension checklist

1. **Design** — cohesion / single responsibility; public-API surface (vs `progenax.__init__`);
   JAX-nativeness (`jnp` only; `lax.scan`/`cond`, no `while_loop` on grad paths; no in-place;
   no `argmax`/`argsort`/dead-branch `where` on grad paths; Equinox immutability); file ≤~600
   LOC, functions ≤100 LOC (over → ticket, not a chop).
2. **Validation** — match analytic/published values within a **regime-anchored, justified**
   tolerance; convergence/order refinement where a method has a formal order; **FD-vs-autodiff
   grad-check on every public differentiable entry point** (no NaN/zero-grad surprises); unit
   consistency (`G`/`units` threaded explicitly; dimensional check); **constant provenance**
   (value → source citation).
3. **Docstrings** — accurate vs implementation; units stated; differentiability noted;
   references resolve to the bibliography.
4. **Test-suite review** — adequacy (edge cases, error paths, grad-check present, tolerances
   justified); anti-patterns (`superpowers:testing-anti-patterns` — no testing-the-mock, no
   test-only methods in `src/`, no assertions on incidental detail, no tautological/no-op
   tests); refactor (DRY, parametrize, 3-tier placement); **remove only** genuinely redundant /
   tautological / dead tests — never to dodge a failure.
   - **Safety rail 1 — coverage-preserved:** `--cov=progenax` line coverage must not drop
     across a removal/refactor (or the drop is justified and Anna approves it at Gate 1).
   - **Safety rail 2 — mutation-sense:** any hardened/kept test is proven RED-sensitive
     (break the code → the test fails), so it is never secretly a no-op.
5. **Findings → severity → disposition.** Each finding gets a severity and a disposition
   (below), recorded in the packet's findings ledger.

## Severity & disposition (severity-gated)

- **Critical / Major** (correctness, units, broken differentiability, wrong/unsourced
  constant) → **fix in this batch's PR**, RED test first.
- **Minor / design-nits / >100-LOC splits** → **log a ticket** in `docs/notes/` and batch
  later — **unless trivial** (≤ a few lines, no behavior change), then fix in-batch.
- A fix that crosses into another batch's module is surfaced at Gate 1 for an explicit
  fix-now-vs-defer decision.

## Three-gate HITL flow (non-negotiable)

Anna approves **before** every state change, never after.

1. Review module(s) — read-only. Run validation (pytest, grad-checks) to gather evidence.
2. Write the `docs/reviews/` packet: checklist + evidence + findings ledger + **proposed fix
   plan**.
3. **🚦 Gate 1 (before any code):** present packet + fix plan → Anna approves which findings
   are fixed now vs deferred.
4. Implement approved fixes, **RED-first** (failing test → minimal pass). No commits yet.
5. **🚦 Gate 2 (before commit):** present diff + verification evidence → Anna approves.
6. Commit on the branch; repeat 4–5 per logical change.
7. **🚦 Gate 3 (before PR/merge):** present full branch + green-CI evidence → Anna approves &
   merges.

## Artifacts

- **This method** → `docs/dev-methods-guides/sota-module-review.md` (you are here).
- **Per-module gate packet** → `docs/reviews/YYYY-MM-DD-<module>.md` (one per batch).
- **Deferred-findings tickets** → `docs/notes/YYYY-MM-DD-<module>-<topic>-ticket.md`.

## Verification commands

```bash
# Targeted module tests (baseline + after fixes)
env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/<module>/ tests/validation/test_<module>_*.py -v
# Coverage (PACKAGE scope only — --cov=<submodule> SIGABRTs via jaxlib/abseil)
env -u VIRTUAL_ENV uv run --no-sync pytest tests/ --cov=progenax --cov-report=term-missing
# FD-vs-autodiff grad-checks: a throwaway script in /tmp (not the repo)
env -u VIRTUAL_ENV uv run --no-sync python /tmp/<module>_gradcheck.py
```

## Gate-packet skeleton

```markdown
# Batch N review — <module> (gate packet)
**Date · Branch · Commit · Reviewer · Engine (manual / +panel)**
## Scope            — files + LOC, public-API surface
## Design review    — cohesion / API / JAX-nativeness findings
## Validation       — expected-vs-measured tables (value, tol, nσ); grad-check table
                      (entry point | FD | autodiff | rel-err); provenance table; unit checks
## Test-suite       — coverage rows; anti-pattern findings; proposed refactors/removals
                      (each with coverage diff + mutation-sense note)
## Findings ledger  — id | finding | severity | disposition (fix-now / ticket-id)
## Proposed fix plan — ordered, RED-first, with the test that will go RED first
## Verification     — exact commands + output (filled at Gate 2)
## Gate status      — G1 / G2 / G3 + Anna's approvals
```

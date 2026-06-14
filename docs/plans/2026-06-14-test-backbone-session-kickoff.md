# KICKOFF — REVIEW + VERIFY the progenax test/validation-backbone design + plan (fresh session)

Paste the block below as the first message of the new session.

---

ROLE & GOAL
You are continuing **progenax** (`~/projects/jaxstro-dev/progenax`, part of the jaxstro ecosystem). A
prior session brainstormed + ratified a **SoTA pre-release validation-backbone** design with Anna and
wrote a 6-phase implementation plan. **Your job THIS session is to critically REVIEW and VERIFY those
two documents BEFORE any implementation** — they are the spec for a large, multi-session arc that will
touch released-core tests, so they must be sound, complete, feasible, and factually correct first. Do
NOT start implementing until Anna explicitly approves after your review.

START STATE
- `main` is clean at `12a732f` (the build_cluster arc + website follow-ups + these two planning docs;
  all LOCAL, nothing pushed). Working tree clean. You are on `main`.
- READ THESE TWO DOCUMENTS IN FULL FIRST (they are the subject of your review):
  1. `docs/plans/2026-06-14-test-validation-backbone-design.md` — the ratified design.
  2. `docs/plans/2026-06-14-test-validation-backbone-plan.md` — the 6-phase TDD implementation plan.
- Context they build on: the existing grad-audit registry
  (`tests/validation/grad_audit/{manifest.py,registry.py,core.py,test_manifest_coverage.py,
  test_json_fresh.py}` + `scripts/audit_gradients.py`) is the TEMPLATE every new registry mirrors. Read
  it. Also `STATUS.md` (the `next:` line) for how this arc was reached.

WHAT THE ARC IS (one paragraph, for orientation — the docs have the detail)
A unified, generated + timestamped + self-policing single source of truth for ALL pre-validation
checks: a **registry layer** (4 registries — differentiability [exists], API-coverage,
physics-validation, provenance-of-constants), a **generated dashboard** that unions them with
`pytest-cov` line coverage + `--durations` into a committed JSON + a MyST matrix page gated by a
staleness test, and a **profiling-driven suite refactor** (`@slow`-mark the measured runtime sinks,
consolidate cross-tier redundancy, fill the coverage holes the registries surface). The release gate =
every registry full (0 holes) + line-cov ≥ 90% floor + dashboard fresh + FULL suite green.

YOUR REVIEW — be adversarial and concrete. Produce a written verdict covering:
1. **Soundness of the design.** Is the registry-vs-test framing correct? Does the 4-registry decomposition
   carve the coverage space cleanly (no overlap, no gaps)? Is "100% = API + line-floor + 0 registry holes"
   the right release gate, or does it miss something (e.g. integration/property/mutation coverage)?
2. **Feasibility / correctness of the plan.** For each phase, are the tasks ordered right (dependencies)?
   Is the keystone (Phase-1 generated dashboard + staleness gate) actually buildable as written? Will
   `pytest-cov` under `-n auto` (xdist) give correct line attribution, or is the watch-item right that it
   needs a non-parallel `--cov` pass? Will line coverage of JAX-traced (`lax.cond`/`vmap`) branches behave
   as the plan assumes?
3. **VERIFY the factual claims against the LIVE repo** (run these, don't trust the docs):
   - `progenax.__all__` length (doc says 114): `python -c "import progenax; print(len(progenax.__all__))"`.
   - Is `pytest-cov` installed? (`python -c "import pytest_cov"`). Is `slow` a registered marker?
   - Re-profile a representative slice (or trust the committed `--durations`?) — spot-check that
     `test_json_fresh` + `test_audit_script` really both regenerate the grad-audit JSON (read them) and
     are the dominant sinks. Confirm the cross-tier redundancy claim (Plummer/EFF/Michie/LIMEPY each have
     unit AND validation physics files that overlap, like the King case already consolidated in `9bb1f79`).
   - Confirm the grad-audit template files exist and the new registries can mirror them.
4. **Risk surface.** Does the plan adequately protect coverage during the Phase-2.3 + Phase-3 test
   removals (per-item Anna approval)? Is the staleness gate itself cheap enough not to become the next
   429s sink (the plan flags this — verify the proposed generator is introspection-only)?
5. **Gaps / improvements.** Anything missing (e.g. a CI-job wiring, a `coverage combine` step, a
   `Date.now()`-in-script caveat, an ordering bug), anything over-engineered (YAGNI), anything that
   contradicts the repo's conventions (JAX-native core, no-numpy-in-src, file/function LOC limits, the
   units policy, the `*.png` gitignore, CI-minutes-exhausted → no PR).

DELIVERABLE
A concise written review + a clear verdict: **ready-to-implement as-is / ready-with-the-following-edits /
needs-rework**, with specific, actionable changes. If edits are warranted, propose them and — with Anna's
approval — apply them to the design/plan docs and commit. Only AFTER Anna signs off on the reviewed docs
do you set up `feat/test-backbone` and begin Phase 0/1 via `superpowers:executing-plans` (or
subagent-driven), with a HITL checkpoint per phase.

STRICT PROTOCOL (NON-NEGOTIABLE)
- **HITL:** Anna approves at every step; no silent decisions; she approves every test deletion before it
  lands (Phases 2.3, 3.2–3.5). Talk to her often.
- **Review BEFORE build:** do not implement until the docs are reviewed + Anna-approved.
- TDD RED→GREEN→REFACTOR when you do implement. NEVER weaken a test/tolerance to pass — fix the root cause.
- JAX-native core only (`jax.numpy`, `equinox`, `lax.scan`; NO numpy/scipy in `src/`); the dashboard
  *generator* is a plain script (subprocess/parsing OK there). Limits: file ≤500 LOC, function ≤100 LOC.
- Units: explicit `G` in core; `units=None → DEFAULT_UNITS` only in convenience wrappers.
- Registry manifests are HAND-CURATED frozen literals (a derived manifest can't catch a deletion) — do
  not "DRY" them away.
- **CI MINUTES ARE EXHAUSTED → verify LOCALLY** (use the env prefix below). GitHub workflows are disabled;
  keep it that way. Do NOT push/merge without Anna's explicit go.
- Commit per verified task; end commit messages with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

ENV (uv, NOT conda):
```
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest <args>
# FAST gate: add  -m "not slow" -n auto ; FULL gate: drop the marker.
```

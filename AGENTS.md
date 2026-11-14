# AGENTS.md (Codex) - progenax

Guidance for Codex when working in `progenax`.

## Integrated delivery

Before implementation, state the next concrete scientific output and its
completion criteria. Keep the approved physical model and criteria fixed.

Component tests support that output; they do not replace it. After repairing a
blocker, return to the integrated pipeline. Do not expand into another modeling
layer, framework or diagnostic campaign without explaining why the agreed
output requires it.

Report implemented, executed, integrated, numerically qualified and physically
tested states separately. “Working,” “fixed” and “complete” must name the
demonstrated scope.

Distinguish physical rejection, numerical failure and an unperformed
calculation. Never present reference catalogs or birth proposals as completed
outputs of a new model.

Own routine engineering autonomously. Seek approval for scientific changes.
Repeated failures require an updated causal diagnosis and recommendation,
rather than another unexplained sequence of experiments.

Preserve compact evidence. Keep the existing status page current.
Documentation and test counts are not substitutes for the requested scientific
output.

## Writing

Remove all mannered prose. State the rule, the measurement and the mechanism in plain
sentences. No rhetorical emphasis, no asides, no metaphors, no dramatised failure stories;
the number, the date and the mechanism carry the point. This applies to chat, docstrings,
commit messages, `STATUS.md` and everything under `docs/`.

## Documentation edits

Before any site edit, read `docs/reviews/site-ownership.md` and locate the page in
`docs/website/myst.yml`. Read connected theory, DF, API, tutorial and validation
pages before proposing additions. Use the existing owner and figure inventory;
repair generated documentation at its source. Update the corresponding Documentation
record in the maintainer-local `docs/tracker/state.json`, when present, with
approval, checks, grade and remaining work. Keep private tracker and review
material out of public commits.
A page inventory or green build is not scientific qualification.

## Read First
- `CLAUDE.md`
- `README.md`
- `pyproject.toml`

## Units Policy
- `DEFAULT_UNITS`: `STELLAR` (Msun, pc, Myr)
- Core APIs require explicit `G` or `units`.
- Convenience wrappers may accept `units=None` and resolve to `DEFAULT_UNITS`.

## JAX Rules
- Use `jax.numpy` only in core code.
- Ensure `jax.grad` works through IC generation.

## Scientific experiment relevance

Scientific demos and qualification experiments must represent the intended physical
population and resolve the numerical quantities being measured. Justify population
size, component occupancy, discretization, observables and computational budget.
Use realistic synthetic-cluster parameter recovery for IMF inference. Tiny exact
enumerations belong only in inexpensive algebraic unit tests; never propose them
as IMF experiments or substitutes for end-to-end physical qualification. Freeze
scientific assumptions and criteria with Anna before running; retain failures.
Use BlackJAX for this recovery demo. Do not add NumPyro or a parallel demo owner.

## Testing
- `pytest tests/ -v`
- `pytest tests/unit/ -v`
- `pytest tests/integration/ -v`
- `pytest tests/validation/ -v`

## Brain hub - this repo is a spoke of ~/brain (read-only from here)

- **Never edit `~/brain` from this session** - not hat homes, ADRs, configs, knowledge, or `_generated/`.
- **One write path home - the inbox, via capture** (works from any directory):
      brain "what happened - short, factual"
- **Cross-cutting insight** (something here that's also relevant to another project/paper)?
      brain "xref: <insight> - touches <other project / paper>"
  It becomes a brain concept and resurfaces there via `/brain-pack` (ADR-0019).
- **Full protocol + conventions:** read `~/brain/AGENTS.md` and `~/brain/guide/` before cross-session work
  (pull-only hub; spec -> session -> log handoffs, ADR-0018; modern mystmd if this is a MyST site).
- **Starting focused work here?** Pull a context pack from the hub: `/brain-pack <this-project>`.
- **Need papers/equations?** Start with that pack's Relevant literature and Equation-critical sources. Read
  source notes in `~/brain/knowledge/sources/`; verify exact equations/tables against
  `~/brain/knowledge/library/<bibkey>.pdf`; use `~/brain/knowledge/derived/equation-digests/` only when rows
  are verified; treat `~/brain/knowledge/raw/` as search-only. Capture needed source-note expansions back to the hub:
      brain "source-note update: <bibkey> - <what this package needs>"

<!-- brain-handshake: keep in sync with ~/brain/guide/how-to/set-up-a-project.md#spoke-stanza -->

<!-- brain-status-convention -->
## Brain status updates
When you make notable progress, hit a blocker, or set the next action, update this repo's `STATUS.md` (`next:` / `blocker:` / `due:` lines) — the brain pulls it into the portfolio dashboard + standup via `federate.py` (see `~/brain/work/meta/status-convention.md`). Brain stays pull-only: never hand-edit `~/brain`; capture events with `brain "…"`.

## Saved module-hardening workflow

For module hardening, equation-registry development and scientific documentation
coverage, use the maintainer-local `.agents/skills/progenax-module-hardening/SKILL.md`
and `docs/reviews/module-hardening-plan.md` when available. Public documentation
ownership is defined in `docs/reviews/site-ownership.md`. Local plans and tracker
records do not grant new scientific execution approvals.

## User-facing scientific documentation

Follow `docs/reviews/site-ownership.md`, especially “User-facing authoring standard”.
Read the full page before targeted edits; preserve existing structure and voice.
Define symbols and physical conventions at first use. Remove stale bug/repair
narratives from teaching pages while retaining current limitations and historical
records. Use MyST callouts and figures deliberately; inspect the rendered reading
width. Keep theory, API, tutorials and validation in their established owners.

## Gravoturb storage limits

Keep current cloud inputs, required reference catalogs, plotting scripts and their
compact data, configurations, seeds, source, results, failures, and figure masters.
Retire superseded native fields and derived workspaces after checking the current
input graph and figure dependencies. Do not delete source or plotting code to
save space. Never reduce scientific precision to fit a disk budget. Preserve
failure results in compact records.

- Current accretion producers use `accretion-law-001/_storage.py`: a 4 GiB logical
  budget for the study plus the shared Green256 cache, with per-output reservations
  and one nonblocking writer lock. A larger run needs a deliberate disk-budget
  revision before computation. Do not evade the guard through another output path.
- Reuse the canonical shared Green cache. The native snapshot CLI owns temporary
  kernel banks; its supervisor closes the worker before removing only that bank.
  Persistent Python API storage remains caller-owned.
- Before a new data-producing experiment, state retained and temporary byte
  estimates in its existing run plan. New producers must use bounded storage and
  clear owned scratch after verified output closeout. Do not accumulate numbered
  full-field copies or unbounded profiler traces by default.
- Bulk Gravoturb arrays, native scratch banks, and profiler output stay out of Git.
  Keep compact plotting inputs and the recipes needed to regenerate retired fields.
- The Progenax search index stays disabled after cleanup. Do not recreate or watch
  it as a side effect of routine work. Explicit reindexing must exclude generated
  scientific outputs, tool caches, and private audit receipts and use a bounded,
  deliberate source selection.

## Test design: quality over quantity (Anna, 2026-10-05)

This rule is identical in the jaxstro-dev root and in every package's `CLAUDE.md` and `AGENTS.md`; keep the
wording the same everywhere.

A test exists to catch a named failure. A few strong tests are better than many weak ones. Inefficient test
design is not allowed in any jaxstro package, and each package's suite is to be brought under this rule file by
file.

**Every test**

- names in its docstring the failure it catches, and fails when that failure is introduced; check this once by
  mutation when it is not obvious;
- asserts against an independent value: an analytic or limiting solution, a vendored reference fixture, a
  conservation law, or a bound whose derivation is stated. A rerun of the same code path is not a reference;
- sets its tolerance from the quantity its error scales with, a few times the measured error. A near-zero
  difference is compared against the scale its roundoff comes from, not against itself (gravax: the contact gap
  against the contact radius, 2026-10-05);
- runs in seconds. A test that needs longer is marked `slow` and says why.

**Compile cost (JAX)**

- Do not call a function that contains `lax.scan`, `fori_loop`, `while_loop` or `cond` eagerly in a test. Call it
  through one module-level `jax.jit` with its static arguments named. An eager call rebuilds the loop body and
  compiles a new program on every call: gravax `test_reach.py` made 575 XLA compiles, 235 after the change, and
  its cold-cache wall went from 17.0 s to 9.2 s (2026-10-05).
- Values that vary between cases (thresholds, spans, seeds, eta, distances) are traced inputs of one compiled
  program: sweep them with `vmap` or `lax.map` inside one `jit`, or parametrize over traced values. A static
  field that differs between tests costs one compile per test.
- Tests in a file share static configurations. Build each distinct configuration once (a module-scoped fixture
  or a shared configuration module) and reuse it. A new static configuration needs a stated reason. Count the
  distinct compiled programs of a file when adding tests to it.
- An expensive result (an integration, a routed state, a replay) is computed once in a module-scoped fixture and
  checked by several tests, not recomputed in each.
- The persistent compilation cache stores every program: `jax_persistent_cache_min_compile_time_secs = 0` and
  `jax_persistent_cache_min_entry_size_bytes = 0` in `tests/conftest.py`. With a 0.5 s floor the small programs
  were recompiled on every run (gravax `test_interaction_policy.py`: 24.0 s to 7.1 s of wall for 4.6 MB of
  cache, 2026-10-05).

**Removal and hardening**

- Delete a test that can no longer fail: its scenario is unreachable, it duplicates another test, or it pins an
  implementation detail with no failure mode. Removing a weak test is progress, not lost coverage.
- Harden a loose test: replace "runs", "is finite" or "changed" with the quantitative claim and its tolerance.
- Never weaken a test to make it pass. Fix the code, or fix the test's premise and say why.

**Measure before changing a slow file.** Record per test the trace, lowering and XLA compile seconds, the cache
hits and the programs compiled (`jax.monitoring` duration listeners and `jax_log_compiles`). Gravax, 2026-10-05,
563 hierarchy unit tests with the cache off: XLA compile was 73% of the wall and execution 12%; 232 distinct
programs took over 0.3 s each to compile.

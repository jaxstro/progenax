# 95-release closeout (v0.1.0) — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: execute via superpowers:subagent-driven-development
> (fresh subagent per task + independent code-review between tasks + HITL checkpoint per phase).

**Goal:** Close the in-scope should-fix items from the release-readiness audit so
progenax v0.1.0 is launch-clean (source/GitHub tag; PyPI still gated on jaxstro per ADR-0015).

**Architecture:** 4 phases (see design doc). Batch ALL `src/progenax/` edits in
Phase 2 → ONE coverage re-stamp. Paper-grounding (Phase 1) verifies load-bearing
kernels before any src fix.

**Tech Stack:** JAX/Equinox src; pytest (XLA caps + `-n auto`); MyST docs; the
test-backbone coverage/dashboard + grad-audit registries.

**Gate commands:**
- FAST: `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -m "not slow" -n auto`
- FULL: same without `-m "not slow"`.
- Coverage re-stamp (after src changes): FULL `--cov` raw → `scripts/build_test_dashboard.py --stamp-coverage <raw> --emit --render`.
- `myst build` from `docs/website/` → 0 content warnings.

---

## PHASE 1 — Paper-grounding (no src edits)

### Task 1.1: B&M82 note + verify `project_dispersion`
**Files:** Read `docs/core-papers/BinneyMamon1982.pdf` (gitignored — do NOT commit).
Create/expand the per-paper note (match the existing per-paper note convention under
`docs/website/`; find it first). Verify against `src/progenax/kinematics/dispersion.py`.

**Steps:**
1. Read the PDF. Transcribe the load-bearing equations: the projected surface
   density Σ(R) and the anisotropic LOS + proper-motion projection integrals
   (σ_los², σ_pmR², σ_pmT² kernels with the `(1-βR²/r²)`, `(1-β+βR²/r²)`, `(1-β)`
   weights). Cross-check the exact kernel forms in `project_dispersion` docstring
   (dispersion.py:677+) line-by-line vs the paper.
2. Write the note with equations + a "code cross-reference" section mapping each
   kernel to its dispersion.py implementation.
3. Record verdict: kernels MATCH (→ just add the held citation) or DISCREPANCY
   (→ document precisely; do NOT fix here — fold into Phase 2, checkpoint Anna).
4. Commit the note (NOT the PDF). Stage by name.
   `docs(refs): Binney & Mamon 1982 per-paper note + project_dispersion verification`

### Task 1.2: Baumgardt&Makino03 note + verify `tidal.py`
**Files:** Read `docs/core-papers/BaumgardtMakino03.pdf` (gitignored). Verify
`src/progenax/tidal.py` (`jacobi_radius` + any cited result).

**Steps:** mirror 1.1 for the Jacobi/tidal-radius result; transcribe the relevant
equation(s); cross-reference `tidal.py`; verdict MATCH/DISCREPANCY; commit the note.
`docs(refs): Baumgardt & Makino 2003 per-paper note + tidal.py verification`

**HITL CHECKPOINT after Phase 1:** report both verdicts. If either is a DISCREPANCY
in a load-bearing kernel, STOP for Anna's go before any physics fix.

---

## PHASE 2 — All src changes, then one re-stamp

### Task 2.1: units-G breaking sweep (TDD)
**Files (src):** `kinematics/{eff,king,limepy,plummer,michie}_df.py` (`.sample_velocities`),
`cluster/multicomponent.py` (`.sample_cluster`), `kinematics/api.py`
(`sample_velocities_pipeline`). **Tests:** the kinematics unit tests +
`tests/integration/test_units_through_pipeline.py`. **Grad-audit:** any registry case
passing the default.

**Steps:**
1. **RED:** add/adjust a test asserting each surface RAISES (TypeError / explicit
   error) when `G` is omitted — i.e. `G` is required. Run → fails (default still present).
2. **GREEN:** drop the `G=None` default + the `if G is None: G = defaults.DEFAULT_UNITS.G`
   block on each surface; signature requires `G`. Update the docstrings to the
   MANDATORY-explicit-units wording (remove "If None, uses DEFAULT_UNITS.G").
3. Fix every now-broken callsite: src already passes `G=G` (builders.py:326,
   api.py:171) — verify; update any test that relied on the default to pass an
   explicit `G` (STELLAR.G). Update grad-audit registry cases similarly.
4. Run kinematics unit + the units-pipeline integration test → green.
5. CHANGELOG: add a **BREAKING** entry under the 0.1.0 cycle (explicit-G required on
   the 7 surfaces; resolves audit A2).
6. Commit (src + tests + CHANGELOG; stage by name).
   `refactor(units)!: require explicit G on sample_velocities/sample_cluster (A2)`

### Task 2.2: Docstring gaps
**Files (src):** `profiles/michie.py` (`from_W0_rc`, `__init__`, `.sample_positions`,
`.density`); `imf/` `TruncatedIMF.ppf`; the IMF mass-ratio `pdf`/`cdf`/`ppf`.
**Steps:** add full docstrings (summary/args/units/returns/differentiability) — match
the exemplary binaries/dispersion docstring style. Forward values bit-identical.
Run a quick import + a docstring-presence check if one exists. Commit.
`docs(api): fill Michie/TruncatedIMF/mass-ratio docstring gaps (D4)`

### Task 2.3: Phase-1 fixes (only if Phase 1 found a discrepancy) + held citations
**Files:** `dispersion.py`, `tidal.py`. Apply any Anna-approved kernel fix; replace
the cited-but-not-held flag with the now-held B&M82 / Baumgardt&Makino03 citation
(+ the per-paper note cross-link). If Phase 1 was MATCH, this is citation-only.
Commit. `fix/docs(refs): held B&M82 + Baumgardt&Makino03 citations [+ kernel fix if any]`

### Task 2.4: ONE coverage re-stamp + dashboard regen
**Steps:**
1. FULL `--cov` run producing the raw coverage file (per the documented producer).
2. `scripts/build_test_dashboard.py --stamp-coverage <raw> --emit --render`.
3. Run the staleness + ratchet gates → green. Confirm `registries_full=True`,
   line_cov ≥ 90 floor. Commit the regenerated `validation/data/` + dashboard md.
   `chore(validation): re-stamp coverage + dashboard after release-closeout src edits`

**HITL CHECKPOINT after Phase 2.**

---

## PHASE 3 — Docs / non-src

### Task 3.1: Fix `validate_king.py`
**Files:** `scripts/validate_king.py:248,451`. Drop the `r_t=...` kwarg (field removed;
the DF derives r_t from the profile). Run the script → exit 0, 5 figures.
Commit. `fix(validation): drop removed r_t kwarg from validate_king.py (T11)`

### Task 3.2: myst.yml repo URL
**Files:** `docs/website/myst.yml`. `drannarosen/progenax` → `jaxstro/progenax`
(match `pyproject.toml` + git remote). `myst build` → 0 warnings. Commit.
`docs(site): reconcile repo URL drannarosen -> jaxstro (D7)`

### Task 3.3: Broken internal-doc links
**Files:** `docs/website/00-getting-started/whats-new.md`,
`docs/website/95-release/checklist.md`,
`docs/website/90-development-log/{whats-changed,index,by-topic}.md`.
Rewrite each `../../plans/…` / `../../notes/…` link to a site-internal target or
remove it (do not expose internal filenames). `myst build` → 0 warnings + no broken
xrefs. Commit. `docs(site): fix broken internal-doc links (D6)`

### Task 3.4: Author the 4 progenax-only how-tos + prune TOC
**Files:** `docs/website/40-howto/{set-up-virial-cluster,gradient-based-r_h-fit,
mix-plummer-positions-king-velocities,add-binary-population}.md` (author real,
runnable content with the "Inputs and assumptions" discipline); remove
`interface-with-gravax` from `docs/website/myst.yml` TOC (backlog); update
`40-howto/index.md`. `myst build` → 0 warnings. Commit.
`docs(howto): author 4 progenax-only how-tos; backlog interface-with-gravax (D6)`

**HITL CHECKPOINT after Phase 3.**

---

## PHASE 4 — Final gate + STATUS + review

### Task 4.1: Full verification
- FULL released-core gate green; `myst build` 0 warnings; `audit_gradients.py`
  0 hazards; `validate_king.py` + spot validate scripts exit 0.
- Update `95-release/checklist.md` (check off closed items).

### Task 4.2: STATUS + CHANGELOG + brain
- STATUS.md `next:`/`blocker:`/`due:` update; brain capture; verify CHANGELOG.
- Commit. `docs(status,changelog): release-closeout v0.1.0 complete`

### Task 4.3: Final whole-arc independent review
Dispatch `superpowers:code-reviewer` over the whole arc (base = closeout branch
point). Address Critical/Important; note Minor.

---

## Definition of done
All in-scope checklist items closed; FULL gate + myst (0 warn) + grad-audit (0 haz)
green; coverage re-stamped + `registries_full=True`; CHANGELOG breaking entry;
STATUS updated; whole-arc review clean. **Nothing merged/pushed without Anna's
separate word.** Branch kept until merged AND pushed.

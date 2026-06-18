# 95-release closeout (v0.1.0) — design

**Date:** 2026-06-18
**Branch:** `feat/release-closeout-v0.1.0` (off `main` after the df_moment lock FF-merge, @ c24a8d4)
**Status:** scope + phasing ratified with Anna (brainstorming)
**Owner:** Anna Rosen (single HITL)

## Scope (Anna-ratified)

Close the actionable should-fix items from the release-readiness audit
(`docs/website/95-release/checklist.md`). **In scope:** quick mechanical fixes,
docstring gaps, the units-policy A2 **breaking sweep**, the cited-but-not-held
refs via **paper-grounding** (Anna added the PDFs), and the `40-howto/` stubs.

**Out of scope (Anna):**
- **CI re-enable/dormant (D8)** — deferred; handled later.
- **OED-demo hold-out (D10)** — deferred until informax is stood up.
- **R2 jaxstro packaging** — already decided (GitHub source tag now; PyPI version
  floor when jaxstro publishes; ADR-0015).
- Minor/polish items (sdist trim, dev-log scrub, experimental count) — not this pass.

## Decisions ratified

| Item | Decision |
|---|---|
| **units A2** | **Do the breaking explicit-`G` sweep now** — drop the `G=None→DEFAULT_UNITS.G` default on the 6 surfaces; require explicit `G`; update all callsites/tests/grad-audit. |
| **Refs** | **Paper-grounding**: ingest the now-held `BinneyMamon1982.pdf` + `BaumgardtMakino03.pdf`, write per-paper notes with all load-bearing equations, verify the code against PDF+notes, fix discrepancies, cite properly. |
| **40-howto** | **Author the 4 progenax-only how-tos** (`set-up-virial-cluster`, `gradient-based-r_h-fit`, `mix-plummer-positions-king-velocities`, `add-binary-population`); **backlog `interface-with-gravax`** (remove from TOC — expand when gravax matures); update `index.md`. |
| **OED hold** | Deferred (see out-of-scope). |

## Phasing (drives ordering)

**Key constraint:** the coverage staleness gate keys on *any* `src/progenax/` change
since the last coverage measurement. So **all src edits are batched** and followed
by **one** FULL `--cov` re-run + re-stamp + dashboard regen.

### Phase 1 — Paper-grounding (no src edits)
- Read `docs/core-papers/BinneyMamon1982.pdf` (the LOS-projection / anisotropic
  Jeans projection kernels behind `project_dispersion`) and
  `docs/core-papers/BaumgardtMakino03.pdf` (the tidal/Jacobi-radius result in
  `tidal.py`). PDFs are **gitignored — read locally, never commit**.
- Write/expand per-paper notes (`docs/website/.../references/` or the established
  per-paper note location) with the load-bearing equations transcribed and a code
  cross-reference, per [[paper-grounding-workflow]] + [[no-assumptions-verify-against-pdfs]].
- **Verify** `project_dispersion`'s B&M82 kernels (Σσ_los², Σσ_pmR², Σσ_pmT²) and
  `tidal.jacobi_radius` against the PDFs cell-by-cell. Record discrepancies (if any)
  → folded into Phase 2 as fixes. If the code is correct, the deliverable is the
  verified note + a held-source citation replacing the unheld-primary flag.

### Phase 2 — All src changes, then one re-stamp
1. **units-G breaking sweep.** Surfaces: `eff_df`, `king_df`, `limepy_df`,
   `plummer_df`, `michie_df` `.sample_velocities`; `multicomponent.sample_cluster`;
   and the `api.sample_velocities_pipeline` wrapper. Drop the `G=None` default →
   require explicit `G`. Update src callsites (`builders.py`, `api.py` already pass
   `G=G`), all tests, and any grad-audit registry cases that relied on the default.
   Update the MANDATORY-explicit-units docstrings. Add a CHANGELOG entry (breaking).
2. **Docstrings.** `MichieProfile.from_W0_rc`/`__init__`/`.sample_positions`/
   `.density`; `TruncatedIMF.ppf`; IMF mass-ratio `pdf`/`cdf`/`ppf` Returns blocks.
3. **Phase-1 fixes** (if any) + held-source citations in `dispersion.py` / `tidal.py`.
4. **One** FULL `--cov` re-run + `build_test_dashboard.py --stamp-coverage <raw>
   --emit --render`; confirm staleness gate green.

### Phase 3 — Docs / non-src
- `validate_king.py`: drop the removed `r_t=` kwarg at lines 248 & 451 (it exits 1
  today). Re-run → exit 0, 5 figures.
- `myst.yml`: reconcile repo URL `drannarosen` → `jaxstro` (match `pyproject.toml`
  + git remote).
- Broken internal-doc links (`../../plans/`, `../../notes/`): rewrite/remove in the
  real-source pages — `whats-new.md`, `95-release/checklist.md`,
  `90-development-log/{whats-changed,index,by-topic}.md` (the `_build/` hits are
  gitignored artifacts, ignored).
- How-tos: author the 4 progenax-only pages (real, runnable, "Inputs and
  assumptions" discipline); drop `interface-with-gravax` from `myst.yml` TOC
  (backlog stub kept on disk or removed — Anna's call at execution); update
  `40-howto/index.md`. `myst build` 0 warnings.

### Phase 4 — Final gate + STATUS + review
- FULL released-core gate green; `myst build` 0 warnings; `audit_gradients.py`
  0 hazards; validate scripts exit 0.
- STATUS update; brain capture; final whole-arc independent review.
- Update `95-release/checklist.md` to check off the closed items.

## Risks
- **units-G sweep is the biggest/riskiest** — a real breaking API change. Mitigate
  with TDD per surface + grad-audit re-run + the explicit-units integration test
  (`test_units_through_pipeline.py`).
- **Paper-grounding may surface a real code discrepancy** in `project_dispersion`
  or `tidal.py` — if so, it's a physics fix (verify-first, gate-gated), not a
  silent edit; checkpoint with Anna before changing a load-bearing kernel.

## Definition of done
Every in-scope checklist item closed at root; FULL gate + myst + grad-audit green;
coverage re-stamped; STATUS + CHANGELOG updated; whole-arc review clean; nothing
merged/pushed without Anna's separate word.

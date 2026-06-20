# progenax docs-site review & hardening — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task (fresh subagent per task + independent code-review between tasks + final whole-site review).

**Goal:** Make every page of the progenax MyST docs site correct, pedagogical, tight, honest, and warning-clean for the v0.1.0 public source/GitHub release — without thinning teaching content.

**Architecture:** Branch `feat/docs-review-hardening` (already created, off `main`). Section-by-section pass. Each section = one fresh subagent that (1) re-reads the real pages, (2) **verifies every agent-flagged finding against the actual `src/` or PDF before editing** — a review-agent claim is a lead, not a fact, (3) applies fixes, (4) runs the verification gate, (5) commits. An independent code-reviewer runs between sections. Verify LOCALLY; nothing merged or pushed without Anna's separate word.

**Tech Stack:** mystmd 1.x (`make build` in `docs/website/`), Python/JAX (`env -u VIRTUAL_ENV uv run --no-sync python ...`), git.

**Full finding corpus:** `/tmp/serial_audit_findings.md` (serial + 3 reviewer reports). Each task below embeds its actionable items; consult the corpus for full per-line detail.

---

## GOVERNING RULES (apply to EVERY task)

1. **Clarity/pedagogy/correctness FIRST. "Trim" = consolidate genuine redundancy/drift/hedging, NEVER cut for length.** A shorter-but-thinner page is a regression. (Anna does not care about length.)
2. **Verify before editing.** Every code-snippet fix: read the real symbol in `src/` and confirm signature/behavior before rewriting. Every citation claim: read the per-paper note AND, for equation/table/coefficient claims, the actual PDF (`docs/core-papers/<f>.pdf` locally or `~/brain/knowledge/library/<bibkey>.pdf`). Never assert from memory or from a reviewer's summary.
3. **NEVER hand-edit generated pages:** `30-api/*.md` and `50-validation/test-dashboard.md`. Fixes go to docstrings / `scripts/build_test_dashboard.py` / the generator, then regenerate.
4. **Test counts:** on any *current-tense* page, replace hardcoded counts with a pointer to the [test dashboard](../50-validation/test-dashboard.md) / "see CI for the live count." Leave *dated changelog/audit* counts frozen (they are point-in-time records) but annotate if they read as current.
5. **Per-section verification gate (the "test"):** after edits, (a) `make build` → exit 0, **0 content warnings**; (b) run the link-integrity + count-drift script (Task 0) → 0 broken `.md` targets in touched files; (c) **run every code snippet you changed** and confirm it executes + matches any stated output.
6. **Commit per section.** Message trailer:
   `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
7. **Brain is read-only.** Do not write `~/brain`. The core-papers→brain migration is a SEPARATE deferred arc; here we only remove dead `docs/core-papers/` pointers.
8. **HITL.** Anna approves each section's scope before the subagent runs, and reviews between sections.

---

## Task 0: Scaffolding — link/count gate + hidden-index mechanism

**Files:**
- Create: `docs/website/scripts/check_links_and_counts.py`
- Create: `docs/website/_unlisted/index.md` (the hidden/unlisted index page)
- Reference: `docs/website/myst.yml`

**Step 1 — Link/count gate script.** Write `check_links_and_counts.py` that, given a list of changed `.md` files (or all): (a) extracts every `](...md)` link, resolves relative to the file, reports targets that don't exist; (b) flags hardcoded `N tests` / `N (released|unit|integration|validation)` patterns; (c) flags `/Users/`, `.claude-work`, `docs/core-papers/`, `docs/plans/`, `docs/notes/` strings. Exit nonzero on broken links. (This is the gate the clean build does NOT provide — MyST silently passes bad `.md` targets.)

**Step 2 — Verify the script reproduces the known 7 broken links** (6× `99-bibliography/index.md`, 1× `contributor-guide.md:28`).
Run: `python docs/website/scripts/check_links_and_counts.py` → Expected: lists exactly those 7 before fixes.

**Step 3 — Hidden-index page + myst mechanics.** Use **myst:myst-expert** to confirm: pages absent from `myst.yml` `toc` still build but drop from nav (reachable by URL). Create `_unlisted/index.md` as a navigable landing page that will link the OED-5, gravoturb-5, and (pending) dev-log pages. Decide with myst-expert whether to (a) leave moved pages out of `toc` entirely, or (b) add a collapsed/unlisted `toc` branch. Document the chosen mechanism at the top of `_unlisted/index.md`.

**Step 4 — Build + commit.**
Run: `cd docs/website && make build` → Expected: exit 0, 0 warnings.
Commit: `chore(docs): add link/count gate + hidden-index scaffold`.

---

## Task 1: Site-wide hygiene sweep

**Files (modify):** `90-development-log/{2026-02-12-imf-hmc-recovery,2026-02-13-binary-aware-imf-recovery-impl,2026-02-13-precision-scaling-panel}.md` (absolute paths — but note Task 10 curates dev-log; coordinate: do path-scrub here OR fold into Task 10. Default: fold dev-log path-scrub into Task 10, do the rest here); `50-validation/{engine-b-eddington.md:185,multimass-equilibrium.md:306}` (.claude-work); all pages with `docs/core-papers/` refs; pages with `docs/plans|notes/` refs.

**Actions:**
- Remove/replace `.claude-work/...` pointers (dead — gitignored) with a neutral phrase or a public design-doc link.
- Replace every `docs/core-papers/<f>.pdf` reference (dead — gitignored copyrighted PDFs) with the corresponding public per-paper note `../99-bibliography/per-paper/<bibkey>.md`. Hits incl: `50-validation/{imf-statistics:87,binary-imf:84,environment-imf:79,zams-relations:23}`, `king-profile:186`, others (run grep).
- Replace `docs/plans/`, `docs/notes/` website links (broken in deployed site + leak internal names) with a proper site cross-ref or drop. For GENERATED pages (`30-api/imf.md:510`, `30-api/binaries.md:812,835`) fix in the DOCSTRING source, not the page (Task 12).
- `whats-new.md:206` "docs/{plans,notes,specs,code-reviews}" — reword to not enumerate internal dirs.

**Gate:** link/count script clean for touched files; `make build` 0 warnings. **Commit:** `chore(docs): scrub dead/internal path references site-wide`.

---

## Task 2: Getting started (00) — onboarding correctness

**Files:** `00-getting-started/{installation,science-capabilities,whats-new,first-plummer-sphere,differentiable-ic,imf-sampling,glossary}.md`

**Verified fixes (already ground-truth-checked):**
- `installation.md` `[dev]` extra is wrong in TWO places. pyproject actual = `pytest, pytest-cov, pytest-xdist, ruff, mypy`. Fix line 24 ("black/isort/flake8" → `ruff`/`mypy`) AND lines 96–97 table (add ruff, mypy). Reconcile both to pyproject.
- `installation.md` smoke-test "Expected output" says `~0.95 pc`; **real run = 1.354 pc**. Fix to `~1.35 pc` (re-run to confirm).
- `installation.md:85` "GPU ~100× speedup at N~1e4" — unverified; soften or mark illustrative.
- `science-capabilities.md:384` "1163 tests (895/34/234)" current-tense → dashboard pointer.

**Also:** read `first-plummer-sphere`, `differentiable-ic`, `imf-sampling`, `glossary` in full; verify every code block runs (`uv run`); check glossary term definitions for correctness; fix any drift.

**Gate:** run every changed snippet; build 0 warnings; link script clean. **Commit:** `fix(docs): correct getting-started onboarding (install extras, smoke-test output, counts)`.

---

## Task 3: Theory — spatial profiles + velocity DFs

**Files:** `10-theory/spatial-profiles/{index,plummer,king,eff,lowered-model-family,multimass-equipartition}.md`, `10-theory/velocity-dfs/{index,plummer-dfs,king-dfs,michie-king,rotation-anisotropy}.md`

**CRITICAL (verify each vs `src/` then fix):**
- C5 `eff.md:68-72,236-237` — remove fabricated "enforces γ>3 ValueError" (no validation in `eff.py`; default γ=3.0). State finite mass holds for any γ via `r_t` truncation.
- C6 `eff.md:148-154` — remove miscited σ_r²(r) "EFF87 Eq.7" (EFF is surface-brightness only); velocities come from Eddington inversion (`eff_df.py`). **PDF-verify** EFF87 has no Eq.7.
- C7 `king.md:160-161,234,243-244` — `solve_king_profile` returns 3-tuple `(xi_grid, psi_clamped, psi_raw)` (verified). Fix the 2-tuple unpack (line 234 crashes).
- C8 `velocity-dfs/index.md:90-116` — delete fabricated `velocity_dispersion` method from the published `VelocityDF` protocol (real `protocols.py:56-84` has only `sample_velocities`).
- C9 `king-dfs.md:128,135-137` — remove non-existent `r_t=` arg from `KingVelocityDF` (constructor `king_df.py:106-113`; derived from W0). `KingVelocityDF(W0=7.0, r_c=1.0)`.

**IMPORTANT:** `spatial-profiles/index.md:9-13,48-49` EFF "param by r_h" wrong (EFF=(a,γ,r_t)); `eff.md:97-107` r_h→a provenance wrong; `michie-king.md:28` β factor-2 inconsistent w/ 3 sibling pages (standardize on single-component `1−σ_t²/σ_r²` or add Merritt reconciliation note); `rotation-anisotropy.md:90` EFF γ=3 sub-virial under equilibrium framing → use γ=5 or add caveat; `plummer-dfs.md` Q=0.5 "exactly" overstates (0.5±5e-3); `index.md:40` drop "production-grade" + move michie-king link to velocity-dfs row (**T1**).

**MINOR/cites:** `plummer.md:71-78` clarify 1.7×(a) vs 1.305×(r_h); `plummer.md:143,147` Plummer DF attribution (really Merritt1985 Eq.42/BT08 — PDF-verify); `plummer-dfs.md:203` Merritt bound Eq.46 not 45; `king.md:124` W₀ range vs index.

**Gate + commit:** `fix(docs/theory): correct profile & velocity-DF API snippets and claims`.

---

## Task 4: Theory — IMFs, binaries, tidal, populations

**Files:** `10-theory/imfs/{index,classic,multiplicity-statistics,mass-ratio-distributions,binary,binary-aware-likelihood,observation-operators,environment}.md`, `10-theory/binaries/{index,kepler-elements,period-distributions,eccentricity}.md`, `10-theory/tidal-and-substructure/{index,tidal,fractal,mass-segregation}.md`, `10-theory/populations/{index,eddington-engine,two-component}.md`

**CRITICAL (verify vs src then fix):**
- C1 `tidal.md:108-138` — `apply_tidal_truncation` documented BACKWARDS (verified: real returns 4-tuple `(positions, velocities, masses_truncated, keep_mask)`, length-N, `@jax.custom_jvp`, JIT/grad-safe, `tidal.py:128-184`). Rewrite to zero-mass/length-N/differentiable reality; real caveat = survivors keep untruncated velocities → super-virial.
- C2 `tidal.md:51-53,92-93,198-201` — `jacobi_radius` fabricated callable API; real `tidal.py:20-51` = point-mass `r_J=R(M_cl/3M_gal)^(1/3)`; reconcile factor-2 vs -3 with `jacobi_radius_isothermal`.
- C3 `eccentricity.md:27-29,90-124,144,163-171` — `MoeEccentricity` wrong (verified real = `e^η(P,M₁)`+Roche, `sample(key, periods, masses)` masses REQUIRED, only field `e_max`; `eccentricity.py:176-273`). Doc call crashes; delete `p_circ_d`.
- C4 `populations/two-component.md:36,66` — `sample_cluster` requires `G` (verified `multicomponent.py:713`). Add `from jaxstro.units import STELLAR`, `G=STELLAR.G`.
- C10 `imfs/index.md:3,17,46` — remove "IGIMF" as implemented (no class; honesty R7) → "environment-dependent IMF (Marks+12/Jeřábková+18 α₃ mapping)".
- C11 `imfs/multiplicity-statistics.md:179` — `progenax.imf._moe17_tables` doesn't exist → real `MoeDiStefano2017`/`imf/binary/`.

**IMPORTANT:** `kepler-elements.md:95-105,190-191` solver is `lax.scan` not for-loop/fori_loop; `max_iter=50` not 10/20; `ic-philosophy.md:211-213` inverted G ratio (write `G_planetary/G_stellar≈8800`); `imfs/environment.md:337` implies non-existent `EnvironmentIMF` class; `binary.md:241` vs `binary-aware-likelihood.md:189` GPU speedup conflict (reconcile to one measured figure); **T3** make `multiplicity-statistics`+`mass-ratio-distributions` canonical for f_b/γ/f_twin tables, `binary.md` cross-refs; populations `index.md:84` "same seed" framed as "independent codepaths".

**Cites to PDF-verify:** Jerabkova2018 "Eq.9/2.83/0.2161" (note has 6/7/8 only); Moe "Tables 10-13" + K/G f_b=0.44 + M-dwarf rows (note=Table13 from 0.8-1.2 f_b=0.40); Gieles2015 §4.1 vs §2.2/3.2; King1966 fill-factor (King has none); Cartwright2004 (D,Q) ladder vs Table1 (`fractal.md:118-140`); Hurley2002 τ_circ factor. DuquennoyMayor1991 needs a bib entry (→ Task 9).

**Gate + commit:** `fix(docs/theory): correct IMF/binary/tidal/populations API and citations`.

---

## Task 5: Theory — gravoturbulence consolidation 10→5, then hidden

**Files:** `10-theory/gravoturbulence/*` (10 pages) → 5; then drop from public `toc`, add to `_unlisted`.

**Consolidation (clarity, NOT length — preserve all teaching depth):**
- G1: merge `density-pdf-fundamentals` + `freefall-density-factor` + `pdf-and-fdf` → `density-pdf-and-fdf.md`. Resolve the f-sfr normalization conflict (`freefall-density-factor.md:126-128` dimensional vs `pdf-and-fdf.md` dimensionless); state α↔p mapping once.
- G2: merge `pp20` + `cored-profiles` + `direct-3d-zeta` → `magnification-factor.md`. One "which ζ-mode when" table. **Fix the 3 stale/fabricated code blocks (C12)** against `gravoturb_fdf/theory/pp20.py` (no `P_MAX=1.95` clip; real `zeta_fdf_direct` formula). Fix `ζ(1.9)=4.93`→**4.44**.
- G3: merge `differentiable-inference` + `projected-beta-inference` → `inference.md`. Fix C13 (`pot_validity_barrier` import — use real symbol or remove); state Gaussianization once; de-orphan.
- G4: keep `bm19.md` (trim α↔p re-derivation) + `index.md` (one canonical "experimental, repo-only, NOT in released wheel" banner; fix `pp20` "implemented in progenax.magnification_factor" → `gravoturb_fdf.theory.pp20`).
- Replace private validation-script names (`projected-beta-inference.md:23-24`) with a validation-page reference.

**Then:** set the gravoturb children to `hidden: true` in `myst.yml` toc (do NOT delete them — this site builds ONLY toc-listed pages, so removal = unbuilt/unreachable; confirmed Task 0). Link the 5 from `_unlisted/index.md`. Fix inbound refs from other theory pages / `index.md:48` (drop "FDF method" billing, **T2**).

**This task also OWNS `10-theory/index.md`** (the shared theory landing, deferred by Task 3): **T1** move the `velocity-dfs/michie-king.md` link from the spatial-profiles row to the velocity-DFs row; drop unearned "production-grade" at `index.md:40`; reframe the `:48` tidal/substructure row as theory-only and drop the removed-"FDF method" capability billing.

**Gate:** all merged-page code blocks RUN against `gravoturb_fdf` (PYTHONPATH=src:src/experimental); build 0 warnings; link script clean. **Commit:** `refactor(docs/gravoturb): consolidate 10→5, fix stale code, move to unlisted`.

---

## Task 6: Architecture (20)

**Files:** `20-architecture/{index,jax-native-philosophy,three-brick-state,protocols,differentiability,units-policy,q-virial-convention,jax-native-substructure-q,contributor-guide,ic-redesign-history}.md`

**Verified fix:** `contributor-guide.md:28` broken link `rotation-anisotropy.md` → `../10-theory/velocity-dfs/rotation-anisotropy.md`.

**Also:** read all 9 sub-pages (not agent-covered); verify code snippets run; check `differentiability.md` roadmap anchor `#roadmap-differentiable-rt` exists (referenced by science-capabilities + lowered-model-family); `differentiability.md:334` docs/plans link (Task 1 class); confirm `units-policy`/`q-virial` correctness (q-virial already read — clean). Verify the King-`r_t` differentiability statement here is consistent with the Task 7 reconciliation.

**Gate + commit:** `fix(docs/architecture): repair cross-refs and verify snippets`.

---

## Task 7: Validation (50) — drift, consolidation, honesty

**Files:** `50-validation/*` EXCEPT `test-dashboard.md` (generated — read-only reference).

**CRITICAL:**
- Replace `__all__` literals with dashboard pointer: `testing-architecture.md:67,110` (114/112 → live 122), `differentiability-audit.md:553,581` (104 → 122; this is the page's own load-bearing invariant — reconcile carefully, it's hand-written/editable). `testing-architecture.md:112` line-cov 94.77% → dashboard (96.1%).
- `.claude-work` dead pointers: `multimass-equilibrium.md:306`, `engine-b-eddington.md:185` (also Task 1).
- `audit-report.md:13` "866 tests" → mark superseded / drop absolute.

**Per-suite test-count drift (10 pages — replace "N tests" with dashboard pointer; live counts for reference):** king-profile 32→35, eff-profile 23→24, michie-anisotropy 12→10, imf-statistics 25→23, binary-imf 24→19, environment-imf 12→15, analytical-test-cases 12→28, fractal-substructure 9→8, rotation-om-anisotropy 10→8, mass-segregation 8→9. (Correct, leave: plummer 20, tidal 9, zams 34, multimass 6, engine-b 6.)

**Honesty:**
- `physics-tests.md` (**V3**) — admits the test files "do not exist in this checkout" yet shows PASS tables + a literal "…" placeholder row. Rewrite as "covered indirectly by listed unit/integration tests" OR make real OR fold into methodology/testing-architecture.
- King-`r_t` CONTRADICTION + SITE-WIDE RECONCILIATION: `king-profile.md` says "blocked/zeroed by argmax"; `differentiability-audit.md:34-38,154` says fixed/differentiable. **VERIFIED (Task 6, independently re-confirmed): r_t IS differentiable** — `jax.grad` of `KingProfile.from_W0_rc(W0).r_t` gives AD=22.88 matching FD to 3.6e-8 (src returns unclamped `psi_raw` so the zero-crossing carries d(xi_t)/dW0). Fix the stale "deferred/not differentiable" claim to "RESOLVED/differentiable" on ALL THREE pages: `50-validation/king-profile.md:178-186`, AND (cross-section, Tasks 2/3 left them stale) `00-getting-started/science-capabilities.md:366-369` and `10-theory/spatial-profiles/lowered-model-family.md:225-228`. (`20-architecture/differentiability.md` already fixed in Task 6.)
- `imf-statistics.md:9` TaperedPowerLaw/Schechter called "released" but not in `__all__` → mark internal.
- `engine-b-eddington.md` evidence-page self-mismatch: 18-seed (0.4947±0.0014, L126) vs 3-seed (0.4917±0.0062, L55) — reconcile (theory pages cite the 18-seed figure).

**Consolidation (V2):** tolerance table 2× (index:114-141 + methodology:43-66), 3-tier table 3× (index/testing-arch/methodology), anchor-lesson 3×. Target: `index`=thin map + "what validated means"; `methodology`=tiers+tolerances+anchor-lesson; `testing-architecture`=registries+gate+dashboard.

**Reorg:** **V1** add `performance-memory`, `zams-relations`, `environment-imf` to `index.md` map. **V4** demote `gravoturbulent-pp20` + fractal *generation* content → unlisted (with gravoturb); keep CW04-Q diagnostic core. Cross-ref fixes: `audit-report.md:11` "dashboard" → `test-dashboard.md`; `plot-gallery.md:62` mass-seg figures target; convert bare-text cites (`rotation-om-anisotropy.md:148-149`, `tidal-truncation.md:165`) to `{cite}` keys.

**Gate + commit:** `fix(docs/validation): defer counts to dashboard, consolidate backbone, fix honesty issues`.

---

## Task 8: Science demos (60) — fixes + OED trim

**Files:** `60-science-demos/*` + `optimal-design/*`

**CRITICAL:**
- C1 `binary-dynamical-mass.md:3` frontmatter "~24%" → "~28%" (body/index say 1.28×).
- C2 `anisotropy.md:167` link text "kinematics" → repoint/relabel (target is "Multi-component populations"; real kinematics page = velocity-dfs/rotation-anisotropy).
- C3 `dynamical-mass.md:239,350` "fluxax" → progenax URL (wrong repo) — fix/drop.

**IMPORTANT:** I2 `background.md:49` "3.7×" → 3.66×; I3 strip internal jargon ("informax-bound","held out of v0.1.0","released-core registry","ADR-0016/0017") from any surviving public page; I4 OED `index.md:138-141` T-optimality "[enabled — high value]" → "design sketch / not built"; "referee-proof" (`binary-robustness.md:3,42`) → "referee-resistant".

**OED trim (D1):** condense `optimal-design/index.md` to the single public overview — KEEP the telescope-time hook + `tbl-oed-examples` (4 headline numbers) + 1-para "any differentiable forward model + likelihood = Fisher = OED"; CUT the `[done]/[B#]/[enabled]` matrices → one honest sentence "prototyped; OED tooling not in v0.1.0"; cut E/I/G/T speculation. Move `background, anisotropy, concentration, dynamical-mass, binary-robustness` → `_unlisted` (prune `myst.yml:199-204` to keep only `optimal-design/index.md`). Verify NO public demo page `{ref}`s a now-hidden OED label (only links to `optimal-design/index.md`). Audit the 5 hidden pages for correctness too (C3/I2/I3 apply there).

**Redundancy (D2):** make `throughline.md` canonical for the "shared method" 4-step recipe; `index.md` links to it.

**Verify:** demos cite Tout1996 + Kuepper2011_McLuster (per-paper notes created in Task 9). Confirm birth-environment IGIMF wording doesn't imply galaxy-wide IGIMF.

**Gate + commit:** `fix(docs/demos): number fixes, OED trim to overview, jargon strip`.

---

## Task 9: Bibliography (99) — missing notes, broken links, PDF verification

**Files:** `99-bibliography/{bibliography,ecosystem-papers,per-paper/*}.md`, `docs/website/references.bib`

**Broken-link fix:** the 6 `[](../99-bibliography/index.md)` links (Task 1/7 pages) — decide target: create `99-bibliography/index.md` OR repoint all to `bibliography.md`/`per-paper/index.md`. (Recommend: create a real `99-bibliography/index.md` landing, add to toc, since multiple pages expect it.)

**f_b table reconciliation (Task 4 review):** on the now-canonical `multiplicity-statistics.md`, the f_b table bins (solar 0.44/0.50, O 0.90) differ slightly from the page's own Moe Table 13 prose (solar 0.40, O 0.94) due to different mass binning. Not fabricated, but add a one-line note reconciling the table vs the Table-13 derivation.

**Missing per-paper notes (create + PDF-verify):** `tout-1996.md` (ZAMS; cited by B4/B12/dyn-mass/bin-robust), `kuepper-2011-mcluster.md` (B9 scale-separation), `king-1962.md` (Jacobi r_t), and notes for `Kritsuk2011` + `TanKrumholzMcKee2006` + `Hurley2002` (gravoturb/binaries). **Add bib entry for `DuquennoyMayor1991`** (LogNormalPeriod μ=4.8,σ=2.3 traces to it; currently no entry).

**PDF-verify the flagged claims** (read `~/brain/knowledge/library/<bibkey>.pdf` or `docs/core-papers/`): EFF87 "Eq.7" (likely none), Plummer1911 DF attribution, Jerabkova "Eq.9/2.83/0.2161", Moe "Tables 10-13"+f_b rows, Gieles2015 §, King1966 fill-factor, Cartwright2004 Table1 (D,Q), Aarseth1974 (note missing but resolves). Fix or correct each citing page.

**Reword residual `docs/core-papers/` provenance strings** (flagged in Task 1 review — these are dead internal paths to a public reader). In `99-bibliography/per-paper/*.md` (strigari, kainulainen, marks-2012, bianchini, baumgardt-makino, peuten, maschberger-clarke, allison) the "Held PDF: `docs/core-papers/X.pdf`" lines and `ecosystem-papers.md:3,8` "stored locally in `docs/core-papers/`" → reword to "verified against the published PDF" (drop the gitignored internal path; do NOT point at `~/brain`, which isn't public). Goal: after Task 9, the only remaining `docs/core-papers/` string anywhere is `audit-report.md:23` (Task 11's file).

**Gate + commit:** `fix(docs/bib): add missing notes + bib entry, repair bib index link, PDF-verify claims`.

---

## Task 10: Development log (90) — curate, modernize, scrub

**Files:** `90-development-log/*` (13)

- **Scrub** all `/Users/anna` absolute paths + conda activation path (`2026-02-12-imf-hmc-recovery`, `2026-02-13-binary-aware-imf-recovery-impl`, `2026-02-13-precision-scaling-panel`).
- **Curate:** merge/retire stale entries; keep genuinely-useful decision records. Make every note + decision CURRENT (Anna's directive).
- Fix stale test counts framed as current; fix internal-path leaks (`code-reviews.md`).
- **Placement:** prepare for hidden/unlisted (likely) — move to `_unlisted` link set; **leave the public-vs-hidden toggle as the final decision for Anna after she reviews the curated result.**

**Gate + commit:** `refactor(docs/dev-log): curate, modernize to current state, scrub paths`.

---

## Task 11: Release section (95) — rewrite to current decisions

**Files:** `95-release/{index,checklist,release-strategy,audit-report}.md`

- `checklist.md:16-19` remove stale "nothing actioned yet" note (contradicts the [x] items).
- `checklist.md:51-53` over-claims internal-doc links fixed / "0 broken xrefs" — reconcile with reality (this arc actually fixes them).
- `release-strategy.md:93-102` rewrite to THIS session's decisions: gravoturb → unlisted/hidden (not just "labelled experimental"); OED → 1 public overview + 5 hidden (not "drop optimal-design entirely"); dev-log → curated + likely hidden.
- `audit-report.md` — relabel as a frozen snapshot at commit 24cb6b9 where it reads as current (e.g. the "six empty TBD stubs" howto claim is stale); add "(superseded — see test-dashboard)" to absolute counts.

**Gate + commit:** `docs(release): reconcile release pages with v0.1.0 docs-hardening decisions`.

---

## Task 12: API (30) — docstring-source fixes only

**Files:** `src/progenax/**` docstrings (NOT `30-api/*.md`), then regenerate.

- `30-api/imf.md:510`, `30-api/binaries.md:812,835` `docs/notes/...` leaks → fix in the source docstrings.
- `30-api/analytical.md:338` "placeholder for future external potential support" → tighten the docstring or remove if not real.
- Regenerate API pages; confirm the leaks are gone and pages build clean.

**Gate + commit:** `fix(src): clean docstring-sourced doc leaks; regen API pages`.

---

## Task 13: Whats-new changelog + hidden-index finalize + final whole-site review

- `whats-new.md` — bring the HEAD current: add the post-2026-06-10 arcs (registry-harness-hoist 2026-06-20, ZAMS migration, OED binary-misspecification) from the dev-log; keep historical dated counts frozen but stop the changelog reading as stale. HEAD count → dashboard pointer.
- Finalize `_unlisted/index.md`: confirm it links OED-5 + gravoturb-5 (+ dev-log if Anna chose hidden).
- **Final whole-site review** (independent reviewer): `make build` 0 warnings; run `check_links_and_counts.py` over ALL pages → 0 broken; spot-run a sample of code snippets; confirm no `/Users/`, `.claude-work`, `docs/core-papers` strings remain; TOC coherent; honesty sweep ("production-grade"/"fully"/"state-of-the-art"/"production-ready" audit).
- Present the curated dev-log + final site to Anna for the public-vs-hidden dev-log decision and merge/push go-ahead.

**Commit:** `docs: update changelog, finalize unlisted index, whole-site verification`.

---

## Execution order & dependencies

Task 0 → 1 first (scaffolding + hygiene). Then 2,3,4,6,8,9 (page fixes, parallel-safe but run sequentially with review between). 5 (gravoturb) before 13. 7 (validation) before 13. 11 (release) reflects decisions from 5/8/10. 12 (API) independent. 13 last. Bibliography (9) before 8's verify of Tout/Kuepper cites — sequence 9 before or alongside 8.

**Nothing merged or pushed without Anna's separate word (merge and push are separate words). Delete the branch after merge + push.**

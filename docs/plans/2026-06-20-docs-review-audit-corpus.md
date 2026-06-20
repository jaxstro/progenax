# Serial audit findings (Claude, by-hand) — progenax docs hardening

Branch: feat/docs-review-hardening. Build: `make build` exit 0, **0 content warnings, 172 pages**.
KEY: MyST does NOT validate relative .md link targets → broken file-links ship silently. The mechanical sweep is the real link gate.

## VERIFIED DEFECTS (ground-truth-checked)

### Broken internal links (build passes but links are dead)
- 6× `[](../99-bibliography/index.md)` — file does NOT exist (only bibliography.md, ecosystem-papers.md, per-paper/index.md). Hits:
  - 50-validation/michie-anisotropy.md:198
  - 50-validation/environment-imf.md:82, 186
  - 50-validation/imf-statistics.md:97, 201
  - 50-validation/multimass-equilibrium.md:300
  → repoint to ../99-bibliography/bibliography.md (or per-paper/index.md)
- 20-architecture/contributor-guide.md:28 → `rotation-anisotropy.md` resolves to 20-architecture/rotation-anisotropy.md (missing); intended ../10-theory/velocity-dfs/rotation-anisotropy.md

### installation.md (00-getting-started)
- [dev] extra WRONG. pyproject actual: pytest, pytest-cov, pytest-xdist, ruff, mypy.
  - line 24 says "pytest, black, isort, flake8, mypy" (black/isort/flake8 gone → ruff)
  - lines 96-97 table says "pytest, pytest-cov, pytest-xdist" (missing ruff, mypy)
  → reconcile BOTH to pyproject
- Smoke-test expected output WRONG: page says "~0.95 pc"; actual run = **1.354 pc**. Fix number.
- line 85: "GPU ~100× speedup at N~1e4" — unverified perf claim; soften or verify.

## TEST-COUNT DRIFT (point to dashboard / "see CI")
- 00-getting-started/science-capabilities.md:384 — "1163 tests (895/34/234)" CURRENT-tense → dashboard
- 00-getting-started/whats-new.md — HEAD entry (2026-06-10) says 1163; changelog STOPS at 2026-06-10, missing ~10 days of arcs (registry-harness-hoist 2026-06-20, ZAMS migration, OED binary-misspec). Historical dated counts OK to freeze; bring HEAD current + add recent arcs.
- 50-validation/audit-report.md:13 — "866 tests" (frozen at commit 24cb6b9)
- reality ~1243 released-core / 1561 full gate (per 95-release/checklist.md:85)

## 95-release section reconciliation (encodes OLD plan; must update to THIS session's decisions)
- checklist.md:16-19 "nothing below has been actioned yet" CONTRADICTS the many [x] (done 2026-06-18) items. Stale note.
- checklist.md:51-53 claims internal-doc links fixed & "myst build 0 broken xrefs" — FALSE: broken bib links exist + docs/notes refs remain in generated API pages.
- release-strategy.md:93-102 says gravoturb "labelled experimental throughout docs" & "Drop optimal-design/ from TOC" → must update to NEW decisions: gravoturb→hidden index; OED→1 public overview + 5 hidden; dev-log→curate + likely hidden.
- audit-report.md:198 "40-howto = six empty TBD stubs" STALE (howtos now full 106-133 lines). audit-report is a frozen 24cb6b9 snapshot — reconcile/ relabel as historical.

## Hygiene leaks
- Absolute /Users/anna paths: 90-development-log/{2026-02-12-imf-hmc-recovery, 2026-02-13-binary-aware-imf-recovery-impl, 2026-02-13-precision-scaling-panel}.md (also a conda activate path)
- .claude-work refs in published pages: 50-validation/engine-b-eddington.md:185, 50-validation/multimass-equilibrium.md:306 (checklist.md hit is meta, fine)
- Internal planning-file names exposed: 20-architecture/differentiability.md:334 (docs/plans/...), 30-api/imf.md:510 + 30-api/binaries.md:812,835 (docs/notes/... — GENERATED → fix in docstrings), whats-new.md:206 (docs/{plans,notes,specs,code-reviews})

## Decided dispositions (this session)
- Governing principle: clarity/pedagogy/correctness FIRST; "trim"=consolidate redundancy/drift, NEVER cut for length.
- Public surface: 00,10(−gravoturb),20,30,40,50(−gravoturb val),60(−OED detail +1 overview),95,99 + hidden index.
- Hidden/unlisted index collects: dev-log (likely), 5 OED detail pages, gravoturbulence subsection (theory+val).
- dev-log: curate+modernize (current notes/decisions), scrub paths; public-vs-hidden decided after Anna reviews.
- OED: 1 public overview + 5 hidden; fix inbound xrefs (60/index B14 row, throughline).
- gravoturb: consolidate for clarity (not length) + clear "experimental, repo-only, not in wheel" banner + hidden.
- Generated (30-api/*, 50-validation/test-dashboard.md): never hand-edit; fix docstrings/generator.
- Reorg: itemized proposals in plan, per-item approval.

## Pages read in full (serial): index.md(root), 00/index, 00/science-capabilities, 00/whats-new, 00/installation, 10/spatial-profiles/plummer, 20/index, 20/q-virial-convention, 40/index, 40/interface-with-gravax, 60/index, 60/optimal-design/index, 60/anisotropy, 95/{index,checklist,release-strategy}
## Still to read in execution: 00/{first-plummer,differentiable-ic,imf-sampling,glossary}, 20/{jax-native-philosophy,three-brick-state,protocols,differentiability,units-policy,jax-native-substructure-q,contributor-guide,ic-redesign-history}, 40 recipes (3), 99 bibliography pages

## ===== AGENT REPORT: 60-science-demos =====
CRITICAL:
- C1 binary-dynamical-mass.md:3 frontmatter "~24%" contradicts body/index "1.28× (28%)" (:154,180,200,238; index:70) → fix to ~28%
- C2 anisotropy.md:167 link text "kinematics" → target ../10-theory/populations/index.md (titled "Multi-component populations"). Relabel or repoint to velocity-dfs/rotation-anisotropy.md
- C3 dynamical-mass.md:239,350 link text "fluxax" → progenax repo URL (wrong repo). Fix/drop (OED hidden page but still)
IMPORTANT:
- I1 MISSING per-paper notes: Tout1996 (ZAMS; B4,B12,dyn-mass,bin-robust) and Kuepper2011_McLuster (B9 central). Create notes + PDF-verify.
- I2 anisotropy star-factor drift: canonical 3.66× (fixed-N), 3.9× (frontier), but background.md:49 says "3.7×" → use 3.66.
- I3 internal jargon leaks ("informax-bound","held out of v0.1.0","released-core registry","ADR-0016/0017") in concentration.md:4,249,476,511 + binary-robustness.md:3,4,538,556 → strip from any surviving public page
- I4 OED index:138-141 T-optimality "[enabled — high value]" overstates (code held out) → "design sketch/not built"
MINOR: M1 binary-robustness 9.0 vs 8.98 rounding; King1962 Jacobi per-paper note check (only king-1966 exists)
INPUTS-STANDARD: all 12 main demos COMPLIANT (✓ all 4 elements). Only OED dynamical-mass.md lacks inputs *table* (prose) — it's going hidden, acceptable.
NUMBER-CONSISTENCY across index/throughline/demo: all ✓ EXCEPT C1 (B12) and I2 (3.7 in background).
REDUNDANCY: "shared method" 4-step recipe triplicated in index.md:85-145, throughline.md:8-23, + each OED page → make throughline canonical, index links to it.
HONESTY: excellent overall. birth-environment.md:158 IGIMF wording OK (confirm not implying galaxy-wide IGIMF — R7). binary-robustness "referee-proof" (:3,42) → soften to "referee-resistant".
OED TRIM PLAN: public overview KEEPS hook + tbl-oed-examples (4 headline numbers) + 1-para "any diff fwd model+likelihood=Fisher=OED". CUTS the [done]/[B#]/[enabled] matrices (roadmap-ish, leans on held-out code) → 1 honest sentence; cut E/I/G/T speculation. PRIMARY STRUCTURAL EDIT = prune myst.yml:199-204 to keep only optimal-design/index.md. No PUBLIC demo page {ref}s a hidden OED label (only link to optimal-design/index.md) → trim is clean. 5 detail pages move together to hidden dir (intra-links survive).
HYGIENE: no abs paths, no .claude-work, no test counts in demos. fluxax URL (C3) only external-link defect. OED figures/ ship .png+.pdf+run_record.json — decide whether to keep PDFs if hidden.

## ===== AGENT REPORT: 50-validation =====
HEADLINE: dashboard (test-dashboard.md, generated) + live repo disagree w/ nearly every hand-written count. live len(__all__)==122.
CRITICAL:
- testing-architecture.md:67 "114 public symbols", :110 "112+2 EXEMPT=114" → live 122. :112 "line-cov 94.77%" → dashboard says 96.1%. Replace literals w/ dashboard pointer.
- differentiability-audit.md:553,581 "104 __all__ symbols" + describes keys-equality invariant set(SYMBOL_CATEGORY)==set(__all__) → live 122. GATE-CRITICAL reconcile (page is hand-written/editable, not generated).
- audit-report.md:13 "866 tests" stale → mark superseded / drop absolute.
- multimass-equilibrium.md:306 + engine-b-eddington.md:185 = .claude-work dead pointers (confirmed, no others).
PER-SUITE TEST-COUNT DRIFT (10 of 14 physics pages WRONG; live via collect 2026-06-20):
- king-profile.md:10,192 says 32 → 35; eff-profile.md:9,161 says 23 → 24; michie-anisotropy.md:9,182 says 12 → 10; imf-statistics.md:11,183 says 25 → 23; binary-imf.md:8,156 says 24 → 19; environment-imf.md:11 says 12 → 15; analytical-test-cases.md:9,155 says 12 → 28; fractal-substructure.md:9,207 says 9 → 8; rotation-om-anisotropy.md:10,136 says 10 → 8; mass-segregation.md:10 says 8 → 9.
- CORRECT (leave): plummer 20✅, tidal 9✅, zams 34✅, multimass 6✅, engine-b 6✅.
- index.md:73 + gravoturbulent-pp20.md:150 "35 tests" = REMOVED PP20 tests → reword as history.
- FIX CONVENTION: replace "N tests" headers w/ "see test dashboard for live count".
HONESTY:
- physics-tests.md:7-9 admits test files "do not exist in this checkout" yet :13-76 shows PASS tables + a literal "…" placeholder row (:67-71). Reframe as "covered indirectly by listed unit/integration tests" or make real. WEAKEST PAGE.
- king-profile.md says r_t "blocked/zeroed by argmax" BUT differentiability-audit.md:34-38,154 says r_t FIXED (differentiable). CONTRADICTION — reconcile (audit page = newer truth).
- imf-statistics.md:9 calls TaperedPowerLaw/Schechter "released" but NOT in __all__ → mark internal.
REDUNDANCY (confirmed substantial): tolerance table 2× (index:114-141 + methodology:43-66); 3-tier table 3× (index:97-108, testing-arch:30-49, methodology:20-35); "how suite runs" 2×; "anchor on defining condition" lesson 3×. PLAN: index.md=thin map+pointer; methodology.md owns tiers+tolerances+anchor lesson; testing-architecture.md owns registries+gate+dashboard; live counts ONLY in test-dashboard.md.
CROSS-REF: audit-report.md:11 "live status dashboard" → links index.md (should be test-dashboard.md). plot-gallery.md:62 mass-seg figures → theory page (likely wrong, should be validation page). bare-text cites (not {cite}): rotation-om-anisotropy.md:148-149, tidal-truncation.md:165 → convert to keys.
INDEX MAP MISSING: performance-memory.md, zams-relations.md, environment-imf.md absent from index.md "Map of section".
CITES TO VERIFY: Aarseth1974 (no per-paper note, resolves), King1962 (tidal, no note), B&T2008/Lynden-Bell1960 (bare text), Kainulainen2014 ζ≈1.789 (marquee, experimental page), Cartwright2004 Table1 (0.79/0.84/0.93 load-bearing), Moe2017 Table13.
HYGIENE Q: docs/core-papers/ (imf-statistics:87, binary-imf:84, environment-imf:79, zams:23) + docs/plans/ + docs/notes/ referenced — VERIFY these ship publicly in source repo or they're dead-pointer class like .claude-work. src/experimental/VALIDATION_SUMMARY.md refs OK if experimental ships.
REORG: demote gravoturbulent-pp20 + fractal generation content (🔬 experimental → hidden); keep CW04-Q diagnostic core. Fold/rewrite physics-tests.md.

## ===== HYGIENE: internal doc dirs (git-checked) =====
- docs/core-papers/ : GITIGNORED (.gitignore:219), 0 tracked, = copyrighted paper PDFs → NEVER ships. ALL refs are DEAD public pointers (imf-statistics:87, binary-imf:84, environment-imf:79, zams:23, + king-profile, 95-release/audit-report:198-area). REDIRECT to per-paper notes (which ship).
- docs/plans/ (86 tracked), docs/notes/ (14), docs/specs/ (1), docs/code-reviews/ (1): SHIP in source repo BUT live OUTSIDE docs/website/ → links broken in DEPLOYED site + leak internal planning filenames. Remove/convert to site xrefs or GitHub URLs. Hits incl: 20-architecture/differentiability.md:334, 30-api/imf.md:510 + binaries.md:812,835 (GENERATED→docstrings), king-profile.md:186, multimass-equilibrium.md:303-305, mass-segregation.md:522,556, whats-new.md:206.
- .claude-work/ : GITIGNORED (.gitignore:212) → dead pointers. (engine-b-eddington:185, multimass-equilibrium:306)

## ===== AGENT REPORT: 10-theory =====
DOMINANT DEFECT CLASS: stale/fabricated API in runnable code snippets — crash, invert reality, or use non-existent symbols. ALL must be VERIFIED against src/ before editing (agent lead, not fact).
CRITICAL (verify each vs src then fix):
- C1 tidal.md:108-138 apply_tidal_truncation documented BACKWARDS (says boolean-remove/len≤N/not-JIT; real tidal.py:128-184 = zero-mass, len N, custom_jvp, JIT/grad-safe). Real caveat: survivors keep untruncated v → super-virial.
- C2 tidal.md:51-53,92-93,198-201 jacobi_radius API fabricated (claims M_gal_enclosed_func callable; real tidal.py:20-51 = point-mass r_J=R(Mcl/3Mgal)^1/3).
- C3 eccentricity.md:27-29,90-124,144,163-171 MoeEccentricity wrong (doc=δ+thermal blend+p_circ_d+sample(periods,key); real eccentricity.py:176-273 = e^η(P,M1)+Roche, sample(key,periods,masses) masses REQUIRED, only field e_max). Doc call CRASHES.
- C4 two-component.md:36,66 sample_cluster crashes (G REQUIRED, multicomponent.py:713,744) + violates own units policy. Add G=STELLAR.G.
- C5 eff.md:68-72,236-237 fabricated "enforces γ>3 ValueError" (eff.py has no validation; default γ=3.0).
- C6 eff.md:148-154 miscited σ_r²(r) "EFF87 Eq.7" (note: EFF is surface-brightness only, no Eq.7). velocities from Eddington inversion.
- C7 king.md:160-161,234,243-244 solve_king_profile shown 2-tuple; real 3-tuple (xi,psi_clamped,psi_raw) king.py:210-222. Line 234 CRASHES.
- C8 velocity-dfs/index.md:90-116 fabricated velocity_dispersion method in VelocityDF protocol (real protocols.py:56-84 only sample_velocities).
- C9 king-dfs.md:128,135-137 passes non-existent r_t= to KingVelocityDF (king_df.py:106-113 has no r_t). TypeError.
- C10 imfs/index.md:3,17,46 lists "IGIMF" as implemented (no IGIMF class; R7 honesty). Reword to env-dependent α₃ mapping.
- C11 imfs/multiplicity-statistics.md:179 progenax.imf._moe17_tables DOES NOT EXIST.
- C12 gravoturb pp20.md:188-200, cored-profiles.md:75-87, direct-3d-zeta.md:94-104: 3 load-bearing code blocks stale/fabricated vs gravoturb_fdf/theory/pp20.py (nonexistent P_MAX=1.95 clip; wrong sigs/formula).
- C13 gravoturb differentiable-inference.md:228 import pot_validity_barrier — symbol doesn't exist.
IMPORTANT (sel): spatial-profiles/index.md:9-13,48-49 EFF "param by r_h"/protocol "r_h always exposed" wrong (EFF=(a,γ,r_t)); michie-king.md:28 β factor-2 inconsistent w/ 3 sibling pages; kepler-elements.md:95-105 shows for-loop/fori_loop but real lax.scan, max_iter=50 not 10/20; ic-philosophy.md:211-213 inverted G ratio (8800 vs 1.1e-4); imfs/environment.md:337 implies EnvironmentIMF class exists (it doesn't); binary.md:241 vs binary-aware-likelihood.md:189 GPU speedup 100-1000× vs ~70×; index.md:40 "production-grade" + michie-king mislisted under spatial row; index.md:48 TOC bills removed "FDF method".
CITES TO PDF-VERIFY: ElsonFallFreeman1987 "Eq.7" (likely none), Plummer1911 DF attribution (really Merritt1985 Eq42/BT08), Jerabkova2018 "Eq.9/2.83/0.2161" (note has Eqs6/7/8 only), MoeDiStefano2017 "Tables 10-13" + K/G f_b=0.44 + M-dwarf rows (note=Table13 starts 0.8-1.2 f_b=0.40), Gieles2015 §4.1 vs §2.2/3.2, King1966 fill-factor F=0.05-0.3 (King has none), Cartwright2004 (D,Q) ladder conflicts Table1, MISSING notes Kritsuk2011 + TanKrumholzMcKee2006 + Hurley2002. Also DuquennoyMayor1991 has NO bib entry (LogNormalPeriod μ=4.8 σ=2.3).
REDUNDANCY: GRAVOTURB 10→5 merge: index; density-pdf-and-fdf (←fundamentals+freefall+pdf-and-fdf, fixes f-sfr norm conflict); magnification-factor (←pp20+cored-profiles+direct-3d-zeta, fixes 3 stale code blocks); bm19 (keep); inference (←differentiable-inference+projected-beta). IMF: f_b/γ/f_twin tables verbatim in multiplicity-statistics+binary+mass-ratio → multiplicity/mass-ratio canonical, binary cross-refs; Maschberger form dup classic+binary.
CROSS-REF: gravoturb pp20.md:30,216 "cored profiles"→pdf-and-fdf (should cored-profiles.md); pp20.md:221 soft-mask→pdf-and-fdf (should direct-3d-zeta); direct-3d-zeta.md:89→binary-imf (wrong); projected-beta-inference orphaned (in myst.yml:112 never linked).
NUMERIC: pp20.md:171 ζ(1.9)=4.93 WRONG (=4.44). engine-b-eddington.md evidence-page 18-seed (0.4947±0.0014, L126) vs 3-seed (0.4917±0.0062, L55) mismatch cited by theory pages.
HYGIENE: no abs paths/no .claude-work. projected-beta-inference.md:23-24 cites private validation script names (_d0*,_v3_logp_sbc etc) → replace w/ validation-page ref.

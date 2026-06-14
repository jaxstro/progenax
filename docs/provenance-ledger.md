# Provenance & Credibility Ledger — progenax release-prep

**Arc:** `feat/provenance-credibility-audit` ·
**Design:** [`docs/plans/2026-06-14-provenance-credibility-audit-design.md`](plans/2026-06-14-provenance-credibility-audit-design.md) ·
**Started:** 2026-06-14

One row per audited scientific claim / constant / citation →
`source (paper + eq/table/page)` → **verdict**:

| verdict | meaning |
|---------|---------|
| ✅ verified | checked against the held PDF (or derivable identity); correct as-is |
| 🔧 fixed | corrected in place; the commit names the source PDF |
| ⚠ flagged | needs Anna's scientific judgment / adjudication |
| 📄 needs-fetch | depends on a paper not held — fetch before verifying |
| 🗑 dead-removed | stale/dead code or doc retired (Anna-approved) |

> **Discipline (locked, design D1–D4):** no scientific value changes without a **held PDF**
> justifying it (never from memory or a review agent); `needs-fetch` is a first-class verdict;
> Anna adjudicates every ⚠ and approves every 🗑 before it lands; `make build` stays 0 warnings;
> the released-core gate stays green after any consolidation deletion.

---

## Phase-1 triage summary (inventory + risk-score)

17 read-only agents over 3 surfaces (citations↔notes, src/ constants, doc claims).
**270 findings — HIGH 67 (released 41 · docs 15 · experimental 11) · MED 137 · LOW 66.**
Verdict-hints: verify-against-pdf 145 · spot-sample 47 · needs-fetch 37 · likely-fine 37.

**Triage verdict: no systemic rot → stay triage-first (D3).** The paper-grounding batches held;
the classic fabrication fingerprints are mostly already-fixed (fake LIMEPY g=1 table removed;
debunked King Table II *not* present; Aarseth1974 figure-eight mis-cite already corrected; README
phantom classes corrected). HIGH items fall into three buckets:

- **A — live phantom-API / stale-doc contradictions** (grep-verifiable, no PDF needed; Anna: *rewrite to real API*)
- **B — held-PDF deep-verify** (PDFs in hand)
- **C — cited-but-not-held** (needs-fetch)

Full 270-row inventory: `.claude-work/provenance-inventory-2026-06-14.md` (untracked working artifact).

### Provenance-hygiene note (workflow false-positives, corrected)
The Phase-1 canonical held-set list ASCII-stripped two accented PDF filenames, spuriously flagging
them needs-fetch. Both **are held**: `Säilynoja_2026.pdf`, `Prša_2016_AJ_152_41.pdf`. Accented PDF
filenames are fragile for tooling (consolidation candidate).

---

## Papers to fetch (CHECKPOINT 1)

| paper | tier | status | needed for |
|-------|------|--------|------------|
| Demircan & Kahraman 1991, Ap&SS 181, 313 | 1 | ✅ fetched | mass–radius fit `builders.py:compute_stellar_radii` |
| Standish & Williams 2012 / JPL Horizons | 1 | 📄 NOT held | planet orbital elements `analytical/base.py` — accepted as standard J2000 (Batch 4, Anna's call); no fetch needed |
| IAU 2009 CBE / Luzum et al. 2011 | 1 | ✅ fetched | planet/Sun mass ratios `analytical/base.py` |
| Zahn 1977, A&A 57, 383 | 2 | ✅ fetched | tidal circularization, `LogisticThermalEccentricity` |
| Hurley, Pols & Tout 2000 (+ Tout, Pols 2002) | 2 | ✅ fetched | tidal-circularization timescale, `eccentricity.md` |
| Lucy 2006, A&A 457, 629 | 2 | ✅ fetched | twin excess, `TwinPeakedMassRatio` |
| King 1962, AJ 67, 471 | 2 | ✅ fetched | tidal-radius definition, `tidal.py`/`king.py` |
| Aarseth, Hénon & Wielen 1974, A&A 37, 183 | 2 | ✅ fetched | Q=T/\|V\| convention (cited 13×); Plummer Beta(3/2,9/2) recipe |
| Kainulainen 2014; Kritsuk 2011; Tan+2006; Hoffman&Gelman 2014; Cook 2006 | 3 | ⏸ deferred | experimental gravoturb/SBC (not in wheel) |
| Binney & Tremaine 2008; Murray & Dermott 1999; Vallado | textbook | ⏸ cite-by-edition | standard formulae — cite edition+eq+page, no PDF |

*Likely no fetch:* **Küpper 2011** (σ_Σ coeffs) is probably the held `McLuster_Methods_2011.pdf` — disambiguate during verify.

---

## Verified / fixed — per batch

### Batch 1 — Marks+2012 / Jerabkova+2018 α₃ relation (2026-06-14)

PDFs read directly: `marks-2014-erratum.pdf`, `Marks-IMF-mnras-2012.pdf`, `Jerabkova-IMF-aa-2018.pdf`.

| item | location | source (held PDF) | verdict |
|------|----------|-------------------|---------|
| α₃ FP **range-of-validity** threshold (−0.87) | `imfs/environment.md` ×4, `marks-2012.md:105`, `differentiability.md` ×2 | Marks et al. **2014 erratum**, MNRAS 442, 3315, Eq. 1 (corrects Marks 2012 Eq. 14, p. 2252) | 🔧 fixed — stale published-typo `+0.87` removed from docs; operative code already `−0.87` |
| FP slope/intercept −0.4072 / 1.9383; ϑ=98° | `coefficients.py:54-55`, `marks-2012.md:105` | Marks 2012 Eq. 14/15, p. 2252 | ✅ verified |
| Table 3 — 4 1-D relations (M_cl −0.94/2.14/0.68; M_ecl −0.77/1.59/0.27; ρ_cl −0.43/1.86/0.095; [Fe/H] 0.66/2.63/<−0.5) | `coefficients.py:63-68`, `marks-2012.md:92-97`, `environment.md:119-147` | Marks 2012 Table 3, p. 2251 | ✅ verified — matches cell-for-cell |
| α₃(x) = 2.3 (x<−0.87), −0.41x+1.94 (x≥−0.87) | `jerabkova-2018.md:72`, `coefficients.py:29-34` | Jerabkova 2018 Eq. 6 (cites the erratum) | ✅ verified |
| Eq. 9 constant 2.83 (4π); progenax 8π deviation → 0.2161 | `environment.md:280-291`, `coefficients.py:13-27` | Jerabkova 2018 Eq. 9 | ✅ verified — doc claim "Eq.9 prints 2.83" is true; 8π deviation ratified 2026-06-09 |
| operative threshold in code | `mapping.py:alpha3_marks_plane`, `coefficients.py` | erratum-corrected −0.87 | ✅ verified — code correct, no change |
| `Marks2014` erratum bibkey | `references.bib` | — | 🔧 added (refs.bib hygiene; cited via `{cite:t}`) |

**Verification:** `make build` → 161 pages, **0 warnings**, exit 0. `src/` untouched → released-core gate unaffected.

### Batch 2 — Moe & Di Stefano (2017) Table 13 + theory-doc reconciliation (2026-06-14)

PDFs read directly (rendered): Table 13 (p. 52), Table 10 (p. 27).

| item | location | source (held PDF) | verdict |
|------|----------|-------------------|---------|
| Table 13 grids: γ_largeq, γ_smallq, F_twin (logP×mass) + companion freq f_logP | `imf/binary/moe_di_stefano.py:186-205, 342-347` | Moe 2017 Table 13, p. 52 | ✅ verified — **all 80 cells match exactly** (incl. the `<0.03→0` twin convention and the −1.1/−2.0 tail) |
| Per-paper note Table 13 (F_{n=0}/F_{n=1}, f_mult, γ_largeq, F_twin) | `moe-distefano-2017.md:35-48` | Moe 2017 Table 13, p. 52 | ✅ verified |
| η(M₁,P) Eqs. 17/18 reproduce Table 13 η; e_max(P) Roche cap | `moe-distefano-2017.md:50-79` | Moe 2017 Eqs. 3/17/18 + Table 13 | ✅ verified (spot: solar logP=2 → 0.6−0.7/1.5=0.13≈0.1; e_max(10 d)=0.66) |
| f_b table mislabeled "Table 13 — companion frequency" | `binary.md:108`, `multiplicity-statistics.md:95` | code `MassDependentBinaryFraction` + Moe Table 13 F_{n=0} row | 🔧 fixed — values correct (match code); relabeled **multiplicity fraction** (1−F_{n=0}, ≤1), NOT the frequency f_mult; fraction-vs-frequency prose clarified |
| γ(M₁) wrongly cited "Moe **Table 10**" | `binary.md:156,161,365`, `multiplicity-statistics.md:154`, `mass-ratio-distributions.md:3,45` | Moe **Table 10 (p. 27) is the VB eccentricity-η table**, NOT γ(M₁) | 🔧 fixed — corrected to "period-averaged reduction of Table 13" (matches the code's own attribution; a progenax approximation) |
| f_twin period-averaged caption | `mass-ratio-distributions.md:74` | Moe Table 13 F_twin (avg over logP) | 🔧 fixed — clarified |

**Verification:** `make build` → 161 pages, **0 warnings**, exit 0. `src/` untouched → released-core gate unaffected.

### Batch 3 — Profiles bundle: King Table II + LIMEPY index + CW04 Q (2026-06-14)

PDFs read directly (rendered): King 1966 Table II (p. 73), LIMEPY Eqs. 8/11 (p. 578) + erratum, CW04 Table 1 (p. 590).

| item | location | source (held PDF) | verdict |
|------|----------|-------------------|---------|
| King Table II ξ_t = r_t/r_c (c = 4.699 / 131.4 / 2272 @ W₀ = 3/9/15) | `profiles/king.py:153` | King 1966 Table II, p. 73 | ✅ verified — matches exactly; re-confirms the real log c (0.672/1.029/1.528/2.119), NOT the debunked 0.84/1.18/1.48/1.76 |
| LIMEPY density index g+3/2 | `profiles/limepy.py:10,92` | GZ15 Eq. 8 (p. 578) + erratum Eqs. 20/21 | ✅ index verified (main-text Eq. 8 = E_γ(g+3/2,φ̂); King g=1 corner) |
| LIMEPY "main-text g+1/2 typo" narrative | `profiles/limepy.py:22-23` | GZ15 p. 578: Eqs. 8 **and** 11 both print g+3/2 — no g+1/2 misprint | 🔧 fixed — removed the false typo claim (a prior-session error; STATUS.md historical echo superseded by this ledger + the corrected docstring) |
| CW04 Table 1 radial Q (3D0/3D1/3D2 = 0.79/0.84/0.93) | `diagnostics/substructure.py:66-68` | CW04 Table 1, p. 590 | ✅ verified — exact (3D1 ± corrected 0.03→0.02) |
| CW04 fractal Q mislabeled "Table 1 reproduced to <0.01" | `diagnostics/substructure.py:69` | CW04 Table 1 F1.5/F2.0/F2.5 = 0.45/0.61/0.73 (p. 590) | 🔧 fixed — listed CW04's real published Q; relabeled 0.47/0.58/0.70 as the A=πR² estimator output (offset ~0.02-0.03; area convention) |

**Verification:** docstring-only `src/` edits (no value/behavior change — imports clean, `limepy_density_hat` forward value bit-identical). Released-core gate unaffected. (API-reference pages `30-api/*` are pre-generated snapshots; regenerating them to propagate these docstrings is tracked as consolidation.)

### Batch 4 — Tier-1 newly-fetched verify: D&K91 + IAU mass ratios (2026-06-14)

PDFs read directly: Demircan & Kahraman 1991 Table II + §4 (scanned, rendered p. 318-319); Luzum et al. 2011 Table 1.

| item | location | source (held PDF) | verdict |
|------|----------|-------------------|---------|
| mass-radius coeffs 1.06/0.945, 1.33/0.555 + 1.66 knee | `builders.py:compute_stellar_radii` | D&K91 Table II **empirical** (R = 10^a M^b; a=0.026/0.124, b=0.945/0.555) + §4 knee 1.66±0.08 | ✅ verified exact |
| docstring labeled these "ZAMS values" | `builders.py` | D&K91's ZAMS fit differs (R ≈ 0.89 M^0.89 / 1.01 M^0.57) | 🔧 fixed — relabeled **empirical** (Anna's call), cite Table II not "Eqs 5-6"; values kept, forward bit-identical |
| 8 planet/Sun mass ratios | `analytical/base.py:46-127` | IAU 2009 / Luzum et al. 2011 Table 1 | ✅ verified all 8 (Mercury 6.0236e6 / Venus 4.08523719e5 / Mars 3.09870359e6 / Jupiter 1.047348644e3 / Saturn 3.4979018e3 / Uranus 2.290298e4 / Neptune 1.941226e4 reciprocals; Earth 332946.05) |
| planet orbital elements (a,e,inc,Ω,ω) | `analytical/base.py` | Standish 2012 / JPL — **NOT held** | 📄 accepted as standard J2000 (Anna's call); source-not-held noted in the comment |

**Verification:** docstring/comment-only `src/` edits (no behavior change — `compute_stellar_radii` forward values bit-identical). Released-core gate unaffected.
**Tier-2 citation-appropriateness checks** (PDFs read directly; Zahn/Aarseth are scanned → rendered):

| citation | claim | source (held PDF) | verdict |
|----------|-------|-------------------|---------|
| Zahn 1977 (`eccentricity.py:119`) | tidal-circularization physics motivating `LogisticThermalEccentricity` | "Tidal Friction in Close Binary Stars", A&A 57, 383 | ✅ appropriate |
| King 1962 (`tidal.py:6,45`) | tidal-radius concept ((m/3M)^⅓ via BT08 Eq. 8.91) | "Structure of Star Clusters I. An Empirical Density Law" (tidal cutoff) | ✅ appropriate |
| Aarseth, Hénon & Wielen 1974 (docs ×13) | Plummer IC recipe + Q≡T/\|V\| convention | A&A 37, 183 — defines the Plummer model (Eq. 1) as the N-body test problem | ✅ appropriate |
| Hurley 2002 (`eccentricity.md:154`) | τ_circ ∝ (a/R)⁸/[q(1+q)] | Hurley, Tout & Pols 2002 §2.3 (equilibrium tide) | ✅ appropriate (standard equilibrium-tide scaling) |
| Lucy 2006 (`mass_ratio.py:233`) | "First systematic study of twin excess" | A&A 457, 629 — twin hypothesis attributed to Lucy & Ricco 1979 | 🔧 fixed — dropped "First"; relabeled "systematic statistical study" (Lucy & Ricco 1979 was first) |

**Verification:** 1 docstring fix (no behavior change). Consolidation: TWO Aarseth-1974 PDFs held (`AarsethHenonWielen1974.pdf` = the cited A&A 37,183; `aarseth1974.pdf` = scanned, likely a different/duplicate Aarseth-1974 — disambiguate). **Bucket C (Tier 1+2) COMPLETE.**

### Batch 5 — IMF classic (Salpeter/Kroupa/Chabrier/Maschberger) + Sana 2012 (2026-06-14)

PDFs read directly: `salpeter-1955.pdf` (scanned, rendered p. 165), `kroupa-2001.pdf` (Eq. 2 p. 234,
⟨m⟩ p. 235), `Chabrier_2003_PASP_115_763.pdf` (Table 1 p. 769), `maschberger-2013.pdf` (Table 1
p. 1727), `maschberger-2011-mnras0416-0541.pdf` (estimator p. 544), `sana-2012.pdf` (main report; p. 3
+ Fig. 1 caption). Inventory fan-out: 5 read-only agents (leads only); **every value adjudicated by
the main loop against the actual PDF**.

| item | location | source (held PDF) | verdict |
|------|----------|-------------------|---------|
| Salpeter α = 2.35 | `power_law.py:147` | Salpeter 1955 p. 165 (ξ ∝ m^−1.35 per d log m = α = 2.35 per dm) | ✅ verified |
| Kroupa exponents [0.3, 1.3, 2.3(, 2.3)] + breaks [0.08, 0.5, 1.0] | `power_law.py:130-131`, `params.py:69-90` | Kroupa 2001 Eq. 2 p. 234 (verbatim) | ✅ verified (3-seg merge of α₂=α₃=2.3 is exact) |
| Kroupa ⟨m⟩ range "0.01–1 M⊙" | `kroupa-2001.md:51` | Kroupa 2001 p. 235: ⟨m⟩ = 0.36 over **0.01–50** M⊙ | 🔧 fixed → 0.01–50 (value 0.36 correct; range was wrong) |
| Chabrier single-star m_c = 0.079≈0.08, σ = 0.69, A = 0.158 | `chabrier.py:65-69` | Chabrier 2003 Table 1 p. 769 (single ≠ system m_c≈0.2/σ≈0.6 — correctly distinguished) | ✅ verified |
| Maschberger α = 2.3, β = 1.4, μ = 0.2, m_l = 0.01 | `smooth.py:74-77` | Maschberger 2013 Table 1 p. 1727 (verbatim) | ✅ verified |
| Σ-estimator (k−1)/(π r_k²) + k = 6 credited only "M&C 2011 Eq. 4" | `segregation_approx.py:237,253` | M&C 2011 p. 544: "follow von Hoerner (1963) / Casertano & Hut (1985)" | 🔧 fixed — credited the upstream origin (formula/value correct) |
| Sana π = −0.55±0.22, κ = −0.10±0.58, f_bin = 0.69±0.09, 71% interact | `period.py:138`, `imf.py:342-343`, `sana-2012.md` | Sana 2012 main text p. 3 + Fig. 1 caption + p. 446 (verbatim) | ✅ verified |
| Sana q-slope cited "Eq. 3" | `mass_ratio.py:99` | Sana 2012 is a Science Report — **no numbered equations** (κ in main text + Fig. 1) | 🔧 fixed → "main text & Fig. 1" |
| **Sana eccentricity η = −0.4 ± 0.2** | `eccentricity.py:188,208`; `sana-2012.md:46`; `moe-distefano-2017.md:79` | **NOT in held main paper** (value lives in supplementary Table S3, paywalled); **absent from ALL held secondaries** (Moe 2017 / D&K 2013 / COMPAS ×2 / Sana-HM 2025 / Raghavan 2010 — searched, 0 hits) | 🔧 fixed — the −0.4±0.2 **backs no code** (`MoeEccentricity` computes η from Moe Eqs. 17–18, held + Batch-2-verified); it was docstring/note context only → **de-asserted to an honest SOM pointer** (Anna-approved). First "value in no held source" finding of the audit. |

**Verification:** `make build` → 161 pages, **0 warnings**, exit 0 (Kroupa ⟨m⟩ journal page re-confirmed
p. 235; M&C 2011 upstream-origin re-confirmed p. 544). All `src/` edits docstring-only (no value/behavior
change). Released-core gate unaffected. **Bucket B held-PDF deep-verify COMPLETE.**

*Low-priority hygiene (NOT applied — optional, Anna's call):* Salpeter code comment omits the
−1.35→−2.35 convention + `m_min=0.1`/`m_max=100` defaults unsourced & outside the fitted ~0.4–10 M⊙
range; Sana period range `[0.15, 3.5]` lower bound is figure-read (cite "Fig. 2" → SOM); Sana π/κ
uncertainty mixing (±0.22/±0.58 vs ±0.2/±0.6) across files; Chabrier `m_trans=1.0` lacks an inline cite.

### Batch 6 — Bucket A: phantom-API doc cleanup (2026-06-14)

Grep-grounded against `src/` (4-lens read-only sweep + main-loop verification; no PDF needed). Real
API confirmed at definition sites: env-IMF = **functional** `BirthEnvironment` + `env_to_imf_params`
(NOT an `IGIMF`/`EnvironmentIMF` class, NOT a galaxy-wide integration — CLAUDE.md R7); builders =
`build_spatial_ic` / `build_binary_cluster`.

| item | location | fix | verdict |
|------|----------|-----|---------|
| `progenax.imf.IGIMF` docstring pointer | `environment.md:292` | → `env_to_imf_params` / `alpha3_jerabkova_*` | 🔧 fixed |
| `gwimf.sample(...)` runnable code | `environment.md:344` | removed (no galaxy-wide sampler) | 🔧 fixed |
| "Both classes … differentiable … SFR" | `environment.md:347` | → the `env_to_imf_params` *function* is diff in ρ_cl/[Fe/H]/M_ecl/ε | 🔧 fixed |
| galaxy-wide-IGIMF overclaim | `environment.md:3,26,seealso` | reframed: progenax implements the cluster-scale α₃ input; the IGIMF integral is background theory | 🔧 fixed |
| §IGIMF reads as implemented | `environment.md:243` | added a **scope admonition** (ξ_cl only; no ECMF integration) | 🔧 fixed |
| "BM19→ECMF→IGIMF differentiable end-to-end" | `environment.md:368` | honest scope (galaxy-wide chain not implemented) | 🔧 fixed |
| "fractal IC generation via the FDF method" | `tidal-and-substructure/index.md:3` | → fractal-substructure *theory* + CW04 Q (generator → experimental) | 🔧 fixed |
| "Energy conservation under fractal substructure" anchor | `physics-tests.md:25-27` | removed (fractal generator retired) | 🗑 removed |
| §7 "IGIMF" — 8 phantom tests | `tests/README.md:380-420` + TOC + refs | replaced with a scope note → real env-IMF tests; a guard test (`test_documented_api.py`) already asserts IGIMF absent | 🗑 + reframed |
| IGIMF/EnvironmentIMF claimed *shipped* | `90-development-log/phase1-complete.md`, `docs/PHASE1_COMPLETE.md` | corrected claims + **Archived banner**; root file **moved** to `docs/archive/` | 🔧 + archived |
| IGIMF/EnvironmentIMF in module table | `2025-12-07-progenax-review.md` | extended the existing retirement banner (body preserved unedited) | 🔧 fixed |

**Deferred to the build_cluster batch** (Anna: make `build_plummer_cluster` a *real* API — design
`docs/plans/2026-06-14-cluster-builder-api-design.md`, commit `0a7a424`): `methodology.md:117` +
`units-policy.md:103` / `three-brick-state.md:151` `build_plummer_cluster` snippets → repoint to the
real builder once it exists (avoids double-editing).

**Verified leave-alone:** correct-negative docs (`environment.md:96/323`, `whats-new` Retired,
`fractal.md`, `gieles-zocchi` MultiMassLIMEPY-superseded) and legitimate **IGIMF-as-physics-theory**
prose (bibliography notes, `imf.md`, `science-capabilities.md`).

**Verification:** `make build` → 161 pages, **0 warnings**. `tests/README.md` is GitHub-rendered (not
in the myst build). **Bucket A COMPLETE** (modulo the deferred build_plummer_cluster repoint).

### Batch 7 — LOW spot-sample + consolidation (2026-06-14)

**LOW-set spot-sample (D3): PASS — no systemic rot in the LOW tail.** Verified via the inventory's
per-surface LOW classifications + concrete spot-checks: the Plummer scale-radius factor is the
canonical √(2^(2/3)−1) form (`plummer.py:48`, algebraically exact — NOT the inverted-a bug); all 116
`__all__` exports resolve; phantom classes confirmed absent; `reg=1e-30` guards carry stated rationale.

| consolidation item | verdict | action |
|--------------------|---------|--------|
| missing per-paper notes (cited) | `prsa-2016`, `kroupa-1995` were live-cited but note-less | 🔧 added 2 grounded notes (read the PDFs) + wired into `myst.yml` toc + `per-paper/index.md` |
| `zocchi2016` note | only an archival dev-log mention (not live-cited) | left — no note (orphan-ish reference PDF) |
| planet table `base.py` vs `solar_system.py` | benign — ss.py *imports* the table (single source); standalone fixtures are independent | 🔧 harmonized the standalone `e_jupiter` 0.0489→0.04839 (table J2000 value; no test pinned it) |
| szapudi "duplicate" PDFs | **FALSE ALARM** — two *distinct* papers (review `arXiv:astro-ph/0505391` vs 4-author `ApJ 631:L1`) | kept both; no deletion |
| Bairagi "duplicate" PDFs | confirmed same paper (Bairagi & Wandelt JCAP03(2026)028) | 🗑 removed the redundant `(1)` download (gitignored core-papers) |
| `references.bib` bibkeys | no duplicate keys | benign (the "Kuepper2011" flag was already resolved) |
| King −9 / Plummer / σ duplicated factors | distinct uses, each internally consistent | benign DRY |
| figure-eight period cite | already cited (Chenciner & Montgomery 2000, `few_body.py:24/64`) | benign |

**NEW ⚠ (surfaced while writing the Kroupa note):** `imf.py:52` cites "Kroupa (1995) MNRAS 277, **1507**"
but the **held** PDF — and the topic-matching paper for "IMF-consistent binary populations" — is MNRAS
277, **1491** (Paper I, inverse dynamical population synthesis). 1507 is the companion Paper II. Likely
a page typo → **recommend correcting to 1491; awaiting Anna's adjudication** (code-citation change).

**Verification:** `make build` → **163 pages** (+2 notes), **0 warnings**; `sun_earth_jupiter_3body`
imports + builds (N=3). Released-core gate unaffected (the `solar_system.py` change is a fixture-value
+ docstring; no test pinned 0.0489).

---

## Open flags (⚠ awaiting adjudication) & queued work

- ✅ **Bucket A — phantom-API docs — COMPLETE (Batch 6):** environment.md IGIMF/`gwimf` reframed to the
  cluster-scale `BirthEnvironment`/`env_to_imf_params` API; `tests/README` §7 + `physics-tests` fractal
  row retired; tidal frontmatter fixed; the `phase1-complete`/`PHASE1_COMPLETE`/`2025-12-07-review`
  snapshots banner-archived (root file moved to `docs/archive/`). **Deferred:** the `build_plummer_cluster`
  doc snippets (`methodology.md`, architecture pages) → repoint to the real builder once it ships
  (design `0a7a424`).
- ✅ **Bucket B — held-PDF deep-verify — COMPLETE:** Moe Table 13 (Batch 2); Profiles bundle — King
  Table II / LIMEPY g+3/2 / CW04 (Batch 3); IMF classic + Sana + Maschberger & Clarke Eq. 4 (Batch 5 —
  M&C estimator/k=6 re-credited to von Hoerner 1963 / Casertano & Hut 1985 per the held M&C 2011 PDF).
- ✅ **⚠ cross-note η (Sana batch) — RESOLVED (Batch 5):** the "Sana η = −0.4 ± 0.2" eccentricity slope
  is **in no held source** (Sana main paper defers it to supplementary Table S3, paywalled; absent from
  all held secondaries) and **backs no code** (`MoeEccentricity` uses Moe Eqs. 17–18). De-asserted to an
  honest SOM pointer in all 3 repo locations. The Sana 2012 SOM (Table S3) is **not fetchable**
  (paywall) — and not needed, since nothing computes from the value.
- ✅ **Bucket C — newly-fetched (Tier 1+2) verify — COMPLETE (Batch 4):** D&K91 (relabeled empirical),
  base.py mass ratios (verified) + elements (accepted-standard), Zahn/Lucy/King1962/Aarseth/Hurley
  citations all appropriate. Remaining: Tier-3 experimental gravoturb/SBC anchors stay deferred.
- **⚠ Remaining adjudications (non-blocking — Anna's call):** (1) ✅ Kroupa 1995 `imf.py:52` page
  **1507 → 1491 — FIXED** (Anna-approved; held-PDF + topic match); (2) the `build_plummer_cluster`
  doc-snippet repoint (deferred to the `build_cluster` implementation, design `0a7a424`); (3) optional
  low-priority hygiene — Salpeter −1.35→−2.35 code comment + unsourced `m_min`/`m_max`; Sana
  period-range "Fig.2" cite; Sana π/κ uncertainty mixing; Chabrier `m_trans` cite; `limepy_df`
  "measured runtime" docstrings; deprecated `G_KMS` 2-sig-fig.

---

## Consolidation (ride-along) tracker

Surfaced in Phase-1 (56 candidates); retired only with Anna's approval. Highlights:

- **Duplicate PDFs — RESOLVED (Batch 7):** `szapudi-2005` vs `Szapudi_2005_ApJ_631_L1` are **two distinct papers** (the higher-order-statistics review `arXiv:astro-ph/0505391` vs the 4-author `ApJ 631:L1`) → both kept; the two `Bairagi_2026` copies **were** the same paper → `(1)` removed; `Kuepper2011` bibkey — **no duplicate found** (already clean).
- **Duplicated code constants — RESOLVED (Batch 7, all benign):** King −9 / Plummer-scale / factor-9 σ are distinct uses or consistent copies (no silent divergence); the planet table is **single-sourced** (`solar_system.py` imports `base.py`) — the standalone-fixture `e_jupiter` 4th-digit mismatch harmonized 0.0489→0.04839.
- **Duplicated Moe doc-tables (now consistent, Batch 2):** the f_b table (`binary.md` + `multiplicity-statistics.md`) and the γ(M₁)/f_twin tables (`binary.md` + `mass-ratio-distributions.md`) — reconciled + correctly labeled; full dedup to one canonical copy deferred (binary.md's narrative uses them inline).
- **Stale docs:** `physics-tests.md` (pre-fractal-removal Measured rows); dup `90-development-log` review; divergent test-count snapshots (1163/866/~1243).
- **Missing per-paper notes — RESOLVED (Batch 7):** `prsa-2016` + `kroupa-1995` (live-cited) → grounded notes added & wired into nav; `zocchi2016` is archival-dev-log-only (left, no note). ~11 held-but-uncited reference PDFs remain (reference material; no notes needed).
- **🔭 IMF ↔ binaries mass-pairing duplication (architectural — DEFERRED, revisit later; brainstormed 2026-06-14, Anna chose keep-as-is + note):**
  the binary **mass-pairing** logic (q-distribution + binary-fraction → `(m₁, m₂, is_binary)`) is implemented
  **twice** — in `imf.binary.BinaryIMF` (mass-only) and *mirrored by hand* inside
  `binaries.companions.CompanionModel` (the IC-wired path used by `build_binary_cluster`; `companions.py`
  comments literally read "matches `BinaryIMF`" / "mirrors `BinaryIMF.sample_mass_ratios`"). Binary-fraction
  *implementations* are also scattered across `imf/binary/binary_fraction.py`, `binaries/mass_dependent.py`,
  and `companions.py` (interface-unified by the `BinaryFractionModel` protocol). The mass-vs-orbit **axis
  split is sound** — only the mass-pairing is duplicated (an evolution artifact: the Batch-4k
  `CompanionModel`/`build_binary_cluster` is the newer canonical IC path; `BinaryIMF` is the older mass-only
  object the new design consciously departs from). **Recommended future fix (not done):** extract ONE shared
  mass-pairing primitive — e.g. `sample_pairing(primary_imf, q_dist, binary_fraction, key) → (m₁, m₂,
  is_binary)` or a small `MassPairing` eqx module — that **both** `BinaryIMF` and `CompanionModel` *compose*
  (kill the hand-mirror → one source of truth); and consolidate the **mass-dependent** f_b models behind
  `BinaryFractionModel` in one home (keep `RadialBinaryFraction` separate — it's a genuinely different
  *spatial* axis). Pairs naturally with the `build_cluster`/builder-API arc (`0a7a424`).

---

## Audit close-out (2026-06-14)

**Scope completed:** Phase-1 triage (270 findings) → deep-verify of every HIGH/MEDIUM held-PDF item
(Batches 1–5), Bucket-A phantom-API cleanup (Batch 6), LOW-set spot-sample + consolidation (Batch 7).

**Verdict: release-credible — NO systemic rot, ZERO fabricated values in released-core.** Every defect
found was a *wrong/incomplete provenance narrative around a correct value*, plus one
untraceable-but-loads-nothing value and the phantom-API docs:

- **Provenance-text fixes (value correct):** Marks +0.87→−0.87 (2014 erratum); Moe "Table 10"→Table 13;
  Moe f_b frequency-vs-fraction; LIMEPY false "g+1/2 typo" narrative; CW04 fractal-Q label; D&K91
  "ZAMS"→empirical; Lucy "First"; Kroupa ⟨m⟩ range 0.01–1→0.01–50; Sana "Eq.3"→main-text; M&C
  estimator → von Hoerner / Casertano & Hut.
- **Untraceable value, de-asserted:** Sana eccentricity η = −0.4 ± 0.2 — in NO held source (paywalled
  SOM Table S3) and load-bearing for nothing (`MoeEccentricity` uses Moe Eqs. 17–18) → honest SOM pointer.
- **Phantom-API docs (Bucket A):** IGIMF/EnvironmentIMF/`gwimf`/`build_plummer_cluster` reframed or
  retired; galaxy-wide-IGIMF overclaim scoped; stale snapshots banner-archived.

**Spin-off (ratified):** the phantom `build_plummer_cluster` → a real SoTA differentiable IC-builder
API (design `docs/plans/2026-06-14-cluster-builder-api-design.md`, `0a7a424`), implementation pending.

**Remaining (Anna's adjudication, non-blocking):** see the ⚠ bullet above (Kroupa 1995 page;
build_plummer_cluster repoint; low-priority hygiene). Needs-fetch deferred (Tier-3/experimental +
textbooks): BT08, Larson 1981, Solomon 1987, Federrath 2013, Kainulainen 2014, Hurley 2000 paper PDF,
Allison MNRAS 395,1449.

**Merge/PR:** the arc is **all-local, nothing pushed** (CI minutes exhausted per Anna); `make build`
0 warnings throughout; `src/` edits are docstring/comment/fixture-value only (no behaviour change) →
released-core gate unaffected. **Recommended:** merge `feat/provenance-credibility-audit` → local main
on Anna's go (no PR while CI minutes are out), with `build_cluster` on its own `feat/cluster-builders`
branch — **or** hold the audit merge until build_cluster lands so the `build_plummer_cluster` doc
repoint ships in one coherent release. **Anna's call.**

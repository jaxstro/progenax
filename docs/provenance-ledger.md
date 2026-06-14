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
| Standish & Williams 2012 / JPL Horizons | 1 | ✅ fetched | planet orbital elements `analytical/base.py` |
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

---

## Open flags (⚠ awaiting adjudication) & queued work

- **Bucket A — phantom-API docs (Anna: rewrite to real API):** `imfs/environment.md` IGIMF section +
  `progenax.imf.IGIMF`/`gwimf.sample()` (240–291, 341); `tests/README.md` §7 IGIMF; `methodology.md`
  `build_plummer_cluster`; `tidal-and-substructure/index.md` removed-fractal frontmatter; `physics-tests.md`
  stale Measured rows. → reframe to `BirthEnvironment`/`env_to_imf_params`/`build_spatial_ic`; drop the
  galaxy-wide-IGIMF overclaim (per CLAUDE.md R7).
- **Bucket B — held-PDF deep-verify:** ✅ Moe Table 13 (Batch 2). **Queued:** King Table II ξ_t / LIMEPY
  g+3/2 / CW04 Table 1 profiles bundle; IMF classic + Sana; Maschberger & Clarke Eq.4 (likely Casertano &
  Hut 1985 mis-attribution).
- **⚠ cross-note η (queued, Sana batch):** `moe-distefano-2017.md:79` states "Sana et al. (2012) measure
  η = −0.4 ± 0.2 for short-period O-stars" — Phase-1 flagged this as inconsistent with `sana-2012.md:47`;
  verify both against the held `sana-2012.pdf` in the Sana/eccentricity batch.
- **Bucket C — newly-fetched (Tier 1+2) verify:** D&K91 mass-radius; base.py planet table (Standish/IAU);
  eccentricity Zahn/Hurley; King1962/Aarseth1974 convention citations.

---

## Consolidation (ride-along) tracker

Surfaced in Phase-1 (56 candidates); retired only with Anna's approval. Highlights:

- **Duplicate PDFs:** `szapudi-2005.pdf` vs `Szapudi_2005_ApJ_631_L1.pdf`; two `Bairagi_2026` copies (diff md5); `Kuepper2011` duplicate bibkey.
- **Duplicated code constants:** King −9 Poisson factor (×5 files); Plummer scale-radius factor (×2); factor-of-9 σ (×3 DF files); planet table `base.py` vs `solar_system.py` (mismatched 4th digits).
- **Duplicated Moe doc-tables (now consistent, Batch 2):** the f_b table (`binary.md` + `multiplicity-statistics.md`) and the γ(M₁)/f_twin tables (`binary.md` + `mass-ratio-distributions.md`) — reconciled + correctly labeled; full dedup to one canonical copy deferred (binary.md's narrative uses them inline).
- **Stale docs:** `physics-tests.md` (pre-fractal-removal Measured rows); dup `90-development-log` review; divergent test-count snapshots (1163/866/~1243).
- **Missing per-paper notes** for cited held PDFs: `kroupa-1995`, `Prša_2016`, `zocchi2016`; 11 held-but-uncited PDFs.

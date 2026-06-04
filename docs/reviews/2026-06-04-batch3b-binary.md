# Batch 3b-binary review — imf/binary + differentiable_binary (gate packet)

**Date:** 2026-06-04 · **Branch:** local `main` (push-direct) · **Base:** `origin/main @ 345db64` ·
**Reviewer:** Claude Opus 4.8 + audit agent · **Engine:** manual + panel · **Status:** 🚦 **AT GATE 1**

## Scope

imf/binary (~999 LOC: `mass_ratio.py` 347, `imf.py` 365, `moe_di_stefano.py` 172,
`binary_fraction.py` 80, `__init__.py` 35) + `differentiable_binary.py` (167). Public API:
`BinaryIMF`, `FlatMassRatio`, `PowerLawMassRatio`, `TwinPeakedMassRatio`, `MoeDiStefano2017`,
`ConstantBinaryFraction`, `MassDependentBinaryFraction`, `DifferentiableBinaryFraction`,
`DifferentiableBinaryModel`. Tests: `tests/unit/imf/{test_binary,test_differentiable_binary,
test_moe2017_provenance,test_imf_extensions}.py`.

## Paper-grounding (DONE, read-only — both PDFs verified by the lead)

| Paper | PDF | Verified facts |
|-------|-----|----------------|
| Sana et al. 2012, Science 337, 444 | sana-2012.pdf | p.3: **intrinsic f_bin = 0.69 ± 0.09**; period **π = −0.55 ± 0.2**; mass-ratio **κ = −0.1 ± 0.6** (uniform). f_obs=40/71=0.56. DOI 10.1126/science.1223344. |
| Moe & Di Stefano 2017, ApJS 230, 15 | Moe_2017_ApJS_230_15.pdf | **Table 13 (p.52):** single-star fraction F_n=0 = {solar 0.60, A 0.41, mid-B 0.24, early-B 0.16, O 0.06}; γ_largeq (logP=1)=−0.5 all masses (→−2.0 at long P); F_twin(logP=1)={0.30,0.22,0.17,0.14,0.08}, →<0.03 at long P. γ is a **two-slope** (γ_smallq 0.1–0.3, γ_largeq 0.3–1.0) + F_twin model, **period-dependent**. |

## Findings ledger (PDF-verified)

| id | finding | sev | disposition |
|----|---------|-----|-------------|
| **B1** | `BinaryIMF.massive_stars()` hardcodes `binary_fraction=0.7` (imf.py:343) but Sana 2012 = **0.69 ± 0.09**, and the docstring already says "f_bin ≈ 0.69". Code value wrong vs its named source. | **Major** | fix-now (RED): 0.7 → 0.69. |
| **B2a** | `MassDependentBinaryFraction` values (0.22→0.90, binary_fraction.py:60-73) are the **multiplicity fraction (1 − F_n=0)** from Moe Table 13 (verified: code 0.60≈Moe-A 0.59, 0.90≈Moe-O 0.94, 0.44≈Moe-solar 0.40), + M-dwarf bins (<0.8 M⊙, not in Moe). Docstring calls them "close binary fraction" / "companion frequencies" — **imprecise** (they are 1−single, not F_n=1 nor f_mult). NOT a misattribution. | Minor (doc) | fix-now: docstring → "multiplicity fraction (1−single-star fraction), Moe Table 13 F_n=0 for ≥0.8 M⊙ + M-dwarf surveys below". |
| **B2b** | `MoeDiStefano2017` single γ (0.4/0.3/0.0/−0.5) is a **single-slope reduction** of Moe's two-slope (γ_smallq+γ_largeq) period-dependent model; qualitative trend (low-mass→equal-q +, OB→small-q −) matches, but it is **not a verbatim Table row** and the reduction is undocumented. | Minor (doc) | fix-now: docstring — state it is an approximate period-averaged single-slope reduction of Moe's two-slope model. |
| **B2c** | `MoeDiStefano2017` f_twin (0.05/0.10/0.08/0.03) are **plausible period-averages** of Moe Table 13 F_twin (solar 0.10 ≈ avg of 0.30/0.20/0.10/<0.03; O 0.03 ≈ avg of 0.08/<0.03). | Minor (doc) | fix-now: docstring — "period-averaged F_twin (Table 13)". |
| **B3** | `differentiable_binary` (a,b,c)=(−0.2799,1.417,0.4755), γ-fit (0.1907,−0.7521) are WLS/linear refits of the Table-13-derived step values; fit method documented (<3% error). | verify | fix-now: numerically confirm the refit reproduces the step values within the claimed tol (PDF-independent); add provenance line. |
| **B4** | Moe γ_largeq & F_twin are **period-dependent** (Table 13 gives them at logP=1,3,5,7); `MoeDiStefano2017` uses single period-averaged values. Docstring says "simplified, period-averaged" — acknowledged. | Minor (doc) | fix-now: make the period-averaging caveat precise + cite Table 13. |
| **Sana-γ** | `massive_stars()` `gamma=-0.1` = Sana κ=−0.1 (uniform mass ratio). | — | verified, no fix. |
| **V-bin** | No FD-vs-autodiff grad-checks on binary entry points (`PowerLawMassRatio.ppf` wrt γ, `TwinPeakedMassRatio.ppf` wrt f_twin/σ, `DifferentiableBinaryModel.sample_systems`); `MoeDiStefano2017.pdf_given_primary` untested; γ≈−1 edge untested. | Major (gap) | fix-now (RED-first): add FD grad-checks + pdf_given_primary test + γ≈−1 edge. |
| **N-bin** | sana-2012.md / moe-distefano-2017.md notes are thin stubs (period-dist only); don't record the f_bin/κ/Table-13 values the code uses. | — | fix-now: expand both to verified depth (michie-1963.md style). |

**JAX-nativeness (agent-verified):** clean — jnp only (jax.scipy.special.erf/erfinv), vmap/lax.cond/
fori_loop, Equinox, no while_loop/argmax/argsort on grad paths.

## Proposed fix plan (RED-first; notes at Gate 2)

1. **B1** (RED): pin `massive_stars().binary_fraction` to 0.69 (Sana 2012) → change code.
2. **V-bin** (RED): FD-vs-autodiff grad-checks (PowerLawMassRatio.ppf/γ, TwinPeaked.ppf/f_twin,σ,
   DifferentiableBinaryModel.sample_systems) + `pdf_given_primary` test + γ≈−1 edge.
3. **B2a/b/c, B4, B3**: docstring-precision fixes (multiplicity-fraction wording; single-slope &
   period-averaging caveats; refit-provenance line) + numerically confirm the B3 refits.
4. **N-bin**: expand sana-2012.md + moe-distefano-2017.md (f_bin=0.69, κ=−0.1, π=−0.55; Table 13
   F_n=0/γ/F_twin), verify DOIs/bibcodes. `myst build` clean.

## Resolution (Gate 1 approved: all as proposed; B2b → document; B5 → fix-now)

| id | resolution |
|----|------------|
| B1 | **Fixed (RED→GREEN).** `massive_stars()` default binary_fraction 0.7 → **0.69** (Sana 2012). New test pins the default to the Sana value; the existing explicit-0.7 wiring test is unaffected. |
| **B5** (NEW) | **Fixed (RED→GREEN).** `PowerLawMassRatio.ppf`/`.cdf` crashed (ZeroDivisionError) at exactly γ=−1.0 — the lax.cond neq-branch's `1/(γ+1)` is traced even at γ=−1. Guarded both denominators with `where(abs(γ+1)<1e-10, 1.0, γ+1)` (the eq/log branch is selected there). Tests: γ=−1 ppf/cdf finite + matches the γ=−1±ε limit. |
| V-bin | **Fixed.** `TestBinaryGradients` (FD-vs-autodiff: PowerLawMassRatio.ppf/γ, TwinPeaked.ppf/f_twin,σ, DifferentiableBinaryModel.sample_systems/γ_intercept), `TestPowerLawGammaMinusOne` (B5), `TestMoeDiStefanoPDF` (pdf_given_primary normalized). |
| B2a | **Fixed (doc).** `MassDependentBinaryFraction` docstring → "multiplicity fraction 1−F_n=0 (Moe Table 13) for ≥0.8 M⊙ + M-dwarf surveys below"; verified code 0.60≈Moe-A 0.59, 0.90≈Moe-O 0.94. |
| B2b/B4 | **Fixed (doc).** `MoeDiStefano2017` docstring → explicit "period-averaged single-slope reduction of Moe's two-slope (γ_smallq/γ_largeq)+F_twin period-dependent model"; cite Table 13; link the ticket. `_gamma_of_mass`/`_ftwin_of_mass` "Table 10" → Table 13 + period-averaged. |
| B2c/B3 | **Fixed (doc).** f_twin documented as period-averaged Table 13 F_twin; `from_moe2017` "<3%" → "≲3% at bin centres, ~6% near step edges" (numerically confirmed). |
| Sana-γ | **Verified** (γ=−0.1 = Sana κ). |
| N-bin | **Fixed.** sana-2012.md + moe-distefano-2017.md expanded to verified depth (f_bin=0.69, κ, π; Table 13 F_n=0/γ/F_twin). Already in TOC + index. |
| (ticket) | Full two-slope period-dependent Moe q-distribution → `docs/notes/2026-06-04-moe-twoslope-q-distribution-ticket.md` (Anna requested). |

## Verification (Gate 2 evidence)

- **Targeted:** `pytest tests/unit/imf/ tests/validation/test_imf_physics.py` → **267 passed** (+24).
- **Full suite:** `pytest tests/` → **1074 passed, 0 failed** (was 1062; +12). No regression.
- **B1 RED→GREEN:** default-f_bin test failed at 0.7 vs 0.69 → 0.69. **B5 RED→GREEN:** γ=−1 ppf ZeroDivisionError → finite + matches limit.
- **Binary grad-checks** (`/tmp/imf_binary_gradcheck.py`): grad rel ~1e-10; pdf_given_primary ∫=1.0000.
- **MyST build:** clean — 127 pages, both binary notes rebuilt, no warnings.

## Gate status
- **G1:** ✅ approved (all as proposed; B2b → document + ticket the full model; B5 → fix-now).
- **G2 — diff + verification before commit:** 🚦 **awaiting Anna** (this packet).
- **G3 — push:** ☐

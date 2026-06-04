# Batch 3b-environment review — imf/environment (IGIMF: `coefficients · density · mapping · birth_environment`) — gate packet

**Date:** 2026-06-03 · **Branch:** local `main` (push-direct) · **Base:** `origin/main @ 3cdf285` ·
**Reviewer:** Claude Opus 4.8 + audit agent · **Engine:** manual + panel · **Status:** 🚦 **AT GATE 1**

## Scope

imf/environment, 1,072 LOC: `coefficients.py` (60), `density.py` (81), `mapping.py` (470),
`birth_environment.py` (407), `__init__.py` (54). Public API: `env_to_imf_params`,
`alpha3_jerabkova_*`, `alpha3_marks_*`, `x_*`, `lowmass_slopes_metallicity`, `BirthEnvironment`,
`compute_r_half/rho_ecl/rho_cl`. Tests: `tests/unit/imf/test_environment.py` (693 LOC).

## Paper-grounding (DONE, read-only — both PDFs verified by the lead)

| Paper | PDF | Verified facts (cite) |
|-------|-----|-----------------------|
| Marks et al. 2012, MNRAS 422, 2246 | Marks-IMF-mnras-2012.pdf | **Table 3 (p.2251, Eq.11):** α₃(λ)=p·λ+q (≷λ_lim, else 2.3): M_cl{−0.94,2.14,0.68}, M_ecl{−0.77,1.59,0.27}, ρ_cl{−0.43,1.86,0.095}, [Fe/H]{0.66,2.63,<−0.5}. **FP (Eq.13-14, p.2252):** x′=cos98°·[Fe/H]+sin98°·log(ρ_cl/10⁶); α₃=−0.4072·x′+1.9383 (x′≥0.87, else 2.3). **Eq.12 (p.2251):** Δα≈0.5/[Fe/H] (low-mass). **DOI** 10.1111/…20767.x ✓ vs PDF. |
| Jerabkova et al. 2018, A&A 620, A39 | Jerabkova-IMF-aa-2018.pdf | **Eq.6 (p.6):** α₃=2.3 (x<−0.87); −0.41x+1.94 (x≥−0.87) [attrib. Marks 2012 + 2014 erratum]. **Eq.7:** x=−0.14[Fe/H]+0.99·log(ρ_cl/10⁶). **Eq.8:** ρ_cl=3M_cl/(4π r_h³); r_h/pc=0.1·M_ecl^0.13 [Marks & Kroupa 2012]; ε=0.33. **Eq.10:** α_i=α_ic+Δα[Fe/H]. **DOI** 10.1051/0004-6361/201833055 ✓ vs PDF. |

## Findings ledger

| id | finding | sev | disposition |
|----|---------|-----|-------------|
| **E1** | **RESOLVED.** Jerabkova `alpha3_slope=−0.41`/`intercept=1.94`/`x_threshold=−0.87` (coefficients.py:28-30) are **exactly Jerabkova 2018 Eq. 6** — correct, just lacked an inline eq citation (the original MNRAS FP −0.4072/1.9383 is correctly kept *separately* in MARKS_COEFFICIENTS). | Minor | fix-now: add "Jerabkova 2018 Eq. 6 (Marks 2012 erratum)" citation. |
| **E2** | **VERIFIED EXACT.** All Marks Table 3 (4 relations) + FP (cos/sin 98°=−0.139/0.990, −0.4072/1.9383, 0.87) + Eq.12 (0.5) coefficients, and Jerabkova Eq.6/7 (−0.14, 0.99) + ε=0.33, match the PDFs exactly. | — | no fix (notes record provenance). |
| **E3** | **RESOLVED — code is correct, no change.** Anna supplied Marks & Kroupa 2012 (A&A 543, A8), which *defines* (p. 2) `ρ_ecl = 3·M_ecl/(8π·r_h³)` — the **8π** half-mass density the code uses. Jerabkova Eq. 8's "4π" is internally inconsistent with her own ρ_ecl=2.08 (8π). Code reproduces Marks 2012 Table 1 exactly (NGC 104=9.54e6). The mass-based constant 0.2161 is correct for 8π. Initial "0.3-dex low" alarm was a red herring from Jerabkova's inconsistency. | — | **No code change.** Documented in density.py + jerabkova/marks-kroupa notes. |
| **E-rh** | **RESOLVED.** r_h=0.1·M_ecl^0.13 (density.py:24) verified vs Marks & Kroupa 2012 (A&A) abstract: r_h ∝ M_ecl^(0.13±0.04); prefactor 0.1 per the paper (also quoted by Jerabkova Eq. 8). Added `MarksKroupa2012` bib entry + marks-kroupa-2012.md note (was uncited in bib). | Minor | fixed (note + bib). |
| **V-env** | No FD-vs-autodiff grad-checks on the env entry points (α₃/x funcs wrt FeH/M_ecl); SFE-extreme (sfe→0/∞) untested. | Major (gap) | fix-now: FD grad-checks + SFE-extreme test (PDF-independent). |
| **N-env** | New per-paper notes needed: marks-2012.md + jerabkova-2018.md (PDFs present, no notes); wire into index.md (add IGIMF/environment subsection). bib DOIs already verified vs PDFs. | — | fix-now (docs). |
| **X1** | Repo cruft: `tests/unit/imf/test_environment.py.bak`, `test_environment_v2.py.bak`. | trivial | fix-now: delete. |

**JAX-nativeness (agent-verified):** clean — jnp only, tanh-smoothed `jnp.where` thresholds
(grad-safe), `jnp.clip` to [0.5,2.3], Equinox; no while_loop/argmax/argsort. Marks/Jerabkova
opposite threshold signs (+0.87 / −0.87) correctly implemented + tested.

## Proposed fix plan (RED-first; notes at Gate 2)

1. **N-env**: write marks-2012.md + jerabkova-2018.md (michie-1963.md depth: Eqs/Table/page/ADS),
   wire into index.md. `myst build` clean.
2. **E1**: add the Jerabkova Eq. 6 citation to coefficients.py.
3. **V-env** (RED): FD-vs-autodiff grad-checks on `alpha3_jerabkova_generalized`,
   `alpha3_marks_plane`, `x_*` wrt FeH/M_ecl; SFE-extreme boundary test (clip keeps α₃∈[0.5,2.3]).
4. **E3**: per D-E3 — docstring note (default) or A&A-PDF cross-check.
5. **X1**: delete the two `.bak` files.

## Resolution (Gate 1 approved; E3 resolved via the supplied A&A PDF)

| id | resolution |
|----|------------|
| E1 | **Fixed.** Added "Jerabkova 2018 Eq. 6 (= Marks 2012 + 2014 erratum)" citation to coefficients.py; clarified it is distinct from the MNRAS FP −0.4072/1.9383. |
| E2 | **Verified, no fix.** All Marks Table 3 / FP / Eq.12 + Jerabkova Eq.6/7 constants match the PDFs exactly. |
| E3 | **Resolved — code correct, no change.** Marks & Kroupa 2012 (A&A) p.2 defines ρ_ecl=3M/(8π r_h³); code matches it + Marks 2012 Table 1 (NGC 104=9.54e6). Documented in density.py + notes. |
| E-rh | **Fixed.** r_h∝M^0.13 verified vs A&A abstract; added `MarksKroupa2012` bib entry + marks-kroupa-2012.md note (was uncited). |
| V-env | **Fixed.** `TestEnvGradients` (FD-vs-autodiff: α₃ Jerabkova/Marks wrt FeH/M_ecl/ρ/sfe) + `TestSFEExtreme` (clip to [0.5,2.3], no NaN). Mutation-sense via the 3b-core pattern. |
| N-env | **Fixed.** Wrote marks-2012.md + jerabkova-2018.md + marks-kroupa-2012.md; wired into index.md + myst.yml TOC. |
| X1 | **Fixed.** Deleted the two `.bak` files. |
| (bonus) | michie/merritt TOC-orphan + michie-poisson xref bug discovered → ticketed (`docs/notes/2026-06-03-michie-merritt-toc-orphan-ticket.md`), NOT fixed (Batch 2c scope). |

## Verification (Gate 2 evidence)

- **Targeted:** `pytest tests/unit/imf/test_environment.py` → **77 passed** (12 new).
- **Full suite:** `pytest tests/` → **1062 passed, 0 failed** (was 1050; +12). No regression.
- **FD grad-checks** (`/tmp/imf_env_gradcheck.py`): 12/12 pass (8 grad rel ~1e-10 + 4 SFE-extreme).
- **MyST build:** clean — **127 pages**, all three new notes build, no warnings.

## Gate status
- **G1:** ✅ approved (E3 → A&A cross-check; scope: all as proposed). E3 then resolved (code correct) via the A&A PDF Anna supplied.
- **G2 — diff + verification before commit:** 🚦 **awaiting Anna** (this packet).
- **G3 — push:** ☐

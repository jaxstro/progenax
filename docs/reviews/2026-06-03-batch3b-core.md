# Batch 3b-core review — imf flat files (`base · power_law · chabrier · smooth · truncated · params · differentiable`) — gate packet

**Date:** 2026-06-03 · **Branch:** local `main` (push-direct) · **Base:** `origin/main @ b41cad0` ·
**Reviewer:** Claude Opus 4.8 + 3 read-only audit agents · **Engine:** manual + panel (physics-heavy) ·
**Status:** 🚦 **AT GATE 1**

## Scope

imf flat files, ~1,730 LOC: `base.py` (279), `power_law.py` (303), `chabrier.py` (391),
`smooth.py` (255, Batch-3a CDF already done), `truncated.py` (106), `params.py` (98),
`differentiable.py` (329). Public API: `PowerLawIMF`, `ChabrierIMF`, `Maschberger`,
`TaperedPowerLaw`, `Schechter`, `TruncatedIMF`, `IMFParams`, `log_prob_masses`,
`sample_masses_from_params`. Tests: `tests/unit/imf/{test_imf_core,test_smooth,test_params,
test_imf_gradients,test_differentiable}.py` + `tests/validation/test_imf_physics.py`.
**Baseline: 223 passed** (`pytest tests/unit/imf/ tests/validation/test_imf_physics.py`).

## Paper-grounding (DONE, read-only — all four PDFs verified by the lead, not an agent)

| Paper | PDF | Verified facts (cite) | Code match |
|-------|-----|-----------------------|------------|
| Salpeter 1955, ApJ 121, 161 | salpeter-1955.pdf | Eq. 5 (p.165): ξ(𝔐)≈0.03(𝔐/𝔐☉)^**−1.35** per *d log m*, valid log(m/m☉)∈[−0.4,+1.0]. "−2.35" is the same slope in the *dN/dm* convention (+1 power from d log m). ADS `1955ApJ...121..161S`. | `salpeter()` = [2.35] ✓ (linear convention) |
| Kroupa 2001, MNRAS 322, 231 | kroupa-2001.pdf | Eq. 2 (p.234): α₀=0.3 (0.01–0.08), α₁=1.3 (0.08–0.5), α₂=2.3 (0.5–1.0), α₃=2.3 (≥1.0); ⟨m⟩=0.36. Salpeter stated as α=2.35, "α=2.3±0.3 adopted." | `kroupa()` [0.3,1.3,2.3]/[0.08,0.5] (3-seg merge) ✓; `IMFParams.kroupa()` 4-seg ✓ |
| Chabrier 2003, PASP 115, 763 | Chabrier_2003_PASP_115_763.pdf | **Table 1 (p.769) single-object disk:** A=0.158, m_c=0.079, σ=0.69; high-mass **x=1.3±0.3 ⇒ α=2.3** linear. **Eq. 18 (p.770) system:** A=0.086, m_c=0.22, σ=0.57. Salpeter x=1.35/α=2.35 (p.765). | `ChabrierIMF` m_c=0.08, σ=0.69, A_ln=0.158 = single-object ✓; **α=2.35 ≠ paper's 2.3** (→C5) |
| Maschberger 2013, MNRAS 429, 1725 | maschberger-2013.pdf | Table 1 (p.1726): L3 canonical single-star α=2.3, β=1.4, μ=0.2, m_l=0.01, **m_u=150** ("only needed for normalization"). G(m)=(1+(m/μ)^(1−α))^(1−β); closed-form CDF/quantile (Eqs. 1–4). §2.1: "Salpeter exponent α=+2.35." | μ=0.2, α=2.3, β=1.4, m_min=0.01 ✓; **m_max=300 ≠ 150** (→C6) |

## Findings ledger (provisional severities firmed up against the PDFs)

| id | finding | sev | disposition |
|----|---------|-----|-------------|
| **C1** | All classic-IMF constants **VERIFIED CORRECT** vs source PDFs (Salpeter 2.35, Kroupa Eq.2, Chabrier Table 1 single-object, Maschberger Table 1). | — | **No code fix.** Notes record precise provenance. |
| **C2** | `chabrier-2003.md` note describes the **system** IMF (m_c≈0.22, σ≈0.57); code correctly implements the **single-object** disk IMF (Table 1) and its *code docstring already says so* (chabrier.py:11–13). The **note is stale**, not the code. | Minor (note) | fix-now: rewrite note to the single-object values the code uses (mention system values + which is implemented). |
| **C3** | Chabrier `A_pl = ξ_ln(m_trans)·m_trans^α` (chabrier.py:69–80) enforces **pdf value-continuity** at m_trans — verified by construction. | — | fix-now: add a continuity test (merges with C7). |
| **C4** | Kroupa 4-seg (`IMFParams`) vs 3-seg merge (`PowerLawIMF.kroupa`) — exact (α₂=α₃=2.3). | Minor | trivial note line; no behavior change. |
| **C5** | **Chabrier high-mass slope: code α=2.35** ("true Salpeter", chabrier.py:65) **vs Chabrier 2003 Table 1 x=1.3 ⇒ α=2.3.** Within Chabrier's ±0.3, but a deviation from the named source. | **Major? — Anna decides** | **D-C5:** (a) change default to 2.3 (paper-faithful; alters sampled masses → ripple to tests) OR (b) keep 2.35 + document the deliberate canonical-Salpeter choice in docstring + note. |
| **C6** | Maschberger code `m_max=300` vs paper fiducial `m_u=150` (normalization-only). Also m_max defaults differ across classes (Chabrier 100 / IMFParams 150 / smooth 300). | Minor | fix-now: document in docstring (note the deviation + that limits only affect normalization); harmonization optional. |
| **C7** | **chabrier.py:218 docstring says "DISCONTINUITY at m_trans by design"** — but A_pl is computed *for continuity*, so the pdf is value-continuous (slope kink only). Code-doc contradiction; the audit agent took the comment at face value. | Minor (doc) | fix-now: correct the comment; the C3/C7 continuity test asserts the right behavior. |
| **V1** | **Grad tests are finiteness/non-zero only** (`test_imf_gradients.py`: Chabrier α/σ, Maschberger μ) — NOT FD-vs-autodiff. The only real FD-vs-autodiff check is `TaperedPowerLaw` (test_smooth.py:67). Same gap Batch 1 fixed as P4. | Major (gap) | fix-now: add FD-vs-autodiff central-diff grad-checks (reuse test_smooth.py:76 pattern) on Chabrier ppf (α,σ,m_c), Maschberger ppf (μ,α,β), PowerLawIMF ppf (slope), Schechter ppf. |
| **V2** | (a) No NaN-grad boundary tests (u→1e-10, 1−1e-10). (b) Weak Salpeter mean-mass bound `0.1<mean<1.0` (test_imf_physics.py:51) — analytic mean over [0.1,100] is ≈0.35. (c) C7 continuity untested. | Minor | fix-now: add boundary NaN-grad tests; tighten Salpeter mean to ≈0.35±few%; add pdf-continuity test. |
| **D1** | Chabrier ppf uses **30** Newton iterations (chabrier.py:368) vs base **20** (base.py:78), unexplained; Tapered/Schechter silently inherit base Newton ppf (Maschberger has analytic ppf). | Minor (doc) | fix-now: one-line justification/notes. |

**JAX-nativeness (verified):** no `numpy`/`scipy` (only `jax.scipy.special.erf/erfinv`); no grad-path
`while_loop`/`argmax`/`argsort`; `fori_loop` fixed-iteration Newton (differentiable); Equinox
throughout; `where`/log/pow denominators guarded with `+1e-30`. The `PowerLawIMF.__init__`
continuity loop is init-time only (not a hot/grad path).

## Proposed fix plan (RED-first; notes shown at Gate 2)

1. **V1 grad-checks** (RED): extend `test_imf_gradients.py` with FD-vs-autodiff central-diff checks
   (Chabrier α/σ/m_c, Maschberger μ/α/β, PowerLaw slope, Schechter) — pattern from test_smooth.py:76.
2. **V2 tests** (RED): NaN-grad boundary (u→1e-10, 1−1e-10, isfinite); tighten Salpeter mean→≈0.35;
   Chabrier pdf-continuity-at-m_trans (C3/C7).
3. **C5** (per D-C5): change `alpha` default 2.35→2.3 **or** keep + document. RED test pins the chosen value to Chabrier 2003 Table 1 provenance.
4. **C7/C6/D1 docstrings**: fix chabrier.py:218 continuity comment; document Maschberger m_max vs paper; document Chabrier 30-iter + Tapered/Schechter Newton-fallback.
5. **Paper-grounding notes**: expand salpeter/kroupa/chabrier/maschberger per-paper notes to verified
   depth (eq/table/page cites); fix C2 in the Chabrier note. (`myst build` clean.)

## Resolution (Gate 1 approved)

| id | resolution |
|----|------------|
| C5 | **Anna approved Option b** — `alpha` default 2.35→2.3 (Chabrier 2003 Table 1, x=1.3⇒α=2.3). RED (`CHABRIER_ALPHA_HIGH`→2.3 made the slope test fail at 2.35) → GREEN (chabrier.py:65→2.3 + docstrings). |
| V1 | **Fixed.** `test_imf_gradients.py` — FD-vs-autodiff central-diff grad-checks (Chabrier α/σ/m_c, Maschberger μ/α/β, Schechter α, PowerLaw exponent+m_min) + boundary NaN-grad (u→1e-10,1−1e-10) on 4 IMFs. Mutation-sense proven (stop_gradient → autodiff 0 vs FD −0.24, rel 1.0). |
| V2 | NaN-grad boundary + Chabrier pdf-continuity-at-m_trans (C3/C7) added. **Salpeter-mean tightening DEFERRED** → exposed **C8** (see below). |
| C2 | **Fixed (note).** `chabrier-2003.md` now describes the single-object disk IMF the code implements (Table 1) + the system IMF (Eq.18) it does NOT; the stale test comment fixed. |
| C6/C7/D1/C4 | **Fixed (docstrings).** C7 chabrier.py:218 corrected (value-continuous, not "discontinuity"); C6 Maschberger m_max=300-vs-150 documented; D1 Chabrier 30-iter + Tapered/Schechter Newton-fallback noted; C4 Kroupa 3-seg≡4-seg noted. |
| paper-grounding | salpeter/kroupa/chabrier/maschberger notes expanded to verified depth (eq/table/page/ADS). |

## C8 — NEW finding (Anna approved: fix all now)

**Linear-grid `mean_mass` under-resolved steep low-mass spikes.** Measured error vs a 200k
log-grid reference: Schechter **427%**, Maschberger 5.7%, Tapered 3.8%, PowerLaw/Salpeter
3.9%; **Chabrier 0.0%** (its lognormal turns over below m_c — no spike). Fix:
`PowerLawIMF.mean_mass` → **exact closed-form** piecewise-power-law mean (reuses the stored
continuity factors + segment integrals; differentiable, d/d m_min finite); `BaseIMF` (smooth
IMFs), `ChabrierIMF`, `TruncatedIMF` → **log-spaced (geomspace) trapezoid**. RED-first:
6 accuracy tests failed (Salpeter 0.365 vs analytic 0.351; Maschberger/Tapered/Schechter/Kroupa
off-grid), Chabrier passed → GREEN after fix; Salpeter mean now exact to 0.35137.

## Verification (Gate 2 evidence)

- **Targeted:** `pytest tests/unit/imf/ tests/validation/test_imf_physics.py` → **243 passed**.
- **Full suite:** `pytest tests/` → **1050 passed, 0 failed** (was 1030; +20 new tests). No regression.
- **C5 RED→GREEN:** slope test failed at measured 2.350 vs expected 2.3 (RED) → 2.3 (GREEN).
- **C8 RED→GREEN:** 6 accuracy tests failed (Salpeter 0.365 vs 0.351; Maschberger/Tapered/Schechter/Kroupa) → GREEN; Salpeter mean exact to 0.35137; Chabrier was already exact.
- **FD grad-checks** (`/tmp/imf_core_gradcheck.py`): 13/13 pass, rel ~1e-10; mutation-sense RED-sensitive (stop_gradient → fails).
- **MyST build:** clean — 124 pages, all four per-paper notes built (nested math-in-admonition OK).

## Gate status

- **G1 — findings + fix plan:** ✅ approved (D-C5 → Option b; "fix all as proposed").
- **G2 — diff + local verification before commit:** ✅ approved (C8 → "fix all now"; "commit + push after C8").
- **G3 — push to origin/main:** ✅ approved (pre-launch direct push).

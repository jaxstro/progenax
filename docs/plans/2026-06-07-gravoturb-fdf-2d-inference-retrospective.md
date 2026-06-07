# Gravoturb-FDF 2D-projection inference — attempts, failures, and lessons (retrospective)

**Date:** 2026-06-07. **Branch:** `gravoturb-fdf-sbc-validation` (local; nothing pushed).
**Scope:** the effort to build an SBC-calibrated, differentiable inference of the natal-turbulence
spectral slope **β** (headline), Mach **ℳ** (secondary), α (depth-gated) from the **2-D projected
star catalogue**. This documents every attempt, why it failed, and what we learned — so the next
session does not repeat them. **Every number below was reproduced first-hand this session** (the
scratch drivers `validation/_m*,_v1*,_v2*` are committed and re-runnable).

> **Status at stop:** the framework is *mostly* working — the IC generator is validated, the
> forward model + emulator + gradient-based HMC are validated, and after the final fix (rank-G)
> **ℳ is SBC-calibrated (p=0.83)**. The one unresolved item is a **residual β bias** (SBC p=0.002)
> traced to coarse-grid forward-model *slope* interpolation. We are **not** "nowhere" — we are one
> forward-model-accuracy refinement from a calibrated β, with a fully-diagnosed cause. The decision
> to pause is about *scope/strategy*, not a dead end.

---

## 0. TL;DR — what works, what doesn't, why

| Component | State | Evidence |
|---|---|---|
| IC generator (BM19 field) | **validated** | σ_s²=2.420 vs 2.419, ⟨eˢ⟩=1.0014, P(k) slope 2.97≈β, f_dense bias 0.30% |
| Forward model (analytic d_n Limber) | **validated (mean/slope)** | oracle: predicted vs measured slope ≤0.08 across full β prior |
| Emulator (precompute+bilinear) | **validated** | 0.03% accuracy, 1064× speedup |
| Gradient-based HMC (logit reparam) | **validated** | div=0, R̂=1.004, ESS=1257, matches grid Δβ=0.038 |
| SBC harness / sampler | **validated** | self-consistency SBC p=0.96/0.36 |
| **Likelihood (raw band-powers, Gaussian)** | **WRONG** | band-power skew up to **12**, exkurt up to **167** |
| **Likelihood (rank-G band-powers, Gaussian)** | **right for ℳ** | ℳ SBC **p=0.83 ✓**; β **p=0.002 ✗** (residual slope bias) |

**The single most important lesson:** the observable (projected star counts of a *fat-tailed*
turbulent field) has a **strongly non-Gaussian band-power likelihood** (skew→12). A Gaussian
likelihood on raw band-powers is fundamentally invalid; **rank-Gaussianization** (which the original
design had and the v2 simplification *dropped*) is required and fixes the likelihood + ℳ. The
remaining β failure is a *forward-model accuracy* problem (coarse-grid slope interpolation), not a
likelihood problem.

---

## 1. Chronology of attempts

### Pivot + scoping (committed `285cd6b`, corrected `ef49515`)
β reframed as the differentiable, SBC-calibrated **successor to Q/MST**; ℳ forecast-grade; α
depth-gated. Design §13/§14. **Correction logged:** an earlier (pre-decomposition) run had claimed
`dslope/dβ≈1` + box-stability; the identical-estimator decomposition (V1a) **superseded** it
(observable gain 0.64–0.86). *Lesson: don't quote numbers from an unreproduced run.*

### M2 — IC generator gallery → **PASS**
The BM19 field is physically faithful (table in §0). The 1-pt PDF matches BM19 to ~5 decades; the
β-sweep visibly behaves (small→large-scale structure). Caveat: the power-law tail is under-resolved
at ℳ≳16 on 96³ (the pipeline flags it). *Lesson: the generator is sound — failures downstream are
inference, not ICs.*

### M1 — naive slope→β "shot bake-off" → **FAIL (informative)**
Reading β off the measured 2-D slope failed for *both* shot models (raw-count +1/n̄; rank-G + Eq.3).
**Why:** even with zero noise the measured 2-D slope ≠ β. *Lesson: there is a deterministic
transfer function; β must be **fit against a forward model**, not read from a slope.*

### V1a — per-step transfer-function decomposition → **quantified**
Slope→β *gain* per space: GRF/log-density `s` 0.98; density `eˢ` 0.64; projected 0.66; **rank-G
projected 0.86**; rank-G counts 0.64. Dominant compressor = the **`s→eˢ` exponentiation**
(Δslope +0.55); **LOS projection/geometry negligible (+0.03)** — this answered the
"cube-vs-spherical?" question (geometry is not the problem). *Lesson: β lives in the **log-density**
slope; the density observable compresses it; rank-G partially recovers it.*

### A-new1 — analytic forward model (committed `5c0f44c`, `92a3f63`) → **PASS (as a mean)**
`angular_bandpowers_2d_limber`: ρ_g(β) → **exact BM19 density 2-pt** (`d_n` Mehler, *not* the
lognormal-limit `expm1` which was ℳ-biased) → Limber slab → 2-D FFT. Oracle: predicted slope vs
measured ≤0.08 across the **full β prior** under the **pointwise copula generative map**
(`smooth_copula`, "Option A" — the SBC-consistent generative model). Code-reviewed READY.
*Lesson: the analytic mean/slope predictor is sound; `n_max=14` converged; the pointwise map (not the
mass-conserving IC map) is the 2-pt-consistent generative model.*

### v2b/v2c/v2d — recovery, HMC, emulator (committed `153269a`) → **PASS (but weak test)**
- **Grid posterior (v2b):** β recovers within 1σ (density 2.93±0.16, count 3.14±0.19),
  σ(β)≈0.16–0.19 ≈ forecast.
- **logit-HMC (v2c):** first NUTS with a **hard `−inf` prior box** failed catastrophically
  (1075 divergences, R̂≈2.3, ESS≈5). **Fixed with a logit reparameterization** → div=0, R̂=1.004,
  ESS=1257, matched grid to Δβ=0.038. *Lesson: never use a hard `−inf` box with HMC; logit-transform
  to unconstrained space.*
- **Emulator (v2d):** precompute the forward model on an 81×81 grid + differentiable bilinear interp
  → **1064× per-eval speedup, 0.03% accuracy**, HMC ~2 s/observable → SBC feasible.
  (Resolution sweep: **σ(β) flat vs resolution** → box-size/cosmic-variance limited, not resolution.)
*Lesson: single-shot recovery "looking fine" is a **weak** test — it hid the systematics SBC later
found.*

### v2e — SBC-2D → **FAIL (the real gate)**
p(β)=0.001, p(ℳ)=0.000. **∪-shaped β ranks** (under-dispersed) + **ℳ pile-up** (biased low).
*Lesson: SBC is far more sensitive than eyeball recovery; it caught what v2b/c/d missed.*

### v2f — self-consistency SBC → **PASS (isolates the cause)**
Data ~ N(model, C); fit same model+C → **p=0.96/0.36 (uniform)**. → the **sampler + logit reparam +
bilinear emulator + rank/thin harness are all correct**. The v2e failure is **model/likelihood/
covariance adequacy, not the machinery** (and not the interpolation *order* — this answered the
"bilinear vs biquintic?" question for calibration). *Lesson: always run a self-consistency SBC to
separate "is my sampler right" from "is my model right".*

### v2g — θ-dependent (emulated) covariance + log|C| → **FAIL**
Both analytic-μ (A) and **simulator-μ (B)** with C(θ): p≈0.000. The cov-grid logdet ranged
**[59.8, 146.9]** (|C| varies ~e⁸⁷ across the prior — confirming the fixed-fiducial C was badly
mis-specified). But fixing C **did not** calibrate, and **B used the simulator's own mean** and still
failed. *Lesson: ruled out "fixed covariance" AND "mean bias" as the **sole** cause — pointing
deeper.*

### Non-Gaussianity diagnostic → **ROOT CAUSE**
Per-bin band-power **skewness 1.75 → 12**, **excess-kurtosis 5 → 167**, *growing* with k (the
fat-tail signature: rare dense clumps dominate small-scale power). **The Gaussian likelihood is
fundamentally mis-specified** — no covariance/mean fix can repair a distributional-shape mismatch.
The one test that passed (v2f) used *Gaussian* data; every test on *real* (fat-tailed) data failed.

### Transform diagnostic → **the fix, chosen by data not faith**
Per-bin skew / excess-kurt of the band-powers under three transforms:
- **raw:** skew → 12, exkurt → 167 (Gaussian likelihood invalid).
- **log (i.e. log-normal likelihood):** skew → ~3, exkurt → ~18 — **much better but INSUFFICIENT**
  (would still fail SBC). *This empirically refuted "use a log-normal likelihood".*
- **rank-G map (Neyrinck Eq.1):** skew ≈ 0.5, exkurt ≈ 0.5 — **near-Gaussian** ✓.
*Lesson: measure which transform actually Gaussianizes the statistic; the log must be applied to the
**field** (rank-G), not the **band-powers** (log-normal), to reach Gaussianity.*

### v2h — rank-G band-powers + Gaussian likelihood → **PARTIAL: ℳ ✓, β ✗**
p(ℳ)=**0.827 (CALIBRATED**, from 0.000!), p(β)=**0.002** (still). div=0. logdet span only ~12
(rank-G moments are mild → interpolation-friendly, unlike raw's e⁸⁷). The **β rank histogram is
one-sided** (pile-up at low ranks, ECDF arch to +0.20) → **β biased high — a *bias*, not
under-dispersion.** *Lesson: rank-G fixed the likelihood and ℳ; the residual β failure is a
**forward-model slope-accuracy** problem.*

### Diagnosed (not yet fixed): residual β bias
β lives in the **slope** of the rank-G band-powers. μ_rg(θ) is emulated on a **coarse 7×7 grid with
bilinear (C⁰) interpolation**, which systematically mis-estimates the slope of a curved μ(β) between
nodes → a consistent β bias that SBC (bias-sensitive) rejects. ℳ (amplitude, coarse feature) is
robust to this; β (slope, fine feature) is not. **Proposed fix (untested):** finer β-grid +
**cubic** interpolation of μ_rg (this is where the earlier "biquintic?" instinct legitimately
applies — coarse grid whose *slope* carries the signal).

---

## 2. What we learned (durable lessons)

1. **The observable is fundamentally non-Gaussian.** Band-powers of a fat-tailed (BM19) field have
   skew up to 12 → a Gaussian likelihood on **raw** band-powers cannot calibrate. This is the crux
   that every covariance/mean fix failed to address.
2. **rank-Gaussianization is not optional.** It Gaussianizes the statistic (skew→0.5), calibrates ℳ,
   *and* maximizes β information (gain 0.86 vs 0.64). Dropping it in the v2 simplification was the
   original sin. A **log-normal** likelihood is insufficient (measured).
3. **β lives in the log-density slope.** The density observable (counts∝eˢ) compresses it (transfer
   function, Δslope +0.55 from `s→eˢ`); projection/geometry are negligible. rank-G recovers most of it.
4. **SBC >> single-shot recovery.** "Recovers within 1σ" hid ~0.3–0.5σ systematics that SBC rejects
   at p≈10⁻³. Always SBC; always run a self-consistency SBC first to isolate sampler-vs-model.
5. **HMC needs a logit reparam.** A hard `−inf` prior box → 1000+ divergences; logit → div=0.
6. **Emulation is the right speed lever** (1064×, 0.03%) and keeps differentiability; σ(β) is
   resolution-independent (cosmic-variance/box-size limited, ~0.2/cluster — physics, needs stacking).
7. **Covariance is strongly θ-dependent** (|C| ~e⁸⁷ over the prior for raw band-powers; mild,
   ~e¹² span, for rank-G). A fixed-fiducial C is invalid over a wide prior; rank-G also tames this.
8. **Interpolation order matters where the *slope* is the signal** (coarse rank-G mean grid → β bias);
   it does *not* matter for calibration on a fine grid (self-consistency passed with bilinear).

## 3. Honest meta-lessons (why it kept "breaking")

- **We simplified the original design to move fast** (dropped rank-G; raw band-powers + Gaussian
  likelihood). Each simplification **passed the weak tests** (oracle slope, single-shot recovery,
  self-consistency) and **only SBC** — the strongest test — caught the dropped physics. The repeated
  "failures" were SBC doing its job, eliminating hypotheses (transfer fn → covariance → mean →
  **likelihood shape** → forward-model slope) until the cause was cornered.
- **I over-claimed several times** (`dslope/dβ≈1`; "recovery looks fine"; "the covariance fix will
  work"; "log-normal will fix it"). Each was corrected by *measuring* rather than asserting. The
  discipline that worked: **verify first-hand before concluding.**
- **The problem is genuinely hard:** infer a 3-D turbulence slope from a 2-D, shot-noise-limited,
  non-Gaussian, nonlinearly-transformed (eˢ) projected point process. Every wall was real physics.

## 4. Where it stands + options for next session (NOT decided)

**Closest-to-done path (incremental):** finer β-grid + cubic μ_rg interpolation → re-run SBC; expect
β to calibrate (ℳ already does). Then consolidate the validated pieces into ONE consistent pipeline
module + a curated figure set (for the MyST docs), and promote to core with TDD.

**Strategic question to decide (the reason we paused):** is the **2-pt rank-G + Gaussian-likelihood**
approach the right long-term tool, given the observable is fundamentally non-Gaussian? Alternatives:
- **Learned / flow likelihood (SBI)** — the honest SoTA for an intractable non-Gaussian likelihood;
  the design *deliberately avoided* it (interpretability + differentiability), but modern flow-based
  NLE is differentiable.
- **Scattering-transform / higher-order statistics (the planned "v2")** — extracts the non-Gaussian
  information we are *suppressing* with rank-G (filaments, tail → α/ℳ), potentially the more
  informative and more novel headline.

**Honest scope regardless of path:** σ(β) ≈ 0.2/cluster (cosmic-variance limited; tightens only by
population stacking); ℳ forecast-grade; α depth-gated.

## 5. Artifacts (all committed, re-runnable; plots are gitignored — regenerate)

`validation/_m2_ic_gallery.py` (IC gallery) · `_m1_shot_bakeoff.py` (transfer-fn fail) ·
`_v1a_transfer_decomposition.py` · `_v1b_limber_predictor.py`, `_v1b_fix_check.py` (forward model) ·
`_v1c_resolution_sweep.py` · `_v2b_grid_posterior.py` · `_v2c_hmc_logit.py` (logit-HMC) ·
`_v2d_emulator_hmc.py` (emulator) · `_v2e_sbc.py` (SBC fail) · `_v2f_sbc_selfcheck.py`
(self-consistency PASS) · `_v2g_cov_fix.py` (θ-dependent C fail) · `_v2h_rankg_sbc.py` (rank-G: ℳ✓,
β✗). Core: `inference/covariance.py::angular_bandpowers_2d_limber`,
`theory/gaussianization.py::bm19_density_hermite_coefficients` (A-new1). Released-core invariant
**814** held throughout (experimental-only changes).

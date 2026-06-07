# Design: gravoturb_fdf → 2D-projection-native, shot-noise-aware inference

**Date:** 2026-06-07
**Status:** Design approved (Anna, gate passed) — pending TDD plan (`superpowers:writing-plans`).
**Branch:** `gravoturb-fdf-sbc-validation` (experimental `gravoturb_fdf`, repo-only, LOCAL @ `a018b93`).
**Continues:** `2026-06-06-gravoturb-fdf-count-model-tail-robustness-{design,plan,handoff}.md`.
**Spike:** `validation/projection_fisher_spike.py` (kept; matures into a forecast AC).

## 1. Context — why this change

The 3D engine infers θ=(ℳ, b, α, β) (b fixed) through three channels: **α** from the POT tail of
the (unobserved) *gas* log-density field, **β** from *field-level* log-density 2-pt band-powers (no
shot noise), **ℳ** from the tail-robust `Var[log₊(N)]` of star counts-in-cells. The ℳ fat-tail bias
is cured at the oracle (AC20 flat <1.5%). But the joint SBC (AC18) **stalled**: the β block is
field-level/shot-noise-free while the ℳ block is shot-noise-limited — *different statistical
realities* — and its fixed-fiducial covariance is mis-scaled across the wide ℳ prior (the Gaussian
likelihood drops log|C|, valid only for θ-independent C). Whack-a-mole, ~40 min/run.

**The pivot:** the real observable is a **2D projected star catalog** — we never observe the 3D
field, and today's α reads a 128³ *gas* density field we never measure. Re-scope to a
**2D-projection-native, shot-noise-aware** inference (HMC on the projected observable). The Limber
operators already exist (`theory/projection.py::{limber_project_grid, limber_project_radial}`) but
are unused by the inference.

## 2. Gate evidence (the de-risking Fisher spike — 3 seeds, stable)

Same star catalog viewed two ways (3D cells vs LOS-projected sky cells); numerical-derivative Fisher
with common random numbers; the decisive metric is the **σ₂D/σ₃D ratio under an identical estimator**.

| quantity | 3D | 2D | 2D/3D |
|---|---|---|---|
| contrast Var(δ) *(model-free)* | ~2.43 | ~0.55 | **0.23** |
| tail skewness *(model-free)* | ~5.6 | ~1.9 | **0.34** |
| σ(β)/fid | ~9% | ~16% | **1.8×** |
| σ(ℳ)/fid | ~24% | ~62% | **2.6×** |
| σ(α)/fid | ~88% | **~225%** | **2.55×** |

The **model-free** contrast/skewness suppression (projection averages ~N_eff≈L/ℓ_corr independent
LOS cells) is estimator-independent: projection suppresses contrast ~4.4× and flattens the tail ~3×.
A flattened tail ⇒ the POT exceedance count N_tail collapses ⇒ σ(α)∝1/√(N_tail·I) blows up. **α is
the unavoidable projection casualty; β (shape) and ℳ (amplitude) survive.** No cheap 2D α rescue
exists (smaller sky cells don't un-project LOS averaging — only genuine depth/3D info recovers α).

## 3. Decision — the three-tier design

- **β primary + honest.** Slope of P(k)∝k⁻ᵝ → measured as the **angular clustering** (angular power
  spectrum / band-powers) of the real star map. Limber projection preserves slope information (a
  known analytic 3D→2D index relation), so β dilutes only ~1.8× and is now **shot-noise-limited and
  honest** (was a "field-level upper bound"). The science pivot (natal turbulence → stellar
  clustering) rides here.
- **ℳ secondary (de-projected, depth nuisance L).** Amplitude via σ_s²=ln(1+(bℳ)²). The observed 2D
  contrast = 3D contrast × projection-suppression(L, β); recovering ℳ requires **de-projecting**
  (dividing out the Limber factor), introducing a **depth nuisance L** (cloud LOS extent) degenerate
  with ℳ — marginalized with a physical prior. Identifiable (spike condition number fine), ~2.6×
  wider; population-stacking tightens as 1/√V. Projection's contrast suppression *also* tames the
  fat-tail bias that broke the 3D ℳ channel.
- **α depth-information-gated.** POT tail slope, lives in the rarest cells projection destroys. Keep
  *all* POT machinery in the likelihood (correct, cheap), but α's information is set by available
  depth: **2D-only** → weak/prior-dominated, reported as a **limit** with AC17-2D quantifying σ(α) vs
  (projected) N_tail; **depth-resolved** (Gaia 3D *star* positions; future) → full strength, an
  *upgrade* (reads the tail from observed stars, not the unobservable gas field). Same code path,
  information gated by the data fed in.

**Headline science = β-spectrum inference** (Anna, 2026-06-07). The depth-resolved 3D-star α mode is
an explicitly-enabled **future extension**, not a v1 deliverable.

## 4. The 2D observable & predicted statistics (differentiable in θ, L)

**Generative mock (reuses field code):** `gaussian_random_field(shape, β)` →
`rank_copula_field(ℳ, b, α)` → s → ρ=eˢ → `sample_cic_counts` (Poisson ∝ ρ) → **sum along the LOS
axis** → 2D sky count map N₂D → bin to sky cells. *Everything below is measured from THIS one
shot-noise-limited observable.*

| θ | measured (data) | predicted (analytic, differentiable) | reuse |
|---|---|---|---|
| **β** | angular band-powers of N₂D (shot-plateau modelled) | Limber-projected linear-ρ 2-pt → FFT → band-powers + shot term | `linear_hermite_coefficients`, `gaussianized_xi`, `limber_project_grid/radial`, `_bin_by_kmag` |
| **ℳ** | `Var[log₊(N₂D)]` at sky-cell scale(s) | projected σ²_{N,2D}=N̄₂D+N̄₂D²·ξ̄_ρ,2D(R; L) → log₊ transform of the projected P(N) | `predict_log_count_variance` (projected), `cell_averaged_xi_rho`+Limber, `box_window_sq_grid` |
| **α** | POT exceedances of N₂D (sparse) | projected tail / `sigma_alpha(α, L_tail, N_tail,2D)` | `tail_exceedance_loglike`, `measure_exceedances`, `sigma_alpha` |
| **L** | — (nuisance) | enters the Limber depth kernel; physical prior (aspect≈1 / distance+size) | `limber_project_radial` |

**Tail-robust 2-pt sub-decision (TDD, oracle-checked):** measure the 2-pt of `log₊(N₂D)` (log-space,
consistent with the cured ℳ channel) vs the linear count 2-pt — projection already tames the tail, so
the linear 2-pt may suffice; settle empirically against a 2D oracle, do not assume.

## 5. The covariance — the actual wall, dissolved

Because all channels now read **one shot-noise-limited 2D reality**, drop the mismatched
fixed-fiducial Gaussian precision. **Primary: a data-derived jackknife covariance** — resample
sky sub-patches of the *realized* observable. It is a deterministic function of the data (truth-
**independent**, applied identically in generation + inference → SBC-valid, exactly like the POT
`s_thr` once truth-keying was removed), and it captures the true non-Gaussian + shot-noise + cross-
block structure with **no log|C(θ)| term needed**. (Optional differentiable analytic-Gaussian C with
a proper ½log|C(θ)| term as a cross-check; the projected field is closer to Gaussian, so this approx
is far better in 2D than 3D.) This removes the fixed-fiducial-mis-scaled-over-wide-prior failure at
its root.

## 6. What carries over vs is re-derived

**Carries over (reuse):** `log₊` tail-robustness (`measure_log_count_variance`,
`predict_log_count_variance`); the σ_s² amplitude channel; the priors (ℳ∈[4,20], α∈[1.5,3], b
fixed); the POT machinery (`tail_exceedance_loglike`, `sigma_alpha`); the Gaussianization Hermite
machine (`gaussianized_xi`, `*_hermite_coefficients`); the Limber operators (now wired in); SBC
infra (`sbc_ranks`, `BM19Prior`, `run_nuts`, jaxstroviz figures); `rank_copula_field`; the AC
framework.

**Re-derived / new:** the Limber-projected predicted statistics; the depth nuisance L + prior; the
2D measurement helpers (projection + sky-cell binning); the jackknife covariance; the 2D SBC mock
(adds the project step); AC15-2D forecast (the spike, matured), AC16-2D recovery, AC18-2D
rank-uniformity, the 2D oracle AC. (`box_window_sq_grid` already generalizes to n-D.)

## 7. SBC validity in 2D (non-negotiable)

Identical data-derived quantities in generation + inference (jackknife C, sky-cell scale, POT
threshold); **covariances truth-independent** (jackknife from the data, never from the trial truth
θ*). The depth L enters as a sampled nuisance with the same prior in mock + posterior. Validity
verified empirically by AC18-2D rank-uniformity.

## 8. Speed

2D grids are N² not N³ (e.g. 64²=4096 vs 64³≈2.6×10⁵ cells); FFTs and the count model collapse
accordingly. Target: ~minutes/SBC-run (was ~40 min) → enough trials for clean calibration.

## 9. Scientific impact & honest scope

- **Contribution:** a calibrated (SBC-validated), differentiable forward model inferring the natal
  turbulence spectrum β (and ℳ) from real projected star positions — galaxy-clustering-style
  inference on star clusters, novel vs the field's non-differentiable, non-calibrated Q/MST.
- **Headline = β** (population trends β(environment); gravoturbulent-paradigm tests; substructure-as-
  a-clock with gravax), robust to absolute-calibration caveats and 1/√V-scaled by stacking.
- **Honest limits:** α is depth-gated (a *limit*, not a measurement, from 2D-only data — the spike
  proves this is physics, not an estimator failure); ℳ–L and ℳ–b degeneracies (L marginalized, b
  fixed); absolute β→ℳ needs validation vs real turbulence sims; optimality holds within the BM19
  model class (3-pt held out as the null test).

## 10. Future extensions (enabled, not built)

Depth-resolved 3D-star α mode (Gaia parallaxes → full α from observed stars); LOS velocities (break
ℳ–b, add a kinematic clock); massive-star tail tracers (partial α rescue in projection).

## 11. Definition of done (for the TDD plan)

- 2D oracle: projected predicted stats match finite-field 2D mocks to ~few-% across the ℳ prior.
- AC15-2D forecast (matured spike): σ(β,ℳ) + the α-degradation curve, with the 1/√V scaling.
- AC16-2D: joint (ℳ, β) recovery covers truth; α reported as a limit (depth-gated).
- AC18-2D: rank-uniformity **passes** for β and ℳ (α per its gated scope).
- Full experimental suite green; **released-core 814 invariant**.

## 12. Refinement (2026-06-07, after the A3 + A4 implementation STOPs)

Two implementation STOPs (the angular-band-power β carrier; the BM19-at-`meff` count carrier) shared
**one root cause: deep LOS projection CLT-Gaussianizes the field, so analytic predictions that carry
the 3D BM19 *shape* through projection are biased in the intermediate regime** (both carriers nearly
worked at full depth, failed where the projected field is neither BM19 nor Gaussian). A4's residual
was mach-dependent (+37%→−8%) from (i) a Jensen gap `Var(avg of log)` vs `log(avg)` and (ii) the
CLT-thinned tail vs the fatter BM19-at-`meff` tail.

**Decision (Anna-approved):** model the deep-projected observable as the **near-Gaussian / lognormal**
field it actually is — *not* a loss of substructure: clustering/substructure lives entirely in the
power spectrum P(k)∝k⁻ᵝ (preserved by projection) and the amplitude σ_s²→ℳ; "near-Gaussian" refers
only to the projected **1-point** distribution, whose fat tail (α) projection erases. Phase-coherent
filaments were always outside the phase-random GRF model (the held-out 3-pt null test) — unchanged.

- **2D-headline inference parameters = (ℳ, β, depth/aspect-ratio nuisance).** **α exits the inference
  loop**, held at a physical fiducial. (It is not marginalized-as-unconstrained; the Gaussian-limit
  prediction barely depends on it.)
- **α scope:** scientifically meaningful (high-density power-law tail = self-gravity beating
  turbulence; shallower α = more-advanced/efficient star formation — a collapse/SF clock), but
  *unmeasurable from 2D star positions* (washed out). It is kept (i) as a fiducial in the realistic
  forward mock (it seeds where the densest stars form), (ii) as the inference target of the future
  **depth-resolved mode** (Gaia 3D star positions; dust/extinction column-PDF), and (iii) as a
  **held-out tail robustness stress-test** of the 2D (ℳ,β) inference.
- **Predicted statistic (lognormal limit, tail-robust, no Hermite tail series):**
  σ_s²=ln(1+(bℳ)²); ξ_s(r)=σ_s²·ρ_g(r;β); **lognormal density 2-pt** ξ_ρ(r)=exp(ξ_s(r))−1 (exact,
  finite — no α≤2 divergence); project + cell-window via `limber_project_slab` → Var[ρ̃_proj,cell](R;
  depth) → σ_s²_proj=ln(1+Var) → Poisson-**lognormal** count model (the α→∞ limit of
  `count_distribution`) → `Var[log₊(N₂D)]`. Differentiable in (ℳ, β, depth).
- **Oracle structure (two tests):** (a) **AC20-2D gate** — predicted vs finite-field **lognormal**
  mock (α→∞) across ℳ, must pass (gen = infer, self-consistent); (b) **tail robustness stress-test**
  — predicted vs finite-field **realistic-tail** mock (α=2.5) across ℳ, report the residual
  (quantifies how safely α is held at fiducial / washed out by deep projection).
- The variance **ladder** over cell scales R carries β (de-risk confirmed: σ(β)/fid≈23%,
  corr(ℳ,β)=−0.25, ℳ–β degeneracy broken).

## 13. β-headline pivot (FINAL, Anna-approved 2026-06-07) — motivation, contributions, scope

After verifying that **β is recoverable from the 2-D projected map** — the observable-space slope→β
transfer gain is **0.64–0.86** (best in rank-Gaussianised projected density: gain 0.86, per-cluster
σ(β)≈0.22; with Poisson shot at N=10⁴: gain 0.64, σ(β)≈0.29), monotonic and degeneracy-free (V1a,
§14) — and that the amplitude (→ℳ) is finite-volume cosmic-variance-limited per cluster (so
data-vector stacking reduces *scatter*, not *bias*), the engine is re-scoped to a **β headline**.
(Naïvely reading β off the measured slope fails: the slope is compressed mainly by the lognormal
s→eˢ step, Δslope≈+0.55, while LOS projection/geometry is negligible, Δslope≈+0.03 — so β must be
**fit against a forward model** that captures the transfer, not read from a slope. The earlier
amplitude per-cluster scatter figure quoted here previously was from an unreproduced run and has been
dropped pending re-measurement; the cosmic-variance *reasoning* for ℳ-forecast-grade stands.)

**Motivation.** Stellar substructure is the field's standard probe of natal conditions, measured today
with *heuristic, non-differentiable, non-calibrated* statistics (Cartwright–Whitworth **Q**, MST,
fractal dimension D). They report "how substructured," but without a forward model, a calibrated
uncertainty, or a gradient.

**Contribution.** A **differentiable, SBC-calibrated, physically-parameterized successor to Q/MST**:
infer the natal-turbulence density power-spectrum slope **β** (P(k)∝k⁻ᵝ) from the projected stellar
positions of *young* clusters, via rank-Gaussianized angular clustering + a Limber forward model + HMC.
β is the *same physical content* as Q/MST (the 2-pt substructure / fractal scaling) but rigorous:
a calibrated number with an error bar and a model behind it. This is the robust, novel deliverable.

**Honest scope (what each parameter is):**
- **β — headline, trustworthy.** Structural (clustering spectrum); recoverable from 2-D positions with
  per-cluster σ(β)≈0.22 (best space, rank-G projected density; V1a). The calibrated Q/MST successor.
  Science: β(environment), **β-vs-age = the substructure clock** (substructure decays in a few crossing
  times → only *young* systems retain it), population tests of the gravoturbulent paradigm.
  (Box/resolution stability was claimed by an earlier run not reproduced this session; not relied upon.)
- **ℳ — secondary, forecast-grade.** A *thermodynamic* amplitude (σ_s²); cosmic-variance-limited
  *per cluster* (improves ∝1/√K with K stacked clusters; the specific per-cluster scatter quoted in an
  earlier draft was not reproduced this session and awaits re-measurement), not a precision per-object
  Mach measurement. β→ℳ is model-dependent (gravoturbulent relation), reported as relative/population
  with honest error bars.
- **α — depth-gated, not inferred in 2-D.** A *gravitational/SF* tail slope; expected to wash out under
  deep-LOS projection (which CLT-Gaussianises the 1-pt tail), and the tail is already sparsely resolved
  even in 3-D at ℳ≈8 (M2: ~5 cells above s_t on 96³). Kept as a forward-model fiducial + held-out tail
  stress-test; a science target only in the future depth-resolved (Gaia 3-D / dust-column) mode.
  (A dedicated 2-D α-washout test has not been run this session; the depth-gating is the conservative
  default, not a fresh first-hand result.)

**Why this is a *better* result than the original (ℳ,α,β)-absolute ambition.** The physics taught us
(through every wall) that 2-D stellar positions trustworthily encode **β**, which is exactly the
quantity the field characterizes heuristically with Q/MST. The deliverable — *the calibrated,
differentiable successor to Q/MST, connected to natal turbulence by a forward model* — is cleaner,
honest, and novel; ℳ/α are honestly-scoped secondary/extension results, not overclaimed.

**Engine (unchanged pillars, β-focused):** rank-Gaussianized projected angular clustering →
**slope→β** (robust), amplitude→ℳ (forecast-grade); analytic-2D **Limber** prediction (speed);
**population stacking** (precision, = the science); α POT retained as fiducial + stress-test, not in
the fit.

## 14. Method validation results (2026-06-07, first-hand, reproduced in-session)

Three verify-first checks (per Anna's "validate the methods before we commit"). Scratch scripts:
`validation/_m2_ic_gallery.py`, `_m1_shot_bakeoff.py`, `_v1a_transfer_decomposition.py`; plots in
`validation/plots/`. Every number below was reproduced first-hand this session (not from memory or a
single subagent report).

**(a) The IC generator is physically faithful — CONFIRMED (M2).** ℳ=8, b=0.4, α=2.5, β=3.0, 96³,
4 seeds: ⟨eˢ⟩=1.0014 (target 1), σ_s²=2.420 vs BM19 2.419, mean(s)=−1.210=−σ²/2, realized P(k) slope
of s = 2.97 vs input β=3.0, and f_dense_realized vs BM19 |bias|=0.30% (AC6 bar 5%); the 1-pt PDF
overlies BM19 across ~5 decades, and the β-sweep is visibly correct (small-scale→large-scale filaments
as β rises). Caveats (not bugs): the marginal is identical across seeds (the mass-conserving copula
assigns the analytic sorted-quantile densities by GRF rank — variation is spatial); the power-law tail
is under-resolved at ℳ≳16 on 96³ (the pipeline raises its own `low_resolution` warning).

**(b) Slope→β transfer function, per step — CONFIRMED with an identical estimator (V1a).** Measured
power-spectrum slope vs β_true (β∈{2,2.5,3,3.5}, n_real=30, 96³); gain ≡ dslope/dβ:

| space | gain | per-cluster σ(β) |
|---|---|---|
| GRF g (3D) / log-density s (3D) | 0.98 / 0.98 | — (β lives here, unobservable) |
| density eˢ (3D) | 0.64 | — |
| projected density (2D) | 0.66 | 0.31 |
| **rank-G projected density (2D)** | **0.86** | **0.22 (best)** |
| rank-G projected counts (2D, N=10⁴) | 0.64 | 0.29 |

Per-step Δslope (positive = compression): g→s +0.00, **s→eˢ +0.55 (dominant compressor)**,
eˢ→projection +0.03 (**geometry negligible — this answers the cube-vs-spherical question**),
projection→rank-G −0.23 (partial restore), density→counts +0.64 (shot). Monotonic in every space, no
degeneracy, no STOP flag. β is usefully recoverable per cluster (σ(β)≈0.22–0.29) → ≈0.07–0.09 stacked
over K≈10.

**(c) Naïve slope→β recovery FAILS for any shot model — CONFIRMED (M1); forward-model-first required.**
Recovering β by fitting the measured 2-D slope (with a shot term) is biased for BOTH a raw-count flat
1/n̄ model (N-stable but biased low: β_rec≈2.58 at β=3, N=10⁴) and a rank-G + flat Neyrinck-Eq.3 model
(strongly N-dependent: β_rec≈1.55→2.74 as N=10³→10⁵). Root cause is the deterministic transfer in (b):
even noiseless, measured-slope ≠ β. Verified sub-results: in raw-count space the flat 1/n̄ plateau is
exact (matches n̄ to ~1%); in rank-G space a *flat* Eq.3 shot is insufficient (Gaussianisation
increases and scale-tilts the shot — Neyrinck+2011 §3.1, verified against the PDF). **Conclusion: β must
be fit against a forward model that predicts the projected, copula/exp-transformed, shot-included
band-powers — not read from a slope.** This *supersedes* the earlier note that a flat +1/n̄ term alone
recovers β unbiased.

**Correction to an earlier in-session draft.** A prior (pre-decomposition) run had reported rank-G
slopes −2.30/−2.81/−3.35 with **dslope/dβ≈1** and σ(β)≈0.3–0.5, and quoted box-stability across
n=48→96. The identical-estimator V1a decomposition supersedes those figures: the **observable-space
gain is 0.64–0.86, not ≈1** (the ≈1 conflated the unobservable s-space gain, 0.98, with the observable
gain), and σ(β)≈0.22–0.31. The box/resolution-stability claim was **not reproduced this session** and
is not relied upon. The qualitative conclusions that survived: rank-G projected density is the best
observable space, β is recoverable, σ(β) is a few tenths per cluster.

**Net:** IC faithful; β recoverable from the projected map (σ(β)≈0.22/cluster, best space); recovery
**requires a forward model** (the analytic-2D Limber predictor — built next as V1b), not a naïve slope
fit; ℳ forecast-grade (cosmic-variance-limited); α depth-gated.

## 15. Follow-up SoTA design (v2): non-Gaussian / scattering-transform block + v1↔v2 comparison

**Status: PLANNED, not built (Anna-approved 2026-06-07 to stay the course on v1 first).** The 2-pt
β engine (v1) ships first as the validated, interpretable baseline. v2 adds a **non-Gaussian
statistic** (wavelet scattering transform / wavelet phase harmonics) as a novelty amplifier. The
**v1↔v2 comparison is itself a deliverable** — it quantifies how much information the non-Gaussian
structure carries beyond the power spectrum.

**Why (the honest gap in v1).** The 2-pt band-power statistic discards the field's *non-Gaussian*
information — the filaments and the BM19 power-law tail — which is the defining signature of
supersonic turbulence. v1 even **depth-gates α** because deep-LOS projection CLT-Gaussianises the
1-pt tail. The methodological frontier for non-Gaussian (turbulent / ISM) fields — the **wavelet
scattering transform (WST)** and **wavelet phase harmonics (WPH)** — is built precisely to capture
projected non-Gaussian structure *robustly and with low estimator variance*. So the very information
v1 writes off is what these statistics recover.

**Method.** WST = a CNN-like cascade of wavelet convolutions + modulus + spatial averaging →
translation-invariant, deformation-stable coefficients: S0 (mean), S1 (≈ power-spectrum content),
S2 (cross-scale interactions = non-Gaussian/clustering-of-structures). It is a **low-variance
estimator** (an advantage for single clusters / few realisations — directly relevant to the
finite-volume cosmic-variance wall), and JAX-differentiable (kymatio-style or custom), so it slots
into the existing differentiable-forward-model + mock/jackknife-covariance + HMC + SBC machinery —
**only the data vector (WST coeffs of the projected, optionally rank-G, star map) and its predicted
statistic change.**

**What we EXPECT it to buy — HYPOTHESES TO TEST, not established results** (held to the same
no-assumptions standard as §14; to be measured, not claimed):
- **(H1) De-gate α.** S2 coefficients may be sensitive to the tail/filament non-Gaussianity that
  survives projection better than the 1-pt tail does → α potentially recoverable from 2-D after all.
- **(H2) Break the β–ℳ degeneracy.** S1 (amplitude-like) vs S2 (structure-like) ratios may separate
  the thermodynamic amplitude (ℳ) from the structural slope (β) more cleanly than (slope, amplitude).
- **(H3) Lower per-cluster variance** → tighter σ(β), possibly a better-than-forecast σ(ℳ),
  mitigating the cosmic-variance wall.

**The hard part (why it is v2, the open methodological risk).** Unlike the 2-pt (closed-form
Mehler + Limber), WST has **no closed-form predicted statistic**. v2 needs either (i) a
*differentiable simulation-based* predicted-WST (mean WST over JAX-generated mocks, gradient through
the generative model — expensive but JAX-feasible) or (ii) a trained emulator; plus SBC calibration
of that estimator and its covariance. v1's analytic 2-pt remains the validated fallback if v2's
predicted-statistic differentiability/calibration proves intractable.

**Reuse (what carries over unchanged).** Generative model (pointwise copula, Option A §14),
projection, priors (ℳ∈[4,20], α∈[1.5,3] or depth-gated, β∈[2,11/3]), mock/jackknife covariance,
HMC/NUTS, SBC driver + integer-aware χ² + ECDF bands. Only the data vector + its predicted statistic
are new.

**Comparison protocol (the publishable result).** Run v1 (band-powers) and v2 (WST) on *identical*
mocks and *identical* SBC, and report: (a) Fisher/posterior σ(β,ℳ) improvement v2-over-v1;
(b) α-recoverability — depth-gated (v1) vs whatever v2 achieves (test H1); (c) shot-noise robustness;
(d) both must SBC-pass. The 2-pt is the interpretable yardstick; **the delta is the contribution.**

**References — VERIFY EACH AGAINST THE PDF before any manuscript use (no-assumptions; these are from
memory):** WST foundations Mallat 2012, Bruna & Mallat 2013; ISM/turbulence applications Allys et al.
(~2019), Regaldo-Saint Blancard et al. (~2020), Saydjari et al. (~2021); weak-lensing Cheng et al.
(~2020); WPH Allys et al. (~2020). Treat all years/venues as provisional until PDF-checked.

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

After verifying the projected-amplitude wall is a **finite-volume cosmic-variance** effect intrinsic
to red fat-tailed fields (the amplitude carries ~16–38% per-cluster scatter + box-drift at n≤128, and
data-vector stacking reduces *scatter* not *bias*), and that the **β-slope is box-stable, monotonic,
and low-scatter** (rank-G angular-clustering slope: −2.30/−2.81/−3.35 for β=2.5/3.0/3.5, drift ≲0.1
across n=48→96, ±0.05–0.07 at n=96), the engine is re-scoped to a **β headline**.

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
- **β — headline, trustworthy.** Structural (clustering spectrum); robustly recoverable from 2-D
  positions; box-stable. The calibrated Q/MST successor. Science: β(environment), **β-vs-age = the
  substructure clock** (substructure decays in a few crossing times → only *young* systems retain it),
  population tests of the gravoturbulent paradigm.
- **ℳ — secondary, forecast-grade.** A *thermodynamic* amplitude (σ_s²); cosmic-variance-limited
  *per cluster* (~16%/√K with K stacked clusters), not a precision per-object Mach measurement. β→ℳ is
  model-dependent (gravoturbulent relation), reported as relative/population with honest error bars.
- **α — depth-gated, not inferred in 2-D.** A *gravitational/SF* tail slope; washed out by projection
  (verified). Kept as a forward-model fiducial + held-out tail stress-test; a science target only in
  the future depth-resolved (Gaia 3-D / dust-column) mode.

**Why this is a *better* result than the original (ℳ,α,β)-absolute ambition.** The physics taught us
(through every wall) that 2-D stellar positions trustworthily encode **β**, which is exactly the
quantity the field characterizes heuristically with Q/MST. The deliverable — *the calibrated,
differentiable successor to Q/MST, connected to natal turbulence by a forward model* — is cleaner,
honest, and novel; ℳ/α are honestly-scoped secondary/extension results, not overclaimed.

**Engine (unchanged pillars, β-focused):** rank-Gaussianized projected angular clustering →
**slope→β** (robust), amplitude→ℳ (forecast-grade); analytic-2D **Limber** prediction (speed);
**population stacking** (precision, = the science); α POT retained as fiducial + stress-test, not in
the fit.

## 14. Method validation results (2026-06-07, before build commit)

Two verify-first checks (per Anna's "validate the methods before we commit"):

**(a) β-slope is the robust observable — CONFIRMED.** rank-G angular-clustering slope vs β_true, box
size: −2.30/−2.81/−3.35 (β=2.5/3.0/3.5 at n=48), drift ≲0.1 across n=48→96, scatter ±0.05–0.07 at
n=96. Monotonic, well-separated (Δslope≈0.5 per Δβ=0.5), box-stable — unlike the amplitude
(±16–38% + box-drift). β is identifiable and stable. dslope/dβ ≈ 1.

**(b) β recoverable at realistic cluster sizes — CONFIRMED, with a required shot-noise ingredient.**
With Poisson star sampling (n=96, full-LOS, 24² sky cells), β_true=3.0:
σ(β) per cluster ≈ 0.46 (N=1000), 0.35 (N=3000), 0.31 (N=10⁴) → ~0.1–0.15 stacked over K≈10. Useful
for resolving β across environment/age. **BUT** shot noise *flattens the measured slope
N-dependently* (−1.94 → −2.86 → −3.25 as N=1000→10⁴→∞): the white shot plateau (1/n̄) contaminates the
angular spectrum.

**Required build ingredient (standard galaxy-clustering):** the forward model **must include the
shot-noise term** — predicted band-powers = clustering(β,ℳ) + 1/n̄ (n̄ known from data) — so β is
recovered **unbiased**. **Build milestone 1 = demonstrate unbiased β recovery WITH the shot model**
(rank-G alone does not remove shot noise; it must be modeled or debiased à la Neyrinck+2011 Eq.3).

**Net:** the β-headline method is validated as viable on real data scales, contingent on the (standard,
well-understood) shot-noise modeling. ℳ (amplitude) remains forecast-grade (cosmic-variance-limited);
α depth-gated.

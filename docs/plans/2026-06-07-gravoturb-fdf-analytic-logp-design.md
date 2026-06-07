# Analytic log₊ 2D-projection inference of β — first-principles re-derivation, Phase-0 evidence, and design

**Date:** 2026-06-07 · **Branch:** `gravoturb-fdf-sbc-validation` (experimental subsystem) · nothing pushed.
**Author context:** deep-dive re-derivation requested by Anna after the previous 2D-inference arc
left β mis-calibrated (SBC p=0.002) while ℳ calibrated (p=0.83). This document supersedes the
rank-G-centric approach for the β headline. Companion: the attempt chronology in
`2026-06-07-gravoturb-fdf-2d-inference-retrospective.md`.

> **Provenance note.** Every Phase-0 number below was reproduced first-hand this session
> (drivers `validation/_d01.._d05`, committed and re-runnable). Literature equations (Neyrinck
> rank-G/log₊, Coles & Jones lognormal, Szapudi & Pan Mehler, Hamimeche & Lewis, Talts SBC) are as
> implemented in the prior-verified code docstrings; **re-verifying them against the held PDFs is a
> tracked Phase-1 close-out item** and is flagged where used.

---

## 1. The problem, re-derived from first principles

We infer the natal-turbulence density power-spectrum slope **β** (P(k)∝k^{−β}) from a 2-D projected
young-star catalogue. The generative chain is: a Gaussian random field g with P(k)=k^{−β} → BM19
log-density `s = T(g)` (copula map, marginal set by ℳ,b,α) → density ρ=eˢ → line-of-sight projection
→ inhomogeneous-Poisson star counts. We want a **differentiable, SBC-calibrated** estimator — the
rigorous successor to the heuristic Q/MST substructure statistics.

### 1.1 Why a Gaussian likelihood on raw band-powers is invalid
Band-powers are quadratic in the field. For a *Gaussian* field a band-power is an average of ~N_modes
χ²₂ variables, so its skew ≈ √(8/N_modes) **decreases** with k. Measured band-power skew instead
**grows** with k and is heavy-tailed and estimator-unstable (D02: skew 3→10+, exkurt 12→127; the
sample skew is dominated by the single largest of N draws). Growing-with-k, sample-size-unstable skew
at large N_modes **cannot** be the estimator's χ² — it is the **field's intrinsic non-Gaussianity**
(rare dense clumps from the BM19 power-law tail → connected trispectrum dominating small-scale power).
A Gaussian likelihood on raw band-powers is therefore fundamentally mis-specified, and no
covariance/mean fix repairs a distributional-shape mismatch.

### 1.2 Why standard analytic non-Gaussian band-power likelihoods do not rescue it
Hamimeche & Lewis 2008 (`g(x)=sign(x−1)√(2(x−ln x−1))`) and the lognormal/log likelihood assume
*Gaussian-field* (Wishart) band-power statistics. Since §1.1's high-k non-Gaussianity is *field*-NG,
these under-correct it. **D3a confirms:** H&L gives per-bin skew 0.78→1.72 (grows with k), no better
than the naïve log (0.33→1.28); both insufficient. *Analytic-likelihood-on-raw is defeated.*

### 1.3 The resolution: Gaussianize the field, keep the β-response analytic
The information and the Gaussianity both live in the **log-density**: in s-space the slope→β gain is
~1 and s is near-Gaussian (a pure lognormal has Gaussian s). The exponentiation ρ=eˢ is what
compresses β and breaks Gaussianity. So the right observable is a *log* transform of the data that
recovers the log-density, and the right forward model lives in log-density space, where the existing
analytic Mehler 2-pt `ξ_s(r)=Σ_{n≥1} c_n²/n! ρ_g(r)^n` (`gaussianized_xi`, c_n from
`bm19_hermite_coefficients`) applies.

---

## 2. Phase-0 evidence (first-hand; `validation/_d01.._d05`)

Config unless noted: 64³, depth=64, b=0.4, α=2.5, k∈[1,28] (10 bins). "gain" = d(slope)/dβ over
β∈{2,2.5,3,3.5,3.667}; slopes are realization-mean log-log band-power slopes.

**(D3b) the analytic forward model is exact for the field; the observable transfer is the issue.**

| channel | gain | vs analytic oracle |
|---|---|---|
| A_s (analytic proj. log-density) | −1.16 | — |
| S_s (sim proj. log-density) | −1.15 | A_s/S_s = **0.99** |
| A_rho (analytic proj. density) | −1.15 | A_rho/S_rho = **0.99** |
| O_logp (sim **log₊ counts**, N=1e5) | −0.90 | shot-suppressed |
| O_rg (sim rank-G counts, N=1e5) | −0.68 | shot-suppressed |

**(D1/D3a) which transform Gaussianizes the band-powers (N=400, well-powered):**
log₊ drives per-bin skew to **≈0** (better than rank-G ~0.3–0.5); raw skew large/unstable; H&L≈log
(insufficient). **log₊ is deterministic + differentiable; rank-G is rank-based (non-differentiable).**

**(D04) N_stars dependence — shot vs cosmic variance** (each field generated once, re-Poissoned):

| N_stars (n̄_sky) | log₊ gain | σ(β)/cluster |
|---|---|---|
| 1e5 (24) [v2h ran here] | −0.90 (71% of max) | 0.113 |
| 1e6 (244) | −1.20 (94%) | 0.099 |
| 1e7 (2441) | −1.27 (99%) | 0.096 |
| ∞ (no shot) | −1.28 | 0.085 |

→ β-response is **shot-limited at N≤1e5** (v2h's regime — a prime suspect for its β bias); recovers
to ~the analytic field value by N≥1e6. σ(β) hits a **cosmic-variance floor ~0.085–0.09/cluster** —
beyond N~1e6 only **stacking** improves it. log₊ recovers faster than rank-G.

**(D05) the enabler — transfer β-stability** (T(k,β)=E[obs]/analytic, per-bin CV across β; N=1e6):

| analytic × observable | median CV |
|---|---|
| **A_s × O_logp (log₊)** | **4.8%** (winner) |
| A_s × S_logSig | 7.8% |
| A_s × O_rg (rank-G) | 39% (β-UNSTABLE) |
| A_rho × O_rg | 91% |

→ For **log₊**, the transfer T(k)=E[obs]/A_s is **β-stable to ~5%** ⇒ `μ(β)=A_s(β)×T_fixed(k)` keeps
the **β-response purely analytic** with only a β-independent amplitude calibrated. For **rank-G** the
transfer is β-*dependent* (39–91%) ⇒ you must emulate the full β-response → the fragile emulated
slope that broke v2h. **This is, quantitatively, why log₊ ≫ rank-G and why v2h's β failed.**

### 2.1 Geometry is not the bias (Anna's question)
The analytic Limber projection predicts the log-observable β-response to **1.7%** (D03) and V1a found
the projection's slope-change negligible (+0.03). The β bias is **shot + the amplitude transfer**,
not 3D→2D geometry. A free "calibration amplitude" to soak up the transfer would be a **fudge**;
physical 3D→2D nuisances (depth L, distance, aspect/inclination) belong in the model as marginalized
parameters, and gen/inference projection operators must match (SBC-validity check).

---

## 3. Design

**Module:** `inference/projected_logp.py` (experimental). Reuses `gaussian_correlation_grid`,
`bm19_hermite_coefficients`, `gaussianized_xi`, `limber_project_slab`,
`_angular_bandpowers_from_xi_rho_2d`, and the logit-NUTS in `hmc.py`.

### 3.1 Observable (identical in generation & inference — the SBC contract v2e broke)
`data = measure_angular_bandpowers_2d( log_plus( project_counts_los(counts) ), k_edges )`
(`log_plus` = Neyrinck Eq.2¹). Generation: GRF(β\*) → `smooth_copula_field` → `sample_cic_counts`
(N_stars) → project → log₊ → band-powers. **rank-G carried in parallel** as the documented contrast.

### 3.2 Forward model (analytic β-response)
`μ(β) = A_s(β, ℳ_fid) × T_fid(k)` where
`A_s = _angular_bandpowers_from_xi_rho_2d( limber_project_slab( gaussianized_xi(ρ_g(β), c_n(ℳ)) ) )`
is differentiable in β; `T_fid(k)` is calibrated **once** at a fixed fiducial θ_fid (≠ per-trial
truth) as the mean ratio `E[data]/A_s` — a **truth-independent constant** (β-response stays analytic).

### 3.3 Likelihood, sampler
Gaussian (D02 justifies it for log₊): `ℓ(β) = −½ rᵀ Cinv_fid r + log_prior + log_jac`,
`r=μ(β)−data`. `Cinv_fid` = Hartlap-corrected inverse of C estimated once at θ_fid
(`N_real ≥ n_bins+2`; drop/widen the lowest-k bin — D05's 14.8% weak point). No log|C| term (C
truth-independent). Sampler: validated logit-reparam NUTS (β→logit on [β_lo,β_hi]; blackjax window
adaptation) — avoids the −inf-box divergences.

### 3.4 SBC-validity invariants (encoded as tests)
(a) gen statistic ≡ inference statistic; (b) `T_fid`, `C_fid` at a fiducial independent of each
trial's truth; (c) generative projection (`project_counts_los`, first-L) ≡ inference projection
(`limber_project_slab`) at the chosen depth.

### 3.5 Scope (first build)
**β-only** SBC (ℳ,α,b fixed) at N_stars=1e6 — the cleanest isolation of the β machinery, avoiding the
unverified T(k) ℳ-stability. Then add ℳ (after checking T's ℳ-stability), then the shot transfer.

---

## 4. Acceptance criteria

1. Forward-model accuracy: `μ(β)` matches `E[data]` bin-by-bin to ≤(per-bin σ/√N_real) across the β
   prior at N=1e6.
2. **SBC: β rank-uniformity p>0.05** (Talts¹ integer-aware χ² + Säilynoja ECDF bands), K≈128, div=0.
3. Differentiability: `jax.grad(ℓ)(β)` finite.
4. Head-to-head: same SBC for rank-G (expected to fail/need emulation) — documents the log₊ choice.

---

## 5. TDD build order (RED→GREEN→REFACTOR; experimental-only; released-core **814** invariant)

1. `predict_logp_bandpowers(β, ℳ, k_edges, T)` = A_s×T — unit test vs the D03/D05 analytic chain.
2. `calibrate_transfer(θ_fid, N_real)`, `calibrate_covariance(θ_fid, N_real)` — truth-independence +
   Hartlap tests.
3. `logp_loglike` + logit-NUTS β driver — recovery test (β in posterior) at N=1e6.
4. SBC driver → the gate (criterion 2); rank-G head-to-head.
5. **Shot-transfer derivation** → make `T` N-agnostic (D04: needed for N≲1e5); re-run SBC at N=1e5.

## 6. Risks / escalation ladder (each a derivation, never a fudge)

- 5% T-residual biases β → improve A_s (T→1) or model T's β-dependence analytically.
- fixed-fiducial C insufficient → C(θ)+log|C|.
- low-N shot transfer harder than expected → state N≳1e6 scope + stacking as the σ(β) lever.

---

¹ PDF re-verification (Neyrinck 2009/2011 Eq.1/2/3; Coles & Jones 1991 lognormal log-2pt; Szapudi &
Pan 2004 Mehler; Hamimeche & Lewis 2008; Talts+2018 / Säilynoja+2022) is a Phase-1 close-out task;
formulas here are as implemented in the prior-verified code.

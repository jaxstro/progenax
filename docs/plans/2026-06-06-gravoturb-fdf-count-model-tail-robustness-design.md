# Design: count-model tail-robustness — a log-space σ_s² channel for the gravoturb_fdf ℳ inference

**Date:** 2026-06-06
**Status:** Design approved (Anna), pending implementation plan (`writing-plans`).
**Branch:** `gravoturb-fdf-sbc-validation` (experimental `gravoturb_fdf`, repo-only).
**Supersedes the open item in:** `docs/plans/2026-06-06-gravoturb-fdf-count-model-tail-robustness-handoff.md`.

## 1. Context — why this change

Simulation-Based Calibration (AC18) **rejects ℳ rank-uniformity** (χ² p≈0.005) while **α and β
pass** (p=0.18, 0.12). Root cause (a genuine forward-model limitation, not a numerics/sampler bug):

the ℳ channel constrains ℳ through the **counts-in-cells over-dispersion**

```
Var(N) = N̄ + N̄² (⟨e^{2s}⟩/μ² − 1),   μ = ⟨e^s⟩,
```

with ⟨e^s⟩, ⟨e^{2s}⟩ integrated **analytically over the infinite-tail BM19 PDF**. The second
**linear** moment ⟨e^{2s}⟩ = e^{σ_s²}-type is **tail-dominated** — it diverges for α≤2 and is
tail-sensitive at α=2.5 — so the model over-predicts the over-dispersion a *finite* star field can
realize (**+9% at ℳ=3 → +36% at ℳ=16** vs a 96³ oracle), biasing ℳ high. AC16 passes only because
it is a single ℳ=5 point; SBC across ℳ∈[2,20] exposes the bias.

**The clue:** the two channels that pass SBC live in **log space** (α via the shift-immune POT tail;
β via log-density Gaussianization). The broken one rides on **linear**-density moments. Ruled out
already (do not re-litigate): prior/Jacobian, quadrature/internal resolution, bigger grid (a hack;
mean bias unchanged), sampler/init/thinning.

## 2. Decision

Replace the linear-count-over-dispersion ℳ-channel with a **log-space σ_s² channel**:

- **Infer σ_s²** (the log-density variance) from a **Gaussianized / log-transformed count-variance**
  statistic, and report **ℳ = √(e^{σ_s²} − 1) / b** as a *derived* quantity.
- **α** (POT tail) and **β** (log-density 2-pt Gaussianization) are **unchanged**.
- **b is fixed** (see §5 — it is not identifiable from density statistics; deferred to the velocity arc).

This gives each parameter its **sufficient, well-conditioned** statistic, all in log space.

## 3. Why this is the optimal SoTA choice (literature, PDF-verified)

- **Carron & Szapudi 2013** (MNRAS 434, 2961) — *Optimal non-linear transformations for LSS statistics.*
  For a lognormal field, **σ² is the sole amplitude parameter** and the **log/power transform is the
  information-optimal *sufficient* observable** (Eq 9: `o ∝ ln²(1+δ)`; Eq 17: `τ²(δ)`; near-optimal
  to σ²~10, Eq 38 / Figs 1–2). ⇒ measuring σ_s² in log space is not merely tail-robust, it is
  **provably sufficient/optimal** for the amplitude.
- **Neyrinck, Szapudi & Szalay 2009** (ApJ 698, L90) — *Restoring information with a log density mapping.*
  The log transform makes the (log-)variance well-behaved (not tail-dominated) and **restores ~10×
  Fisher S/N** vs the linear field; large-scale log-vs-linear bias = `e^{−Var[ln(1+δ_cell)]}` (Eq 1).
  ⇒ the direct cure for our linear-moment tail fragility.
- **Neyrinck, Szapudi & Szalay 2011** (ApJ 731, 116) — *The Gaussianized galaxy density field (discreteness).*
  Gaussianization works **with shot noise** (our counts): exact rank-Gaussianization
  `G(δ)=√2 σ erf⁻¹(2 f_{<δ} − 1 + 1/N)` (Eq 1), shot-tolerant modified log `log₊(δ)` (Eq 2), a
  shot-noise estimate (Eq 3); and the Fisher gain **peaks at a cell scale a few× coarser than Nyquist,
  where shot ≈ clustering** — *do not use cells too small*. ⇒ the recipe + cell-scale rule for star counts.

## 4. Method / architecture

**Predicted statistic (analytic, differentiable in θ):**
- σ_s²(R) — **already exists** as `theory/cic.py::smoothed_log_variance` (tail-robust, log-space;
  Route B already uses it internally to set the width). Reuse directly.
- **plus an analytic shot-noise term** `s(N̄)` (Neyrinck-2011-style) so the prediction matches the
  measured *discrete* statistic: `predicted = σ_s²(R) + shot(N̄)`.

**Measured statistic (data side, done once; non-diff is fine):**
- log-transform the observed cell counts (Neyrinck-2011 exact rank-map **or** `log₊`; `N=0` handled),
  **shot-noise-debiased variance** of the log/Gaussianized cell field, at a **cell scale R chosen
  where shot ≈ clustering** (Neyrinck-2011 optimal-resolution rule).

**Likelihood:** Gaussian on this scalar (optionally a small set of cell scales R), parametrized by σ_s².

**Removed from the inference path:** `inference/likelihood.py::count_loglike` (Route-B compound-Poisson
over-dispersion). `theory/cic.py::count_distribution` is **retained only as a labelled
validation/diagnostic**, not in the θ-fit.

## 5. Parametrization & the b question

Infer **(σ_s², α, β)**; ℳ derived; **b fixed**. Density statistics depend on (ℳ, b) only through
σ_s² = ln(1+(bℳ)²) — only the *product* bℳ is identifiable (AC15 rank-3 singular Fisher; Carron &
Szapudi 2013 "σ² is the sole parameter"). Freeing b now ⇒ an unidentifiable ridge ⇒ singular Fisher,
broken SBC, bad NUTS geometry. **b is identifiable only with an independent observable that pins ℳ
separately — the velocity field** (ℳ=σ_v/c_s; the solenoidal/compressive mix also informs b). ⇒
**deferred to the velocity arc.**

**Prior floor ℳ≥4 (Anna-approved 2026-06-06).** The calibrated prior is **LogUniform[4, 20]** (was
[2,20]). The AC20 oracle (§9) exposed a real low-ℳ over-prediction (+7–9% at ℳ≲2.5; +3.8% at ℳ=3) —
**not** a cell-*variance* error (Route A, with a 6%-*low* variance, *still* over-predicts +9%) but a
cell-density **shape/discreteness** effect in the transonic, shot-noise-dominated regime: cell-averaging
makes the realized cell density *less skewed* than the lognormal Poisson-mixture the model assumes.
Rather than fit that shot-dominated corner, we restrict to ℳ≥4 (residual <1.5% there) — losing **no
science**: cluster-forming GMCs are highly supersonic (ℳ~5–20; ℳ<4 is the transonic dense-core regime,
α_vir~1–2, c_s≈0.19 km/s). This is an honest scope statement, not a hack.

## 6. SBC-validity (non-negotiable)

Any **data-derived quantity** (the cell scale R, the shot-noise debias) must be computed **identically
in mock-generation and in inference** — the lesson from the POT `s_thr` / I2 fix. The measurement is a
deterministic function of the counts, so consistency is structural; the cell-scale rule must be a pure
function of the realized data.

## 7. Reuse map

- **Reuse:** `theory/cic.py::smoothed_log_variance` (the prediction), `theory/gaussianization.py`
  (Hermite/Gaussianization machinery), `theory/bm19.py::sigma_s_squared` (ℳ↔σ_s²),
  `field/sampling.py::sample_cic_counts` (the generator), the SBC driver `inference/sbc.py`.
- **New:** analytic shot-noise term; the data-side log-transform variance estimator (shot-debiased);
  the Gaussian σ_s² likelihood block; the cell-scale rule; wiring into `build_logdensity`; oracle.

## 8. Open technical decisions (to settle in the TDD, grounded in Neyrinck 2011)

1. Data estimator: exact rank-Gaussianization (Eq 1) vs modified `log₊` (Eq 2) — and `N=0` handling.
2. Shot-noise debiasing of `Var[ln(1+δ_cell)]` (Eq 3 vs an analytic Poisson-lognormal correction).
3. The principled cell-scale R rule (shot ≈ clustering); whether one R or a small ladder.

## 9. Validation plan (definition of done)

- **Oracle (AC20):** the §1 over-prediction table must flatten to **<6% across the calibrated prior
  ℳ∈[4,20]** — ✅ DONE: worst |rel|=1.44% (ℳ=8), residual flat (slope −6e-4), ℳ=4 edge +0.49%,
  ℳ=20 −0.31% (64³, n_real=6). The decisive quantitative gate. [publication plots → Task 9]
- **AC16** (single-point recovery) stays green.
- **AC18-ℳ** rank-uniformity becomes **uniform** (the xfail flips to pass).
- **σ(ℳ)-vs-N_star** forecast (honest, shot-noise-limited; mirrors AC17 for α).
- **Released-core 814 invariant**; full experimental suite green.

## 10. Scientific impact & honest scope

- **Method (the contribution):** a calibrated (SBC-validated), differentiable forward model that
  reframes star-cluster substructure as galaxy-clustering-style inference — novel vs the field's
  non-differentiable, non-calibrated diagnostics (Q, MST); imports cosmology Gaussianization SoTA to
  the stellar regime.
- **Headline science = relative/population trends:** β(environment), natal-turbulence→stellar-clustering,
  population tests of the gravoturbulent paradigm, substructure-as-a-clock (with gravax) — robust to the
  absolute-calibration caveats.
- **Honest limits:** σ_s²/ℳ is shot-noise-limited for sparse clusters; ℳ–b degenerate (b fixed);
  absolute ℳ needs validation vs real turbulence sims; optimality holds **within** the BM19
  lognormal+tail model class (3-pt held out as the null test).

## 11. Future arcs / extension points (explicitly enabled, not built now)

- **Velocity structure** → breaks ℳ–b, makes b inferrable, adds a kinematic dynamical clock.
- **2D Limber projection / light-cone** → real (projected) data; σ_s² is sample-size-independent and
  the log-density 2-pt projects cleanly.
- **SBI/WST frontier** (PatchNet, Wavelet Scattering Transform) → the genuinely phase-coherent /
  higher-order information beyond the GRF+marginal model; a differentiable WST amplitude is a natural
  tail-robust cross-check.

## 12. Considered alternatives (recorded)

- **Philosophy A — finite-N truncation of the mixing PDF.** Also physically correct (the observable is
  finite), but makes the *prediction* sample-size-dependent (conflates field property with sampling),
  less clean for population studies. Documented runner-up.
- **Bulk-of-P(N) (minimal patch).** Hides the fragile tail rather than fixing the representation;
  rejected as not-SoTA per Anna's "correct/versatile, not minimal" directive.
- **2D projection as the v1 mechanism.** Deferred to the larger re-scope arc (a re-architecture; the
  projected marginal is not BM19); kept as a §11 extension.

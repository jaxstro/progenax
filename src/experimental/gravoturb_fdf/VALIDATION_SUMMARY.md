# gravoturb_fdf — Validation Summary

**Status: EXPERIMENTAL** — follow-up paper, **not** part of the initial progenax/jaxstro
release and **not** shipped in the progenax wheel.

**Generated:** 2026-06-05 (clean-room rewrite); **differentiable-inference addendum 2026-06-06**
(AC11–AC17, see the Differentiable Inference section below).
**Every number below was printed by a committed acceptance script on the current tree** —
no prose claim of correctness exists without a fresh artifact behind it. Reproduce with:

```bash
cd progenax
PYTHONPATH=src:src/experimental python -m gravoturb_fdf.validation.acceptance
```

The acceptance suite lives in `gravoturb_fdf/validation/acceptance.py` (AC1–AC17) +
`gravoturb_fdf/validation/calibration.py` (the Q(f_sub) driver behind AC7). numpy/scipy are
permitted on this validation/diagnostics side; the `theory/`, `field/`, and `inference/` cores
are JAX-native.

---

## Acceptance criteria — fresh run (production scale)

| AC | Criterion | Threshold | Measured | Verdict |
|----|-----------|-----------|----------|---------|
| **AC1** | BM19 σ_s²(ℳ=5,b=0.4)=ln5; s_t(α=2)=1.5·ln5 | <1e-6 | 1.60944 / 2.41416 (abserr 0) | **PASS** |
| **AC1** | f_dense vs numeric quadrature (3 cases) | rel <1e-4 | relerr ≤5.8e-9 | **PASS** |
| **AC2** | Mass conservation ∫eˢ p_LN ds = 1 | =1 ± 1e-3 | 1.000000 (abserr 0) | **PASS** |
| **AC3** | ζ(p) anchors: ζ(0)=1, ζ(1)=1.0887, ζ(1.5)=√2, ζ(1.67)=1.79 | <0.1% | relerr ≤5.6e-4 | **PASS** |
| **AC4** | ζ_FDF (direct field) vs analytic ζ(p), p∈{0.5,1.0,1.5} | within few % | relerr ≤0.55% | **PASS** |
| **AC5** | CW04 Q vs Table 1 (3D radial models, A=πR²) | within CW04 σ | see below | **PASS** |
| **AC6** | **Cornerstone**: f_dense_realized vs BM19 f_dense (mass-conserving copula, 128³×8) | ens<1%, single<5% | see below | **PASS** |
| **AC7** | **Headline**: Q(f_sub) monotone↓ + Q∈[0.4,0.8] (64³×10×500★) | trend↓ + band | see below | **PASS** |
| **AC8** | Gradient signs: ∂σ_s²/∂ℳ>0, ∂f_dense/∂ℳ<0, ∂f_dense/∂α<0, ∂ζ/∂α<0 | sign | all correct | **PASS** |
| **AC9** | FD-vs-autodiff on f_dense(ℳ) and ζ(p) | rel <1e-4 | relerr ≤2e-9 | **PASS** |
| **AC10** | Full suite (uv) | 100% | 1065 passed (815 released-core + 250 experimental) | **PASS** |

---

## AC5 — CW04 Q estimator vs Cartwright & Whitworth (2004) Table 1

Estimator: `Q = m̄/s̄`, `m̄ = L_MST/√(N·A)` with **A = πR²** (the CW04 area convention;
reproduces Table 1 — convex-hull area biases Q high by ~+0.1). 2D-projected, 30 realizations,
N=200 points drawn from analytic 3D radial models `ρ(r)∝r^(−α)`.

| Model | Q (meas) ± σ | CW04 Table 1 | \|dev\| | Verdict |
|-------|--------------|--------------|--------|---------|
| 3D, α=0 (uniform) | 0.782 ± 0.019 | 0.79 ± 0.02 | 0.008 | PASS |
| 3D, α=1 | 0.832 ± 0.020 | 0.84 ± 0.03 | 0.008 | PASS |
| 3D, α=2 | 0.924 ± 0.031 | 0.93 ± 0.03 | 0.006 | PASS |

This is the same physics the released `progenax.diagnostics.substructure.compute_q_parameter`
uses; an equivalence test pins the two implementations together.

---

## AC6 — Cornerstone (the make-or-break)

Does a 3D realization reproduce the 1D BM19 dense-mass fraction? The mass-conserving rank
copula assigns each cell its slab-mass-averaged density at exact volume quantiles, so the
realized dense-mass fraction matches BM19 `f_dense` to O(1/N). **128³ grid, 8 realizations:**

| Grid | ℳ | b | α | f_dense (BM19) | ensemble bias | single-real max | Verdict |
|------|----|----|----|----------------|---------------|-----------------|---------|
| 128³ | 10.0 | 0.40 | 2.0 | 0.0568 | +0.004% | 0.004% | PASS |
| 128³ | 8.0 | 0.50 | 1.8 | 0.1161 | +0.003% | 0.003% | PASS |
| 128³ | 12.0 | 0.33 | 1.6 | 0.2194 | +0.001% | 0.001% | PASS |

Tolerances: ensemble \|bias\| < 1%, single-realization \|bias\| < 5%. **Measured biases are
~1000× inside tolerance.** (The superseded white-noise + point-value-copula pipeline gave a
−37% cornerstone bias; that path and its summary were deleted in the 2026-06 rewrite.)

---

## AC7 — Headline: f_sub → Q substructure calibration

Paired-realization design (one FBM field per realization, sampled at every f_sub) isolates the
f_sub effect from field-to-field scatter. **64³ grid, 10 realizations, 500 stars** (ℳ=8, b=0.5,
α=1.8, β=3.5):

| f_sub | Q ± σ |
|-------|-------|
| 0.00 | 0.647 ± 0.044 |
| 0.20 | 0.621 ± 0.059 |
| 0.40 | 0.599 ± 0.080 |
| 0.60 | 0.566 ± 0.100 |
| 0.80 | 0.519 ± 0.119 |

Linear slope **−0.156** (trend↓ PASS), strict-monotone **True**, all means within **[0.4, 0.8]**.
Q decreases as more stars are drawn from the dense FBM tail — i.e. more substructure → lower Q,
in the direction of CW04's fractal-D ladder. Per-point scatter grows with f_sub (a genuine FBM
property), so the headline is the negative slope + decreasing endpoints, reported with bands.

A smooth differentiable surrogate `q_surrogate(f_sub, σ_s, β)` (in `gravoturb_fdf.surrogate`)
emulates this calibration for gradient-based inference, since the star-sampling (categorical) and
CW04 Q (MST/scipy) steps are themselves non-differentiable.

---

## AC8 / AC9 — Gradients

| Quantity | sign / value | Verdict |
|----------|--------------|---------|
| ∂σ_s²/∂ℳ (M=5, b=0.4) | +3.20e-01 (>0) | PASS |
| ∂f_dense/∂ℳ (b=⅓, α=1.8) | −1.91e-02 (<0) | PASS |
| ∂f_dense/∂α (ℳ=8, b=⅓) | −4.55e-01 (<0) | PASS |
| ∂ζ/∂α (via p=3/α) | −1.06e+00 (<0) | PASS |
| AC9 f_dense'(ℳ) autodiff vs FD | relerr 4.5e-9 | PASS |
| AC9 ζ'(p) autodiff vs FD | relerr 1.6e-10 | PASS |

The public differentiable entry points (`sigma_s_squared`, `f_dense_bm19_full`,
`magnification_factor`, `bm19_icdf`) survive `jax.grad`; finite-difference and autodiff agree to
machine precision, including across the α→1 and p→2 guard regions.

---

## Differentiable inference (Phases 5–6) — AC11–AC17

The `inference/` layer predicts summary statistics analytically as smooth functions of
θ=(ℳ,b,α,β) and differentiates *those* (cosmology playbook: analytic prediction + likelihood +
blackjax NUTS), rather than differentiating the stochastic simulator. AC11–AC15 (log-density 2-pt
ξ_s via Gaussianization, ρ_g(β)/Limber projection, CIC moments + P(N), gradient validation, the
field-level Fisher forecast) are asserted in `acceptance.py`. The Phase-6 finish (2026-06-06) makes
**α — the BM19 power-law tail slope — inferable**:

### AC16 — joint (ℳ, α, β) HMC recovery (the headline)

α had been un-recoverable: a finite N-cell field truncates the power-law tail at
`s_max ≈ s_t + ln(N)/α`, so fitting the full infinite-tail PDF biased α high. The fix is a
**peaks-over-threshold (POT) truncated-exponential** likelihood on the gas-density exceedances
above a fixed `s_thr` — exact (the lognormal norm cancels → decoupled from σ_s²), shift-immune, and
geometry-free. Stellar counts-in-cells (clean Poisson sampler, matched grid) carry (ℳ,β); the POT
block carries α; a soft barrier keeps the chain where `s_t(θ) ≤ s_thr`. **160³ gas map
(N_tail=510), 500/1000 NUTS, injection θ=(ℳ5, α2.5, β3; b0.4 fixed):**

| param | posterior | truth | deviation | cover |
|-------|-----------|-------|-----------|-------|
| ℳ | 4.88 ± 0.65 | 5.0 | 0.18σ | ✓ |
| **α** | **2.533 ± 0.112** | **2.5** | **0.30σ** | ✓ |
| β | 3.20 ± 0.43 | 3.0 | 0.45σ | ✓ |

α posterior width 0.112 matches the truncation-corrected Fisher 0.113 (**ratio 0.99** — calibrated,
not covering-by-being-wide); `corr(ℳ,α) = −0.11` (the old ℳ–α degeneracy is broken). This is an
**injection-recovery** test of the inference machinery (mock drawn from the same BM19 model).

### AC17 — σ(α) vs N_tail forecast (the transferable result)

"How many independent tail elements N a gas map needs to measure α." Per-exceedance Fisher info of
the truncated exponential is `I(α) = 1/α² − L²e^{−αL}/(1−e^{−αL})²` (→ `α/√N` as `L→∞`). Validated
with genuine i.i.d. truncated-exponential draws (the rank copula's marginal is *deterministic* — the
exact order statistics — so it has zero 1-pt scatter and cannot validate this):

| N_tail | σ_emp (i.i.d.) | σ_Fisher | rel |
|--------|----------------|----------|-----|
| 61 | 0.366 | 0.356 | 0.03 |
| 158 | 0.204 | 0.210 | 0.03 |
| 327 | 0.145 | 0.143 | 0.01 |

√N law: empirical slope −0.556 (Fisher −0.544; ideal −0.5). **Caveat (honest, cf. AC15):** a
*realistic correlated* field (smooth copula, β=3) scatters ~2.5× wider than the i.i.d. bound
(N_eff ≈ N_tail/6) — red-spectrum tail cells are not independent. Option B cross-check: a
mass-conserving realization's dense-mass fraction matches `f_dense_bm19_full` to rel 0.000
(convergent, truncation-robust).

---

## What this validates (and what it does not)

**Validated:** the BM19 1D density-PDF scalars and mass normalization; the PP20 ζ(p) law and its
direct-field estimator; the CW04 Q estimator against Table 1; the mass-conserving copula
cornerstone; the f_sub→Q calibration trend; gradient correctness end-to-end.

**Not claimed:** that the f_sub→Q calibration reproduces a specific published fractal-D Q value to
high precision (it tracks the CW04 ladder *direction* with realistic scatter, reported as bands —
not tuned to hit a target); that the β(ℳ) interpolation is a derived law (it is a Kim & Ryu 2005
log-linear fit to measured density-spectrum slopes, clipped to physical limits — see the per-paper
note). Star positions from categorical sampling and the CW04 Q metric are non-differentiable by
construction; the differentiable interface is the fitted surrogate.

**Grounding:** Burkhart & Mocz (2019), Parmentier & Pasquali (2020), Padoan & Nordlund (2011),
Federrath et al. (2010), Heyer (2009), Kim & Ryu (2005), Lomax et al. (2018), Cartwright &
Whitworth (2004). Per-paper notes: `docs/website/99-bibliography/per-paper/`.

# gravoturb — Validation Summary

**Status: EXPERIMENTAL** — follow-up paper, **not** part of the initial progenax/jaxstro
release and **not** shipped in the progenax wheel.

**Generated:** 2026-06-05 (clean-room rewrite); **differentiable-inference addendum 2026-06-06**
(AC11–AC17, see the Differentiable Inference section below).
**Every number below was printed by a committed acceptance script on the current tree** —
no prose claim of correctness exists without a fresh artifact behind it. Reproduce with:

```bash
cd progenax
PYTHONPATH=src:src/experimental python -m gravoturb.validation.acceptance
```

The acceptance suite lives in `gravoturb/validation/acceptance.py` (AC1–AC17) +
`gravoturb/validation/calibration.py` (the Q(f_sub) driver behind AC7). numpy/scipy are
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
| **AC4** | ζ (direct field) vs analytic ζ(p), p∈{0.5,1.0,1.5} | within few % | relerr ≤0.55% | **PASS** |
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

The star-sampling (categorical) and CW04 Q (MST/scipy) steps are non-differentiable, so Q is a
validation/demo diagnostic only. Differentiable inference predicts summary statistics analytically
and differentiates *those* (the `inference/` layer, AC11–AC17 / Differentiable Inference section).
(The earlier fitted `q_surrogate` prototype was retired in Phase 7.)

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

The public differentiable entry points (`sigma_s_squared`, `dense_mass_fraction`,
`magnification_factor`, `log_density_icdf`) survive `jax.grad`; finite-difference and autodiff agree to
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
mass-conserving realization's dense-mass fraction matches `dense_mass_fraction` to rel 0.000
(convergent, truncation-robust).

---

## Cluster IC acceptance (Build 4 forward tool) — AC-IC0–AC-IC8

The `build_cluster_ic` layer (turbulent field → spherical envelope → star placement → coherent
velocities → COM + virial scaling) has its own acceptance suite,
`gravoturb/validation/cluster_acceptance.py`, under the same discipline: every number below
was printed by the committed script. Reproduce with:

```bash
PYTHONPATH=src:src/experimental python -m gravoturb.validation.cluster_acceptance
```

**Fresh run 2026-07-16** (fiducial ℳ=8, b=0.5, α=1.8, box=4 pc, 32³): **6/6 PASS.**

| AC | Criterion | Measured | Verdict |
|----|-----------|----------|---------|
| **AC-IC0** | envelope fidelity map (2026-07-16): turbulence-OFF realized r_half vs requested r_h | OFF bias 2.7% (32³ and 64³); ON: 1.23×/1.59×/1.71× at ℳ=4/8/12 (32³), resolution-converged → turbulent relocation is physical; **r_h is a SHAPE parameter** (re-run at Phase 1 close, amendment A4) | **PASS** |
| **AC-IC1** | envelope: median radius ↑ with r_h, concentrated vs uniform box | 0.412/0.641/0.837 pc for r_h=0.3/0.5/0.8; 0.837 < 0.85·1.678 | **PASS** |
| **AC-IC2** | realized virial ratio Q=T/\|V\| hits Q_target | \|err\| = 0.0 at Q_target ∈ {0.3, 0.5, 0.75} | **PASS** |
| **AC-IC3** | CW04 (m̄,s̄) plane separates β-substructure from concentration | Q↓ in β (1.119→0.837) and in r_h; m̄ swing: β 0.08 vs conc 0.46 | **PASS** |
| **AC-IC4** | velocity coherence (nearby stars co-move) | cosθ near +0.675 vs far −0.038 | **PASS** |
| **AC-IC5** | envelope+field construction differentiable | d(core mass)/d r_h = +626.8, finite/nonzero | **PASS** |
| **AC-IC6** | input β recovered from log-density P(k) slope | max \|err\| = 0.012; recovery-line slope 0.995 | **PASS** |

**AC-IC7 — FK12 multi-freefall placement (Phase 1 gate, fresh 2026-07-16): PASS.**
The default placement law is now ``p_⋆ ∝ w(s_turb)·ρ_total^{3/2}`` (FK12 Eq. 7 integrand,
t_ff ∝ ρ^{−1/2} Eq. 8, verified against the held PDF; ε/φ_t cancel in the PMF — see the
federrath-klessen-2012 per-paper note), with the former free ``f_sub`` replaced by the
derived ``tail_star_fraction`` (hard, from the actual placement PMF — the f_sub
successor) plus the smooth ``collapse_eligible_fraction`` (AD=FD < 1e-6). NB the two are
materially different (~0.97 vs ~0.44 at the fiducial): the eligible fraction is a
measure share, not a star fraction — conflating them was a caught review finding. Measured: (a) turbulence-OFF
envelope control vs an independent numpy ρ^{3/2} oracle, two-sample KS = 0.006 (< 0.015);
(b) collapse_eligible_fraction monotone ↓ in α at every ℳ (printed 4×3 table; ℳ-direction
characterized, regime-dependent per AC8); (c) Q(β) re-baselined under multi-freefall (β=2→4 ordering
preserved); (d) matched-fraction legacy comparison printed. The legacy ``two_population``
mode is retained for ablations and reproduces the pre-rename pins byte-exactly.
**A4 re-run of AC-IC0 under multi-freefall: PASS** against the placement-consistent
ρ^{3/2}-weighted reference (OFF bias 5.4%); note the ℳ-relocation is STRONGER than legacy
(realized/requested r_h up to ~1.9× at ℳ=12, 64³) and resolution-sensitive at 32³ for
ℳ≥8 — **use ≥64³ grids with multi-freefall at high Mach** (the low-resolution guard is
more binding under ρ^{3/2} weighting).

**AC-IC8 — physical velocity mode (Phase 2 gate, fresh 2026-07-16): PASS.**
``VelocitySpec(mode='physical', c_s=…, eta_v=1.0)`` sets the stellar mass-weighted 3-D
dispersion to σ_⋆ = η_v·ℳ·c_s (stars inherit the gas turbulence amplitude; η_v<1 for
subvirial-star studies, cf. Foster+2015) and **Q_virial becomes an output**; the
Bertoldi & McKee (1992) ``alpha_vir`` = 5σ_1D²r_h/(GM) on the realized cluster
(**1-D literature convention**, σ_1D = σ_3D/√3, so α_vir ~ 1 reads as virial on the
GMC scale — 2026-07-16 review fix: the 3-D form inflated the diagnostic 3×) is
reported in both modes as the consistency diagnostic. ``c_s`` is in km/s; the builder
requires ``units`` (a jaxstro UnitSystem consistent with G) for the conversion.
Measured: (a) σ_⋆ round trip exact to ≤2e-16 across ℳ∈{4,8,12} × η_v∈{0.5,1}
(gate bound <1%; scaling happens after COM removal); (b) emergent Q grid over
(ℳ, r_h) with 3-seed bands — Q monotone ↑ in both ℳ and r_h (0.028±0.006 at
ℳ=4/r_h=0.3 → 0.480±0.139 at ℳ=12/r_h=0.8; fiducial ℳ=8/r_h=0.5: Q≈0.10–0.16,
α_vir≈0.18 — the Larson-chain amplitude leaves these N=2000×1 M⊙ clusters strongly
SUBVIRIAL on both scales, the physically expected cold-birth regime), and Q(η_v=0.5)/Q(η_v=1) =
0.250000000000 (exact, frozen positions); (c) units pin 1 pc/Myr = 0.97779 km/s
(0.9778 ± 2e-4); (d) physical-mode gravax seam re-run PASS (COM ~1e-16, σ_⋆
round-trip < 1e-8 across the handoff, |ΔE/E| < 5e-3 leapfrog smoke). NB the emergent Q
couples the cloud ℳ to the STELLAR mass/size only through the user's choice of
(masses, r_h); ``cloud_spec_from_larson`` (M_ecl, SFE, ρ_cl → ℳ, β, b, box=2R_cloud)
closes that loop through the released Larson chain when cloud-level consistency is
wanted. ``mode='virial_target'`` is byte-identical to the pre-Phase-2 pipeline
(rename-pin gate re-passed).

**AC-IC9 — Helmholtz-coupled density–velocity construction (Phase 3 gate, fresh
2026-07-16, new script ``coupling_acceptance.py``): 5/5 PASS.**
``CloudSpec(coupling='helmholtz', beta=None)``: ONE white field drives both the velocity
realization (Helmholtz projectors P∥/P⊥, per-mode compressive power fraction exactly χ;
default χ = chi_f10(b) = b/√3, PDF-verified against F10 Eqs. 21–22/Fig. 8 — the radical
is over D only, and forced turbulence never reaches χ=1) and, via linearized continuity
(ĝ ∝ −i k·v̂∥, no new randomness), the density Gaussian carrier — so **β is DERIVED
(= β_v − 2)** and resolved with χ at builder entry by ``validate_spec_bundle``
(ADR-0041 Option A). Measured: (a) coupled log-density slope 1.629±0.025 / 1.951±0.025
at β_v = 11/3, 4 (derived 1.667/2.0, |err| < 0.05); (b) coupling strength
C = corr(s,−∇·v)·√(E_long/E_tot) tracks √χ (0.333/0.569/0.776 vs 0.316/0.548/0.760),
independent ablation −0.002; (c) coupled-mode re-pass of AC6 (rel 2e-4), σ_⋆ round trip
(1e-16), AC-IC4 coherence (near +0.827/far −0.237) at unweakened thresholds; (d) carrier
slope → 2.0 at 96³ + coupled≡independent equivalence per resolution; (e) mass-weighted
convergence signature ⟨−∇·v⟩ρ/σ_div = +1.735±0.007 coupled vs −0.006 independent.
TDD caught TWO instrument defects en route, both root-caused with discriminating
experiments and fixed without touching thresholds: (i) mixed-Nyquist bins break
transversality under the ``.real`` Hermitian symmetrization (2e-4 longitudinal leak,
corr 0.9896) → Nyquist planes zeroed in the vector construction; (ii) the draft gate
statistic corr(s,−∇·v) is scale-invariant and CANNOT depend on χ (measured 1.000
everywhere) → replaced by the amplitude-weighted C above (gate-statistic correction,
surfaced to Anna); an apparent β=2 slope depression was integer-|k| binning bias in
narrow windows (a pure k⁻² GRF read 1.83) → unbiased mode-level regression estimator.
Honest claim: perfect correlation on the compressive channel, none on the solenoidal —
the "frozen flow at star-formation epoch" limit, not resolved turbulence.

**AC-G1–G8 — Phase-4a stars+gas handoff (`TurbulentCloudIC`, fresh 2026-07-16, new
script ``gas_acceptance.py``): 8/8 PASS.** The Aim 2 handoff: the SAME cloud realization
is normalized to a physical parent cloud (ρ_cl = M_cl·ρ̃/∫ρ̃dV; M_cl = Σmᵢ/ε_global,
masses-first) and partitioned by the local free-fall model ε⋆ = 1−exp(−τ⋆w/t_ff)
(t_ff = √(3π/32Gρ_cl); τ⋆ by 120-step scan-bisection with the IFT derivative — AD/FD
agree to 6e-11). ``ClusterIC`` was REPLACED outright by the nested ``TurbulentCloudIC``
(stars/gas/fields/geometry/physics/ledger; gas=None star-only path; no shim), and the
physical velocity mode became **field-first** (ratified): the gas grid's volume-weighted
rms is σ_g = ℳ·c_s EXACTLY (measured rel err <1e-10), while the stellar COM-frame
dispersion is EMERGENT — characterized at σ_⋆/σ_g = 0.75–0.89 ± 0.07–0.14 (COM removal
strips the coherent box-scale bulk at β_v=4; AC-IC8(a) re-scoped to grid-exactness +
band, thresholds set from measurement). Measured: mass closure |residual|/M_cl ≤ 3e-15;
pointwise ρ⋆+ρ_g=ρ_cl to 2e-16 with positivity; continuous-partition SFE reproduced to
≤8e-16 across (ℳ, sfe); the sfe→0 limit reproduces the AC-IC7 w·ρ^{3/2} law (ratio-field
CV 4e-4); normalization exact + envelope-offset invariant at 16³/32³; joint stars+gas
COM and momentum close to <1e-8; deterministic at fixed seed. **Physics found by the
gates:** the freefall partition CAPS the reachable SFE at the collapse-eligible mass
share — the ceiling falls with ℳ (≈0.79/0.36/0.16 at ℳ=4/8/12 for the printed fiducial
normalization) and is seed-dependent; over-ceiling requests RAISE loudly (never clip),
and fully-consumed cells (ε⋆→1, ρ_g = 0 exactly) are counted, not hidden. virial_target
+ gas is refused (an imposed Q has no cloud meaning); ``ledger.gas_included`` labels
star-only products loudly. Ownership boundary held: NO feedback/B-splines/gas
gravity/evolution here — gravax owns those (Aim 2 Phases 2+).

**2026-07-16 review remediation (adversarial 8-angle code review, 10 verified findings):**
(i) the former ``f_sub_derived`` conflated two materially different quantities — replaced by
``tail_star_fraction`` (Σ_{s>s_t} p under the actual placement PMF; ~0.97 at the fiducial —
the honest f_sub successor, now the AC-IC7(d) matching knob) and ``collapse_eligible_fraction``
(the smooth ungated measure share, ~0.44 — the analytic hook); a unit test pins them apart.
(ii) AC-IC1/AC-IC4 now run in BOTH placement modes; **AC-IC4 initially FAILED under the
multi-freefall default at the 32³ fiducial** (near-alignment +0.013): the star sample
concentrates into ~8 cells and COM subtraction erases coherence — the recorded ≥64³-at-ℳ≥8
caveat biting our own fiducial. At 64³: near +0.631 / far +0.018, PASS. Resolution fix per the
caveat, thresholds untouched; ``ClusterIC.placement_n_eff`` (PMF inverse participation ratio)
added as the resolution-monitoring diagnostic (11.5→18→22 across 32³→64³→128³ at the fiducial).
(iii) Q_target=0 (cold collapse) and traced spec construction restored (regressions vs main);
byte-identity gate hardened (exact env fingerprint + GRAVOTURB_BYTE_GATE=1 strict mode).

**Honest caveats (2026-07-16 science audit):** (i) Q alone conflates β with concentration —
AC-IC3's (m̄,s̄) plane is the separable statistic; per-realization β identifiability from Q is
marginal (β=4 seed scatter ±0.196 vs β 2→4 response ≈0.28). (ii) The requested envelope r_h is a
*shape parameter*, not the realized concentration: with turbulence ON the sampled profile is
centrally suppressed (~3×) and wing-enhanced (~3×) relative to the analytic profile at 32³
(turbulence-OFF control matches the analytic profile to few %). (iii) The velocity field is an
independent GRF — spatially coherent; its AMPLITUDE is now Mach-set under the Phase-2
``mode='physical'`` (σ_⋆ = η_v·ℳ·c_s, AC-IC8), but the field remains statistically independent
of the density (no density–velocity coupling until Phase 3), and β_v is a free parameter. (iv) A Gravax seam + 2-crossing-time
smoke (2026-07-16, N=500) transferred cleanly (COM ~1e-16, Q=0.5000, \|ΔE/E\| ≤ 1e-3) and the
β=2-vs-β=4 substructure ordering survived evolution at ~1.5–2σ (3 seeds). The finalization design
(`docs/plans/2026-07-16-gravoturb-cluster-ic-finalization-design.md`, maintainer-local) addresses
(ii)–(iv). (v) **No gas potential**: the IC is the stellar system alone. In the embedded phase the
gas (1/SFE − 1 ≈ 4× the stellar mass at SFE=0.2) deepens the true potential, so the embedded state
is even MORE subvirial than the emergent Q reports; the classic SUPER-virial post-gas-expulsion
state (Hills 1980; Baumgardt & Kroupa 2007) is the other physical regime — representable via
``virial_target`` with Q>0.5 or η_v>1, but physical mode as-built reads "stars at birth, gas
potential neglected". (vi) **Hydro-only (no magnetic fields)**: B would suppress the density-PDF
variance (Molina+2012: σ_s² = ln(1 + b²ℳ²·β_plasma/(β_plasma+1))), the PN11 MHD critical density
(their Eq. 18) is explicitly not implemented (HD limit only), and magnetic support raises the gas
α_vir — so at fixed (ℳ, b) we somewhat overproduce dense-tail substructure. Internally consistent
(FK10's b calibration and BM19's PDF are hydro), but strongly magnetized applications need the
Molina extension. (vii) **IMF characterization (2026-07-16, Maschberger, Larson-closed cloud, physical mode,
3 seeds; committed artifact: ``cluster_acceptance.characterize_imf``)**: emergent Q tracks
TOTAL mass, not the mass-function shape — matched M_tot≈2029 M⊙ gives Q=0.167±0.027 vs
equal-mass 0.163±0.023 (|ΔQ|=0.004 < 3σ_seed, the gated check), while fixed-N=2000
(M_tot=737) warms to Q=0.340±0.029; the massive tail broadens the α_vir seed scatter
(±0.012→±0.042 in the 1-D convention; one draw hit m_max≈130 M⊙ — consider
``TruncatedIMF``/an m_max–M_ecl cap for production). Velocities are mass-independent by
construction (stars inherit gas velocity; NO primordial equipartition — the correct birth
state) and mass↔position assignment is random until Phase 4's λ_corr wiring (primordial
segregation, gate AC-IC10).

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
construction; the differentiable interface is the analytic predicted-statistics inference layer
(`inference/`, AC11–AC17), which differentiates the predicted summary statistics rather than the
stochastic simulator.

**Grounding:** Burkhart & Mocz (2019), Parmentier & Pasquali (2020), Padoan & Nordlund (2011),
Federrath et al. (2010), Heyer (2009), Kim & Ryu (2005), Lomax et al. (2018), Cartwright &
Whitworth (2004). Per-paper notes: `docs/website/99-bibliography/per-paper/`.

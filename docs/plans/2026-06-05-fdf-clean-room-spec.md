---
title: Gravoturbulent-FDF subsystem — clean-room publication-readiness spec
date: 2026-06-05
status: FINALIZED 2026-06-05 (decisions resolved with Anna; §8 is authoritative). Next: TDD implementation plan + build in a fresh session.
author: Claude Opus 4.8 (1M) under Anna's HITL direction
---

# Gravoturbulent-FDF subsystem — clean-room spec

## 0. Why this document exists

The FDF / gravoturbulent IC subsystem (`gravoturb/` + `cluster/fdf*`, ~3.3k LOC)
was built experimentally by older models with a documented tendency to fabricate,
exaggerate, and self-certify. This session already caught concrete instances:

- a PP20 ζ(p) **transcription typo** with a fake `p=1.3` pole, rationalised in
  docstrings as a "domain limit" (fixed in `pp20_magnification.py`; guide 6g; F10);
- a **fabricated** BM19 `s_t` formula in the website docs (6b);
- a PN11 prefactor off by ~2.3× (0.242 vs 0.547) (6a);
- the Paper-A validation suite **claiming** "BM19 implementation is correct, all
  formulas match" while actually running a **white-noise** density field through a
  **√N-less** Q estimator → a nonphysical headline `Q≈0.13`.

**Therefore: trust nothing.** Every formula, constant, citation, and "validated"
claim is re-derived from the held PDFs or re-measured before it is believed. The
existing code/docs/summaries are treated as *untrusted reference*, not truth.

## 1. Governing principles

1. **Trust-nothing / PDF-grounded.** The spec is the held PDFs
   (`docs/core-papers/`: BM19, PP20, PN11, FK10, Heyer+2009, Kim&Ryu) + first
   principles. The 6g-corrected guide is a *guide*, not gospel.
2. **Clean-room.** New code is authored from this spec, **not** copy-pasted from
   the old modules. A function is "ported" only after its formula is re-derived and
   its output re-checked numerically. Old code is **quarantined** (read-only) until
   the new path is validated end-to-end, then **deleted wholesale**.
3. **Validation-first / TDD.** The acceptance criteria (§4) are written before code;
   every function is RED→GREEN against a physics test. JAX-native, differentiable,
   both-env verified (jax 0.10.1 + conda/jax 0.7.0), atomic commits, HITL-gated.
4. **No self-certification.** "Validated" is only written next to a number that a
   committed, reproducible script just printed. No prose claims of correctness
   without a fresh artifact.

## 2. Scope

**IN** — the gravoturbulent FDF IC-generation pipeline:
cloud params (ℳ, b, α, η, Σ, R) → 1D PDF (BM19) → 3D field (GRF + rank copula) →
dense-tail selection → star sampling → Q diagnostic; plus PP20 ζ for the dense-gas
SFR interpretation, and the **f_sub→Q calibration** (the headline result).

**OUT (quarantine + delete, not rebuilt):** the displacement-field `fdf.py`,
`fractal_gw_legacy.py`, the `FDFCalibration` D→χ stub mapping, and the
fractal-dimension `D`-based path. **OUT (left as-is** unless the inventory finds an
FDF dependency): `mass_segregation`, `tidal`, two-component populations.

> Open decision Q-S1: confirm `fdf.py` displacement path is deleted, not rebuilt.

## 3. Physics spec (per layer)

Verification key: **[V]** re-grounded against the held PDF this session;
**[⚠P]** grounding pending (must verify against the named PDF before coding).

### 3.1 BM19 1D density PDF  — *[V] vs Burkhart & Mocz 2019 (ApJ 879, 129)*

- PDF width (Eq. 1): `σ_s² = ln(1 + b²ℳ²)`.
- Lognormal body with mass-conservation mean `s₀ = −σ_s²/2` (so ∫eˢ p_LN ds = 1).
- Transition density (Eq. 2): `s_t = (α − ½) σ_s²` — **derived, not a free parameter**.
- Power-law tail `p_PL(s) ∝ e^{−αs}` for `s ≥ s_t`; continuity sets the amplitude.
- Self-gravitating fraction (Eq. 19–20): mass-weighted piecewise integral
  `f_dense = M_PL / (M_LN(−∞,s_t) + M_PL)`, with
  `M_LN(−∞,s_t) = ½[1 + erf((s_t − σ_s²/2)/(√2 σ_s))]` (no extra e^{σ_s²/2}),
  `M_PL = A/(α−1) · e^{(1−α)s_t}`, `A = p_LN(s_t) e^{αs_t}`.
- PDF-slope ↔ radial-slope map: `p = 3/α` (κ = 3/α).
- Lognormal limit (α→∞, comparison only): `f_dense ≈ ½ erfc((s_t−σ_s²/2)/(√2σ_s))`.

### 3.2 PP20 magnification ζ  — *[V] vs Parmentier & Pasquali 2020 (ApJ 903, 56)*

- Analytic (Eq. 6): `ζ(p) = (3−p)^{3/2} / [2.6 (2−p)]` ≡ `2(3−p)^{3/2}/[3^{3/2}(2−p)]`
  (2.6 ≈ 3^{3/2}/2 = 2.598). **Well-behaved on 0 ≤ p < 2; ζ(0)=1; diverges only at p=2.**
  Anchors: ζ(1)=1.089, ζ(1.5)=√2, ζ(1.67)=1.79.
- Cored profile `ρ(r)=ρ_c[1+(r/r_c)²]^{−p/2}`: numerical ζ via trapezoid on r/R∈[0,1]
  (regularizes p→2; finite for p≥2).
- Direct field measurement `ζ_FDF = Σ ρ^{3/2} w / (M_dg ⟨ρ_dg⟩^{1/2})` — preferred for
  realistic (cored, non-power-law) geometry; ground truth.

### 3.3 PN11 classical critical density (alt only)  — *[V] vs Padoan & Nordlund 2011 (ApJ 730, 40)*

- Eq. 8: `ρ_crit/ρ₀ = 0.067 θ^{−2} α_vir ℳ²`, θ≈0.35 (PN11 p.3) ⇒ prefactor 0.547 (Eq. 11);
  `s_crit = ln(0.067 θ^{−2} α_vir ℳ²)`. Distinct from the FK12 `(π²φ_x²/5)` form.
- Kept as a clearly-labelled *classical alternative* to BM19's s_t; not the default path.

### 3.4 Turbulence / cloud-physics relations  — *[⚠P] GROUNDING PENDING*

These are the fabrication-risk items (F4/F7/F15). To be verified before coding:

- `σ_s² = ln(1+b²ℳ²)` attribution — **verify against FK10 (A&A 512, A81)**; confirm the
  exact equation number (code says "Eq. 14" — unverified) and that the b∈[1/3,1]
  driving range is theirs.  **[⚠P FK10]**
- β(ℳ) power-spectrum slope — currently a `tanh` interpolation between Kolmogorov
  `11/3` and a supersonic/Burgers value `≈4`. This is a **heuristic, not a derived law.**
  Verify the limiting values vs **FK10** (Kolmogorov subsonic) and **Kim&Ryu 2005
  (ApJL 630, L45)** (supersonic density-spectrum slope); the interpolation itself must
  be labelled a modelling choice, not "Federrath+2010 scaling".  **[⚠P FK10, Kim&Ryu]**
- α_vir(Σ) ∝ Σ⁻¹ and the normalization Σ₀ (PN11 path only) — **verify against
  Heyer+2009 (ApJ 699, 1092)**; the code's "Heyer & Dame 2015" cite is wrong (F4) and
  Σ₀=85 is unsourced.  **[⚠P Heyer+2009]**
- Larson σ_v(R)=σ_v0 R^α and ℳ=σ_v/c_s — standard; cite Larson 1981 / Solomon+1987;
  state the σ_v0/α convention chosen.

### 3.5 3D field realization (GRF + rank copula)  — *[V] (code read this session)*

Goal: a field with **both** the BM19 one-point PDF **and** turbulent P(k)∝k^{−β}.
1. Gaussian random field g(x) with P(k)∝k^{−β} (Hermitian-symmetrized, DC=0).
2. Standardize g→~N(0,1).
3. **Rank copula** (empirical CDF): `u = (rank(g)+0.5)/N` via double-argsort → exactly
   uniform regardless of g's realized (non-Gaussian) marginal.
4. `s = F_BM19⁻¹(u)` (tabulated CDF + interp); `ρ = exp(s)`; normalize.
- Why rank not Φ: a finite-grid GRF is only approximately Gaussian; at steep β the Φ
  remap collapses the dense tail. Rank is distribution-free → exact BM19 marginal at
  any β. Monotone → preserves spatial rank/clumps. P(k) mildly distorted by the
  nonlinear remap (accepted; marginal + clump locations are what set f_dense).
- Differentiability: BM19 params enter via the CDF table (smooth); ranks are frozen →
  grad-safe in (ℳ,b,α).
- A resolution guard must warn when < ~5 cells are expected above s_t.

### 3.6 Tail selection + star sampling  — *[V] (code read; to re-author clean)*

- Direct s>s_t soft-sigmoid tail weights `w = σ(κ(s−s_t))` (differentiable);
  `f_tail_actual = Σ w ρ / Σ ρ` must reproduce f_dense.
- PMFs: `p_tail ∝ w ρ`, `p_smooth ∝ ρ`. Sample `N_tail=round(f_sub N_*)` from p_tail,
  rest from p_smooth, sub-voxel jitter. (No white-noise field; no D-mapping.)

### 3.7 CW04 Q diagnostic  — *[V] vs the CW04-referenced `compute_q_parameter`*

- `Q = m̄/s̄`, `m̄ = L_MST/√(N·A)` (**√N normalization is mandatory**), `s̄ = ⟨r⟩/R`,
  2D-projected (CW04 methodology). Numpy/scipy (analysis-side; non-differentiable).
- Must reproduce CW04 anchors before use (see §4).

### 3.8 f_sub→Q calibration (headline)

Run FDF realizations across (ℳ,b,α) and f_sub, sample N_* stars, measure Q with the
**correct** estimator, fit Q(f_sub; σ_s,β) with realization scatter bands. N_* ≳ 500
(test N_* scaling). This is to be **measured and reported honestly**, not assumed.

## 4. Acceptance criteria (validation-first; each → a committed script that prints it)

| # | Criterion | Threshold |
|---|-----------|-----------|
| AC1 | BM19 scalars vs analytic (σ_s², s_t, f_dense, lognormal limit α→∞) | machine precision / <1e-6 |
| AC2 | Mass conservation ∫eˢ p_LN ds | =1 ± 1e-3 |
| AC3 | ζ analytic anchors ζ(0)=1, ζ(1)=1.089, ζ(1.5)=√2, ζ(1.67)=1.79 | <0.1% |
| AC4 | ζ_FDF vs analytic ζ(p), pure power law, p<1.7 | within few % |
| AC5 | **Q estimator sanity** vs CW04: uniform sphere 0.79±0.04; fractal D=1.5→0.47, 2.0→0.58, 2.5→0.70 | within CW04 σ |
| AC6 | **Cornerstone** f_tail_actual vs f_dense (rank copula, 128³) | \|bias\| < ~5% single, <1% ensemble |
| AC7 | Q(f_sub) physical & monotone↓, real dynamic range | Q∈[0.4,0.8], reported with bands |
| AC8 | Gradient signs ∂f_dense/∂ℳ<0, ∂f_dense/∂α<0, ∂ζ/∂α<0 | sign + FD-vs-autodiff agree |
| AC9 | FD-vs-autodiff on public differentiable entry points | rel err < 1e-4 |
| AC10 | Full suite both envs | 100% pass |

AC6 is the make-or-break: the old suite got −37% with white noise; the rank-copula
path must do far better, or the framework claim is not earned.

## 5. Target architecture (proposed — advise)

Consolidate the verified 1D theory under `gravoturb/`, move the 3D realization to a
clean `fdf/`, dedup (`sigma_s_squared` currently in two modules), enforce LOC limits:

```
progenax/gravoturb/   bm19.py  pp20.py  pn11.py  pdf.py  turbulence.py   (1D theory)
progenax/fdf/         field.py  tail.py  sampling.py  pipeline.py        (3D realization)
progenax/diagnostics/ q.py (CW04 Q)  substructure.py                     (analysis)
validation/gravoturb/ acceptance suite (AC1–AC10) + paper-figure scripts
```

> Open decision Q-S2: this layout (rename `pp20_magnification`→`pp20`, `bm19_pdf`→`pdf`,
> new `fdf/` package) vs minimal in-place. Trade-off: cleaner API + callsite churn.

## 6. Quarantine → delete map (PROVISIONAL — finalized by the Phase-0 inventory)

**Delete (demonstrated wrong/dead):**
- suite `compute_q_parameter_mst` (√N-less) + the white-noise field in `sample_ic_from_fdf`
- `_legacy_chi_to_beta`, `init_turbulent_density_field` (uncalibrated χ→β), if superseded
- `compute_tail_pmfs_pn11_legacy` / `local_overdensity` / `mode='pn11_legacy'`
- `FDFCalibration` v1_stub + `fractal_layer_from_D` / `density_layer_from_D`
- `fdf.py` (displacement) + `fractal_gw_legacy.py`  (per §2)
- stale `VALIDATION_SUMMARY.md` claims (B5/B6 ζ, dead `physics/*.py` paths)

**Keep but re-verify before porting:** bm19_model, bm19_pdf (rank copula),
pp20_magnification (post-fix), pn11_model, turbulence (after §3.4 grounding),
init_bm19_density_field, fdf_tail BM19 path + sampling, diagnostics/compute_q_parameter.

> Quarantine mechanism (Q-S3): move condemned code to `legacy/gravoturb-fdf-pre-rewrite/`
> (git-tracked, clearly marked, importer-free) for one commit, confirm suite green +
> nothing imports it, then delete in the next commit. Avoids "delete then discover a
> dependency."

## 7. Build sequence (each phase HITL-gated + both-env verified)

- **P0 — Adversarial inventory.** Read every in-scope file; classify VERIFIED /
  STALE-WRONG / DEAD with evidence; finalize §6. (Trust nothing.)
- **P1 — Acceptance harness.** Implement AC1–AC5 + AC8/AC9 as committed scripts/tests
  against the *kept* core; this is the "is the core actually correct?" gate. Includes
  the Q-estimator CW04 sanity (AC5) before any Q science.
- **P2 — Correct IC pipeline + cornerstone.** Re-author field/tail/sampling clean;
  wire rank copula; hit AC6. Decision point: does the cornerstone pass?
- **P3 — f_sub→Q calibration.** AC7; honest result + bands; figure scripts.
- **P4 — §3.4 turbulence grounding** (FK10/Heyer/Kim&Ryu) + PN11 alt.
- **P5 — Quarantine → delete** (§6) + consolidation/refactor (§5).
- **P6 — Docs.** Regenerate VALIDATION_SUMMARY from real runs; update website
  `10-theory/`, the canonical guide, READMEs, per-paper notes, CLAUDE.md counts.

> Build mechanism (Q-S4): in-place with quarantine (above) vs a git worktree for the
> rewrite. Recommend in-place + quarantine — the new code coexists with quarantined
> old until validated, then old is deleted.

## 8. Resolved decisions (2026-06-05, with Anna) — AUTHORITATIVE

Where this section differs from the draft body above, **this section wins.**

**Strategy & location**
- **Clean-room rewrite** from the PDF-grounded theory (trust nothing; author from spec, not old code).
- New package lives in **`src/experimental/`** — gravoturb+FDF is a **follow-up paper**, excluded from the initial progenax/jaxstro release.
- **Boundary = option A:** ONLY the gravoturb-FDF IC pipeline moves to experimental — `gravoturb/` (1D theory) + the density-field FDF (`fdf_density/`, `fdf_tail`, `gravoturbulent`, `fdf_calibration`) + the bm19_fdf validation suite. The **legacy displacement `fdf.py` + `fractal_gw_legacy.py` are DELETED** (not rebuilt). **General cluster machinery stays in core** (mass-seg, two-component, `build_binary_cluster` — all FDF-free). **`turbulence.py` (σ_ln_ρ/Larson/Mach) stays in core** because the released `EnvironmentIMF` needs it. Phase-0 severs 3 core→subsystem links: (1) drop `core.py`'s legacy-FDF IC branch, (2) keep turbulence in core, (3) drop the `fractal_gw_legacy` export.

**Naming.** "FDF" = the **density-field** realization (GRF + rank copula). The model = **combined gravoturb(1D) + FDF(3D)**. The legacy *displacement*-field `fdf.py` is a different, deleted thing.

**Physics semantics — split f_sub.** `SFE_dense ≡ η·f_dense` = gas→surviving-stars efficiency (physics OUTPUT/diagnostic); **`f_sub^spatial`** = fraction of *stars* in the substructured component (the IC knob). State the linking assumption explicitly (clustered SF traces dense-gas mass); never silently equate a gas-mass fraction with a stellar-count fraction.

**The 6-step algorithm is the centerpiece:**
1. **gravoturb 1D** → σ_s²=ln(1+b²ℳ²); s_t=(α−½)σ_s²; f_dense (Eq.19-20); SFE_dense=η·f_dense.  *(how much)*
2. **FDF 3D field** → GRF P(k)∝k^{−β} + **rank copula** → ρ(x) with BM19 marginal AND turbulent clumps.  *(where)*
3. **dense-tail mask** s(x)>s_t (soft sigmoid); check **f_tail_actual ≈ f_dense** (cornerstone).
4. **star sampling** N_tail=f_sub^spatial·N⋆ from tail, rest smooth.
5. **measure Q** (CW04, non-diff).
6. **calibrate** Q(f_sub^spatial; σ_s,β) with scatter bands.  *(headline)*
(PP20 ζ is a Part-III side branch for dense-gas SFR; not needed to generate ICs.)

**Q & differentiability.** CW04 **Q is the non-diff truth metric** (MST + convex hull); must reproduce CW04 anchors (AC5) before any Q science. The **differentiable interface is the smooth Q(f_sub; σ_s,β) surrogate** (fit once from measured Q). Differentiable-MST (stop-grad topology / perturbed optimizers / matrix-tree) is a documented OPTIONAL future extension, NOT a dependency — and note the upstream categorical star-sampling is also non-diff, so a diff-MST alone would not yield cloud→Q gradients.

**Acceptance bars (publication-grade).** AC6 cornerstone |f_tail−f_dense| < 5% single / <1% ensemble. AC7 Q(f_sub) monotone↓ AND near **CW04 fractal-D-derived targets** (D=2.0→Q≈0.58, D=1.5→Q≈0.47, …), with realization scatter bands. AC1–AC5, AC8–AC10 as in §4.

**Compute (P3): tiered.** Fast 64³ smoke (small grid × ~5 realizations × N⋆=500, minutes) to confirm trend/wiring; then 128³ production (denser grid × ~10–20 realizations × N⋆≤~2000), gated on the smoke passing.

**Layout: structured (§5)** under `src/experimental/<pkg>/`: `theory/` (bm19, pp20, pn11, pdf), `field/` (field, tail, sampling, pipeline), `diagnostics/` (q), `validation/`. (turbulence stays in core; experimental imports it.)

**Retirement: quarantine-in-place → delete.** Sever the 3 links + mark old modules deprecated/importer-free in one commit; confirm both-env suite green + nothing imports them; DELETE in a follow-up commit. New experimental code coexists during the transition. No worktree.

**Sequence:** P0 inventory → P1 acceptance harness (incl. AC5 Q-sanity) → P2 IC pipeline + cornerstone (AC6) → P3 f_sub→Q (AC7) → P4 turbulence grounding (FK10/Heyer/Kim&Ryu) → P5 quarantine→delete + consolidation → P6 docs. Each HITL-gated, TDD RED→GREEN, both-env verified, atomic commits.

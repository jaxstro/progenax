# Batch B: Cross-Engine Figure + Science Demos — Design

**Date:** 2026-06-10 (brainstormed with Anna; all four scoping decisions hers)
**Status:** APPROVED design; implementation plan to follow (superpowers:writing-plans)

## Purpose (Anna's choice: methods-paper material)

Publication-grade demos + figures aimed at the jaxstro methods paper (reusable
for the Cottrell census): end-to-end differentiable-inference showcases with
quantified recovery. Demo scripts live in `scripts/demo_*.py` (house
`validate_*` discipline: expected-vs-measured tables, exit-nonzero gates,
`_plotstyle` figures); one paper-section-prose page per demo in a NEW
`docs/website/60-science-demos/` section. gravax forward chain DEFERRED
(crosses package boundaries; own batch).

## Architecture: one shared physics-direct likelihood layer

Mock stars sampled at truth → binned kinematic summaries with finite-N errors
→ Gaussian likelihood against the model's ANALYTIC predictions → gradients
through the analytic side only (no resampling in the loss; the gravoturb
physics-direct precedent). Scripts-local helper `scripts/_demo_inference.py`
(sibling of `_plotstyle.py`; NOT a package API addition yet — YAGNI):

- `binned_dispersions(pos, vel, group_ids, r_edges)` → per-group σ̂(r), SE = σ/√(2n)
- Gaussian χ² likelihood builder over a `predict_fn(θ) → σ_model(r_bins)`
- MLE driver (optimistix BFGS vs optax Adam — pick empirically at plan time)
  + Fisher errors via `jax.hessian`
- blackjax NUTS reused exactly as the experimental gravoturb layer does

Unconstrained reparametrization everywhere (logit mass fraction, log radii);
Engine B realizability diagnostic checked at θ̂ and reported.

## Deliverables

### B1 — Cross-engine agreement figure (`scripts/demo_cross_engine.py`)

One King model (W₀=5, g=1) built TWICE: Engine A (DF-defined coupled ODE) and
Engine B (density-defined Eddington). Overlay ρ(r), σ_1d(r), f(E) + residual
strip with freshly measured KS / σ-dev numbers (ledger anchors: 2e-4 / 3e-4).
The paper's "two independent derivations agree" credibility figure; assembles
already-validated machinery. ~1 day.

### B2 — Self-consistent IMF + equipartition recovery (`scripts/demo_delta_recovery.py`)

**Anna's extension (2026-06-10): the IMF is part of the inference.**

- **Truth:** Maschberger IMF, high-mass slope α_true=2.3 (other shape params
  fixed at literature defaults), mass range chosen so the top group is
  well-populated; δ_true=0.4, W₀=5, g=1; J=4 mass groups on FIXED log-m edges
  (μ_j(α), N_frac_j(α) smooth in α — no re-binning discontinuities);
  N=3×10⁴ (10⁵ ensembles affordable post-fusion if shot noise demands).
- **Joint likelihood:** L = L_kin(σ̂_j(r) | δ, W₀, α) × L_massfn(m_i | α).
  α enters BOTH channels: the observed mass sample AND the group weights/means
  that the equipartition law w_j = μ_j^(−δ) couples to kinematics.
- **Inference:** joint gradient MLE over (α, δ, W₀) + Fisher; blackjax NUTS
  corner on one dataset.
- **Figures:** (a) per-group σ(r) data vs best fit + mass function with fitted
  Maschberger; (b) Fisher-ellipse panel: kinematics-only (α, δ) degeneracy vs
  the joint constraint (the referee-memorable panel); (c) NUTS corner
  (α, δ, W₀); (d) **wrong-IMF bias curve** δ̂(α_assumed), α_assumed ∈
  [1.9, 2.7], kinematics-only, quoting dδ̂/dα ± uncertainty — the actionable
  "assumed-IMF bias" number (HEADLINE panel, Anna's pick: both); (e)
  **robustness grid** α_true ∈ {1.9, 2.3, 2.7} × joint recovery
  (unbiasedness across IMF shapes).
- **Gates:** joint (α̂, δ̂, Ŵ₀) within 3σ per seed; ensemble bias consistent
  with zero; bias-curve slope REPORTED with uncertainty, not asserted.
- **Plan-time risks to pin:** (1) `Maschberger` differentiability w.r.t. its
  slope parameter (binaries demo used the segmented-power-law layer; a
  Jacobian addition would be contained package TDD); (2) exact
  `from_imf`/`from_mass_segregation` signature for traced-α group
  construction; (3) top-group occupancy vs N (≥~500 stars).

### B3 — Halo+core two-population recovery (`scripts/demo_halo_core.py`)

Truth = the validated Engine B headline mix (Plummer halo r_h=2 + EFF core,
fracs 0.6/0.4, OM r_a on the halo). Recover (t, r_a, r_h) — the triple with
ledger-validated AD — by MLE on σ̂(r) + β̂(r); Fisher errors; same gates.
MLE only (no second NUTS).

## Estimator policy (Anna's choice)

Gradient MLE headline + bias-vs-N panels on both inference demos; ONE NUTS
posterior corner (B2). Timings reported, recovery gated.

## Testing & close-out discipline

- Library-like helpers (`_demo_inference.py`; any Maschberger-differentiability
  package addition) get full TDD; demo scripts are gated CLIs.
- Figures → `validation/plots/` → copied to `docs/website/60-science-demos/figures/`.
- New section wired into `myst.yml`; dashboard cross-link from 50-validation.
- Close-out: FULL gate, memory gates, myst build, completion doc, merge to
  local main on Anna's go. NO push without explicit go.

## Out of scope

gravax forward chain (deferred, own batch); promoting `_demo_inference.py`
into the package API (revisit after the demos prove the shapes); SBC-grade
calibration of the NUTS posterior (cite honestly as future work on the page).

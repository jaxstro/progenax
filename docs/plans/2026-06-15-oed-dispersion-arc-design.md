# OED demo + differentiable dispersion capability — ratified design

**Date:** 2026-06-15
**Status:** RATIFIED (Anna HITL, brainstorming complete). Supersedes the OED slice (idea #2) of
`2026-06-15-five-novel-demos-design.md` with a concrete, code-grounded two-phase design.
**Branch:** `docs/five-novel-demos-design` (local only).
**Scope guard:** progenax-only — single-epoch inference from differentiable equilibrium ICs. No
gravax N-body, no fluxax photometry, no startrax tracks.

This is the first build of the five-novel-demos arc: **idea #2, Bayesian optimal experimental
design (OED)**. Anna's ratified ordering is **#2 first, then brainstorm #1 (hierarchical)** — with a
`tinygp` emulator carried forward as *reusable expensive-forward-model infrastructure* for #1, not
just a speed hack. Idea #1 is deferred to its own later brainstorm.

## Why OED, and why a capability phase first

OED optimizes the *observation itself*: given a differentiable Fisher, compute
`∂(information)/∂(design)` and gradient-ascend the observing strategy. Almost no astro simulator can
do this. The headline target is the **Osipkov–Merritt anisotropy radius `r_a`**, chosen because the
**RV-vs-PM design split is physically an anisotropy lever** (radial velocities measure `σ_r`, proper
motions measure `σ_t`, and `r_a` *is* the `σ_r/σ_t` asymmetry). The optimizer therefore *discovers*
that PM stars belong in the outskirts — an emergent, interpretable, telescope-time result.

**Adversarial verification (live repo, 2026-06-15) found one load-bearing gap:** the velocity DFs are
**samplers** — they expose `sample_velocities(...)` but **no** `σ_r(r)`, `σ_t(r)` getter. A Fisher
needs the smooth sensitivity of the *predicted* observable, not the gradient of a noisy Monte-Carlo
realization. So the arc splits into two phases, **Phase 0 gated separately** (it is packaged `src/`
and must meet the full Definition of Complete before any demo code).

## Phase 0 — packaged differentiable dispersion capability

**New public API** (explicit-units policy; population moments, not sample estimates):

```python
result = df.dispersion_profile(r, M, G)   # r:(R,) radii; M: total mass; G explicit
# -> NamedTuple DispersionProfile(sigma_r, sigma_t, sigma_1d, beta)  each (R,)
```

Added to `PlummerVelocityDF`, `EFFVelocityDF`, `MichieVelocityDF`. (Michie's anisotropy arg is `r_a`,
not `anisotropy_radius` — a known naming difference, preserved.)

**Implementation (ratified):**
- **Plummer-OM → analytic.** Isotropic `σ_1d²(r) = GM/(6√(r²+a²))` is closed-form (already in the DF
  docstring). OM via a single differentiable anisotropic-Jeans quadrature with `β(r)=r²/(r²+r_a²)`
  and the analytic Plummer ρ, M, Φ.
- **EFF-OM & Michie → second velocity moment of the existing `f`-table.** `σ_r²(r) = ∫ v_r² f d³v / ρ`
  by quadrature over the same `(E_grid, f_grid)` that `eddington_invert` already builds for the
  sampler. **Self-consistent with the sampler by construction** — no re-derivation of `M(r)`.
- `σ_t = σ_r / √(1 + r²/r_a²)`; `σ_1d² = (σ_r² + 2σ_t²)/3` (exact OM relation, Merritt 1985 Eq. 15).

**Shared primitive:** `src/progenax/kinematics/dispersion.py` holds the Jeans quadrature + the
f-table second-moment kernel; the three DF methods are thin wrappers.

**Mixed-pairing caveat (documented in the API):** the getter returns the dispersion self-consistent
with **the DF's own potential**. In a mixed pairing (e.g. Plummer positions + King velocities) that is
not the dispersion of the position profile — stated explicitly in the docstring.

**Gates (Definition of Complete):**
- **Physics (3-way anchor):** analytic/f-table `σ_r(r)` agrees with the empirical binned dispersion
  from `binned_sigma_beta` on a large sample (to MC error); Plummer analytic-Jeans agrees with its
  f-table second moment. Anchors are physical (no fitted fudge).
- **Differentiability:** AD-vs-FD on `∂σ_r/∂(r_a, M, r_h)` — pure jnp, reverse-mode (`jacrev`;
  forward-mode forbidden by diffrax `custom_vjp`).
- **Registries (self-policing — must update or the gate fails):**
  `api_coverage/manifest.py` (new symbol), `physics_registry/manifest.py` (the anchor),
  `grad_audit/manifest.py` + `registry.py` (the AD-vs-FD entry),
  `provenance_registry/manifest.py` (Merritt-1985 / Jeans provenance).
- **Full released-core gate** green (`tests/unit tests/integration tests/validation -n auto`).

**Out of scope for Phase 0:** the tidal-truncation `r_t` hard-kink differentiability hazard is owned
by the differentiability-gradient-audit arc, not this one. Flagged, not solved.

## Phase 1 — the scaffolded pedagogical OED demo (B14)

**Forward model & likelihood.** Mock = stars from the OM DF → binned `σ_1d(r)`, `β(r)` with honest
standard errors (`binned_sigma_beta`). The *predicted* observable for the Fisher is the deterministic
Phase-0 `dispersion_profile()`. Likelihood = weighted Gaussian on the binned dispersions
(`gaussian_loglike`).

**Additive-Fisher backbone (the load-bearing idea).** Fisher information is additive over independent
data, so

```
F(design) = Σ_bins Σ_channels  w[bin, channel] · F_block[bin, channel]
```

where each per-(radius-bin × {RV→σ_r, PM→σ_t}) `F_block` is computed **once** via `jacrev` through
`dispersion_profile`. The design variables are the weights `w`; optimizing them is a tiny smooth
problem with **gradients trivial and linear in `w`** — the forward model is never re-differentiated,
and the forward-mode ban never bites. The RV/PM coupling then *emerges*: since `β(r)` grows outward,
the c-optimal design spends PM stars in the outskirts.

**Optimality criterion (ratified):** **c-optimality** headlines — minimize the marginal target
variance `(F⁻¹)_{target,target}` after marginalizing nuisances (`r_h`, `M`, …). **D-optimality**
(`log det F`) and **A-optimality** (`tr F⁻¹`) are computed alongside as the pedagogical contrast that
shows *why* they differ from c-optimality under nuisance degeneracy.

**Three pedagogical stages (growing design space):**
1. radial weights × RV/PM split, fixed-N, equal cost → the `r_a` headline figure.
2. + **magnitude limit** via `zams_luminosity` (Tout+1996) — a differentiable depth-vs-area selection
   weight; powers the **M_dyn** cross-check (`M` enters through `GM`).
3. + **epochs / unequal cost** — PM precision ∝ epochs, RV needs spectroscopy; one defensible
   cost-ratio knob; a realistic telescope budget.

**Optimizer:** `optax` Adam on the unconstrained design (softmax→simplex allocation; `expit` for
bounded knobs — both already in `_demo_inference.py`). Multi-start for the non-convex landscape.

**Gates:**
1. Optimized design reaches target `σ(r_a)` with **N× fewer stars** than uniform — report the factor.
2. Optimum is interpretable (PMs in the outskirts; recovers B7-style "info in the outskirts" from
   first principles).
3. **AD-vs-FD** on `∂(log det F)/∂design` and `∂(F⁻¹)_target/∂design`.

**Figures:** (1) information-vs-design optimization path; (2) optimal radial weighting + RV/PM split
over `σ(r)`/`β(r)`; (3) precision-vs-budget frontier (designed vs uniform); (4) c-vs-D-vs-A contrast;
(5) M_dyn cross-check design.

**Scope/non-goals:** static single-shot design (no sequential/online); equilibrium OM models only;
self-consistent mock (no real catalog, no cross-channel systematics).

## File layout

```
src/progenax/kinematics/dispersion.py            # NEW (Phase 0): Jeans + f-table 2nd-moment primitives
  └ dispersion_profile() on Plummer/EFF/Michie DF classes
tests/unit/kinematics/test_dispersion.py         # NEW: API, shapes, grads
tests/validation/test_dispersion_physics.py      # NEW: 3-way physics anchor
  └ + registry updates (api_coverage, physics_registry, grad_audit, provenance_registry)
scripts/_demo_oed.py                             # NEW (Phase 1): additive Fisher, optimality criteria, optimizer
scripts/demo_oed.py                              # NEW (Phase 1): gated CLI, 3 stages, run-record
docs/website/60-science-demos/optimal-design.md  # NEW (Phase 1): B14 MyST page
docs/website/60-science-demos/figures/demo_oed*.png
.claude-work/OED_DEMO_COMPLETE.md                # completion doc
```

**Helper placement (cross-cutting Q6):** OED helpers live in a new scripts-local `_demo_oed.py` that
*imports* `_demo_inference.py`; the dispersion capability is packaged `src/`.

**Dependencies:** zero new. Pure jnp + existing diffrax (Phase 0); `optax` (already `[experimental]`,
already imported by `_demo_inference.py`) for Phase 1.

## Sequencing & HITL

1. **Phase 0** mini-arc: TDD → full gate → **Anna merge-go** → (per branch-lifecycle) the dispersion
   capability lands before any Phase-1 code.
2. **Phase 1** OED demo: TDD harness → gated CLI exit 0 + run-record → figures inspected → B14 MyST
   page `myst build --html` 0 warnings → completion doc → STATUS + brain.

Anna approves every design decision, plan, and merge. Verify **locally** (CI minutes tight); nothing
pushed/merged without explicit go. Commit per task; messages end with the Co-Authored-By trailer.

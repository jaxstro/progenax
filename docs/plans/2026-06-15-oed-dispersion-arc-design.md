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

**New public API** (explicit-units policy; population moments, not sample estimates). Placement
**revised 2026-06-15 (Anna)**: a **free function on the profile**, not a method on the DF — the
dispersion is a property of the *(potential, anisotropy)* pair, so it lives with the profile (which
owns ρ, M, Φ), not the sampler. This kills the ρ/M duplication and the mixed-pairing footgun (you pass
the *actual* profile), and decouples the forward model from the stochastic sampler:

```python
from progenax import jeans_dispersion        # exported in progenax.__all__
result = jeans_dispersion(profile, r_a, r, M, G)   # profile: SpatialProfile; r_a: OM radius or None
# -> NamedTuple DispersionProfile(r, sigma_r, sigma_t, sigma_1d, beta)  each (R,)
```

Works for `PlummerProfile`, `EFFProfile`, `MichieProfile` (and any profile exposing `density(r)`).

**Implementation (ratified, revised):**
- **Unified anisotropic-Jeans quadrature** over the profile's own `density(s)`:
  `ρσ_r²(r) = 1/(r²+r_a²) · ∫_r^∞ (s²+r_a²) ρ(s) GM(<s)/s² ds`, with the **enclosed mass from a
  quadrature of `profile.density`** (`M(<s) = M·∫₀ˢρs'²ds'/∫₀^∞ρs'²ds'`) — builder-quality, NOT a
  re-differentiated Ψ (Anna's call: avoids boundary noise near r=0/r_t). For Plummer the closed forms
  agree analytically; the isotropic closed form `σ_1d²=GM/(6√(r²+a²))` is the validation truth.
- `σ_t² = σ_r²·r_a²/(r_a²+r²)`; `σ_1d² = (σ_r²+2σ_t²)/3`; `β = r²/(r²+r_a²)` (exact OM, Merritt 1985 Eq. 15).
- **f-table second moment is the cross-check, ISOTROPIC-ONLY** (`σ_r²=∫v_r²f d³v/ρ`, a clean 1-D speed
  moment over the DF's stored `f`-table): per Anna's "both, cross-checked" call, Jeans (primary) and
  f-moment are gated to agree to table resolution — a numerical-convergence discriminator, not a
  loosened tolerance. (The OM f-moment is a 2-D `(v_r,v_t)` integral — *not* the 1-D speed moment; for
  the anisotropic case the cross-check is Jeans-vs-empirical-sampler instead.)

**Projected (OBSERVED) dispersions — `project_dispersion` (ratified 2026-06-15, Anna).** The 3-D
σ_r/σ_t are *not* observable; telescopes measure **projections** along the line of sight. A companion
free function delivers the three observables via the Binney & Mamon (1982) integrals:

```python
from progenax import project_dispersion
proj = project_dispersion(profile, r_a, R, M, G)   # R: projected (on-sky) radii
# -> ProjectedDispersion(R, sigma_los, sigma_pm_r, sigma_pm_t, beta_proj)
#    sigma_los  = RV channel; sigma_pm_r/sigma_pm_t = PM channels (B&M82 kernels)
```
The singular LOS integral `∫_R^∞ g(r) r/√(r²−R²) dr` is evaluated via the substitution `r²=R²+u²`
(→ `∫₀^∞ g(√(R²+u²)) du`, singularity cancels, fully differentiable). Isotropic limit (β=0):
σ_los = σ_pm_r = σ_pm_t = σ_1d (a free correctness check). This makes the OED's "RV↔σ_r, PM↔σ_t"
claim physically honest and the capability Gaia-ready.

**Validation strengthening (Anna, gaps 2–5):** (a) f-moment cross-check isotropic-only (above);
(b) tight **analytic Merritt (1985) OM-Plummer oracle** (rtol ~1e-3) replaces the 5% MC anchor where a
closed form exists; (c) a **quadrature self-convergence study** (trapezoid O(h²) under s-grid
refinement); (d) cheap **invariants** (`σ²(2M)=2σ²(M)`, `σ²(λG)=λσ²(G)`), an **r_a validity-domain
guard** (the free fn must not silently accept `r_a < 0.75a` where the Plummer OM DF is unphysical), and
a **jit smoke test** (OED evaluates this many times).

**Shared module:** `src/progenax/kinematics/dispersion.py` holds `DispersionProfile`,
`jeans_dispersion`, and the `f`-table second-moment kernel.

**Registry cost of the free-function placement:** `jeans_dispersion` is a new `progenax.__all__`
symbol → it is categorized in all four registries (api_coverage SYMBOL_TESTS; physics
EXEMPT_NON_MODEL — a forward-model helper, not an equilibrium model; grad_audit SYMBOL_CATEGORY=AUDITED
+ MUST_AUDIT pairs + Cases; provenance the Jeans/Merritt row). Accepted: the capability should be
first-class and gated.

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

**Forward model & likelihood.** Mock = stars from the OM DF, projected to on-sky radius `R` → binned
**projected** dispersions `σ_los(R)` (RV channel) and `σ_pm,R(R)`/`σ_pm,T(R)` (PM channels) with honest
standard errors. The *predicted* observable for the Fisher is the deterministic Phase-0
**`project_dispersion()`** (B&M82) — RV and PM are genuine, distinct projections of the *same* σ_r/β,
not an idealized "measure σ_r directly." Likelihood = weighted Gaussian on the binned projected
dispersions (`gaussian_loglike`).

**Additive-Fisher backbone (the load-bearing idea).** Fisher information is additive over independent
data, so

```
F(design) = Σ_bins Σ_channels  w[bin, channel] · F_block[bin, channel]
```

where each per-(radius-bin × {RV→σ_los, PM_R→σ_pm,R, PM_T→σ_pm,T}) `F_block` is computed **once** via
`jacrev` through `project_dispersion`. The design variables are the weights `w`; optimizing them is a
tiny smooth problem with **gradients trivial and linear in `w`** — the forward model is never
re-differentiated, and the forward-mode ban never bites. The RV/PM coupling then *emerges* from the
B&M82 kernels: the σ_pm,T/σ_los and σ_pm,R/σ_los ratios carry β(r), which grows outward, so the
c-optimal design spends PM stars in the outskirts.

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
src/progenax/kinematics/dispersion.py            # NEW (Phase 0): jeans_dispersion() + project_dispersion() (B&M82) free fns
  └ both exported in progenax.__all__ (profile-based; DF only for the isotropic f-moment cross-check)
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

## Deferred caveats & versatility roadmap (FOLLOW-UP — Anna, 2026-06-15)

Phase 0 ships the **spherical, single-population, mass-follows-light** dispersion forward model. These
are *documented scope boundaries*, not flaws — but Anna's stated goal is to eventually make the
kinematic forward model **fully complete and versatile**. Captured here as a tracked follow-up so the
caveats become a roadmap, not a permanent ceiling. Each is a self-contained later increment:

1. **Rotation.** B&M82 assumes no streaming. Real clusters rotate; `apply_solid_body_rotation` /
   `apply_differential_rotation` already exist. Add a rotational mean-velocity field to the projected
   observables (σ_los acquires a `v_los(R, φ)` mean; PM gains a systemic-rotation signature).
2. **Non-sphericity / flattening.** B&M82 is spherical. A versatile model needs axisymmetric (or
   triaxial) Jeans — a much larger increment; flag as a separate research arc, not a quick add.
3. **Tracer ≠ mass (mass-follows-light relaxation).** Phase 0 uses `ν = ρ`. Generalize to an
   independent tracer density `ν(r)` ≠ mass density `ρ(r)` (the standard Jeans-modeling separation of
   luminous tracers from total mass) so the forward model can fit real photometric tracers in a DM/
   total-mass potential.
4. **The `r_t` truncation kink** (hard cutoff) — owned by the differentiability-gradient-audit arc;
   confirm the dispersion paths inherit a clean (or documented-blocked) gradient there.
5. **Multi-population / multi-mass** kinematics (per-species σ via `MultiComponentCluster`).
6. **Arbitrary / native anisotropy β(r)** (Phase-0 finding, Anna 2026-06-15). `jeans_dispersion`
   currently imposes the **Osipkov–Merritt** law β(r)=r²/(r²+r_a²) on any profile — exact for
   Plummer/EFF, but the **Michie–King** model has its OWN anisotropy law (agrees in the core, diverges
   outward), so `jeans_dispersion(MichieProfile, r_a)` is "Michie density under OM," not the Michie
   equilibrium (validated only at inner radii in Phase 0). Generalize the solver to accept a native
   β(r) via the general integrating factor `f(r)=exp(2∫β(s)/s ds)`,
   `ρσ_r²(r)=1/f(r)·∫_r^∞ f(s)ρ(s)GM/s² ds` — makes Michie and any custom anisotropy fully correct at
   all radii. Self-contained later increment.

**Action:** these go in the B14 MyST caveat audit as "current scope / planned extensions," in
`STATUS.md`, and as a `brain` capture so they resurface. Revisit after the OED demo (Phase 1) lands —
ideally fold (1) and (3) in first (smallest, highest-value for real-data versatility).

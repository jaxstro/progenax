# OED demo — Phase 1, Stage 1 (the `r_a` headline) — ratified design

**Date:** 2026-06-16
**Status:** RATIFIED (Anna HITL, brainstorming complete).
**Branch:** `feat/oed-demo-stage1` (local only).
**Parent design:** `docs/plans/2026-06-15-oed-dispersion-arc-design.md` (the two-phase arc).
**Builds on:** Phase 0 + Phase 0.5 (merged, `origin/main` @ `1db8838`) — the differentiable
`project_dispersion` (B&M82) forward model with clean reverse-mode grads wrt `r_a, M, r_h`.

This design covers **only Stage 1** of the three-stage OED demo. Per Anna's HITL call, Stage 1
ships and gates on its own; **Stages 2 (magnitude-limit / M_dyn) and 3 (epochs / cost budget) are a
later brainstorm** after Stage 1 is reviewed and merged.

## Honored decisions (do not re-litigate)

- **ADRs 0001–0010** (c-optimality headline; two-phase architecture; `jeans_dispersion` free
  function; **additive-Fisher backbone**; scaffolded stages; grid forward model; OED-first ordering;
  `beta_fn`/`df_moment_dispersion`; Michie-W0 deferral; polar quadrature).
- The headline forward model is **OM-Plummer** via `project_dispersion`, *not* the exact Michie
  `df_moment_dispersion` — the latter's `∂/∂W0` gradient is deferred/xfail (ADR 0009), so it cannot
  back a Fisher. Plummer-OM has clean grads across the whole radial grid (Phase 0.5 fixed the two
  sqrt(0)→NaN-grad Criticals that would otherwise have poisoned the Fisher).

## What the demo *is* (locked via brainstorm, 2026-06-16)

A **pure pre-data Bayesian optimal experimental design**: given a fixed star budget, compute where
on the sky (radius) and in which kinematic channel (RV vs proper-motion) to spend stars so as to
**minimize the marginal variance of the Osipkov–Merritt anisotropy radius `r_a`**. The optimization
is fully deterministic (no mock sampling in the loop); a single mock draw + MLE afterwards
*calibrates* the Fisher as a gate.

The scientific punchline we are hunting: the optimizer **discovers** that proper-motion stars belong
in the **outskirts** and RV stars in the centre — an emergent, interpretable telescope-time result,
because the B&M82 PM-tangential kernel carries `β(r)` which grows outward.

## The mock cluster (generic GC-scale, unnamed — no overclaim)

| Quantity | Value | Note |
|---|---|---|
| `M` | `1e5` M⊙ | total mass (nuisance) |
| `r_h` | `3.0` pc | half-mass radius (nuisance); `a = r_h·√(2^{2/3}−1)` |
| `r_a` | `2·r_h = 6.0` pc | **OM anisotropy radius — the TARGET** (well inside validity; `r_a ≫ 0.75a`) |
| `d` | `4.0` kpc | distance — converts PM error mas/yr → km/s |
| `σ_RV` | `1.0` km/s | per-star spectroscopic RV error |
| `σ_PM` | `0.05` mas/yr → `≈0.95` km/s @ 4 kpc | per-star astrometric error per PM axis |
| `N_total` | swept (few×10³ scale) | budget; the frontier figure sweeps it |
| units | `STELLAR` (M⊙, pc, Myr); convert km/s explicitly | |

The σ_RV ≈ σ_PM **parity at 4 kpc is deliberate**: neither channel trivially dominates, so the
optimal split is decided purely by *where each channel carries more information about `r_a`*.

## Model and the additive-Fisher reduction

**Parameters** θ = (`r_a`, `M`, `r_h`). `r_a` is the **target**; `M`, `r_h` are **nuisances**.
c-optimality minimizes `(F⁻¹)_{r_a,r_a}` — the marginal target variance after profiling nuisances
(automatically given by the (r_a,r_a) element of the *full* inverse).

**Predicted observable.** Three channels from `project_dispersion(profile, r_a, R, M, G)` evaluated
at `K=12` log-spaced on-sky bin-centre radii `R_k` out to `~3 r_h`:

```
g(θ) = { σ_los(R_k), σ_pm,r(R_k), σ_pm,t(R_k) }     k = 1..K
```

**Per-datum error.** A dispersion from `n` stars (per-star error `ε`, predicted dispersion `σ`)
has Gaussian error `δσ² = (σ² + ε²) / (2n)`.

**The backbone (ADR 0004 made concrete).**

```
F(design) = Σ_{bin b, channel c}  n_eff,{b,c} · M_{b,c}
   M_{b,c}  = 2 · J_{b,c} J_{b,c}ᵀ / (σ²_{b,c} + ε²_c)     # 3×3 per-STAR block (design-INDEPENDENT)
   J_{b,c}  = ∂ σ_pred,{b,c} / ∂θ                          # via jacrev(project_dispersion), ONCE
   n_eff,{b,c} = n_{b,c} · completeness_b                  # fixed realism multiplier (below)
```

`M_{b,c}` is computed **once** (reverse-mode `jacrev` through `project_dispersion`, the only place
the forward model is differentiated). The design enters only as the linear weights `n_{b,c}`, so
the entire optimization is `F = Σ n·(c·M)` → invert 3×3 → read an element: trivial gradients. We use
`jacrev` because it is the supported/tested AD path for all profiles and stays correct under a
King/Michie swap (those equilibrium-solver profiles hit a `custom_vjp` ODE with no forward-mode
rule). On the Plummer path here there is no ODE, so forward-mode would also work — `jacrev` is the
robust choice, not a forced one.

## Design space, criteria, optimizer

**Design variables.** Allocate budget `N_total` over (`K=12` bins × 3 channels = 36 cells).
Unconstrained `z ∈ ℝ³⁶`, `n_{b,c} = N_total · softmax(z)_{b,c}` (differentiable simplex; the
`softmax`/`expit` reparam already in `scripts/_demo_inference.py`).

**Fixed completeness (realism, NOT a design knob).** `completeness_b = c(R_b)` is a smooth faint-end
roll-off (≈1 in the core, <1 in the outskirts), applied **identically to all three channels** for
Stage 1 (channel-dependent depth is a Stage-2/3 refinement). It folds into `M_{b,c}` as above, so the
optimizer must **over-target the faint outskirts to extract information there** — making "PMs →
outskirts" a harder-won, more credible result. Functional form: a logistic in `R` with a defensible
turnover near `~2 r_h` (tuned in the plan; documented as illustrative, not a real selection function).

**Three criteria (same `F = Σ n·(c·M)`):**
- **c-optimality (headline):** minimize `(F⁻¹)_{r_a,r_a}`.
- **D-optimality:** maximize `log det F`.
- **A-optimality:** minimize `tr F⁻¹`.

Pedagogy: D and A weight all three parameters democratically; c targets `r_a` after marginalizing
`M`, `r_h`. Under the `M`↔`r_a` (through `GM`) and `r_h`↔`r_a` degeneracies, c allocates to
**different radii** than D/A. We plot the three allocations side-by-side so *why they differ* is
visible, not asserted.

**Optimizer.** `optax` Adam on `z`, **multi-start** (non-convex c-landscape), keep best. ~36 design
vars → milliseconds/start. Criteria and `∂/∂z` are pure linear algebra over precomputed `M` blocks
→ AD-clean.

## Calibration draw (one stochastic check — a GATE, not in the loop)

At `design*`: allocate `n_{b,c}` stars, draw a mock catalog from the existing OM-Plummer sampler
(`PlummerProfile.sample_positions` + `PlummerVelocityDF(anisotropy_radius=r_a).sample_velocities`),
project onto the sky (LOS = ẑ: `R=√(x²+y²)`, `v_los=v_z`, decompose plane-of-sky velocity into
`v_pm,r`/`v_pm,t`), bin, measure σ̂ per (bin, channel), run MLE (`mle_adam` / `fisher_cov` from
`_demo_inference.py`). **Assert realized `Cov̂_{r_a,r_a} ≈ (F⁻¹)_{r_a,r_a}`** within tolerance —
proving the pre-data Fisher is calibrated and the design is trustworthy.

## Figures (5)

1. Optimization path — c-criterion (σ(r_a)) vs iteration, multi-start traces.
2. **Headline** — optimal radial weighting + RV/PM split, overlaid on `σ(r)` / `β(r)`.
3. c-vs-D-vs-A allocations side-by-side (the criterion-disagreement lesson).
4. Precision-vs-budget frontier — designed vs uniform → the **"N× fewer stars"** number.
5. Calibration — realized MLE Cov vs `F⁻¹` (the gate, visualized).

## Definition of Complete (gates)

1. `design*` reaches the target `σ(r_a)` with **N× fewer stars** than a uniform allocation — report
   the factor.
2. **Interpretability** — quantitative assertion that the optimal PM fraction *increases* with radius.
3. **AD-vs-FD** on `∂(criterion)/∂design` for c, D, and A.
4. **Calibration** assertion (realized MLE `Cov_{r_a,r_a}` ≈ `(F⁻¹)_{r_a,r_a}`) passes.
5. Unit + harness tests green; gated CLI exits 0 + writes a run-record; B14 MyST page `myst build`
   0 warnings; completion doc; STATUS + brain.

## File layout

```
scripts/_demo_oed.py        # additive Fisher (Σ n·c·M), c/D/A criteria, optimizer, sky-projection helper
scripts/demo_oed.py         # gated CLI: Stage 1, run-record
tests/unit/test_demo_oed.py # Fisher additivity, criteria correctness, AD-vs-FD, calibration
docs/website/60-science-demos/optimal-design.md       # B14 page (+ caveat audit: scope = Stage 1)
docs/website/60-science-demos/figures/demo_oed_*.png
.claude-work/OED_DEMO_STAGE1_COMPLETE.md              # completion doc (gitignored, local)
```

**Dependencies:** zero new (`jnp` + `diffrax` + `optax`, all present). The demo is **scripts-level**,
not packaged `src/` → **no registry burden** (Phase 0 already registered the packaged symbols; this
is a consumer).

## Scope / non-goals (Stage 1)

- Single-shot, pre-data design (no sequential/online OED).
- OM-Plummer equilibrium mock only; self-consistent (no real catalog, no cross-channel systematics).
- Magnitude limit enters as **fixed realism** only (depth is not yet a design variable — Stage 2).
- Single LOS projection (no flattening/rotation — roadmap items).

## Sequencing & HITL

1. `writing-plans` → TDD plan (`docs/plans/2026-06-16-oed-demo-stage1-plan.md`).
2. Subagent-driven TDD, code review per task.
3. Full released-core gate green locally (CI minutes tight — verify LOCALLY).
4. Gated CLI exit 0 + run-record; figures inspected; B14 MyST `myst build` 0 warnings.
5. Completion doc; STATUS + brain.
6. **Anna merge-go** → merge to local main → push on her word.

Anna approves every design decision, plan, and merge. Nothing pushed/merged without explicit go.
Commit per task; messages end with the `Co-Authored-By` trailer.

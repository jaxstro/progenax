# Follow-up: gravax dynamical mass-segregation experiment (DEFERRED)

**Date:** 2026-06-08 · **Status:** deferred (do progenax-side validation + plots first).
**Decision (Anna, 2026-06-08):** do **not** validate with gravax yet. Validate and plot the
segregation *diagnostics and IC machinery in progenax* (static configs, no time evolution); leave the
gravax N-body runs for a later session. This note records the deferred experiment so it can be picked
up without re-deriving.

## Why deferred
The forward-science "ICs → dynamical outcome" figures (Fig 4 Λ_MSR(t), Fig 5 knob→outcome grid)
require **gravax N-body evolution** of the FDF cluster ICs — a real mini-project (integrator choice,
N, timescales, ensemble, controls) that depends on `gravax` being wired to `build_cluster_ic` output.
The progenax-side pieces (the Λ_MSR diagnostic, the new density-correlated mass placement, the
mass-weighted substructure metric, and all the *static* IC + diagnostic validation/plots) are
independent of gravax and worth nailing first.

## What IS being done now (progenax-only, no gravax)
- Tier A: validate `compute_lambda_msr` on **static constructed** configs (unsegregated→1,
  hand-constructed exact, maximal→≫1, inverse→<1, estimator convergence, binary caveat) + plots.
- Tier C: build + validate `correlated_mass_assignment` (density-rank↔mass-rank, λ_corr knob) and a
  grounded **mass-weighted substructure** metric — all measured on *static* ICs.
- Figs 1–3 (hero / fidelity / knobs) + the segregation-diagnostic validation figures.

## DEFERRED to the gravax session (this ticket)
**Goal:** reproduce the Allison et al. (2009, ApJ 700 L99) short-timescale result with our ICs and turn
it into Figs 4–5.

1. **Wiring.** `build_cluster_ic` → `gravax.ParticleSystem.from_ic(ic, units=STELLAR)` →
   integrator. Confirm units (pc, M⊙, Myr / pc·Myr⁻¹) thread cleanly.
2. **Integrator.** Violent-relaxation (collapse-driven) segregation over ~1 crossing time ⇒ a
   **softened collisionless** integrator (PEFRL, ε≈0.05·d_mean) is adequate and cheap; not slow
   two-body relaxation. Add an **energy-conservation check** per run. (Consider Hermite/IAS15 only if
   a 2-body-segregation regime is wanted.)
3. **ICs / arms.** IMF masses (Chabrier/Kroupa). Headline contrast: substructured + sub-virial
   (envelope-light config so turbulent substructure dominates — recall the spherical envelope pushes
   CW04 Q up to ~0.8–0.9 while the *pure* turbulent field is Q≈0.6) vs smooth Plummer + virial.
   **Two mass-placement conditions** (Anna's choice): λ_corr=0 (dynamical-only) and λ_corr>0
   (primordial); the mass-weighted metric quantifies the primordial case.
4. **Diagnostic over time.** `compute_lambda_msr(pos, masses, N_massive≈10,20,50,100)` per snapshot →
   Λ_MSR(t). Ensemble ~15–50 seeds/arm (fractal/IC scatter is large — L99 used 50).
5. **Pre-flight gate.** 1–2 seeds, single arm pair: is the substructured-vs-smooth Λ_MSR contrast
   real before committing to the full ensemble? Honest caveat: our turbulence reaches Q≈0.6
   (*moderately* substructured), not Allison's deep-fractal Q≈0.2–0.4, so the contrast may be milder.
6. **Figs.** Fig 4 = Λ_MSR(t) two arms + ensemble bands + snapshot row (t=0, 1, ~5 t_cross) coloured
   by mass. Fig 5 = Λ_MSR(t≈1–2 t_cross) over the (β, Q, λ_corr) knob grid.

**Literature anchor (Tier B end-to-end).** A cool (Q=0.3) fractal (D=1.6) N=1000 cluster should
evolve Λ(N=10) from 1 to a *few* within ~1 Myr, segregating only down to ~2–4 M⊙ (L99 Fig. 2).
For the exact ONC Λ value + the formal Λ_MSR equation, **obtain the companion MNRAS 395,1449 PDF**
(not currently held) — see [allison-2009 per-paper note](../website/99-bibliography/per-paper/allison-2009.md).

## Released-core note found during grounding
`progenax.diagnostics.compute_lambda_msr` docstring mis-cites "Allison 2009, ApJ 700, L99, **Eq. 1**"
for Λ_MSR — but **Eq. 1 of L99 is the Spitzer t_seg relation**, not Λ_MSR (formal eq. is in MNRAS
395,1449). Fix the citation during the Tier-A released-core pass.

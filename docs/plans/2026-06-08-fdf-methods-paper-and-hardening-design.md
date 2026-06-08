# FDF methods paper — figure set, forward-science experiment & hardening plan (DEFERRED)

**Date:** 2026-06-08 · **Branch:** `gravoturb-fdf-sbc-validation` · nothing pushed.
**Status: DEFERRED design reference.** The methods-paper figures are *not* being built now; this
document captures the full design so a future session can pick it up once the experimental
`gravoturb_fdf` modules are hardened and trusted (see the Hardening checklist at the end). It
consolidates the 2026-06-07/08 brainstorm (forward-science-enabler thesis, mass-segregation headline,
the (m̄,s̄)/Λ_MSR/Σ diagnostics, Tier-C primordial placement) into one place.

## Thesis & scope

**Forward-science enabler:** "Realistic, substructured, spherical young-cluster ICs (turbulent BM19
density + chosen virial Q + coherent velocities) map to *dynamical outcomes*." The ICs are the setup;
the **headline dynamical outcome is mass segregation** (Allison et al. 2009): natal substructure +
sub-virial state drives rapid dynamical mass segregation. The distinctive asset is the *combination*
— a differentiable-forward FDF IC generator feeding N-body, with validated substructure/segregation
diagnostics, all in the jaxstro stack.

## The 5-figure set

| # | Figure | Thesis | Status |
|---|--------|--------|--------|
| **1** | **Hero / pipeline** (1×4): turbulent field → +envelope → stars → +velocities → 3D+projection | "what the tool produces" | buildable now (curate `cluster_acceptance`) |
| **2** | **Physical fidelity** (1×2): BM19 density PDF (sampled vs analytic) ∥ β recovery (log-density slope vs input β, 1:1, ≤0.5%) | "ICs faithful; β controllable & recoverable" | buildable now |
| **3** | **Knobs + diagnostic** (1×3): (m̄,s̄) plane (β vs concentration independent) + radial-profile turbulence-OFF/ON control + Q-vs-β/Q-vs-r_h conflation | "clean, separable substructure/shape knobs" | buildable now |
| **4** | **HEADLINE: dynamical mass segregation** — Λ_MSR(t): substructured+sub-virial vs smooth+virial, ensemble bands + snapshot row | "natal substructure → rapid segregation; the ICs matter" | **needs gravax runs** |
| **5** | **Enabler payoff** — outcome vs IC knobs: Λ_MSR(t≈1–2 t_cross) over the (β, Q, λ_corr) grid; + primordial-vs-dynamical (m–Σ) | "the IC knobs map to a controllable, measurable outcome" | **needs gravax runs** |

Figs 1–3 are replots/curation of already-validated `cluster_acceptance.py` data (+ the committed
website figures). Figs 4–5 are the forward-science payload and require the deferred gravax experiment.

## The gravax dynamical-segregation experiment (Figs 4–5)

Full spec lives in `docs/notes/2026-06-08-gravax-segregation-validation-followup.md`. Essentials:
`build_cluster_ic` → `gravax.ParticleSystem.from_ic(units=STELLAR)` → **PEFRL** (collisionless,
ε≈0.05·d_mean; this is *violent-relaxation* segregation, Allison's regime, not slow 2-body),
energy-conservation check, ~5–10 crossing times, ensemble ~15–50 seeds/arm (fractal scatter is large).
Arms: substructured+sub-virial (envelope-light so turbulence dominates; recall the spherical envelope
pushes CW04 Q to ~0.8–0.9 while the *pure* turbulent field is Q≈0.6) vs smooth Plummer+virial; **two
mass-placement conditions** — λ_corr=0 (dynamical-only) and λ_corr>0 (primordial, via
`gravoturb_fdf.masses.correlated_mass_assignment`). Diagnostics over time: `compute_lambda_msr`
(concentration) + `mass_density_segregation` (m–Σ, substructure-robust). **Pre-flight gate:** 1–2
seeds confirm the substructured-vs-smooth Λ_MSR contrast is real before the full ensemble. Honest
caveat: our turbulence reaches Q≈0.6 (*moderately* substructured), not Allison's deep-fractal Q≈0.2–0.4,
so the contrast may be milder. Literature anchor: a cool (Q=0.3) fractal (D=1.6) N=1000 cluster should
evolve Λ(N=10) from 1 → a few within ~1 Myr, segregating only down to ~2–4 M⊙ (Allison L99 Fig. 2;
exact ONC Λ + formal eq. need the MNRAS 395,1449 PDF, not yet held).

## What is BUILT and trustworthy now (this session)

- **Λ_MSR diagnostic** (`progenax.diagnostics.compute_lambda_msr`, released-core): validated against
  analytic ground truth (`tests/validation/test_mass_segregation_physics.py`, 8 tests); definition
  verified vs the held Allison L99 PDF. Non-differentiable (scipy MST), by design.
- **Mass-weighted substructure metric** (`gravoturb_fdf.diagnostics.mass_density`, experimental):
  M&C 2011 local Σ=(k−1)/(πr_k²) + m–Σ plane; 6 tests incl. exact formula + primordial detection;
  verified vs the held M&C 2011 PDF. Non-differentiable (kNN).
- **Density-correlated mass placement** (`gravoturb_fdf.masses.correlated_mass_assignment`,
  experimental): McLuster partial-shuffle on a density key, λ_corr knob; 5 tests. Non-diff (ranking).
- **FDF cluster IC forward tool** (`gravoturb_fdf.cluster.build_cluster_ic`): 6/6 acceptance, 5-fig
  gallery, β recovery ≤0.5%, fixed radial-profile control.

## Hardening checklist — promote `gravoturb_fdf` from experimental → trusted (before the paper)

The FDF subsystem is **repo-only experimental** (not in the released wheel). Before the methods paper,
the following must hold (this is the "trust gate"):

1. **gravax integration validated end-to-end** — `build_cluster_ic` output evolves stably in gravax
   (energy conservation |ΔE/E| within integrator spec; COM/units sane); a regression test pins it.
2. **The Figs-4/5 experiment run + pre-flight passed** — the substructure→segregation contrast is
   real at our achievable Q≈0.6, with quantified magnitude (or the honest null reported).
3. **β_v grounded** — the velocity-spectrum slope is currently *ungrounded* (flagged in
   `field/velocity.py`); ground against Heyer 2009 / Federrath compressible-turbulence spectra before
   any β_v-dependent claim.
4. **Resolution/convergence** — the cluster-IC results (envelope median, Λ_MSR contrast) shown stable
   vs grid resolution (32³→64³→128³) and N_stars.
5. **A differentiable inference path decided** (see below) — or the paper explicitly scopes inference
   as SBI-only.
6. **jaxstroviz port** — publication figure styling currently ad-hoc in validation scripts; the
   committed website figures are the interim curated set.
7. **Filamentary morphology** — remains Nice-To-Do (docs/plans/2026-06-07-…spherical-ic-design § NICE-TO-DO);
   only needed if a dynamical observable distinguishes filament-vs-blob ICs at matched P(k)+Q.

## Differentiability & inference status (design-relevant)

- **Forward generation is differentiable** in the segregation strength `lambda_seg`
  (`generate_cluster_ic` + `MassSegregationLayer`; verified `∂/∂λ_seg ≈ 8300`, finite) and in the FDF
  density construction (envelope/field). Star *placement* is categorical (non-diff), as is the
  positions output — accepted (spec §8).
- **The segregation diagnostics are non-differentiable** — Λ_MSR (scipy MST) and m–Σ (kNN) cannot be
  backpropagated. `q_approx` exists as a *differentiable substructure-geometry* surrogate, but there is
  **no differentiable mass-segregation diagnostic**.
- **Consequence for inference:** SBI (simulation-based) is the clean path — the forward model is fast
  and samplable over (β, Q, λ_seg, λ_corr); the validated diagnostics serve as summaries; no gradients
  through observables required. **Gradient-based (HMC with an analytic/diff likelihood) is NOT yet
  available** for segregation, because there is no differentiable segregation observable. Closing that
  gap (a differentiable Λ_MSR/Σ surrogate, à la `q_approx`) is the prerequisite for clean
  gradient-based inference of segregation parameters.

# B12 — The binary-inflated dynamical mass: a "confidently wrong" virial mass + its differentiable fix

**Date:** 2026-06-14
**Status:** Design ratified (brainstorm, Anna HITL — 4 decisions confirmed). Next: implementation plan.
**Branch (impl):** `feat/binary-dynamical-mass-demo` (off `main`).
**Kind:** Science demo (B12), `scripts/` + `docs/` only — **no `src/progenax/` change → released-core gate unaffected** (Batch-C pattern). NOT a released-core feature; no registry/coverage dance.

## Scientific premise

Unresolved binaries inflate the measured line-of-sight velocity dispersion of a stellar
system, biasing the **virial / dynamical mass high**. In low-dispersion systems
(ultra-faint dwarfs, low-mass GCs) this is a *large fractional* effect and has driven real
debates about M/L ratios and dark-matter content. The demo (a) quantifies the bias, (b) shows
that a **dispersion-only** analysis cannot remove it (the `σ_true`–`f_b` degeneracy is rank-1),
and (c) demonstrates a **differentiable joint recovery** from the *non-Gaussian wings* of the
velocity distribution that returns an unbiased dynamical mass — with a Fisher/CRLB forecast vs
sample size `N` and RV precision `ε`.

This is the kinematic companion to **B4** (which measures `f_b` photometrically from the
unresolved-binary mass function); B4 → B12 is a natural multi-channel story.

## Ratified decisions (Anna HITL)

1. **Scope = C** (bias → differentiable fix → Fisher forecast), one coherent demo.
2. **RV realism = A** — flux-weighted SB2 blend via ZAMS luminosity:
   `v_obs = (L₁v₁,los + L₂v₂,los)/(L₁+L₂)`, `L = progenax.stellar.zams_luminosity(m)`. High-q
   binaries self-cancel (`Δ→0`); low-q are primary-dominated (`Δ→v₁`). Uses BOTH the Moe P–q
   coupling AND the just-shipped ZAMS relations.
3. **Likelihood = A** — binned global `v_los` distribution, analytic mixture
   `μ_k(σ,f_b) = N·[(1−f_b)𝒩_k(σ) + f_b·(𝒩(σ)⊛K_orb)_k]`, Poisson likelihood; reuses
   `_demo_inference` (poisson_loglike / mle_adam / poisson_fisher_information / logit/expit).
4. **Recovery = A** — joint `(σ_true, f_b)`; headline = **dispersion-only is rank-1, the wings
   make it full-rank**; Moe period dist treated as known (fixed `K_orb`); honest **ε-precision
   floor** (`f_b → f_b(P<P_max(ε))`). Period-slope recovery is a noted STRETCH, not the headline.

## §1 — Forward model & contamination kernel

One isotropic single-population cluster (King or Plummer) → each star's LOS COM velocity
`v_COM ~ 𝒩(0, σ_true²)`; `σ_true` sets the true virial mass `M ∝ σ_true² r_h / G`. A fraction
`f_b` are unresolved Moe binaries (`MoeJointOrbit` over P–q–e).

**Internal blend velocity** for a binary (choice A): components orbit the COM with
`v₁=(m₂/M)v_rel`, `v₂=−(m₁/M)v_rel`; the unresolved centroid is `v_obs = v_COM + Δ` with
`Δ = (L₁v₁,los + L₂v₂,los)/(L₁+L₂)`, `L₁,L₂ = zams_luminosity(m₁,m₂; Z)`. Algebra:
`Δ = [L₁m₂ − L₂m₁]/[(L₁+L₂)M] · v_rel,los`. q→1 ⇒ Δ→0 (cancellation); low q ⇒ Δ→v₁ (SB1).

**Kernel `K_orb`** = distribution of `Δ` over a large Moe pool (sample P,q,e → `to_state` →
component v's → random orbital phase + isotropic inclination → flux-weight by ZAMS L → project
to LOS). **σ_true-independent** ⇒ precomputed once.

**Observed distribution:** `p(v) = [(1−f_b)𝒩(0,σ²) + f_b·(𝒩(0,σ²)⊛K_orb)] ⊛ 𝒩(0,ε²)`, `ε` =
per-star RV precision (noise convolution → sub-ε binaries blend into the singles = the floor).

## §2 — Bias, degeneracy, differentiable fix

- **Bias:** `σ_obs² = σ_true² + Var(K_orb) + ε²` ⇒ `M_naive ∝ σ_obs²` biased high by
  `σ_obs²/σ_true²`; reported vs `f_b`.
- **Degeneracy:** with only `σ_obs`, `(σ_true, f_b)` are perfectly degenerate ⇒ dispersion-only
  Fisher **rank-1**.
- **Fix:** the binary term injects **non-Gaussian wings** (`K_orb` tails) no Gaussian `σ_true`
  can mimic. Recover `(σ_true,f_b)` by `mle_adam` on the binned Poisson likelihood (unconstrained
  via `logit/expit`); recovered `σ_true` → `M_dyn` is **unbiased**. Full-distribution Fisher is
  **full-rank** (the wings supply the 2nd constraint).
- **Forecast:** `poisson_fisher_information` → `Cov(σ_true, f_b)` ⇒ `σ(σ_true)`, `σ(f_b)`, ρ vs
  `(N, ε)`. **ε-floor:** as `ε` grows `K_orb` smears into noise, sub-ε binaries vanish ⇒
  recoverable `f_b → f_b(P<P_max(ε))` (honest scope panel).

## §3 — Regime, gates, artifacts

**Regime defaults (tunable):** UFD-like — `σ_true ≈ 5 km/s`, metal-poor `Z ≈ 10⁻³`, isotropic
King/Plummer, Kroupa/Maschberger IMF, Moe P–q–e, `f_b ≈ 0.5`, `N ≈ 1500` RV stars, `ε ≈ 1 km/s`.

**Gates (exit 0 = all pass):**
1. **Bias exists** — `σ_obs > σ_true`; `M_naive/M_true` reported high at `f_b=0.5`.
2. **Degeneracy** — dispersion-only Fisher rank-1 (≈zero eigenvalue / huge condition number).
3. **Recovery** — joint `(σ_true,f_b)` `<3σ`; recovered `σ_true`→`M_dyn` unbiased; full Fisher
   full-rank / well-conditioned.
4. **ε-floor** — bias-removal degrades monotonically with `ε`; recovered `f_b → f_b(P<P_max(ε))`.
5. **Null** — `f_b=0` ⇒ no bias, recovered `f_b ≈ 0`.
6. **AD-vs-FD** — differentiable `μ_k(σ,f_b)` Fisher matches finite differences (gradient
   integrity, the suite style).

**Artifacts:** `scripts/demo_binary_dynamical_mass.py` (gated CLI, B12); figures
`validation/plots/demo_binary_dynamical_mass_*.png` — (1) `v_los` singles-vs-observed showing
the wings, (2) `M_dyn` bias vs `f_b`, (3) the `(σ_true,f_b)` constraint: degenerate ridge
(dispersion-only) vs tight ellipse (full distribution), (4) Fisher `σ(σ_true)`,`σ(f_b)` vs `N`,
(5) bias-removal vs `ε`. Website page `docs/website/60-science-demos/binary-dynamical-mass.md`
(B12) + nav.

**Harness reuse:** `_demo_inference` (poisson_loglike, mle_adam, poisson_fisher_information,
fisher_cov, logit/expit), `MoeJointOrbit`, `KeplerElements.to_state`, `resolve_binary_components`
(or direct to_state), `progenax.stellar.zams_luminosity`, a `project_los` lift from
`demo_rotation.py`. **New demo helpers** (the `K_orb` kernel builder + the mixture `predict_fn` +
the bias formula) live in demo-helper land (e.g. `scripts/_demo_inference.py` or a small
`_demo_binaries.py`) with a few unit tests in `tests/` (demo-harness, not released-core).

**Scope / non-goals (YAGNI):** single population; Moe period dist KNOWN; global `v_los` (no
radial `σ(r)` profile); no selection/incompleteness beyond `ε`; no real data. Period-slope
recovery = noted stretch.

**Definition of done:** gated CLI exit 0 (all 6 gates) with a captured run-record; figures
visually inspected (publication-style, physically correct); demo-harness unit tests pass; website
page builds (0 warnings); FULL released-core gate re-run at close-out (expected unaffected —
scripts/+docs/ only); completion doc. Merge → main on Anna's go.

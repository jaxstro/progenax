---
title: "Validation audit report — 2026-06"
description: "Point-in-time validation-audit report for progenax: what is verified and how, scientific-trustworthiness tiers with explicit limits, implementation completeness, modelling capability and example science questions, improvement recommendations, and the remaining/incomplete modules still to harden, validate, and plot."
---
# Validation audit report — 2026-06

:::{admonition} Scope & date
:class: note
**Completed: 2026-06-08.** This report covers the validation-audit campaign for the
released-core **spatial profiles, velocity DFs / kinematics, substructure, and
diagnostics**. It is a point-in-time companion to the live [status dashboard](index.md):
the dashboard is the current state; this page records *how* trustworthy that state is,
what is still missing, and what must be hardened first. Released-core: **866 tests**
(all differentiable, all collect clean, every figure build-verified).
:::

## 1. What was verified

Every row is backed by a committed test **and** a regenerated figure whose measured
value is shown against the tested tolerance (no self-consistency-only checks).

```{list-table}
:header-rows: 1

* - Module
  - Page
  - Tests
  - Figures
  - Key anchor(s)
  - Date
* - profiles/Plummer
  - [](plummer-equilibrium.md)
  - 20
  - 5
  - $a=0.7664\,r_h$ exact; $\mathrm{Beta}(3/2,9/2)$ speed law; $Q=0.50$
  - 2026-06-08
* - profiles/King
  - [](king-profile.md)
  - 32
  - 5
  - $c(W_0)$ vs King 1966 Table II (Δ≤0.002); density vs DF oracle (1e-10)
  - 2026-06-08
* - profiles/EFF
  - [](eff-profile.md)
  - 23
  - 5
  - slope $\to-\gamma$; $\gamma=5\equiv$ Plummer (exact); $f(\mathcal{E})\ge0$
  - 2026-06-08
* - kinematics/Michie
  - [](michie-anisotropy.md)
  - 12
  - 5
  - $\beta(r)$ vs DF moment oracle; isotropic King limit (2.7e-3)
  - 2026-06-08
* - kinematics/rotation + OM anisotropy
  - [](rotation-om-anisotropy.md)
  - 10
  - 5
  - $v_\phi=\Omega R$ (exact); OM $\beta=r^2/(r^2+r_a^2)$ exact
  - 2026-06-08
* - substructure/CW04 Q + azimuthal
  - [](fractal-substructure.md)
  - 14
  - 8
  - CW04 2004 Table 1 (3D0/1/2); $\sigma_\Sigma$–Q anti-correlation
  - 2026-06-08
* - diagnostics/Λ_MSR + segregation
  - [](mass-segregation.md)
  - 8
  - 4
  - Λ_MSR vs analytic ground truth; energy-sorted generator
  - 2026-06-08
* - cluster/two-component
  - [](two-component.md)
  - ✅
  - 1
  - per-component mass + dynamical-state recovery
  - 2026-06-08
```

All four velocity DFs are **true equilibria with no external virial rescale**
(unscaled $Q=T/|V|$: Plummer 0.50, King 0.505, EFF 0.502, Michie 0.501; 100% bound),
and every released structural parameter ($r_h, r_c, a, W_0, \gamma, M, \Omega, r_a$)
passes an autodiff-vs-finite-difference gradient check (agreement $10^{-6}$–$10^{-11}$).

## 2. Scientific trustworthiness

Verification rests on three independent legs:

1. **Physics-anchored tests** — each asserts a quantitative match to an *analytic*
   result, an *independent oracle*, or a *published table* — never self-consistency.
2. **Independent oracles** where no closed form exists (King density via direct
   velocity integral; Michie $\beta$ via DF 2nd moments) — these catch implementation
   bugs a self-referential test cannot.
3. **Paper-grounding** against held PDFs (King 1966; Cartwright & Whitworth 2004;
   Merritt 1985; Elson, Fall & Freeman 1987), not memory.

A useful trust grading:

```{list-table}
:header-rows: 1

* - Tier
  - Meaning
  - Examples
* - **A — exact/analytic**
  - matches a closed form to ~machine precision
  - Plummer $a$, EFF $\rho(a)$ & $\gamma=5\equiv$Plummer, rotation $v_\phi=\Omega R$, OM $\beta$, all gradients
* - **B — published anchor**
  - reproduces a published table within stated tolerance
  - King $c(W_0)$ vs Table II; CW04 Q vs Table 1
* - **C — approximate / finite-N**
  - statistically anchored or a documented surrogate
  - Monte-Carlo $Q$/dispersions (±0.02–0.04); differentiable `q_approx` (kNN)
```

### Limits stated honestly (not papered over)

```{list-table}
:header-rows: 1

* - Limitation
  - Status
* - **EFF sharp truncation ($\gamma=3$) is ~5–8% sub-virial**
  - intrinsic to truncating an empirical (non-DF) profile; use King for a strict lowered-DF equilibrium, or mild EFF truncation. Documented on the page.
* - **Michie $\beta$ is *suppressed below* pure Osipkov-Merritt**
  - real physics (lowering term breaks $f(Q)$); validated vs the DF's own moment oracle, not textbook OM.
* - **`q_approx` over-reads ~0.1 for concentrated configs ($Q>0.85$)**
  - faithful in the substructure regime it is used for; gradient is a kNN soft-surrogate (median AD-FD 0.9%, cell-boundary worst-case ~20%).
* - **King scalar $r_t$ not differentiable** (argmax crossing)
  - deferred + documented; the profile *shape* in $W_0$ is differentiable.
* - **No LIMEPY cross-validation**
  - by design — validated against the *original* papers, not the external package.
* - **Inference loop not validated end-to-end on data**
  - gradients verified at the IC / forward-model level; structural-parameter recovery on real/mock observations is the research program, not a shipped result.
```

**Bottom line:** trustworthy for production ICs and a methods paper. *Not* a claim that the differentiable-inference pipeline is validated end-to-end against data.

## 3. Implementation completeness

```{list-table}
:header-rows: 1

* - Layer
  - State
* - **Validated (publication tier: unit + integration + validation + figures)**
  - Plummer, King, EFF, Michie; matched isotropic DFs + OM anisotropy + Michie anisotropy + rotation; CW04 Q + azimuthal variation; Λ_MSR + energy-sorted segregation; two-component clusters. IMF + binary engine + tidal truncation validated in prior work.
* - **Released, tested, *not yet figure-validated*** ⚠️
  - [imf-statistics](imf-statistics.md), [binary-imf](binary-imf.md), [analytical-test-cases](analytical-test-cases.md) — 61 validation tests pass combined, but no figures/measured tables yet.
* - **Released, *unit-tested only*** ⚠️
  - [tidal-truncation](tidal-truncation.md) (`jacobi_radius`, `apply_tidal_truncation`) — `tests/unit/test_tidal.py` only; no validation-tier test or figures.
* - **Experimental (repo-only, not in the wheel)** 🔬
  - `gravoturb_fdf` turbulent/fractal density-field subsystem (`src/experimental/`, AC1–AC17).
* - **Deferred / not implemented** ⏳ 🚧
  - differentiable King $r_t$ (IFT, plan written); unified differentiable lowered-model family (Wilson/Woolley/multi-mass); multi-mass equipartition; self-consistent rotating equilibria.
```

## 4. What it lets us model

The architecture is **compositional and differentiable end-to-end**: any
`SpatialProfile` pairs with any `VelocityDF`, plus IMF, binaries, anisotropy,
rotation, and substructure diagnostics — flowing into gravax (dynamics) and fluxax
(photometry). You can generate, as differentiable initial conditions:

- Equilibrium clusters of chosen concentration (Plummer / King $W_0$ / EFF $\gamma$) in true virial balance;
- **radially anisotropic** clusters (Michie self-consistent, or OM via $r_a$) and **rotating** clusters (solid-body or differential);
- **realistic stellar populations** — IMF-sampled masses with a faithful binary population (Moe & Di Stefano $P$–$q$–$e$ coupling), split across components;
- **tidally truncated** clusters tied to a Jacobi radius;
- and **quantify substructure** (Q, azimuthal variation, mass segregation) on any of them.

## 5. Example science questions

Scoped to the *validated, differentiable* capability:

1. **Structural inference** — jointly infer $(W_0\text{ or }\gamma, r_c, M, r_a, \Omega)$ from projected density + kinematics by HMC, with calibrated uncertainties.
2. **Anisotropy-vs-rotation degeneracy** — two distinct anisotropy routes (exact-OM stretch vs suppressed self-consistent Michie) + rotation transforms: *can projected data distinguish radial anisotropy from rotation?*
3. **Primordial vs dynamical substructure** — generate clumpy/smooth ICs, quantify with the validated Q / azimuthal metrics, evolve in gravax: *how fast does measurable substructure erase, and does mass segregation build dynamically vs primordially?*
4. **Binary imprint on the IMF** — with the Moe & Di Stefano engine + binary-aware likelihood: *how biased is a single-star IMF fit with unresolved binaries, and at what $N$ is it "confidently wrong"?*
5. **Tidal-field coupling** (once $r_t$ is differentiable — see roadmap) — *can a population of cluster limiting radii constrain the Galactic potential?*

## 6. Improvement recommendations

```{list-table}
:header-rows: 1

* - #
  - Recommendation
  - Why / payoff
* - R1
  - **Differentiable King $r_t$** (Approach B / implicit function theorem)
  - unlocks tidal-field inference (science Q5); plan already written, ~15–20 LOC + a 6th gradient panel.
* - R2
  - **Figure-validate IMF / binary / analytical** (61 tests already pass)
  - closes the three ⚠️ released gaps; `scripts/validate_imfs.py` already exists — needs pub-figure rewrite + measured tables + pages.
* - R3
  - **Add a tidal-truncation validation tier** (currently unit-only)
  - validate `jacobi_radius` / `apply_tidal_truncation` against the analytic Jacobi radius + a truncation figure.
* - R4
  - **`q_approx`: extend calibration above $Q\sim0.85$ or hard-document scope; consider a soft-rank neighbour surrogate**
  - removes the concentrated-regime over-read and smooths the gradient cell-boundary noise.
* - R5
  - **EFF sharp-truncation: optional virial rescale or more prominent guidance**
  - lets users get a virial $\gamma=3$ IC without silently inheriting the ~5–8% sub-virial offset.
* - R6
  - **End-to-end inference validation** (SBC-style recovery on mocks through gravax/fluxax)
  - this is the real outstanding gap: gradients are verified at the IC level, not against data.
* - R7
  - **Retrofit a Measured column onto the older figured pages** (mass-segregation, two-component)
  - consistency with the new pages' tolerance/measured convention.
```

## 7. Remaining modules to validate + plot

**Released, tested, missing only figures/pages (do these next):**

- [imf-statistics](imf-statistics.md) — Salpeter/Kroupa/Chabrier/Maschberger sampling + recovered $\alpha$ (a `validate_imfs.py` script exists already).
- [binary-imf](binary-imf.md) — forward-model + binary-aware likelihood ("confidently wrong" regime).
- [analytical-test-cases](analytical-test-cases.md) — two-body Kepler, figure-eight, harmonic oscillator (exact-solution checks).
- [tidal-truncation](tidal-truncation.md) — **needs a validation-tier test first**, then a Jacobi-radius/truncation figure.

## 8. Incomplete — harden first, then validate + plot

These are **not** merely unplotted; they need implementation/hardening before a
validation tier is meaningful:

- **`gravoturb_fdf`** (experimental, repo-only) → harden to release quality, then validate turbulent ICs + the Küpper $\sigma_\Sigma$–$D$ slope (which the released azimuthal test deliberately does *not* claim, pending fractal-$D$ models).
- **Differentiable King $r_t$** → implement (R1), then add the gradient panel.
- **Unified differentiable lowered-model family** (Wilson/Woolley/multi-mass) → implement per the [roadmap](../10-theory/spatial-profiles/lowered-model-family.md), then validate against the original models.
- **Self-consistent rotating equilibria** → implement (current rotation is a streaming transform on a non-rotating equilibrium), then validate.

## 9. Changelog

```{list-table}
:header-rows: 1

* - Date
  - Milestone
* - 2026-06-08
  - Plummer, King (retrofit), EFF, Michie, rotation+OM-anisotropy, substructure/CW04 Q, azimuthal-variation — all to publication standard (5 figures each; +measured tables). Shared `scripts/_plotstyle.py`; LIMEPY reframed across 13 pages + lowered-model-family roadmap; `validate_profiles.py` retired. Released-core 830 → 866 (+36 validation tests). This report authored.
* - (prior)
  - Λ_MSR + energy-sorted segregation, two-component clusters, IMF + Moe binary engine, tidal machinery — implemented and tested.
```

# gravoturb_fdf — Differentiable, physics-direct inference layer (DESIGN / BRAINSTORM CHECKPOINT)

> **Status: BRAINSTORM IN PROGRESS (2026-06-05).** This is a *living design doc* to carry
> context into a new session, not a finalized spec. Decisions converged so far are marked
> **[DECIDED]**; things still open are marked **[OPEN]**. The next session should continue
> the brainstorm from the **Open questions** list, then turn this into a TDD plan.
>
> **Sub-skills for the executor:** `superpowers:brainstorming` (continue convergence) →
> `superpowers:writing-plans` (TDD plan) → `superpowers:test-driven-development`.
> **Rules in force:** trust-nothing/verify-against-PDFs, HITL-approve-everything,
> evidence-before-done, JAX-native core, single-env `uv` verification. See
> [[gravoturb-fdf-clean-room]] memory.

## 0. Where this sits

The clean-room gravoturb-FDF rewrite (P0–P6) is **complete and in PR #5**
(`gravoturb-fdf-clean-room → main`, 965 tests green, isolation=0). The experimental
`src/experimental/gravoturb_fdf/` package now has: PDF-grounded 1D theory
(BM19/PP20/PN11/PDF), a 3D realization simulator (GRF + mass-conserving rank copula →
BM19 marginal → dense-tail mask → categorical star sampling), a CW04 Q diagnostic, and a
**prototype** 7-parameter, mean-only, 64³, α-frozen `q_surrogate` that Anna correctly
rejected as incomplete.

**This doc designs the *next* phase: the differentiability + inference layer.**

## 1. Motivation / problem statement

The current `q_surrogate` conflates two things: it makes a *fitted emulator* differentiable,
not the *generator*. It is mean-only (discards the physically-dominant scatter), fit on a
single resolution / narrow parameter box (64³, α=1.8, N⋆=500, 3×3 σ_s×β grid), and
low-capacity (linear-in-7-features). It should be **retired** as the inference interface.

We want a **versatile, differentiable, physics-direct** inference layer for inferring
molecular-cloud parameters `θ = (ℳ, b, α, β)` from observed star-cluster substructure
(LSST-era science) — **without** burying the physics in SBI/neural emulators if we can keep
it direct.

## 2. Goals **[DECIDED]**

Anna selected **all four** (complementary, composable):

- **(A)** A fast smooth emulator that models **mean + scatter**, at converged resolution
  over the full parameter box (not the current mean-only linear toy).
- **(B)** True end-to-end pipeline gradients (where they genuinely pay off).
- **(C)** Calibrated inference of `θ` from cluster substructure.
- **(D)** Field-level differentiable observables (skip stars/Q) — for versatile science.

**Anchor / first deliverable [DECIDED]:** a **versatile differentiable-observables toolkit**
(the foundation), built so inference/emulator/SBI layer on top later.

## 3. Key insight — differentiate the *predicted statistic*, not the *simulator*

The breakthrough that makes "differentiable + physics-direct + (largely) SBI-free" possible:

> Don't push autodiff through a stochastic realization (sort + categorical + MST are
> non-differentiable). Instead **predict the summary statistic analytically as a smooth
> function of θ** and differentiate *that*. This is the cosmology playbook (differentiable
> theory prediction + Gaussian likelihood + HMC), and it fits this problem unusually well.

Supporting facts (verified against the code/PDFs this session):

- **The copula already separates value from arrangement** (`mass_conserving_copula_field`):
  the density *values* are an analytic function of `(ℳ,b,α)` on a fixed quantile grid; **β
  only sets the spatial *arrangement*** via `argsort(g(β))`. So:
  - `∂(field observable)/∂(ℳ,b,α)` already flows cleanly (values differentiable, ranks frozen).
  - Permutation-invariant observables (PDF, f_dense, σ_s², moments) are differentiable in
    (ℳ,b,α) **and independent of β**.
  - Only `∂/∂β` (spatial) and the point-process path (stars/Q) hit the non-diff breaks.
- **Phase-randomness stops being a misspecification for 1pt+2pt inference.** A Gaussian
  field is *fully* specified by its 1-point + 2-point statistics. If inference uses only
  the 1-pt PDF + 2-pt correlation, we use *exactly* the statistics the GRF model represents
  faithfully — the lack of real-turbulence intermittency (higher-order phase structure)
  only matters for higher-order/morphological statistics, which we deliberately don't
  invert. So the 1pt+2pt likelihood is the *most powerful AND most honest* use of the model.

## 4. Physics of the fat tail (why observable/field choice matters)

BM19 density PDF = lognormal body + power-law tail. In linear density `p(ρ) ∝ ρ^{−(α+1)}`
(a **fat / heavy tail**). Moments `⟨ρ^k⟩ ∝ ∫ρ^{k−α−1}dρ` diverge for `k ≥ α`:

- `⟨ρ⟩` finite (α>1); **`⟨ρ²⟩` infinite for α ≤ 2** (the canonical collapsing slopes!).
- **Tail-sensitive** statistics (density variance, *linear*-density 2-pt ⟨ρρ⟩) are formally
  divergent / resolution-dependent / wildly scattered — this is *why* the old realization
  pipeline had ~90% scatter. `f_dense` is NOT tail-sensitive (a mass fraction, ∫ρ^{−α} converges).

**Consequences for observable choice [DECIDED]:**
- For **β**: use the **log-density / Gaussian-space 2-point** `ξ_s(r)=⟨s·s⟩` — `s` has finite
  variance for any α, so it's well-behaved and differentiable (β-carrier). Avoid the
  tail-sensitive linear-density 2-pt.
- For the **stellar observable**: lean toward **counts-in-cells (CIC)** over the angular
  pair-correlation — CIC is tail-robust (finite N), and does double duty:
  `σ²_N(R) = N̄ + N̄² ξ̄(R)` → CIC **variance constrains β** (integrated 2-pt), CIC
  **distribution shape** (over-dispersion / high-count tail) reflects the density PDF →
  **constrains (ℳ,b,α)**. Angular `w(θ)` is the alternative (more window/edge-sensitive).

## 5. The Gaussianization machinery (how β stays analytic)

Field = Gaussian `g` (spectrum `k^{−β}`) → monotone copula → BM19 marginal. Classic result
(Coles & Jones 1991; Szapudi & Pan 2004) for a monotone transform `Y=T(g)`:

  `ξ_Y(r) = Σ_{n≥1} (c_n²/n!) ξ_g(r)^n`, with `c_n = ⟨T(g) H_n(g)⟩` (Hermite coefficients).

- `c_n` = 1-D integrals of our copula map → smooth in `(ℳ,b,α)` (use the iCDF map `T=F⁻¹∘Φ`
  in **log space** so variance/coefficients are finite — avoids the fat-tail divergence).
- `ξ_g(r)` = FT of `k^{−β}` → analytic in **β**.
- ⇒ `ξ(r; ℳ,b,α,β)` is differentiable, semi-analytic, **β included, no sort, no realization.**

## 6. Proposed architecture **[DRAFT]**

```
θ=(ℳ,b,α,β)
  ├─ [theory] BM19 1-pt PDF  (analytic, differentiable — EXISTS today)
  ├─ [theory] log-density 2-pt ξ_s(r) via Gaussianization (NEW; β analytic)   ──┐
  └─ [forward] Cox/Poisson sampling + max-density cutoff + Limber 2D projection ─┤
                                                                                 ▼
            predicted observable: CIC distribution σ²_N(R), shape  (+ optional w(θ))  (differentiable)
                                                                                 ▼
            Gaussian (or count-) likelihood vs data  ─▶  HMC over θ   (physics-direct, no SBI)

  [realization simulator + CW04 Q]  ─▶  RETAINED for: "we reproduce fractal clusters"
                                        demo (CW04 Q–fractal-D ladder), mocks, covariance,
                                        validation, and feeding gravax.

  [3-pt / bispectrum]  ─▶  SEPARATE validation/diagnostic branch (NOT in the θ-likelihood):
                           null test of the phase-randomness assumption + filament detector. See §6b.
```

**Two-tier differentiability boundary [LEANING, not final]:** Tier-1 = analytic predicted
statistics (1-pt PDF + log-space ξ_s + CIC) fully autodiff in (ℳ,b,α), with **β via either
the analytic Gaussianization path or paired finite-difference on the one scalar** (common
random numbers — the paired trick the calibration already uses) — *no soft-sort*. Tier-2 =
exact non-diff realization statistics (CW04 Q, true MST) for validation/mocks. Soft-sort /
Gumbel relaxations are **deferred** (YAGNI) unless full-fused backprop to gravax/render is
later needed.

## 6b. Higher-order statistics — the 3-point function **[DECIDED: validation/diagnostic only, NOT inference]**

**Decision:** the inference likelihood stays at **1-pt + 2-pt**; the **3-pt (bispectrum) is
implemented as a *separate* validation / diagnostic, never in the θ-fit.** Reasons, in order
of importance:

1. **Sufficiency.** Our field is a Gaussian `g` (2-pt set by β) through a monotone marginal
   map (set by ℳ,b,α). Such a model is *fully specified by its 1-pt marginal + 2-pt function*;
   every higher-order statistic is a **derived** quantity. So in the ideal (noise-free,
   model-correct) limit the 3-pt carries **no independent information about θ** — 1-pt+2-pt
   are sufficient statistics. (With noise it adds marginal info, but see #3.)

2. **Two sources of 3-pt — only one is in the model.** A Gaussian field has *exactly zero*
   connected 3-pt. So our model's 3-pt is **purely marginal-induced** (the skewed PDF, à la
   the lognormal bispectrum) — analytic from the same Gaussianization machinery extended to
   3 points, and fully fixed by (1-pt, 2-pt). Real clusters add a **second** source:
   genuine **phase coherence** (filaments/sheets/chains) that a phase-random GRF **cannot**
   produce.

3. **Putting 3-pt in the θ-likelihood would BIAS θ.** If real filamentary 3-pt is present,
   the model can only chase it by *distorting (ℳ,b,α,β)* — importing the phase-randomness
   misspecification directly into the parameter estimates. The 3-pt is *where the
   misspecification lives*, so it must stay out of the fit. This is the safety argument that
   dominates the modest noise-limit information gain in #1.

**Its right role — a held-out null test + discovery channel.** Predict the model 3-pt
(marginal-induced, analytic) from the θ inferred via 1-pt+2-pt, then compare to the measured
3-pt:
- **Agree** → the cluster's non-Gaussianity is consistent with "skewed PDF + random phases";
  the 1-pt+2-pt inference (and its sufficiency assumption) is validated.
- **Excess over prediction** → detection of **genuine filamentary/coherent structure** beyond
  a lognormal-random field — a science result in itself, and the signpost that a richer
  (phase-correlated) field model is warranted.

This closes the logic: **the 3-pt tests the very assumption (phase-randomness is adequate)
that justifies using only 1-pt+2-pt for inference.** Pass → inference validated; fail →
discovery.

**Observational caveat.** 3-pt needs triplet counts → poor S/N at the ~10²–10³ members of a
single cluster; feasible mainly **population-stacked** or as the model-validation null test.
A small-N-robust substitute (MST-topology / higher-order tree statistics) is an **[OPEN]**
alternative to the raw bispectrum — see Open Questions.

**Implementation note:** lives in the validation/diagnostic layer alongside CW04 Q — it
consumes the *realization simulator's* mocks (measured 3-pt) and the *analytic* predicted 3-pt
(Gaussianization-to-3-pt), and reports the null-test residual. It is **not** wired into the
HMC likelihood.

## 7. 2D projection **[DECIDED it's required]**

Data is 2D (sky positions; LOS depth within a cluster unresolved). The forward model must
collapse 3D → 2D via the **Limber projection** (`w(θ)=∫ξ₃D(√(r⊥²+ℓ²))dℓ`); the projected
column-density PDF is narrower than the volumetric PDF (FK10 §3.5). Projection (a) is
differentiable, (b) introduces cluster **distance/depth** as a nuisance parameter to
marginalize, (c) erases LOS information.

## 8. LSST science program (the "why")

Reframes gravoturb_fdf as a **differentiable forward model for cluster counts-in-cells /
two-point** — galaxy-clustering-style inference applied to star clusters. β is the *pivot*
(natal turbulence → density power spectrum → spatial clustering of newborn stars → observed
substructure), and it's the parameter the *spatial* data most directly constrains.

- **Fisher forecast** on CIC/ξ (pure gradient eval): "how well will Rubin constrain natal
  turbulence (β→ℳ)?" — cheapest, timely, standalone.
- **Population inference of β(environment)** across thousands of LSST clusters
  (galactocentric radius, Σ, metallicity, cluster mass) — hierarchical Bayesian.
- **Joint (β, α_IMF)** with the EnvironmentIMF work — do turbulence morphology and the IMF
  share an environmental driver?
- **Substructure as a clock** (with gravax): disentangle initial β from dynamical age from a
  single snapshot (needs end-to-end coupling; β-diff is the upstream piece).
- **Population test of the gravoturbulent paradigm**: inferred β→ℳ vs independent cloud
  line-widths.
- **Stacked-3-pt "filament memory"** (novel LSST angle): population-stacked 3-pt residual
  (measured − model-predicted) vs. environment/age → do young clusters retain primordial
  *coherent* (phase-correlated) substructure beyond a lognormal-random field, and how fast
  does it erase? Uses the 3-pt as the discovery channel of §6b, not as a θ-constraint.

## 9. Honest caveats / limitations

- **Phase-random model** — fine for 1pt+2pt inference (§3), but do NOT invert higher-order /
  morphological statistics with it.
- **Fat-tail cutoff [OPEN, important]** — the Cox step needs a max-density (resolution vs. a
  physical opacity/Jeans floor); this is the one modeling choice touching *absolute* results.
- **Covariance + survey window** — the real practical machinery (as in LSS); needs modeling
  or mock-estimation; must be (smoothly) θ-dependent or θ-independent for clean HMC.
- **Absolute β→ℳ mapping** requires validating the model morphology against real
  gravo-turbulent sims; **relative/population trends are far more defensible** than absolute
  per-cluster Mach numbers — so lead with Fisher forecasts + relative trends.
- **Per-cluster LSST realities**: membership/contamination, distance, extinction,
  completeness — population statistics are where the power is, not single clusters.

## 10. Open questions for the next session **[OPEN]**

1. **Fat-tail cutoff**: how to set the max density honestly (resolution-based vs.
   physical Jeans/opacity floor)? Does it become an inferred nuisance parameter?
2. **CIC vs angular w(θ)** as the primary stellar observable (lean: CIC) — or both?
3. **β gradient**: commit to analytic Gaussianization, paired finite-difference, or both
   (FD as the validation check on the analytic path)?
4. **Covariance**: analytic Gaussian+shot-noise model vs. mock-estimated; θ-dependence.
5. **Likelihood**: Gaussian on CIC moments vs. a proper count likelihood (negative-binomial
   / compound-Poisson for the over-dispersed counts)?
6. **Hermite/Gaussianization convergence** with the BM19 power-law tail — confirm finite
   coefficients in log space; how many terms; where it breaks.
7. **Library / inference engine**: numpyro vs blackjax for HMC; do we need SBI at all (keep
   as optional higher-order cross-check)?
8. **Module layout / what stays experimental**: a new `theory/statistics.py`
   (Gaussianization, projection, CIC prediction) + `inference/` (likelihood, HMC driver)?
   Field-level observables possibly general enough to live in shared `diagnostics`.
9. **Resolution/compute budget** for the *validation* (realization vs prediction agreement)
   and the emulator-of-scatter (goal A).
10. **Validation targets / ACs**: predicted ξ_s / CIC vs realization-measured (agreement);
    Fisher forecast sanity; HMC recovery of injected θ on mocks; convergence of the
    Gaussianization series; **3-pt null test** — model-predicted vs realization-measured 3-pt
    agree (since the simulator IS phase-random, the null test must PASS on its own mocks — a
    clean self-consistency check that the 3-pt machinery is correct).
11. **3-pt estimator for small N**: raw bispectrum vs. an MST-topology / higher-order-tree
    statistic as the small-N-robust substitute for per-cluster use (bispectrum likely
    population-stacked only). Which to implement first?

## 11. Concrete next steps

1. Continue the brainstorm on the **Open questions** (start with #1 fat-tail cutoff and
   #5 likelihood form — they shape everything).
2. Convert to a TDD plan (`superpowers:writing-plans`): the differentiable predicted-statistics
   module first (Gaussianization + projection + CIC), validated *against the existing
   realization simulator* (the simulator becomes the ground-truth oracle for the analytic
   predictions — a beautiful internal-consistency test we already have the machinery for).
3. Then the Fisher-forecast demo (cheapest science deliverable), then HMC recovery on mocks.

## 12. Session handoff (paste this to bootstrap the next session)

```text
Continue the gravoturb_fdf differentiable-inference design (phase 2, after the clean-room
rewrite that's in PR #5).

READ FIRST, in order:
  1. docs/plans/2026-06-05-gravoturb-fdf-differentiable-inference-design.md  (this living
     design doc — note the [DECIDED] vs [OPEN] tags and the §10 Open Questions).
  2. Memory: gravoturb-fdf-differentiable-inference.md AND gravoturb-fdf-clean-room.md.
  3. The code: src/experimental/gravoturb_fdf/ (theory/, field/, diagnostics/) — the
     realization simulator is your GROUND-TRUTH ORACLE for validating analytic predictions.

WHERE WE ARE (converged — do not relitigate without reason):
  - Direction: physics-direct, predicted-statistics inference — differentiate the PREDICTED
    summary statistic, NOT the stochastic simulator (cosmology playbook: analytic prediction
    + Gaussian likelihood + HMC). SBI optional, not the backbone.
  - Inference likelihood = 1-pt PDF + 2-pt (log-density ξ_s via Gaussianization, β analytic)
    + counts-in-cells. CW04 Q → validation-only ("we reproduce fractal clusters"). 3-pt →
    validation/diagnostic null-test + filament detector, NEVER in the θ-likelihood (§6b).
  - 2D Limber projection required (data is 2D; distance/depth = nuisance). Two-tier diff
    boundary; NO soft-sort (deferred, YAGNI). Fat-tail handled via log-space + max-density
    cutoff (the cutoff choice is OPEN). Realization simulator retained as oracle/mocks.

DO THIS, in order:
  1. BRAINSTORM with me first (superpowers:brainstorming) to close the §10 Open Questions —
     START with #1 (fat-tail max-density cutoff: resolution vs physical Jeans/opacity floor;
     inferred nuisance?) and #5 (likelihood form: Gaussian-on-CIC-moments vs a proper
     over-dispersed count likelihood). These gate everything else.
  2. THEN superpowers:writing-plans → a detailed TDD implementation plan. First module to
     plan: the differentiable predicted-statistics module (Gaussianization ξ_s + Limber
     projection + CIC prediction), validated AGAINST THE REALIZATION SIMULATOR as oracle;
     then the Fisher-forecast demo (cheapest science deliverable).

RULES IN FORCE:
  - Trust-nothing / verify against held PDFs (read the actual PDFs). OBTAIN the Gaussianization
    refs first — Coles & Jones (1991), Szapudi & Pan (2004), Carron & Szapudi — NOT yet in
    docs/core-papers/; do not assert their formulae from memory.
  - HITL-approve-everything (gate before acting); evidence-before-done (fresh command output
    on the current tree); JAX-native core (numpy/scipy only in diagnostics/validation);
    single-env uv verification: `env -u VIRTUAL_ENV uv run --no-sync pytest`.
  - NO git push / no PR merge without Anna's explicit go.

REPO STATE: on branch gravoturb-fdf-clean-room (PR #5 OPEN → main, green, 44 commits,
965 released-core+experimental tests). The clean-room rewrite + this design doc live there.
Do not merge PR #5 or push without Anna's go.
```

## References (held PDFs, verified this session unless noted)

BM19 (Burkhart & Mocz 2019), PP20 (Parmentier & Pasquali 2020), PN11 (Padoan & Nordlund
2011), FK10 (Federrath+2010), FK12 (Federrath & Klessen 2012), Kim & Ryu (2005), Heyer
(2009), Lomax+ (2018), CW04 (Cartwright & Whitworth 2004). Gaussianization: Coles & Jones
(1991), Szapudi & Pan (2004), Carron & Szapudi — *to be obtained/verified next session*.
Kainulainen (2014) — no held PDF (values cross-referenced via PP20/BM19 only).

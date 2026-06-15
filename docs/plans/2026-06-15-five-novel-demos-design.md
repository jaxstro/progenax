# Five novel progenax-only demos — design portfolio

**Date:** 2026-06-15
**Status:** DRAFT design portfolio — **not yet ratified.** Per CLAUDE.md, each idea must go
through the `superpowers:brainstorming` skill + Anna HITL ratification before any solver code.
This document is the strategic menu and dependency strategy, not five frozen specs.
**Scope guard:** every idea is **progenax-only** — single-epoch inference from *differentiable
equilibrium initial conditions*. No N-body time evolution (that is gravax), no real SED/bandpass
photometry (fluxax), no stellar tracks (startrax). The forward models are the existing profile/DF
zoo + IMF/binary machinery; the inference rides `scripts/_demo_inference.py` and the `[experimental]`
toolkit.

## Why these five (the gap above B1–B12)

The shipped demos are all *one model, one cluster, mostly one channel* — parameter recovery or a
degeneracy diagnosis. These five go up a level — across **clusters** (populations), across **models**
(selection), across **channels** (fusion), across **observing strategies** (design), and across
**inference paradigms** (exact vs neural). Three of them convert existing demos' *negative* results
(B5 environment rank-1, B12 weak f_b) into *positive* recoveries.

---

## Dependency strategy

progenax is JAX-native (CLAUDE.md): **no PyTorch-based inference libraries** (rules out `sbi`,
`lampe`). The `[experimental]` extra already carries the inference stack; these demos extend it.

| Package | State today | Role in this portfolio |
|---------|-------------|------------------------|
| `jax`, `equinox`, `diffrax`, `optax` | core / present | forward models, autodiff, optimization (OED, MLE) |
| `blackjax>=1.2` | `[experimental]` | NUTS/HMC — fusion (#5), reference posteriors (#3) |
| `flowjax>=17` | **already present** | conditional normalizing flows — NPE/NRE (#3); equinox-native (perfect fit) |
| `arviz` | `[experimental]` | R-hat/ESS/divergence diagnostics for all HMC |
| `numpyro` | **reserved (commented)** | **uncomment** → hierarchical population model (#1); its NUTS + plate/`sample` DSL + non-centered reparam are ideal |
| `jaxns` | **new — recommend** | JAX-native nested sampling → Bayesian evidence cross-check (#4). Laplace evidence needs no dep; jaxns validates it |
| `tinygp` | new — optional | JAX-native GP emulator of the Engine-A ODE solve if inference cost bites (infra for #1/#4 at scale) |
| `sbijax` | new — optional | convenience JAX SBI framework (#3); not required — flowjax can hand-roll NPE/NRE |
| `sbi`, `lampe` | **excluded** | PyTorch — violates the JAX-native invariant |

**Recommended adds, in order:** (1) uncomment `numpyro` (already reserved), (2) `jaxns` for evidence,
(3) `tinygp` only if the ODE-solve cost forces emulation, (4) `sbijax` only as ergonomic sugar over
flowjax. Each new dep is `[experimental]` (validation-side, CPU-only) — the released wheel stays lean.

---

## Idea 1 — Hierarchical population inference (breaks the single-cluster degeneracies) ⭐

**Science gap.** B5 proved the birth environment is *rank-1 unrecoverable from one cluster* (the
3→1 env→α₃ map); B12/B4 found f_b only weakly constrained per cluster. The real science is done over
**populations**: different clusters sample different points along the single-cluster degenerate
ridge, so a hierarchical fit over an ensemble *closes the flat direction*. This is the headline arc
— a demo that turns a published-style negative result into a positive one.

**Headline & gates.**
- Generate an ensemble of clusters spanning `log M_ecl` (each observed sparsely, e.g. N~few×10²–10³).
- Hierarchical model: per-cluster α₃ⱼ drawn from the Jeřábková relation α₃(env; θ_pop) with
  population hyper-parameters θ_pop; partial-pool via non-centered reparametrization.
- **Gate 1:** the hyper-posterior on the α₃–log M_ecl *slope* is recovered and excludes zero, while
  the *same data fit one-cluster-at-a-time* leaves it unconstrained (the degeneracy-breaking).
- **Gate 2:** shrinkage works — sparse clusters' α₃ⱼ pulled toward the population relation; coverage
  honest (SBC on the hierarchy, ranks uniform).
- **Gate 3 (HMC health):** 0 divergences, R-hat<1.01, BFMI ok (funnel handled by non-centering).

**progenax machinery.** `build_cluster`/`build_binary_cluster` over an M_ecl grid; `BirthEnvironment`
+ `env_to_imf_params`; the per-star IMF log-pdf channel (B5) and/or the f_b channels (B4/B12).
**Inference / AI/ML.** Hierarchical Bayes, non-centered HMC (NUTS), partial pooling/shrinkage, SBC of
a hierarchical model. **Deps:** uncomment `numpyro` (plates + non-centered priors) — or blackjax.
**Figures.** (1) one-cluster ridge vs population-constrained slope; (2) shrinkage plot
(per-cluster α₃ⱼ raw vs pooled); (3) hyper-posterior corner; (4) SBC rank histogram.
**Scope/non-goals.** Synthetic ensemble (no real catalog); fixed relation *form* (recover its
parameters, not discover its functional shape).
**Risk.** Funnel pathologies (mitigated by non-centering); cost scales with ensemble size → may need
the GP emulator (tinygp) if Engine-A rebuilds dominate.
**Effort.** Medium-high (the richest payoff).

---

## Idea 2 — Bayesian optimal experimental design ("where should the telescope point?")

**Science gap.** Every demo *analyzes* a fixed mock; none *designs* the observation. progenax's
differentiable Fisher makes `∂(information)/∂(design)` cheap — a capability almost no astro simulator
has — so we can optimize the observing strategy itself.

**Headline & gates.**
- Design variables: radial annulus weights, the RV-vs-proper-motion star split, the magnitude limit
  (via ZAMS L), number of epochs — all continuous, all differentiable.
- Objective: a D- or A-optimality criterion on the Fisher of a target (M_dyn, r_a, or f_b);
  gradient-ascent the design with optax.
- **Gate 1:** the optimized design reaches target σ(parameter) with **N× fewer stars** than uniform
  sampling (report the factor).
- **Gate 2:** the optimum is interpretable and matches physical intuition (e.g. tidal-radius info
  concentrates in the outskirts — recovers B7's "93% from the outer bins" *from first principles*).
- **Gate 3 (AD-vs-FD):** ∂(logdet F)/∂(design) matches finite differences.

**progenax machinery.** The Fisher routines (`poisson_fisher_information`, `fisher_information_gn`);
a differentiable selection/weighting operator; `zams_luminosity` for the mag limit.
**Inference / AI/ML.** Bayesian optimal experimental design, information-theoretic acquisition
(D/A/E-optimality), gradient-based design / active learning. **Deps:** none new (jax + optax).
**Figures.** (1) information vs design (optimized path); (2) optimal radial weighting overlaid on the
profile; (3) precision-vs-budget frontier, designed vs uniform.
**Scope/non-goals.** Static (single-shot) design; no sequential/adaptive online design (a stretch).
**Risk.** Non-convex design landscape → multi-start; defining a realistic but differentiable
selection operator.
**Effort.** Medium. **Highest practical impact (telescope time).**

---

## Idea 3 — SBI calibration testbed ("when does simulation-based inference lie?")

**Science gap.** progenax has what almost no astro simulator has: an **exact, differentiable
likelihood** for many channels. That makes it the ideal *ground-truth* bench to validate — and
falsify — neural simulation-based inference, which is normally used precisely because a likelihood is
*absent*. (The `flowjax` dep is already staged for this.)

**Headline & gates.**
- Train a conditional normalizing flow (NPE) and/or a neural ratio estimator (NRE) on progenax mocks
  for a recovery we can also do exactly (e.g. B12's (σ_true, f_b) or B11's (W₀, r_c)).
- **Gate 1:** the amortized SBI posterior matches the **exact differentiable HMC** posterior
  (per-parameter KS / C2ST near chance) in the well-specified regime.
- **Gate 2:** SBC ranks are uniform when SBI is trustworthy.
- **Gate 3 (the payoff):** a *map of where SBI fails* — push to low N, model misspecification, or the
  degenerate ridge and show SBI becomes overconfident (SBC ranks pile up) exactly where the exact
  posterior says it should widen. "Here is where the flow lies, and here is the truth."

**progenax machinery.** Mocks from any model; exact HMC (blackjax) for the reference; the binned
summaries as flow inputs (or Deep-Sets on the raw catalog as a stretch).
**Inference / AI/ML.** Normalizing flows (NPE), neural ratio estimation (NRE), simulation-based
calibration (SBC), C2ST posterior comparison, amortized inference. **Deps:** `flowjax` (present);
optional `sbijax` for ergonomics. **Figures.** (1) SBI-vs-exact overlaid corners; (2) SBC rank
histograms across regimes; (3) the failure map (coverage vs N / misspecification).
**Scope/non-goals.** Methodological showcase, not a new astrophysical result; single-cluster.
**Risk.** Flow training stability/compute; fair comparison hinges on identical summaries.
**Effort.** Medium-high. **Community-level methodological impact.**

---

## Idea 4 — Differentiable model selection (recover the *family*, not the parameters)

**Science gap.** Every demo assumes the model family (King, or Plummer, or EFF) is known. The
load-bearing real-world question is *which family is it?* — and B-series showed mild misspecification
masquerades as a parameter shift (B6's OM-vs-Michie). This demo tackles family selection head-on.

**Headline & gates.**
- Compute the Bayesian evidence for King vs Wilson vs Plummer vs EFF vs Michie from one snapshot via
  the **Laplace approximation** (max-likelihood + ½log|2πF⁻¹| — the Occam factor falls out of the
  Fisher determinant already computed), cross-checked with **nested sampling** (jaxns).
- **Gate 1:** the true family wins the evidence; report the Bayes-factor-vs-N curve (how many stars to
  *distinguish* a King from a Wilson).
- **Gate 2:** Laplace evidence agrees with jaxns log-Z within stated tolerance.
- **Gate 3 (misspecification honesty):** fit the *wrong* family and show the recovered parameters
  shift to absorb the mismatch (the danger the evidence guards against).

**progenax machinery.** The full profile/DF zoo; the existing Fisher (for Laplace); binned count or
σ(r) likelihoods. **Inference / AI/ML.** Bayesian model comparison, Laplace/Occam factor, nested
sampling, model-misspecification analysis. **Deps:** `jaxns` (nested-sampling cross-check); Laplace
itself needs none. **Figures.** (1) log-Z bar chart across families; (2) Bayes factor vs N; (3)
wrong-family parameter-shift panel. **Scope/non-goals.** Equilibrium families only; no
non-parametric model. **Risk.** Laplace under-counts evidence for multimodal/curved posteriors →
jaxns is the guardrail; ODE-domain limits for some families. **Effort.** Medium.

---

## Idea 5 — Multi-channel fusion (one joint posterior from photometry + kinematics + counts)

**Science gap.** The `60-science-demos/throughline.md` page *argues* the channels break each other's
degeneracies; **no demo fuses them.** This makes the synthesis concrete and quantitative.

**Headline & gates.**
- Jointly fit (α, f_b, W₀, r_c) from three simultaneous channels on the *same* mock cluster: the
  unresolved-blend mass function (B4), the velocity-distribution wings (B12), and the radial counts
  (B11) — summing the existing per-channel Poisson/Gaussian log-likelihoods over one shared θ.
- **Gate 1:** each *pairwise* degeneracy present in a single channel is closed in the joint fit —
  quantified by the combined Fisher condition number collapsing as channels are added (a
  cond-vs-channels staircase).
- **Gate 2:** the joint MLE/posterior is unbiased and tighter than any single channel (ellipse-area
  shrinkage reported).
- **Gate 3 (information accounting):** leave-one-channel-out Fisher decomposition attributes how much
  each observable contributes to each parameter.

**progenax machinery.** The three shipped likelihoods + their forward models, one parameter vector;
blackjax NUTS for the joint posterior. **Inference / AI/ML.** Joint/factorized likelihoods,
information decomposition (leave-one-out), active-subspace analysis of the combined Fisher.
**Deps:** none new. **Figures.** (1) single-channel vs fused corner; (2) cond-number staircase; (3)
leave-one-out information matrix heatmap. **Scope/non-goals.** Shared-truth mock (channels are
self-consistent); no cross-channel systematic. **Risk.** Channel weighting / relative
normalization; keeping all three differentiable in one graph. **Effort.** Medium (most reuse).

---

## Sequencing & priority

| Rank | Idea | Impact | Novelty | Effort | New deps |
|------|------|--------|---------|--------|----------|
| 1 | #1 Hierarchical population | very high | high | med-high | uncomment numpyro |
| 2 | #2 Optimal experimental design | high (practical) | high | med | none |
| 3 | #3 SBI calibration testbed | high (community) | high | med-high | flowjax (present) |
| 4 | #5 Multi-channel fusion | high | med | med | none |
| 5 | #4 Model selection / evidence | med-high | med | med | jaxns |

Recommended order: **#1 → #2 → #5 → #3 → #4** (lead with the negative-result-breaker; #2 and #5
need no new deps; #3 builds the SBI muscle flowjax is already staged for; #4 last). #1, #3 each could
seed a methods paper.

## Open questions for ratification (per idea, before brainstorming)

1. **#1:** synthetic ensemble size & M_ecl range? recover the relation *slope* only, or its scatter too?
2. **#2:** which single target parameter is the headline (M_dyn vs r_a vs f_b)? which design variables in v1?
3. **#3:** NPE only, or NPE+NRE? which recovery as the bench (B12 σ–f_b vs B11 W₀–r_c)?
4. **#4:** Laplace-only v1 with jaxns deferred, or both from the start? which family pair is the headline?
5. **#5:** MLE+Fisher v1, or straight to joint NUTS? include the active-subspace panel or defer?
6. **Cross-cutting:** do we want a shared `scripts/_demo_inference2.py` for the new
   hierarchical/flow/nested-sampling helpers, or extend the existing module?

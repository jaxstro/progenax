---
title: What's new
description: Release-style changelog for progenax — most recent change first. Curated from the development log.
---
# What's new

Release-style changelog. Most recent change first. Curated from the
[development log](../90-development-log/index.md).

## 2026-06-10 — Memory-bounded kernels, table-routed standalone DFs, numerics consolidation

A production-scale memory + consolidation pass over the cluster and kinematics stack:

- **Blocked + remat'd virial kernels** — the $O(N^2)$ pairwise potential-energy /
  virial kernels now run in fixed-size blocks with rematerialization, bounding
  peak memory in the forward pass *and* the gradient at $O(\mathrm{block}\cdot N)$.
  Measured peak RSS (`scripts/profile_cluster_memory.py`, all stages PASS):
  virial scaling at $N=2\times10^4$ → 0.41 GB; Engine A iso/aniso at $N=10^5$ →
  1.4 / 2.2 GB; Engine B halo+core at $N=10^5$ → 2.5 GB.
- **Table-routed standalone DFs** — the LIMEPY / King / Michie *standalone*
  velocity DFs now route through the differentiable DF tables by default, with
  the exact quadrature retained as the `speed_method="quadrature"` oracle
  (anisotropic standalone DF sampling at $N=2\times10^4$ → 2.3 GB measured).
- **`progenax.numerics` consolidation** — five cumulative-trapezoid
  implementations and the inverse-CDF draw helpers unified in one module;
  unbounded oracle maps now chunked via `lax.map(batch_size=2048)`.
- **Grouped engine state** — Engine-A-only internals live in a `_EngineAState`
  group; accessing them on an Engine-B model raises an informative
  `AttributeError` (shared fields like `total_density` dispatch correctly on
  both engines).

**Gate.** Released-core **1150 tests passing** (unit 882, integration 34,
validation 234).

## 2026-06-09 — Unified multi-component equilibrium clusters (two engines)

The multimass arc (Phases 1+2) merged to main: ONE differentiable
`MultiComponentCluster` (eqx.Module → `ICResult`) spans mass segregation,
GC 1G/2G, binaries-vs-singles, and halo+core populations in a single
self-consistent shared potential, via **two independent engines**:

- **Engine A (DF-defined)** — the multi-mass lowered-isothermal/LIMEPY
  family (Gieles & Zocchi 2015): `from_components` (direct velocity-scale
  ratios $w_j = s_j/s$), `from_mass_segregation` (the equipartition law
  $w_j = \mu_j^{-\delta}$, differentiable in $\delta$), `from_imf`
  (eigenvalue solve for the central density fractions $\alpha_j$). Optional
  per-component Michie/Osipkov–Merritt anisotropy; differentiable DF tables
  accelerate construction ~5.6× and sampling 21–67× with the exact
  quadrature retained as a selectable oracle.
- **Engine B (density-defined)** — `from_density_profiles`: Plummer/EFF/King
  *density* components in one shared potential (single quadrature Poisson
  pass, no ODE), per-component DFs via generic Eddington inversion
  (`eddington_invert`), optional per-component OM `r_a_j`, and an
  $f_j \ge 0$ realizability gate. Validated against Engine A on the same
  King model (two independent engines agree: $\sigma_{1d}(r)$ deviation
  $3\times10^{-4}$, radial KS $2\times10^{-4}$).

`sample_cluster` returns an `ICResult` whose new `component_id` field labels
each star's generating component; per-component theory oracles prove
$Q_j = 0.5$ for every component. `energy_sorted_segregation` stays as the
labeled PRIMORDIAL (non-equilibrium) generator.

**Retired (pre-launch, no backwards compat):** the `populations` module
(`generate_two_component_cluster`/`TwoComponentConfig` — it fed the *full*
cluster mass to each sub-population's velocity DF, a real physics bug for
unequal-mass IMFs), the string-dispatch `generate_cluster_ic`/`ClusterState`/
`SpatialStructureParams`/`MassSegregationLayer`, the `lambda_seg` catalog
blend (intermediate states drift from per-mass-group virial balance), and
`MultiMassLIMEPY` (subsumed by `MultiComponentCluster`). `src/` is now
dataclass-free (all containers are Equinox modules).

## 2026-06-04 — Binaries SoTA arc: review, completion, and faithful Moe composition

A paper-grounded review + completion pass over the whole `binaries/` module (Batches 4a–4k).

**Correctness (4a–4e).** Fixed a Critical STELLAR-unit mean-motion bug (an absolute `a³`
floor made realistic stellar-binary velocities ~100% wrong in pc), a `SanaOBPeriod(power=-1)`
divide-by-zero + NaN gradient, an `e→1` gradient singularity, and a non-PyTree config; split
`population.py` and `kepler.py` to the size limits; replaced the mislabelled eccentricity
heuristic with the **faithful `MoeEccentricity`** `p(e) ∝ e^η(logP, M₁)` + Roche cap (Moe &
Di Stefano 2017 Eqs. 3, 17–18), keeping the old logistic blend honestly cited as
`LogisticThermalEccentricity` (Duquennoy & Mayor 1991).

**Completion (4f–4k).** The binary-population engine is now end-to-end:

- **Connector** `resolve_binary_components()` — binary COMs → masked 2N components, COM-preserving
  and jit/grad-safe — plus the `build_binary_cluster` orchestrator.
- **Faithful Moe P–q–e** (`MoeDiStefano2017Full`, `MoePeriod`, `MoeJointOrbit`) — the two-slope,
  period-dependent mass-ratio + the joint interrelation (an FD-vs-autodiff grad-check caught that a
  multi-uniform sampler drops the mixture-weight gradient; the grid inverse-CDF is reparameterized).
- **SoTA composition** — `build_binary_cluster(primary_imf, companion_model, target, …)`. A
  `CompanionModel` owns the binary statistics (f_b + q + P + e); `IndependentCompanions` reproduces
  the period-averaged default, `MoeCompanions` wires the faithful joint (the same q sets m₂).
  Population-size budgets `Systems(n)` / `Stars(n)` / `TotalMass(M)` let you fix systems
  (companions not counted; the observational convention), resolved stars, or total mass.
- **Diagnostics** — `find_bound_pairs`/`find_bound_multiples`/`primordial_survival` and
  `binary_energy_budget`, which separates the cluster COM virial (`Q_com`, what the build scales to)
  from the internal binary binding-energy reservoir (grounded in the McLuster CoM-virialization
  convention, Küpper+2011 §A8).

**Impact.** Realistic primordial-binary cluster ICs with the empirically-calibrated Moe & Di Stefano
(2017) interrelation, differentiable end-to-end, paper-grounded, green under jax 0.10.1 and 0.7.0.
The mass-only `BinaryIMF.sample_systems` path (used by the *Confidently Wrong* IMF-inference paper) is
unchanged. Suite 1074 → 1201.

## 2026-06-03 — Engineering hardening: CI coverage decoupling + coverage to 91%

Decoupled coverage from the CI pass/fail gate — a coverage-tooling crash (the documented
jaxlib/abseil narrow-scope class) can no longer red a build whose tests pass — and added
~60 discriminating tests lifting five under-covered modules (`imf/binary/mass_ratio` 53→96,
`imf/smooth` 64→**100**, `imf/binary/imf` 61→93, `imf/environment/mapping` 69→96,
`kinematics/api` 72→**100**). Package coverage **86 % → 91 %**; full suite passing, no
existing test weakened. The 500-LOC file limit was relaxed to a guideline (cohesive files
≤~600 accepted). A deliberate **SoTA-design + per-module validation pass** — including the
`fdf.py` split and a cumulative-shared-grid CDF for `smooth.py` — is queued for release; see
[`2026-06-03-pre-release-sota-agenda.md`](../../notes/2026-06-03-pre-release-sota-agenda.md).

## 2026-06-03 — Follow-up audit: two launch-blockers closed

A five-lane post-hardening audit re-verified every 2026-06 fix and surfaced two
**Critical "untested twins"**, now closed: `build_spatial_ic` crashed under
`jax.grad` (`float(softening)` → `jnp.asarray`), and the default `mode="bm19"`
tail sampler still OOM'd at production scale (`random.categorical` Gumbel-max →
`cumsum`+`searchsorted` inverse-CDF). Also fixed two Major (a NaN gradient in
`compute_potential_energy` at the default `softening=0`, via a double-`where`;
a seed-fragile BM19 test) and ~10 Minor (a differentiable BM19 resolution guard,
the `energy_sorted_segregation` top-level export, `profiles/api.py` coverage
37 % → 100 %, a `c(W₀)`↔King-1966-Table-II regression guard, doc fixes). Test
suite spans three tiers (unit / integration / validation). These were the
blockers the audit said would take it from A− to a solid A.

**Impact.** Gradient-based inference through `build_spatial_ic` and
production-scale `mode="bm19"` substructure now work. No previously-trusted
result was affected — both Criticals manifested as a crash / OOM, never as a
silent wrong number.

## 2026-06-02 — Audit hardening: true King/EFF velocity DFs

Resolved the 2026-06-01 expert audit (2 Critical, 9 Major). The King
and EFF velocity DFs are now sampled in **detailed equilibrium** — King
via the lowered-Maxwellian $f(E)\propto e^{E/\sigma^2}-1$ with a
self-consistent $\sigma$, EFF via exact Eddington inversion $f(E)$ of
the truncated density — so both are virial ($Q=T/|V|\approx0.5$) with
**no external rescale**. A latent density–potential bug (the King ODE
sampled a non-King, 2–30× over-extended profile) was fixed, so the
concentration $c(W_0)$ now matches King (1966) Table II to $\le 0.02$.
Also fixed: $G$ was silently dropped from velocity sampling (C1); the
King $K$-function had a NaN gradient and a non-JIT-able constructor (C2).
Test suite: 848 across 3 tiers (724 / 21 / 103), with tightened,
regime-anchored tolerances.

**Impact.** ICs built with King or EFF velocities before this release
were not in equilibrium and should be regenerated. The unused
`king_K_function` was removed; the corrected density is
`king_lowered_maxwellian_density`.

**Reference.** [](../90-development-log/code-reviews.md).

## 2026-04-28 — PP20 ζ(p) transcription bug fix

The `magnification_factor(p)` function was producing systematically
wrong values for typical $p \in [1, 2]$ due to a transcription error
of {cite:t}`ParmentierPasquali2020` Eq. 6. Fixed; replaced with the
canonical analytic form $\zeta(p) = 2(3-p)^{3/2}/[3^{3/2}(2-p)]$,
which is equivalent to PP20 Eq. 6 to 0.08% across the physical
domain. 35 new regression tests anchor every spot value on PP20 /
analytic / Kainulainen+14 references.

**Impact.** All BM19 forward-chain ζ outputs computed before this
fix are systematically wrong. The validation plots
`b5_zeta_comparison.png`, `b6_pp20_diagram.png`, and
`e5_pp20_diagram.png` need regeneration.

**Reference.** [](../90-development-log/2026-04-28-pp20-fix.md);
chapter at [](../10-theory/gravoturbulence/pp20.md).

## 2026-04-28 — progenax docs website launched

The single-source-of-truth documentation site you are reading now
went live. Migrated and rewrote ~12K lines of source material from
`docs/methods/`, `docs/dev-methods-guides/`, and `docs/core-papers/`
into ~120 chapters of structured MyST-MD content. The legacy raw
docs are preserved under `docs/{plans,notes,specs,code-reviews}/`
for archaeological purposes.

## 2026-02-13 — Binary-aware IMF recovery v1

End-to-end forward-model + likelihood for binary-aware IMF
inference. Reproduces the "confidently wrong" regime where the
naive single-star likelihood produces a posterior 95% CI that
shrinks below the bias and excludes the truth at $N \gtrsim 10^4$.
The binary-aware likelihood eliminates the bias.

**Reference.** [](../10-theory/imfs/binary.md);
[](../50-validation/binary-imf.md).

## 2026-02-12 — IC redesign: protocol-based composition

Comprehensive redesign of the IC API. Replaced the inheritance-based,
partially-stateful, implicitly-united legacy design with the
protocol-based, immutable, explicitly-united architecture. Breaking
change; legacy preserved at `progenax_legacy`.

**Reference.** [](../20-architecture/ic-redesign-history.md);
spec at [](../90-development-log/2026-02-12-ic-redesign.md).

## 2025-12-07 — Phase 1 milestone

First production-grade release with full IMF / spatial profile /
velocity DF / mass segregation / fractal substructure / binary
support. Validation suite reaches 432 tests across 3 tiers.

**Reference.** [](../90-development-log/phase1-complete.md).

## 2025-12-07 — IMF stack fix

Correction to the cumulative-mass integration in `PowerLawIMF` for
multi-segment forms; prior version produced subtle off-by-one errors
at segment boundaries.

**Reference.** [](../90-development-log/2025-12-07-imf-stack-fix.md).

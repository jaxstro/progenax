---
title: What's new
description: Release-style changelog for progenax — most recent change first. Curated from the development log.
---
# What's new

Release-style changelog. Most recent change first. Curated from the
[development log](../90-development-log/index.md).

## 2026-06-03 — Engineering hardening: CI coverage decoupling + coverage to 91%

Decoupled coverage from the CI pass/fail gate — a coverage-tooling crash (the documented
jaxlib/abseil narrow-scope class) can no longer red a build whose tests pass — and added
~60 discriminating tests lifting five under-covered modules (`imf/binary/mass_ratio` 53→96,
`imf/smooth` 64→**100**, `imf/binary/imf` 61→93, `imf/environment/mapping` 69→96,
`kinematics/api` 72→**100**). Package coverage **86 % → 91 %**; suite **949** passing, no
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
suite: **874** across 3 tiers (742 / 24 / 108), **86 % coverage**. These were the
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

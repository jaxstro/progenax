---
title: What's new
description: Release-style changelog for progenax — most recent change first. Curated from the development log.
---
# What's new

Release-style changelog. Most recent change first. Curated from the
[development log](../90-development-log/index.md).

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

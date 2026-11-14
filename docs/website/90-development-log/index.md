---
title: Development log
description: >-
  Internal development history of progenax: dated design specs, code-review
  audits, and recovery/scaling notes, preserved as historical records.
---

# Development log

This section is progenax's **internal development history** — dated design
specifications, full-package code-review audits, and validation recovery/scaling
notes, kept for the record. The pages are **point-in-time snapshots**: each one
describes the state of the package on its date, and several reference APIs that
were later renamed or retired. Where that happens the page carries an
admonition pointing at the current entry point. Historical grades, completion
claims, proposed criteria, and reported measurements do not establish current
release readiness or scientific qualification. The original calculations have
not all been independently revalidated.

For the *current* state of the package, start from the
[release notes](../00-getting-started/whats-new.md), the
[architecture overview](../20-architecture/index.md), and the
[validation backbone](../50-validation/index.md). The
[test dashboard](../50-validation/test-dashboard.md) displays a dated stored
census and coverage artifact; read its freshness limits before using those numbers.

## Code-review audits

Full-package technical reviews (scientific correctness, numerics,
JAX/differentiability, testing, provenance), newest first:

- [](code-reviews.md) — the landing record for the **2026-06** post-hardening
  and expert-audit reviews, including their historical grades and resolution tables.
- [](2025-12-07-progenax-review.md) — the original **2025-12-07** comprehensive
  code/architecture/science review. Its internally differing grade labels are
  preserved as part of that record.

## Design specs

- [](2026-02-12-ic-redesign.md) — the original specification for the
  protocol-based, composable, differentiable IC architecture (single-population
  `build_spatial_ic` + matching `SpatialProfile`/`VelocityDF` protocols).

## Milestones, fixes, and recovery notes

- [](phase1-complete.md) — the **2025-12-06** Phase 1 completion snapshot
  (legacy `gravax.ic` ported to a standalone JAX-native package).
- [](2025-12-07-imf-stack-fix.md) — the IMF-stack correction plan
  (Chabrier log₁₀ basis, parameter-gradient and domain-behavior fixes).
- [](2026-02-12-imf-hmc-recovery.md) — the IMF-only NUTS slope-recovery
  proposal figure (α₃ across 1.7–2.8 via NumPyro).
- [](2026-02-13-binary-aware-imf-recovery.md) — the binary-aware IMF recovery
  **spec**: the Moe+17-aware mixture likelihood that removes the binary bias.
- [](2026-02-13-binary-aware-imf-recovery-impl.md) — the matching
  **implementation plan** (task-by-task build of the validation script).
- [](2026-02-13-precision-scaling-panel.md) — the σ(α)-vs-N precision-scaling
  panel added to the binary-aware recovery figure.
- [](2026-04-28-pp20-fix.md) — the PP20 magnification-factor ζ(p)
  transcription-bug fix (now superseded by the experimental `gravoturb`
  rewrite; see the page's update banner).

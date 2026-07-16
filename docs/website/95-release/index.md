---
title: Release readiness
subtitle: Is progenax ready to release — and what stands between here and a public tag?
description: >-
  The evidence-backed release-readiness audit of progenax — verdict, audit
  report, blocker checklist, and release strategy. Audited at commit 24cb6b9
  (2026-06-17).
---

(release-readiness-index)=
# Release readiness

You are looking at a research package that has been hardened across many arcs and
now wants to leave the laptop. The question this section answers is narrow and
honest: **can progenax be released today, and if not, exactly what blocks it?**
Not "does it feel done" — *what did we run, read, or check against a primary
source, and what did that evidence say.*

This is the output of a verify-everything audit performed at commit `24cb6b9`
(2026-06-17), re-checking the fixes claimed in the [CHANGELOG](https://github.com/jaxstro/progenax/blob/main/CHANGELOG.md)
against the live code, the test suite, the gradient registries, and the source
PDFs. Every number on these pages traces to a command run or a document read in
that session.

(release-readiness-verdict)=
## Top-line verdict

**progenax v0.1.0 is release-ready as a source/GitHub tag. It is not yet
installable from PyPI, and its continuous-integration gate is currently switched
off.** Neither gap is a defect in the science: the equilibrium machinery, the
gradient integrity, the packaging mechanics, and the documentation are all
verified sound. The two things standing between here and a public release are at
the *boundary*, and both were already known and recorded:

1. **The runtime depends on the unpublished sibling `jaxstro`** — so an outsider
   cannot `pip install progenax` yet. This is a deliberately-deferred packaging
   decision, not a code fault (see [release strategy](#release-strategy)).
2. **Both GitHub Actions workflows are disabled** — the CI configuration is
   excellent, but nothing currently runs it, so the advertised gate is enforced
   only on this machine.

Everything else the audit found is editorial polish or a documented, deferred
limitation. The prior 2026-06-11 adversarial audit raised ten release blockers
(R1–R10); **nine are verified fixed**, and the tenth (R2, the `jaxstro`
dependency) is the deferred decision above.

(release-readiness-paths)=
## Where to go next

- Want the evidence? → **[Audit report](#release-audit-report)** — every dimension,
  what was checked, the command/number, the verdict.
- Want the to-do list? → **[Release checklist](#release-checklist)** — actionable
  items, blockers first, each tracing to a finding.
- Want the plan? → **[Release strategy](#release-strategy)** — version, channel,
  the `jaxstro` dependency story, CHANGELOG/DOI, and the public-API freeze.

:::{note} Scope
This audit covers the **released-core** package (`src/progenax/`, the wheel
contents). The experimental `gravoturb` subsystem (`src/experimental/`, not in
the wheel) was confirmed present, green, and correctly quarantined, but is out of
scope for the public-API and release-blocker assessment.
:::

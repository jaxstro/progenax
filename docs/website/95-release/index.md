---
title: Release readiness
subtitle: Is progenax ready to release — and what stands between here and a public tag?
description: >-
  Current release qualification requirements and remaining gates, with links
  to the historical audit and release checklist.
---

(release-readiness-index)=
# Release readiness

Release qualification is in progress. The audit report and historical checklist
below describe earlier revisions; they do not establish readiness of the current
implementation.

(release-readiness-verdict)=
## Current verdict

**The current revision is not yet qualified for a public release.** Selected
Plummer documentation and numerical checks, shared lowered-profile repairs, and
retained Engine A first-derivative chains pass their recorded criteria. Those
results do not qualify every model, parameter regime or sampling observable.

Before a release, complete current physical validation and figures, declare and
verify the supported domain, run the integrated test and documentation gates,
verify installed artifacts and registry resources, and confirm remote CI actually
runs. Resolve the reproducible `jaxstro` installation route for the chosen release
channel. Remote workflow settings have been inspected; no CI was run for this publication. Any future remote verification requires explicit authorization.

Fixed-realization first derivatives are distinct from population-expectation
gradients across discrete component assignments. Population-expectation gradients for supported population observables are a
release requirement. Their estimator and validation contract remain to be
qualified. Higher derivatives remain unqualified.

The June 17 audit at `24cb6b9` is retained as historical evidence. Its conclusion
that only distribution and CI issues remained must not be applied to the current
revision without renewed qualification.

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
contents). The experimental `gravoturb` subsystem (`src/experimental/`) remains outside
the intended released core. Verify artifact exclusion during packaging acceptance;
historical tests do not establish its current qualification.
:::

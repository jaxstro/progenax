---
title: Release readiness
subtitle: Current engineering checks and scientific qualification limits
description: >-
  Current release verdict, completed local checks, remaining scientific and
  distribution requirements, and links to their evidence owners.
---

(release-readiness-index)=
# Release readiness

(release-readiness-verdict)=
## Current verdict

**Progenax is not yet qualified for a public release.** The primary local
released-core suite completed with 1,993 passed and one intentionally skipped
timing case; all 24 core validation CLIs passed. Measured coverage is 97.32% line,
89.09% branch and 96.58% combined, above the unchanged 90% combined floor. The
local release gate passed, including 19 serial artifact checks and gradient
regeneration/freshness. The separate serial timing benchmark also passed. See the
[checklist](checklist.md#checklist-verified) for scope.

The full local measurement uses Progenax `59b96f30`, Python 3.13.7, JAX 0.11
and local `jaxstro` revision `6451e183`. A separate isolated installed-wheel
check uses public `jaxstro` revision `94b83705` and passes its smoke check and six
existing compatibility tests. These results do not qualify every Python/JAX
combination, accelerator backend or PyPI installation.
The source-checkout route and the remaining PyPI dependency decision are described
in the [installation guide](../00-getting-started/installation.md) and
[release strategy](release-strategy.md).

Scientific qualification remains model-, parameter- and observable-specific.
The retained [gradient audit](../50-validation/differentiability-audit.md) records
99 named cases: 98 clean, one pinned non-differentiable binned-data count and
zero unresolved hazards. Serial regeneration and freshness checks reproduced
this baseline.
Those checks do not establish general population-expectation gradients across
discrete assignments. Their estimator and validation contract remain a release
requirement. Higher derivatives, float32 and accelerator numerics remain
unqualified. Each intended application also needs physical and inference checks
within a declared numerical domain.

This assessment covers the released core. Experimental Gravoturb and private
research outputs are outside its supported artifacts and qualification scope.
No remote CI, tag, deployment or package publication is established by a local
check.

(release-readiness-paths)=
## Evidence and next actions

- [Release checklist](checklist.md): completed local checks and remaining gates.
- [Test dashboard](../50-validation/test-dashboard.md): recorded test and coverage evidence.
- [Release strategy](release-strategy.md): source and PyPI channel decisions.
- [June audit](audit-report.md): a historical snapshot, not the current verdict.

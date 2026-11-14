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

**Progenax is not yet qualified for a public release.** Focused local checks
pass for finite-model construction under eager execution, JIT and supported
first-order AD, including derivative-only and batched calls. A finite saved
potential must start positive and bracket its tidal zero. This enforces model
validity; numerical accuracy and supported physical domains still need qualification.
See the [checklist](checklist.md#checklist-verified) for the measured scope.

The full-suite and coverage baseline predates this validity-guard patch; its
measurements have not been restamped. It uses Progenax `59b96f30`, Python 3.13.7, JAX 0.11
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

---
title: Release checklist
subtitle: Current local checks and remaining release decisions
description: >-
  Completed local engineering checks and outstanding scientific, distribution,
  and publication requirements for progenax v0.1.0.
---

(release-checklist)=
# Release checklist

The [current verdict](index.md#release-readiness-verdict) remains **not qualified
for release**. The validity-guard patch has focused local checks. The full
release gate and uninstrumented timing baseline below predate that patch.

(checklist-blockers)=
## Qualification still required

- [ ] Declare and verify each supported model's physical and numerical domain,
  including grid resolution, finite boundaries and approximation error.
- [ ] Qualify population-expectation gradients for supported observables across
  discrete component assignments. Specify the estimator and randomness contract;
  fixed-realization derivative checks do not close this gate.
- [ ] Establish application-specific physical and inference accuracy. Higher
  derivatives, float32 and accelerator backends remain unqualified.

(checklist-verified)=
## Completed local checks

The full-suite and coverage baseline uses Progenax `59b96f30`, Python 3.13.7, JAX 0.11 and
local `jaxstro` revision `6451e183`. The separate installed-wheel check below uses
public `jaxstro`; its scope is narrower than the local suite.

- [x] **Finite-model validity guards:** focused checks cover eager execution, JIT,
  forward/reverse first-order AD, derivative-only outputs and mixed valid/invalid
  batches. Profile factories, standalone velocity DFs and completed Engine A
  models require an actual tidal crossing. Physics, grids and solver tolerances
  are unchanged; raw diagnostics and manual precomputed-profile constructors
  retain their contracts. These checks do not refresh full-selector coverage.
- [x] **Primary released-core suite:** 1,993 passed and one intentionally skipped
  timing case in 5,017.25 seconds. That case passed in the separate serial run below.
- [x] **Measured coverage:** 97.32% line, 89.09% branch and 96.58% combined. The
  unchanged 90% gate applies to the combined percentage.
- [x] **Core validation CLIs:** all 24 passed.
- [x] **Local release gate:** exit 0; all five predicates passed. The serial
  artifact phase passed 19 tests in 511.31 seconds, including fresh regeneration
  of the 99-case gradient audit: 98 clean, one pinned binned-data-count limitation
  and zero hazards.
- [x] **Serial LIMEPY table timing test:** one passed in 22.92 seconds without
  coverage or xdist. It meets the existing threefold warm-median speedup criterion
  for that test configuration; it is not a general hardware performance claim.
- [x] **MyST compatibility and rendering.** CLI 1.10.1 with book theme 1.4.1
  builds 215 HTML pages with zero content warnings and zero broken links.
- [x] **Artifact boundaries.** The explicit source-archive allowlist and wheel
  policy exclude private review material and experimental code. The wheel rebuilt
  from the source archive matches the direct build.
- [x] **Installed-wheel smoke test.** An isolated Python 3.13.7 environment with
  JAX 0.11 and public `jaxstro` revision `94b83705` passes the smoke check and six
  existing compatibility tests, and loads 13 registry TOMLs containing 37
  equations and six bundles, without an editable checkout fallback. This does not
  qualify every supported version or backend.

(checklist-important)=
## Distribution channels

- [ ] **Source release:** record the exact supported environment and reproducible
  installation route, final gate results and approved API scope.
- [ ] **PyPI:** provide a resolvable `jaxstro` distribution and verify installation
  through that channel. The isolated wheel check does not establish a standard
  `pip install progenax` route.

See the [installation guide](../00-getting-started/installation.md) and
[release strategy](release-strategy.md) for the source-checkout route and channel
decisions.

(checklist-slice-d)=
## Before a tag or publication

- [ ] Obtain Anna's approval for the release scope, channel and version.
- [ ] Inspect current remote workflow state and obtain explicit authorization
  before any CI run or deployment. No remote CI is part of these local checks.
- [ ] Finalize `CITATION.cff`, archival DOI and the dated `CHANGELOG.md` for the
  approved release.

(checklist-minor)=
## Contributor support

- [ ] Add `CONTRIBUTING.md` with the supported environment, local gates and
  documentation workflow.
- [ ] Execute the tutorials and examples relevant to the supported release scope.

(checklist-docs-hardening)=
## Documentation evidence

The [test dashboard](../50-validation/test-dashboard.md) and
[gradient audit](../50-validation/differentiability-audit.md) own their recorded
checks and limitations. The [June audit](audit-report.md) is historical evidence;
its test counts and readiness findings do not replace the current gates above.

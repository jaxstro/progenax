---
title: Unlisted pages (internal / preview)
description: >-
  In-repo pages intentionally kept out of the public navigation: OED detail
  pages, the experimental gravoturbulence theory, and the development log.
---

# Unlisted pages

This is the landing page for documentation that lives **in the repository and
builds with the site, but is intentionally absent from the public navigation**.
These pages are preview, historical, or experimental material: they are reachable
by direct URL (and from the links below) but do not appear in the top-of-page
nav, so the public table of contents stays focused on the released, supported
package.

:::{important} Navigation visibility is not access control
Hidden pages remain part of the site build and can be reached directly.
`hidden: true` does not make unpublished content private or authenticate a
reader. A deployment's access policy must cover the complete built site.
:::

:::{note} Unlisted navigation
This site's explicit `project.toc` in `myst.yml` retains hidden entries so they
are built and linked while omitted from navigation. Removing an entry is
different from hiding it and can leave a linked page without a built route.
This landing page is registered as:

```yaml
- file: _unlisted/index.md
  hidden: true
```

The OED detail pages, the consolidated gravoturbulence theory pages, and the
development log are all `hidden: true` (built and reachable, but out of the
public navigation) and are linked below.
:::

## Optimal experimental design (detail pages)

The science-demos section keeps a single
[public OED overview](../60-science-demos/optimal-design/index.md). The longer
worked-example detail pages are `hidden: true` in the toc (built and
URL-reachable, kept out of the public navigation). The OED tooling itself is
planned for a separate package and is not part of v0.1.0; these pages document
the prototyped worked designs:

- [](../60-science-demos/optimal-design/background.md) — the shared OED formalism:
  Fisher information, the additive design-linear backbone, the c/D/A optimality
  criteria, and the sky-projection geometry, built once for the worked examples.
- [](../60-science-demos/optimal-design/anisotropy.md) — Stage 1: allocating a star
  budget across radius and RV/PM channels for an anisotropy-radius measurement
  in the reported worked design.
- [](../60-science-demos/optimal-design/dynamical-mass.md) — Stage 2: promoting survey
  depth (limiting magnitude) to a design variable for the dynamical mass $M$; the
  Fisher peaks at an interior optimal depth.
- [](../60-science-demos/optimal-design/concentration.md) — Stage 1 redux: the same
  channel-allocation machinery targeting the concentration $W_0$; proper motions go
  to the core (the mirror image of anisotropy).
- [](../60-science-demos/optimal-design/binary-robustness.md) — Stage 4: comparison
  of binary-blind and binary-aware designs on the reported mock configuration.
  Read the exposure and limitations on the owning page.

## Gravoturbulence (experimental, repo-only)

The gravoturbulent / fractal-density-field theory documents the experimental
`gravoturb` subsystem, which is **repo-only and not part of the released
wheel** (`src/experimental/`). The five theory pages and their validation
companion are hidden in the TOC and built with the site. They describe analytic
background and historical experiments, not released-core qualification:

- [](../10-theory/gravoturbulence/index.md) — section landing: the
  density-PDF/FDF background, reading order, and experimental scope.
- [](../10-theory/gravoturbulence/density-pdf-and-fdf.md) — the
  {cite:t}`FederrathKlessen2012` lognormal+power-law density PDF, the $\rho^{3/2}$
  freefall-density kernel, the cloud-integrated SFR, and the density-law mapping.
- [](../10-theory/gravoturbulence/magnification-factor.md) — the magnification
  factor from analytic {cite:t}`ParmentierPasquali2020`, cored, and direct-3D forms.
- [](../10-theory/gravoturbulence/bm19.md) — the
  {cite:t}`Burkhart2018,BurkhartMocz2019` dense-gas SFR forward model.
- [](../10-theory/gravoturbulence/inference.md) — historical analytic inference
  experiments and their stated scope.
- [](../50-validation/gravoturbulent-pp20.md) — the validation-side companion:
  the PP20 ζ(p) regression suite + BM19 unit coverage (now backed by the
  experimental `gravoturb` AC suite), and the historical record of the
  2026-04-28 transcription-bug fix. Hidden to match the gravoturb theory subsection.

## Development log

The development log is progenax's **internal development history** — dated design
specs, full-package code-review audits, and validation recovery/scaling notes.
The pages are point-in-time snapshots (several reference since-renamed APIs and
carry pointers to the current entry point). Their historical grades, expected
results, and completion claims do not establish current readiness. The whole section is `hidden: true`
in the toc (built and URL-reachable, kept out of the public navigation):

- [](../90-development-log/index.md) — section landing: the curated map of every
  surviving entry, grouped by code-review audit / design spec / milestone, with
  pointers to the current package state.
- [](../90-development-log/code-reviews.md) — landing record for the 2026-06
  post-hardening (A−) and expert-audit (B+) reviews + resolution tables.
- [](../90-development-log/2025-12-07-progenax-review.md) — the original
  2025-12-07 comprehensive review, with its differing historical grade labels retained.
- [](../90-development-log/2026-02-12-ic-redesign.md) — the protocol-based,
  composable, differentiable IC architecture design spec.
- [](../90-development-log/phase1-complete.md) — the 2025-12-06 Phase 1
  completion snapshot.
- [](../90-development-log/2025-12-07-imf-stack-fix.md) — the IMF-stack
  correction plan (Chabrier base-10 logarithms, gradients, and domain behavior).
- [](../90-development-log/2026-02-12-imf-hmc-recovery.md) — the IMF-only NUTS
  high-mass-slope recovery proposal figure.
- [](../90-development-log/2026-02-13-binary-aware-imf-recovery.md) — the
  binary-aware IMF recovery spec (Moe+17 mixture likelihood).
- [](../90-development-log/2026-02-13-binary-aware-imf-recovery-impl.md) — the
  matching task-by-task implementation plan.
- [](../90-development-log/2026-02-13-precision-scaling-panel.md) — the
  sample-size versus posterior-precision panel proposal.
- [](../90-development-log/2026-04-28-pp20-fix.md) — the PP20 ζ(p)
  transcription-bug fix (superseded by the `gravoturb` rewrite).

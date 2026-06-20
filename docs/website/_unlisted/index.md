---
title: Unlisted pages (internal / preview)
description: >-
  In-repo pages intentionally kept out of the public navigation: OED detail
  pages, the experimental gravoturbulence theory, and the development log.
---

# Unlisted pages

This is the landing page for documentation that lives **in the repository and
builds with the site, but is intentionally absent from the public navigation**.
These pages are preview, internal, or experimental material: they are reachable
by direct URL (and from the links below) but do not appear in the top-of-page
nav, so the public table of contents stays focused on the released, supported
package.

:::{note} Unlisted-page mechanism (confirmed via myst-expert + mystmd docs, v1.10.1)
In mystmd the navigation TOC is **explicit** (`project.toc` in `myst.yml`), and
this site builds **only** the pages listed in that toc (verified empirically:
177 toc `file:` entries → 177 pages built; a file absent from the toc is *not*
built and gets no URL). So merely omitting a page from the toc would leave it
unbuilt and unreachable — wrong for a landing page that other pages must link
to.

**Chosen mechanism: `hidden: true` on the toc entry.** Per the mystmd docs
(`table-of-contents.md`), a `project.toc` entry marked `hidden: true` *is* built,
*is* reachable by its URL slug, *can* be referenced/linked by other pages, and
does **not** appear in the site navigation — the official
"built-but-not-in-the-nav" mechanism. This page is registered in `myst.yml` as:

```yaml
- file: _unlisted/index.md
  hidden: true
```

Later tasks add the OED detail pages, the consolidated gravoturbulence pages,
and (pending Anna's decision) the development log as further `hidden: true`
entries and wire the links below. No existing page is moved here yet — that is
later tasks' work.
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
  budget across (radius × {RV, PM}) to pin the Osipkov–Merritt anisotropy radius
  $r_a$; proper motions go to the outskirts ($3.66\times$ fewer stars at equal
  precision).
- [](../60-science-demos/optimal-design/dynamical-mass.md) — Stage 2: promoting survey
  depth (limiting magnitude) to a design variable for the dynamical mass $M$; the
  Fisher peaks at an interior optimal depth.
- [](../60-science-demos/optimal-design/concentration.md) — Stage 1 redux: the same
  channel-allocation machinery targeting the concentration $W_0$; proper motions go
  to the core (the mirror image of anisotropy).
- [](../60-science-demos/optimal-design/binary-robustness.md) — Stage 4: a
  robustness design where a binary-blind survey biases the mass by $+184\%$ with
  $41\times$ false confidence, and a binary-aware design removes it.

## Gravoturbulence (experimental, repo-only)

The gravoturbulent / fractal-density-field theory documents the experimental
`gravoturb_fdf` subsystem, which is **repo-only and not part of the released
wheel** (`src/experimental/`). The section was consolidated 10 → 5 pages; all
five are `hidden: true` in the toc (built and URL-reachable, kept out of the
public navigation):

- [](../10-theory/gravoturbulence/index.md) — section landing: the
  PDF → FDF → ζ → BM19 → inference chain, reading order, and the experimental banner.
- [](../10-theory/gravoturbulence/density-pdf-and-fdf.md) — the
  {cite:t}`FederrathKlessen2012` lognormal+power-law density PDF, the $\rho^{3/2}$
  freefall-density kernel, the cloud-integrated SFR, and the canonical α↔p mapping.
- [](../10-theory/gravoturbulence/magnification-factor.md) — the magnification
  factor ζ three ways (analytic {cite:t}`ParmentierPasquali2020` ζ(p), cored, direct-3D).
- [](../10-theory/gravoturbulence/bm19.md) — the
  {cite:t}`Burkhart2018,BurkhartMocz2019` dense-gas SFR forward model.
- [](../10-theory/gravoturbulence/inference.md) — the differentiable
  physics-direct inference layer (3-D α recovery + projected β estimator).
- [](../50-validation/gravoturbulent-pp20.md) — the validation-side companion:
  the PP20 ζ(p) regression suite + BM19 unit coverage (now backed by the
  experimental `gravoturb_fdf` AC suite), and the historical record of the
  2026-04-28 transcription-bug fix. Hidden to match the gravoturb theory subsection.

## Development log

Dated development-log entries (decisions, recovery notes, scaling panels) may be
curated and relocated here, pending a final public-vs-unlisted decision.

*Links will be wired in by a later task.*

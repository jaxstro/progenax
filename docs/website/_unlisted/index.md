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

The science-demos section keeps a single public overview of optimal experimental
design (OED). The longer worked-example detail pages (background, anisotropy,
concentration, dynamical-mass, binary-robustness) will be relocated here.

*Links will be wired in by a later task.*

## Gravoturbulence (experimental, repo-only)

The gravoturbulent / fractal-density-field theory documents the experimental
`gravoturb_fdf` subsystem, which is **repo-only and not part of the released
wheel**. The consolidated gravoturbulence theory pages will be linked here.

*Links will be wired in by a later task.*

## Development log

Dated development-log entries (decisions, recovery notes, scaling panels) may be
curated and relocated here, pending a final public-vs-unlisted decision.

*Links will be wired in by a later task.*

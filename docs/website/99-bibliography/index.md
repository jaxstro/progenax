---
title: Bibliography
description: Overview of the progenax bibliography section — the full reference list, ecosystem papers, per-paper annotated notes, and the paper-grounding policy.
---
# Bibliography

Every quantitative claim in these docs — every coefficient, every
formula, every distribution — is traceable to a published source.
This section is where those sources live.

## What's here

```{list-table}
:header-rows: 1

* - Page
  - Contents
* - [](bibliography.md)
  - The **full reference list**, rendered from `references.bib` (the
    single source of truth for citations). Every `{cite}` key used
    anywhere in the docs resolves here.
* - [](ecosystem-papers.md)
  - **Ecosystem / methodology references** — comparison codes
    (McLuster, COMPAS), survey papers, and other work that informs
    progenax's design but is not cited line-by-line in a single
    chapter.
* - [](per-paper/index.md)
  - **Per-paper annotated notes** — one page per cited paper with a
    paraphrased abstract, ADS / arXiv links, the equations or tables
    progenax actually uses, and the list of modules and chapters that
    depend on it.
```

## The paper-grounding policy

progenax follows a strict provenance discipline: a primary-source
fact (an equation, a coefficient, a table value) is never asserted
from memory or from a secondary summary. Each per-paper note records
what was **verified against the published PDF** — including, where
relevant, the specific equation numbers, the page, and any errata
that correct the printed paper. Claims that could not be verified
against the source are flagged as such rather than invented.

This is why the per-paper notes carry equation-level detail: they are
the audit trail linking the code's constants back to the literature.

## Adding a new paper

1. Add the BibTeX entry to `references.bib`.
2. (Optional but encouraged) Add a per-paper note under `per-paper/`,
   following the template described in [](per-paper/index.md).
3. Use the new `{cite}` key in any chapter; it will render on
   [](bibliography.md) automatically.

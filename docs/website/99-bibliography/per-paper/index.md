---
title: Per-paper detail pages
description: Index of per-paper detail pages with abstract, ADS / arXiv links, and the progenax modules each paper underpins.
---
# Per-paper detail pages

For each paper progenax cites, this section provides:

- The full bibliographic reference.
- A paraphrased abstract.
- A list of progenax modules and chapters that depend on the paper.
- Notes on caveats, conventions, and use cases.

The pages are organised alphabetically by first-author surname.

## Spatial profiles

- [](plummer-1911.md)
- [](king-1966.md)
- [](elson-fall-freeman-1987.md)

## Velocity distribution functions

- [](merritt-1985.md)
- [](michie-1963.md)

## Initial mass functions

- [](salpeter-1955.md)
- [](kroupa-2001.md)
- [](chabrier-2003.md)
- [](maschberger-2013.md)

## Environment-dependent IMF (IGIMF)

- [](marks-2012.md)
- [](marks-kroupa-2012.md)
- [](jerabkova-2018.md)

## Binaries

- [](duquennoy-mayor-1991.md)
- [](sana-2012.md)
- [](moe-distefano-2017.md)
- [](moe-2019.md)
- [](raghavan-2010.md)
- [](heggie-1975.md)

## Substructure & mass segregation

- [](cartwright-2004.md)
- [](goodwin-2004.md)
- [](allison-2009.md)
- [](baumgardt-2008.md)

## Gravoturbulence

- [](federrath-klessen-2012.md)
- [](burkhart-2018.md)
- [](burkhart-mocz-2019.md)
- [](parmentier-pasquali-2020.md)
- [](kainulainen-2014.md)

## Stellar evolution

- [](hurley-2000.md)

## How to add a new per-paper page

1. Add the BibTeX entry to `references.bib`.
2. Create a new file `99-bibliography/per-paper/<author>-<year>.md`
   following the template of an existing page.
3. Add a link to this index under the appropriate category.
4. Reference the new page from any chapters that cite the paper.

The per-paper detail page format is:

- **Frontmatter** with `title` and `description`.
- **First section.** `admonition` block with title, authors,
  reference, DOI, optional arXiv.
- **Abstract (paraphrased).** 4–6 sentences in your own words; do
  not copy the publisher abstract.
- **Use in progenax.** Bullet list of modules and chapters that
  cite the paper.
- **Notes.** Caveats, conventions, edge cases.

For an exemplar see [](parmentier-pasquali-2020.md), which has the
fullest treatment because of its central role in the BM19
framework.

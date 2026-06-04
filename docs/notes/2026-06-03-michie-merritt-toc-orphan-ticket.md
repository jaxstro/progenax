# Ticket: michie-1963 / merritt-1985 per-paper notes are TOC-orphaned + a broken xref

**Opened:** 2026-06-03 (discovered during Batch 3b-environment index wiring) · **Severity:** Minor (docs) · **Owner:** Anna

## What

The per-paper notes `99-bibliography/per-paper/michie-1963.md` and `merritt-1985.md` (added in
Batch 2c) are **not listed in `docs/website/myst.yml`'s TOC**, so they are not navigable in the
built site (they are referenced from `per-paper/index.md` and via `{cite}` keys, but have no nav
entry). They are the velocity-DF notes and should sit after `elson-fall-freeman-1987.md`.

When `michie-1963.md` is added to the TOC and built, MyST emits:

```
⚠️  99-bibliography/per-paper/michie-1963.md Cross reference target was not found: michie-poisson
⚠️  99-bibliography/per-paper/michie-1963.md Cross reference text is empty for <undefined>
```

The `{math}` block carries `:label: michie-poisson` (line 75) and it is referenced by
`{eq}`michie-poisson`` (line 98), yet the label does not register — most likely the
nested ```` ``` ````-fence `{math}`-inside-`{admonition}` earlier in the file (the
`michie-king-df` admonition) mis-closes and swallows the later label. Needs the outer
admonition fences widened to 4 backticks (or `:::{admonition}` colon fences), then re-add both
notes to the TOC.

## Why deferred (not fixed in Batch 3b-environment)

The fix is in a Batch-2c file (the Michie note), outside the 3b-environment scope. Batch 3b-env
added marks-2012 / marks-kroupa-2012 / jerabkova-2018 to the TOC (they build clean); michie/merritt
were intentionally left out of the TOC to avoid shipping the michie-poisson warning. Surfaced to
Anna at the 3b-env Gate 2.

## Fix (later)

1. In `michie-1963.md`, fix the `michie-poisson` label registration (widen the `michie-king-df`
   admonition's outer fence to 4 backticks, or convert admonitions to `:::` colon fences).
2. Add `merritt-1985.md` + `michie-1963.md` to `myst.yml` after `elson-fall-freeman-1987.md`.
3. `myst build` → confirm 0 warnings and both pages navigable.

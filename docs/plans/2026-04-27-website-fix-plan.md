# progenax website fix plan

**Date:** 2026-04-27  
**Source audit:** `docs/reviews/2026-04-27-website-verification.md`  
**Policy:** Catalogue-first. This plan proposes fixes; it does not decide whether aspirational docs should become implementation work.

## High priority

| Item | Why it matters | Suggested fix | Effort |
|---|---|---|---:|
| Repair getting-started examples | Phase E notebooks will fail immediately: `sample(N, key)`, `Chabrier`, and `PowerLawIMF(breaks=..., alphas=...)` do not match code. | Rewrite `first-plummer-sphere.md`, `differentiable-ic.md`, `imf-sampling.md` around live constructors and `sample(key, n)`. | 2-4 h |
| Rewrite two-component docs around live API | Current docs use non-existent `ComponentSpec`, `Q_target_global`, and wrong return order. | Use `TwoComponentConfig(f_A, profile_A, profile_B, velocity_df_A, velocity_df_B)` and `positions, velocities, pop_id = ...`. | 1-2 h |
| Fix mass-segregation implementation path | Current chapter imports removed `progenax.profiles.mass_segregation.apply_mass_segregation_baumgardt`; integration test still imports it too. | Decide whether docs should use `MassSegregationLayer` / `generate_cluster_ic` or introduce a wrapper later. Mark current page accordingly. | 2-3 h |
| Fix FDF example path/name | `generate_fdf_positions` is not exported; live API is lower-level `FractalDisplacementLayer`, `apply_displacement`, `generate_fractal_ic`, etc. | Rewrite example to live API or label FDF tutorial aspirational. | 2-4 h |
| Demote validation pages without validation files | 7+ pages claim `tests/validation/test_*.py` files that do not exist. | Change to "unit/offline validation" or add real validation tests later. | 2-3 h |
| Correct bad bibliography metadata | PP20, Burkhart & Mocz, Subr entries have wrong DOI/volume/page/title mapping. | Correct `references.bib` and matching per-paper pages. | 1-2 h |
| Fix MyST build-blocking glossary errors | Build exits nonzero and glossary directive emits many errors. | Convert glossary syntax to MyST definition list or remove glossary directive wrapper. | 1 h |

## Medium priority

| Item | Why it matters | Suggested fix | Effort |
|---|---|---|---:|
| Clean API reference anchors | Build emits many "No target for internal reference" warnings in `30-api/*` and full symbol index. | Update `scripts/build_api_reference.py` to emit stable MyST anchors matching links. | 2-4 h |
| Split implemented vs planned architecture | `ParticleSystem`, `SystemParams`, `BaseProfile`, `EnvironmentIMF`, `CMDOperator`, `LIMEPYProfile` read as current but are absent. | Add status badges or "planned design" callouts. | 2-3 h |
| Align protocol chapter with actual protocols | Docs claim `cumulative_mass`, `evaluate_density`, `velocity_dispersion` protocol members that are absent. | Rewrite required-method tables from `src/progenax/protocols.py`. | 1 h |
| Add missing per-paper detail pages | Several papers are cited/code-referenced but lack detail pages. | Add detail pages after DOI metadata correction. | 3-5 h |
| Refresh PP20 stale validation artifacts | Existing note flags stale BM19/PP20 figures and summary text. | Regenerate `b5_zeta_comparison.png`, `b6_pp20_diagram.png`, `e5_pp20_diagram.png`; update summary. | 2-4 h |
| Mark pseudocode snippets | Binary-aware NumPyro likelihood references undefined globals. | Add setup or label as schematic pseudocode. | 1 h |
| Fix missing citations | MyST reports missing `Heggie`; software citations may need robust URLs. | Add bib entries or remove citations. | 30 min |

## Low priority

| Item | Why it matters | Suggested fix | Effort |
|---|---|---|---:|
| Reduce list-table warnings | Many MyST `widths` options are ignored. | Remove unsupported options or use MyST-supported table syntax. | 1-2 h |
| Improve orphan cross-links | Some how-to/dev-log pages have no inbound markdown links beyond the TOC. | Add contextual links from theory/API/validation pages. | 1-2 h |
| Add advanced cluster-internals docs | Substantial `progenax.cluster` and diagnostics logic is under-documented. | Add "Advanced cluster internals" or "experimental cluster API" section. | 4-8 h |
| Verify ADS abstract fidelity | Crossref checked DOI metadata; full abstract fidelity was not exhaustively ADS-checked. | Do a dedicated bibliography pass with ADS bibcodes. | 2-4 h |
| Clarify differentiability rule scope | Docs say no while-loop, but some IMF internals use `jax.lax.while_loop`. | Explain acceptable fixed-shape `while_loop` vs sampling guidance. | 30-60 min |

## Recommended triage order

1. Fix executable tutorials and two-component/mass-segregation/FDF examples before any notebook conversion.
2. Correct bibliography metadata and MyST build errors so the site can be trusted as a reference.
3. Reclassify validation pages into real pytest validation, unit-test-backed claims, and offline/aspirational results.
4. Decide with Anna which aspirational APIs should become implementation work versus documentation rollback.


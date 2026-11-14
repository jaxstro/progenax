# MyST site ownership and hardening guide

`docs/website/myst.yml` owns navigation for the scientific documentation.
Maintainer planning records are separate from the user-facing site and do not
define scientific contracts or qualification.

## Read before editing any page

1. Find its navigation branch in `myst.yml`, then read its section overview.
2. Read the same model's spatial, velocity, architecture, API, validation, tutorial,
   model-card, and source-note pages that exist. Search for the equation, symbol,
   claim, example, and figure across the site before adding another explanation.
3. Identify the canonical owner below. Link to that owner rather than copying a
   derivation, measured result, tolerance table, or runnable example into another page.
4. Check the implementation and tests behind claims. Generated pages must be repaired
   at their source and regenerated. A successful build proves rendering, not physics.
5. Present scientific edits and consequential changes for Anna's approval. Keep new
   assumptions, thresholds, experiments, deletions, and figure designs explicit.
6. Apply the approved batch, execute affected examples with explicit float64 when
   required, check links/build, and inspect rendered equations and figures.
7. Update the page record with approval, changed files, commands, results, limitations,
   and next action. Reopen connected pages when their owner changes.

## User-facing authoring standard

- Read the whole page before editing; preserve its structure, voice, and useful content.
  Make targeted corrections, not unsolicited rewrites. Check later explanations before
  adding an earlier one; introduce the idea first and derive it once in its owning section.
- Define every symbol at first use on each page, with physical meaning, units or
  dimensionless status where relevant. Define vector directions and projected versus
  three-dimensional quantities explicitly. A name such as “scale radius” needs context.
- Describe current behavior. Remove stale bug narratives and repair warnings from
  teaching prose; preserve historical evidence in development logs and review records.
  Keep active limitations and scientifically useful cautions visible.
- Use MyST components strategically and consistently: interpretation and convention
  callouts, warnings for consequential limits, and concise related-page navigation.
  Do not replace a readable page with a wall of callouts.
- Keep runnable examples in tutorials/how-to, contracts in API/architecture, and
  measured qualification in validation. Link to their owners rather than duplicate them.
- Cite sources beside the supported claim; let MyST render the bibliography. Avoid
  a redundant manual reference summary and internal audit narration in user-facing prose.
- Use figures to answer a physical question. Define symbols, scales, normalization,
  and boundaries in the figure or caption. Keep captions concise without hiding caveats.
- Inspect the rendered page at its actual reading width: figure labels, equations,
  callout hierarchy, spacing, and links. Build success alone is not a visual review.

## Section ownership

| Section | Owns | Avoid duplicating |
|---|---|---|
| Landing | Scope and entry paths | Universal readiness/differentiability claims |
| Getting started | Executable learning sequences, installation and glossary | Full derivations and independent tolerance tables |
| Theory & methods: spatial profiles | Density, mass, potential, radius conventions and position sampling | Full velocity-DF derivations |
| Theory & methods: velocity DFs | Distribution functions, velocity moments, conditional sampling and admissibility | Spatial derivations already explained upstream |
| Other theory families | Physical assumptions, derivations and numerical methods for that family | Software interfaces and measured qualification tables |
| Model reference | Generated equation/source/code/test links | Manually maintained copies of registry-owned relations |
| Architecture | Ownership, protocols, execution, units and cross-package contracts | Re-deriving each physical model |
| API | Generated signatures, parameters, shapes and docstrings | Authored competing implementation descriptions |
| How-to | Short task-specific recipes | A second introductory tutorial |
| Validation | Test exposures, criteria, measured outcomes, limits and evidence figures | Unsupported guarantees inferred from theory |
| Science demos | End-to-end experiments, assumptions and bounded inference claims | General package qualification |
| Development log | Historical decisions and changes | Current status inferred from old completion claims |
| Release | Release-specific gates and outstanding actions | Treating approval or a local build as publication |
| Bibliography | Source identity, source notes and verified locators | Claiming a source supports an unverified equation |
| Unlisted | Navigation to hidden historical/experimental pages | Treating hidden pages as deleted or qualified |

The existing navigation combines theory and methods. Keep physical derivations and
numerical algorithms in separate subsections on their owning model page. Put JAX
execution and software contracts in architecture, signatures in API, and outcomes
in validation. Do not introduce another methods section just to resolve duplication.

Generated owners: `docs/website/scripts/build_api_reference.py` for API;
`scripts/build_provenance_registry.py` with `docs/provenance/registry/` and packaged
symbolic records for model cards; `scripts/build_test_dashboard.py` for the test
dashboard. Inspect each generated header before editing; the inventory's generated
flag is discovery metadata, not permission to overwrite a file.

## Plummer reading path and figure ownership

| Owner | Content / figures |
|---|---|
| `10-theory/spatial-profiles/plummer.md` | Density, CDF, radius conversion, potential, inverse radial sampling; `plummer_density_to_radius`, `plummer_projected_density` |
| `10-theory/velocity-dfs/plummer-dfs.md` | Eddington derivation, velocity moments, Beta speed law; `eddington_triptych`, `plummer_dispersion_oracles`, `plummer_velocity_components` |
| `10-theory/velocity-dfs/index.md` | DF meaning, compatibility, equilibrium versus necessary diagnostics; existing phase-space family comparison |
| `10-theory/velocity-dfs/rotation-anisotropy.md` | Intrinsic OM anisotropy versus additive rotation; existing anisotropy comparison |
| `00-getting-started/first-plummer-sphere.md` | One end-to-end runnable construction sequence |
| `50-validation/plummer-equilibrium.md` | Five existing validation figures, actual exposure/criteria and gradient limitations |
| `30-api/kinematics.md` | Generated public interface; fix source docstrings before regenerating |

The velocity-components illustration belongs to the DF page.
No additional Plummer figure is needed to explain the Beta law or Eddington inversion.
A possible new figure would distinguish continuum equilibrium, a finite realization,
and an energy-rescaled realization. Its design is a proposal, not an approved numerical
experiment. Do not choose new samples, thresholds, or equilibrium claims implicitly.

## Review checklist for each page

- [ ] Map owner and read connected pages; find duplicates and contradictions.
- [ ] Check equations, definitions, units, domains and source attribution.
- [ ] Check APIs, code paths, examples, random keys and dtype configuration.
- [ ] Match validation claims to current tests and recorded exposures.
- [ ] Audit figures: purpose, existing alternatives, generator, normalization,
      labels, caption, provenance and rendered readability.
- [ ] Cut stale prose/comments and propose removals; preserve relevant history.
- [ ] Show before/after edits and record approval before applying them.
- [ ] Implement at the authoritative source; regenerate dependent material.
- [ ] Run focused checks, inspect the rendered page, and record exact evidence.
- [ ] Record grade with scope, remaining risks, and re-verification trigger.

Grades are evidence-based assessments, not progress counters. Use “ungraded” until
reviewed. A/A+ requires correct claims and conventions, reproducible examples,
useful pedagogy and figures, maintained ownership, and documented limitations.
A+ additionally needs a concrete improvement in explanation or verification; it is
not awarded for adding more words, figures, or tests.

## Navigation and review scope

`docs/website/myst.yml` owns navigation. A missing navigation entry may be an
intentional include. Hidden TOC pages are built and remain directly reachable.
A completed page review does not certify the full scientific module.

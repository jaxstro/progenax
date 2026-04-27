# progenax website follow-up audit

**Date:** 2026-04-27  
**Scope:** docs-only correction pass over `docs/website/`, followed by a second code-vs-docs audit against `src/progenax/`, `tests/`, and `validation/`.  
**Source plan:** `docs/plans/2026-04-27-website-fix-plan.md`

## Honest summary

After the fix pass, the website is roughly **82% factually accurate**. The highest-risk user-facing failures from the first audit have been corrected: getting-started examples now use live constructor names and `sample(key, n)` ordering; two-component, tidal, mass-segregation, and fractal examples now point at implemented APIs or are explicitly labelled planned/offline; validation pages no longer present missing pytest suites as real; and the known bad PP20/Burkhart/Subr/Heggie bibliography metadata has been corrected.

The most important remaining fixes are:

1. **API reference anchors:** `30-api/*` still emits many internal-reference warnings; this is probably a generator/anchor problem rather than prose drift.
2. **Validation regeneration:** several numerical tables are now labelled unit-backed/offline/illustrative, but they are not freshly regenerated from runnable validation commands in this checkout.
3. **Aspirational science features:** binary likelihood conveniences, CMD operators, three-brick state containers, LIMEPY backends, and richer N-component population builders remain design directions rather than implemented APIs.

Phase E is **not safe to promote wholesale to executed notebooks today**. The prose is now much safer, but the local Python environment cannot import `jax`/`jaxstro`, so notebook promotion should wait for an environment fix plus execution smoke tests. It is reasonable to start notebook scaffolding for the three core tutorials after that environment check.

## Verification evidence

| Check | Result | Evidence / notes |
|---|---:|---|
| High-risk symbol grep after fixes | PASS with planned/negative residuals | `rg` no longer finds live-example uses of `sample(N, key)`, `Chabrier`, `PowerLawIMF(breaks=...)`, `ComponentSpec`, `Q_target_global`, `Moe17BinaryPopulation`, `apply_mass_segregation_baumgardt`, or `apply_tidal_truncation_collapsed` outside explicit "not exported/planned" statements. |
| Missing validation-file grep | PASS with explicit reclassification | Remaining hits say "There is no `tests/validation/test_*.py` file" or mark offline/unit-backed status. |
| MyST build | CONTENT MOSTLY PASS / PROCESS BLOCKED | `myst build --html` rendered 123 pages but exited on `listen EPERM: operation not permitted 0.0.0.0:3100`. Previous build-blockers from glossary syntax, missing `Heggie`, long subject text, and YAML frontmatter were resolved. |
| Snippet execution | BLOCKED | `python3` cannot import `jax`; `uv run` cannot resolve local `jaxstro`; `uv run --no-sync` also lacks `jax`. Snippets were grep/static-checked but not executed. |
| Bibliography DOI spot fixes | PASS for corrected entries | `references.bib` now has corrected PP20, Burkhart & Mocz, Subr, and Heggie DOI metadata. ADS-level abstract fidelity remains a separate low-priority pass. |

## Part 1 - module status after fixes

| Module | Status | Evidence | Remaining action |
|---|---:|---|---|
| profiles | Mostly fixed | `PlummerProfile`, `KingProfile`, `EFFProfile`, `UniformSphereProfile`, and `solve_king_profile` are live. King docs now say `solve_king_profile` returns arrays, not `KingSolution`; `LIMEPYProfile` is labelled future backend. | Fix API-page anchors and rerun rendered-link audit. |
| kinematics | Mostly fixed | Velocity DF chapters use live profile/DF pairing. Protocol docs no longer require a nonexistent `velocity_dispersion` method. | Some conceptual Eddington-inversion prose still describes theory more broadly than implemented samplers. |
| imf | Substantially fixed | Getting-started IMF examples use `ChabrierIMF`, `PowerLawIMF(exponents=..., breakpoints=...)`, `PowerLawIMF.kroupa()`, and `sample(key, n)`. Environment docs use `BirthEnvironment` and `env_to_imf_params`, not `EnvironmentIMF`. | `binary-aware-likelihood.md`, observation operators, and period-conditional likelihoods are now labelled schematic/planned, but the science decision remains: build them or keep them as design notes. |
| binaries | Substantially fixed | Period/eccentricity pages use `LogUniformPeriod`, `LogNormalPeriod`, `SanaOBPeriod`, `ThermalEccentricity`, `UniformEccentricity`, `MoeEccentricity`, and `sample(key, n)`. `Moe17Period`/`Moe17BinaryPopulation` are not presented as current APIs. | Full Moe+17 joint period/q/e sampler remains partial by design. |
| analytical | Reclassified | Validation page now points to current unit coverage rather than a missing validation file. | Numerical spot tables still need fresh execution after environment repair. |
| builders | Fixed for tutorials | `virial_scale` examples now treat the return value as velocities only; `to_com_frame` examples return positions and velocities only. | `ic-redesign-history.md` is historically framed; keep it out of tutorial promotion until snippets are executed. |
| tidal | Fixed | `apply_tidal_truncation` docs now show `(positions, velocities, masses, keep_mask)` and describe shape-collapsed output. | Dedicated validation file remains absent; page is unit-backed. |
| populations | Fixed for live API | Two-component docs now use `TwoComponentConfig(f_A, profile_A, profile_B, velocity_df_A, velocity_df_B)` and `generate_two_component_cluster(masses, cfg, key, G)` returning `(positions, velocities, pop_id)`. | N-component/per-component-IMF/global-Q machinery remains a composition pattern, not an API. |
| gravoturb | Partly fixed | PP20 DOI and PP20 validation status corrected; BM19 PDF docs use `bm19_volume_pdf`, not `density_pdf_full`; `posterior_chain` is labelled absent/planned. | Regenerate stale PP20/BM19 figures and summaries flagged in `docs/notes/2026-04-28-pp20-fix.md`. |
| protocols | Fixed | `SpatialProfile`, `VelocityDF`, and `IMFProtocol` tables now match `src/progenax/protocols.py`: `sample_positions`, `characteristic_radius`, `sample_velocities`, `logpdf`, `cdf`, `ppf`, `sample`, `mean_mass`. | Three-brick architecture remains explicitly planned rather than current public API. |

## Part 2 - validation suite status after fixes

| Page | Test file status now documented? | Properties claimed vs actually tested | Numbers match? | Remaining action |
|---|---:|---|---:|---|
| `plummer-equilibrium.md` | Yes, real validation file exists. | Still broadly aligned with `tests/validation/test_plummer_physics.py`. | Not rerun. | Re-execute after env repair and trim any unsupported energy-conservation prose if still present. |
| `king-profile.md` | Yes, real validation file exists. | Tests King ODE/profile behaviour, but LIMEPY comparison remains theoretical/future. | Not rerun. | Either remove unanchored LIMEPY spot values or add validation tests. |
| `eff-profile.md` | Yes, real validation file exists. | Core EFF tests exist; some spot numbers were not regenerated. | Not rerun. | Regenerate or demote remaining numerical tables. |
| `imf-statistics.md` | Fixed filename. | Now references `tests/validation/test_imf_physics.py`; recovered-slope table is labelled offline/illustrative. | Not rerun. | Add a pytest validation for recovered slopes or keep table offline. |
| `binary-imf.md` | Reclassified. | Correctly says no `tests/validation/test_binary_imf_recovery.py`; points to offline script under `validation/imf/`. | Not rerun. | Decide whether offline recovery should become pytest validation. |
| `fractal-substructure.md` | Reclassified. | Correctly says no dedicated validation file; unit/offline coverage described. | Not rerun. | Add reproducible validation if D/velocity-structure tables are meant to be normative. |
| `mass-segregation.md` | Reclassified. | Correctly says no dedicated validation file; unit-backed Lambda_MSR behaviour described. | Not rerun. | Add deterministic validation for published Lambda_MSR-style spot results. |
| `gravoturbulent-pp20.md` | Split status fixed. | PP20 unit tests real; BM19 forward pytest absent. | PP20 constants source-anchored; plots stale. | Regenerate `b5_zeta_comparison.png`, `b6_pp20_diagram.png`, `e5_pp20_diagram.png`, and summaries. |
| `two-component.md` | Reclassified. | Points to `tests/unit/test_populations.py`; no validation suite. | Not rerun. | Add validation if two-component examples become tutorial material. |
| `tidal-truncation.md` | Reclassified. | Points to `tests/unit/test_tidal.py`; no validation suite. | Not rerun. | Add validation file only if Jacobi-radius spot table should be a formal benchmark. |
| `analytical-test-cases.md` | Reclassified. | Points to `tests/unit/analytical/test_analytical.py`; no validation suite. | Not rerun. | Add validation wrapper if the page should stay in `50-validation`. |
| `physics-tests.md` | Reclassified. | Now says cross-cutting validation files are absent. | N/A | Decide whether to delete, rename to methodology, or add the claimed tests. |

## Part 3 - bibliography status after fixes

| Entry / page | Status | Evidence | Remaining action |
|---|---:|---|---|
| `ParmentierPasquali2020` | Fixed | DOI corrected to `10.3847/1538-4357/abb8d3` in `references.bib` and per-paper page. | Refresh PP20 validation artifacts. |
| `Burkhart2021` legacy key | Fixed metadata, legacy key retained | Entry/page now describe Burkhart & Mocz 2019, ApJ 879, 129, DOI `10.3847/1538-4357/ab25ed`. | Consider renaming key later; that is a larger citation churn. |
| `Subr2008` | Fixed | Title and DOI corrected to the mass-segregation paper, `10.1111/j.1365-2966.2008.12993.x`. | Do an ADS bibcode check before publication if exact page metadata matters. |
| `Heggie1975` | Fixed | DOI `10.1093/mnras/173.3.729` added; eccentricity citation no longer missing. | Add per-paper detail page if binary-dynamics coverage grows. |
| `Jerabkova2018` | Improved but still flagged | Title corrected to DOI metadata. | Verify intended paper against ADS; add/repair per-paper page if needed. |
| Per-paper pages | Partly fixed | Baumgardt page now points at `MassSegregationLayer`/`energy_sorted_segregation`; Moe & Di Stefano page no longer names `Moe17BinaryPopulation`. | Missing detail pages remain for several cited/code-referenced papers and software references. |

## Part 4 - gap analysis after fixes

### Code covered but still under-documented

- `src/progenax/cluster/fdf_density.py`, `fdf_tail.py`, `fdf_config.py`, and `fdf_calibration.py` contain substantial density-field and FDF calibration logic that the public site only touches indirectly.
- `src/progenax/cluster/turbulence.py` has useful Mach/forcing/Larson helpers that deserve either an advanced cluster-internals page or explicit "experimental" status.
- `src/progenax/cluster/validation.py` contains sweeps/inverse-problem helpers that could become the backbone of formal validation pages.
- `src/progenax/diagnostics/*` has Lambda_MSR, Q, and azimuthal diagnostics that are mentioned in validation prose but not documented as APIs.
- `src/progenax/dynamics/virial.py` is not surfaced in the API/tutorial narrative.

### Docs covered but implementation remains planned or unclear

- Planned/design: `ParticleSystem`, `SystemParams`, richer three-brick state containers, `CMDOperator`, photometric/CMD likelihood operators, period-conditional binary likelihoods, hierarchical binary-statistics inference, LIMEPY backend, and N-component population builders.
- Partially implemented: mass segregation and FDF/fractal layers exist under `progenax.cluster`, but their public/tutorial shape is still immature and mutually exclusive in the high-level `generate_cluster_ic` route.
- Implemented/current: canonical profiles, velocity DFs, IMF samplers, basic binary orbital/population distributions, PP20 magnification, BM19 PDF helpers, tidal utilities, simple two-component generation, and protocols.

### Per-paper bibliography gaps

The corrected bibliography still lacks per-paper pages for several cited or code-referenced entries: Aarseth 1974, Gieles & Zocchi 2015, Marks 2012, Jerabkova 2018, Subr 2008, Kuepper 2011/McLuster, Tan/Krumholz/McKee 2006, Kritsuk 2011, Hurley 2002, JAX, and Equinox.

### Cross-link density and build hygiene

The build still reports many MyST warnings, especially unsupported `list-table :widths:` options, duplicate identifiers, API anchor failures, and generated API cross-reference misses. These do not carry the same factual risk as the original false API examples, but they make the rendered site feel less trustworthy and should be handled before public release.

### Phase E tutorial readiness

The three getting-started pages are now much closer to notebook-ready, but promotion should be gated on:

1. A working local env that can import `jax`, `jaxstro`, and `progenax`.
2. A smoke-executed notebook version of each tutorial.
3. A small snippet harness for code fences in `00-getting-started`, `10-theory/populations/two-component.md`, and the tidal/substructure chapters.

### Validation regenerations

Known stale outputs from the PP20 fix still need regeneration: `b5_zeta_comparison.png`, `b6_pp20_diagram.png`, and `e5_pp20_diagram.png`. The follow-up audit also recommends refreshing any committed JSON/summary outputs under `validation/imf/` and `validation/turbulence/` after the environment is fixed, because several pages now correctly label those as offline results rather than pytest validation.

## Residual punch list

| Priority | Item | Effort | Why |
|---|---|---:|---|
| High | Fix Python/uv environment so docs snippets and validation can execute. | 1-3 h | Without this, Phase E notebooks cannot be certified. |
| High | Regenerate PP20/BM19 stale plots and summaries. | 2-4 h | PP20 was the exemplar anchor bug; stale artifacts undermine the corrected docs. |
| High | Repair API-reference anchors in `30-api/*`. | 2-4 h | Rendered API docs still produce many broken internal links. |
| Medium | Add formal validation files or permanently reclassify pages that are unit/offline backed. | 4-8 h | Prevents validation prose from drifting back into unverified claims. |
| Medium | Decide fate of planned binary likelihood/CMD/three-brick APIs. | Anna triage | These are science/product choices, not prose edits. |
| Low | Full ADS/per-paper bibliography audit and missing detail pages. | 3-6 h | Important for publication polish, lower risk for API correctness. |
| Low | Remove unsupported MyST table options and duplicate IDs. | 1-2 h | Build hygiene and trust polish. |

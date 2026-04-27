# progenax website verification audit

**Date:** 2026-04-27  
**Scope:** `docs/website/` checked against `src/progenax/`, `tests/`, `validation/`, and DOI/Crossref metadata.  
**Mode:** catalogue only; no website chapters were re-authored.

## Honest summary

The website is roughly **62% factually accurate**: the core profile, IMF, binary, tidal, and PP20 API pages mostly describe real code, but several student-facing examples and validation pages describe interfaces or suites that do not exist in this checkout. The most important fixes are: **A. repair executable tutorials/snippets before Phase E**, **B. demote or rewrite aspirational validation pages**, and **C. correct bibliography DOI/detail-page drift for PP20, Burkhart & Mocz, and Subr et al.** Phase E tutorials are **not safe to start as notebooks yet** because the prose examples currently use wrong IMF sampling signatures and aspirational two-component/FDF/mass-segregation APIs.

## Scope limits

- `pytest` was not available directly, and `uv run pytest tests/validation -q` could not resolve the local `jaxstro` dependency from the registry. Validation numbers below are therefore cross-checked against source test files and committed outputs, not fresh runtime output.
- DOI checks used Crossref DOI metadata plus targeted web lookups. ADS-specific abstract/volume/page verification was approximated from DOI metadata where ADS was not queried directly.
- MyST was run with `myst build --html` from `docs/website`. It rendered pages but exited nonzero after trying to listen on `0.0.0.0:3100` (`EPERM`); content warnings are still useful evidence.

## Evidence

- Rollout design: `docs/plans/2026-04-28-progenax-website-design.md`, especially the section 3 source-doc mapping and section 4 API strategy.
- PP20 bug pattern: `docs/notes/2026-04-28-pp20-fix.md`, especially the "anchor on published value" test-count and stale-artifact notes.
- Repo convention: `CLAUDE.md` documents `DEFAULT_UNITS = STELLAR`, JAX-native core, and validation command expectations. The requested `progenax/CLAUDE.md` path is absent in this checkout.
- Site TOC: `docs/website/myst.yml`.
- Render check: `myst build --html` produced glossary directive errors, missing citation `Heggie`, duplicate identifiers, and many missing API anchors.

## Part 1 - code-vs-docs audit per module

Extraction method: grep-audit of backticked code symbols in the relevant theory, architecture, and API chapters, excluding MyST citation keys. Statuses below list all meaningful unresolved symbols and representative verified claims; symbols not listed in a module table resolved by grep in `src/progenax/`.

### profiles

| Symbol / claim | Status | Evidence | Action needed |
|---|---:|---|---|
| `PlummerProfile`, `KingProfile`, `EFFProfile`, `UniformSphereProfile`, `solve_king_profile` | ✓ | `src/progenax/profiles/__init__.py`; `src/progenax/profiles/plummer.py:14`, `king.py:137`, `eff.py:20`, `uniform.py:15` | Keep. |
| `make_profile`, `sample_density_profile`, `compute_profile_potential` | ✓ | `src/progenax/profiles/api.py:42`, `:123`, `:177` | Keep, but API anchors fail in build. |
| `EFFProfile.r_h_to_a` | ✗ | Mentioned in `docs/website/10-theory/spatial-profiles/eff.md:98`; no grep hit in `src/progenax/`. | Reword as formula/helper not implemented, or add method later. |
| `KingSolution` | ✗ | Mentioned in `docs/website/10-theory/spatial-profiles/king.md:164`; `solve_king_profile` returns arrays, not a class. | Remove/demote. |
| `progenax.profiles.LIMEPYProfile` | ✗ | Mentioned in `docs/website/10-theory/spatial-profiles/king.md:199`; no source module. | Mark planned or delete. |
| protocol method `cumulative_mass` | ✗ | Site claims in `docs/website/10-theory/spatial-profiles/index.md:66`; `SpatialProfile` only has `sample_positions` and `characteristic_radius` in `src/progenax/protocols.py:14`. | Fix protocol prose. |
| Plummer scale-radius `a = r_h sqrt(2^(2/3)-1)` | ✓ | Test asserts exact formula in `tests/validation/test_plummer_physics.py:20`. | Keep. |

### kinematics

| Symbol / claim | Status | Evidence | Action needed |
|---|---:|---|---|
| `PlummerVelocityDF`, `KingVelocityDF`, `EFFVelocityDF` | ✓ | `src/progenax/kinematics/plummer_df.py:15`, `king_df.py:16`, `eff_df.py` | Keep. |
| `apply_osipkov_merritt`, `apply_solid_body_rotation`, `apply_differential_rotation` | ✓ | `src/progenax/kinematics/anisotropy.py:16`, `rotation.py:14`, `rotation.py:61` | Keep. |
| `sample_velocities_pipeline`, `VelocityModel`, `AnisotropyParams`, `RotationParams` | ✓ | `src/progenax/kinematics/api.py` exports these; API page generated them. | Keep, but anchors fail in build. |
| `KingSolution` | ✗ | Mentioned in `docs/website/10-theory/velocity-dfs/king-dfs.md:119`; no source class. | Replace with `xi_grid, psi_grid`. |
| `velocity_dispersion` protocol member | ✗ | Claimed in `docs/website/10-theory/velocity-dfs/index.md:110`; `VelocityDF` only requires `sample_velocities` in `src/progenax/protocols.py:55`. | Fix protocol claims. |
| Eddington-inversion implementation | ⚠ | Theory chapter says Eddington machinery; code uses distribution-specific samplers, not a generic `eddington` implementation. | Clarify theory vs code. |

### imf

| Symbol / claim | Status | Evidence | Action needed |
|---|---:|---|---|
| `PowerLawIMF`, `ChabrierIMF`, `Maschberger`, `TruncatedIMF` | ✓ | `src/progenax/imf/__init__.py:3-13`; constructors in `power_law.py:49`, `chabrier.py:24`, `smooth.py:54`. | Keep. |
| `BinaryIMF`, `FlatMassRatio`, `PowerLawMassRatio`, `TwinPeakedMassRatio`, `MoeDiStefano2017` | ✓ | `src/progenax/imf/binary.py:110`, `:145`, `:265`, `:398`, `:627`. | Keep. |
| `BirthEnvironment`, `env_to_imf_params`, `alpha3_*`, `x_*` | ✓ | `src/progenax/imf/environment.py:171`, `:861`, `:669`, `:604`. | Keep. |
| `EnvironmentIMF` | ✗ | Claimed in theory/architecture; exports are functions + `BirthEnvironment`, not a class. | Replace with actual API. |
| `BinaryIMF.with_inferred_binary_stats()` / `with_period_conditional()` | ✗ | Mentioned in IMF theory; no methods in `src/progenax/imf/binary.py`. | Mark planned or remove. |
| `CMDOperator` | ✗ | Mentioned in `observation-operators.md`; no source hit. | Mark aspirational. |
| `progenax.imf._moe17_tables`, `ftwin_for_primary`, `gamma_for_primary` | ✗ | Mentioned in mass-ratio/multiplicity chapters; no source hit. | Rewrite as conceptual tables unless implemented. |
| Getting-started IMF examples | ✗ | Docs call `imf.sample(N, key)` at `first-plummer-sphere.md:46`; code signature is `sample(self, key, n)` in `src/progenax/imf/base.py:138`. `imf-sampling.md` imports `Chabrier`, but code exports `ChabrierIMF`. | High-priority tutorial fix. |

### binaries

| Symbol / claim | Status | Evidence | Action needed |
|---|---:|---|---|
| `KeplerElements`, `compute_period`, `period_to_semimajor_axis` | ✓ | `src/progenax/binaries/kepler.py:15`, `:473`, `:515`. | Keep. |
| `BinaryOrbitalState`, resolved-state helpers | ✓ | `src/progenax/binaries/orbital_state.py:23`, `:220`, `:236`, `:285`. | Keep. |
| `LogUniformPeriod`, `LogNormalPeriod`, `SanaOBPeriod`, `ThermalEccentricity`, `UniformEccentricity`, `MoeEccentricity` | ✓ | `src/progenax/binaries/population.py:26`, `:68`, `:234`, `:115`, `:151`, `:295`. | Keep. |
| `Moe17Period`, `Moe17BinaryPopulation` | ✗ | Mentioned in binary theory; code uses `MoeDiStefano2017` in IMF and `MoeEccentricity` in binary population. | Rename to actual APIs or mark planned. |
| `binary_id`, `primary_masses` outputs | ⚠ | Mentioned as data fields; code APIs return arrays/states but not these names as public API. | Clarify as example variable names. |
| Heggie citation | ✗ | MyST build: missing citation label `Heggie` in `eccentricity.md`. | Add bib entry or remove cite. |

### analytical

| Symbol / claim | Status | Evidence | Action needed |
|---|---:|---|---|
| Analytical public functions | ✓ | `src/progenax/analytical/core.py:196`, `:390`, `:531`, `:659`, `:832`, `:959`. | Keep. |
| Validation page `tests/validation/test_analytical_physics.py` | ✗ | Page says file at `docs/website/50-validation/analytical-test-cases.md:8`; actual tests are `tests/unit/analytical/test_analytical.py`. | Rename page claim or create validation file later. |
| Spot-result table | ❓ | Cannot run tests; no matching validation file to anchor page values. | Treat as unverified. |

### builders

| Symbol / claim | Status | Evidence | Action needed |
|---|---:|---|---|
| `ICResult`, `build_spatial_ic`, `to_com_frame`, `virial_scale`, energies | ✓ | `src/progenax/builders.py:13`, `:212`, `:146`, `:171`, `:75`, `:107`. | Keep. |
| `ParticleSystem`, `SystemParams`, `BinState`, `ExternalState`, `populations.MultiBinSystem` | ✗ | Claimed in `20-architecture/three-brick-state.md`; no source hits. | Mark as architecture-history/proposed, not current implementation. |
| `gravity_policy` | ✗ | Claimed in three-brick state; no source hit. | Remove or mark planned. |
| Q convention `Q = T/|V|` | ✓ | `CLAUDE.md`; `tests/validation/test_plummer_physics.py` computes `Q = T / abs(V_analytical)`. | Keep. |

### tidal

| Symbol / claim | Status | Evidence | Action needed |
|---|---:|---|---|
| `jacobi_radius`, `jacobi_radius_isothermal`, `apply_tidal_truncation`, `fill_factor_to_r_h` | ✓ | `src/progenax/tidal.py:17`, `:51`, `:82`, `:119`. | Keep. |
| `apply_tidal_truncation_collapsed` | ✗ | Mentioned in `docs/website/10-theory/tidal-and-substructure/tidal.md:140`; no source hit. | Remove or implement later. |
| `M_gal_enclosed_func` | ⚠ | Mentioned as conceptual extension; current `jacobi_radius` accepts scalar `M_galaxy` only. | Mark planned. |
| Validation page `tests/validation/test_tidal_truncation.py` | ✗ | Actual file is `tests/unit/test_tidal.py`; no validation file. | Fix page. |

### populations

| Symbol / claim | Status | Evidence | Action needed |
|---|---:|---|---|
| `TwoComponentConfig`, `generate_two_component_cluster` | ✓ | `src/progenax/populations.py:23`, `:70`. | Keep but rewrite examples. |
| `ComponentSpec`, `Q_target_global` | ✗ | Docs use these at `docs/website/10-theory/populations/two-component.md:45-48`; actual config fields are `f_A`, `profile_A`, `profile_B`, `velocity_df_A`, `velocity_df_B` at `src/progenax/populations.py:63-67`. | High-priority rewrite. |
| Return order in docs | ✗ | Docs claim `masses, positions, velocities = generate_two_component_cluster(...)` at `two-component.md:56`; code returns `(positions, velocities, pop_id)` at `src/progenax/populations.py:70-76`. | High-priority rewrite. |
| Validation page `tests/validation/test_two_component.py` | ✗ | No file; unit file is `tests/unit/test_populations.py`. | Fix page. |

### gravoturb

| Symbol / claim | Status | Evidence | Action needed |
|---|---:|---|---|
| `magnification_factor`, `magnification_factor_with_core`, `zeta_fdf_direct`, `sfr_per_dense_gas` | ✓ | `src/progenax/gravoturb/pp20_magnification.py:151`, `:208`, `:301`, `:387`. | Keep. |
| PP20 ζ analytic values | ✓ | `tests/unit/physics/test_parmentier.py:45-83` anchors ζ(0.5), ζ(1), ζ(1.5), ζ(1.67). | Keep. |
| `P_MAX = 1.95` | ✓ | `src/progenax/gravoturb/pp20_magnification.py:142`. | Keep. |
| `density_pdf_full` | ✗ | Mentioned in density-PDF fundamentals; code has `bm19_volume_pdf`, `build_bm19_cdf_table`, `gaussian_to_bm19`. | Rename or mark planned. |
| `progenax.gravoturb.bm19_inference.posterior_chain` | ✗ | Mentioned in `bm19.md`; no module. | Mark aspirational. |
| `tff` | ⚠ | Conceptual symbol, not API; no source function. | Avoid backticks if not code. |
| Stale PP20 artifacts | ⚠ | `docs/notes/2026-04-28-pp20-fix.md` flags `validation/turbulence/bm19_fdf_suite/VALIDATION_SUMMARY.md` and three plots; source docstring still says cored form "avoids singularity at p = 1.3" in `pp20_magnification.py`. | Refresh artifacts and docstring later. |

### protocols

| Symbol / claim | Status | Evidence | Action needed |
|---|---:|---|---|
| `SpatialProfile`, `VelocityDF`, `IMFProtocol` | ✓ | `src/progenax/protocols.py:14`, `:55`, `:87`. | Keep. |
| `BaseProfile` | ✗ | Claimed in `20-architecture/protocols.md`; no source hit. | Remove or mark historical. |
| `cumulative_mass`, `evaluate_density`, `velocity_dispersion` as required methods | ✗ | Actual protocols do not include these; see `src/progenax/protocols.py`. | Fix protocol chapter. |
| `EnvironmentIMF` | ✗ | No class; see IMF section. | Fix protocol chapter. |
| Differentiability rule "no while_loop" | ⚠ | `CLAUDE.md` says fixed-iteration sampling uses scan, but live code still uses `jax.lax.while_loop` in `src/progenax/imf/base.py:77` and docs mention while-loop as anti-pattern. | Clarify rule scope. |

## Code snippet spot-checks

| Snippet | Status | Evidence | Action needed |
|---|---:|---|---|
| Binary-aware NumPyro likelihood | ⚠ | Pseudocode references undefined globals (`q_nodes_scaled`, `imf`, `mass_ratio_log_prob`, `binary_fraction`) and is not copy-paste runnable. | Label pseudocode or provide runnable setup. |
| Two-component pipeline | ✗ | Uses `ComponentSpec`, `Q_target_global`, and wrong return order; actual API in `src/progenax/populations.py:23-76`. | Rewrite before tutorials. |
| FDF positional displacement | ✗ | Imports `generate_fdf_positions` from `progenax.cluster.fdf`; source exports `generate_fractal_ic`, `FractalDisplacementLayer`, `apply_displacement`, not that function (`src/progenax/cluster/fdf.py:746-756`). | Rename to current API or mark planned. |
| Mass segregation example | ✗ | Imports `progenax.profiles.mass_segregation.apply_mass_segregation_baumgardt`; live code is `progenax.cluster.mass_segregation.energy_sorted_segregation` plus `MassSegregationLayer` in `cluster.core`. Integration test `test_knobs_pipeline.py` still imports removed path. | Rewrite docs and later fix stale integration test. |
| Getting-started IMF sampling | ✗ | Docs call `sample(N, key)`; code uses `sample(key, n)`. Docs import `Chabrier`; code exports `ChabrierIMF`. | Do not promote to notebooks until fixed. |

## Part 2 - validation suite completeness audit

| Page | Test file present? | Properties claimed -> actually tested? | Numbers match? | Action needed |
|---|---:|---|---|---|
| `plummer-equilibrium.md` | ✓ `tests/validation/test_plummer_physics.py` | Mostly: scale radius, CDF, velocity dispersion, virial Q, bound particles, beta distribution. Energy conservation over time is not in this file. | ❓ Not rerun; tolerances in docs are tighter/different in places (`Q 0.4995 ± 0.005` vs test uses `tolerances.VIRIAL_RATIO`). | Keep but trim unsupported "all twelve / energy conservation" claims. |
| `king-profile.md` | ✓ `tests/validation/test_king_physics.py` | Tests K-function, ODE monotonicity, truncation, concentration, DF isotropy. Does not test LIMEPY reference values `xi_t=30.94`, `r_h/r_c=1.491`. | ✗ Spot table not anchored in the test file. | Remove LIMEPY spot-result table or add tests. |
| `eff-profile.md` | ✓ `tests/validation/test_eff_physics.py` | Tests density formula, truncation, gamma concentration, rough half-mass, velocity DF. Does not test doc formula `r_h = a sqrt(2^{2/(gamma-3)}-1)` or KS p-values. | ✗ Spot values reuse Plummer-like `M(<a)=0.354`, not anchored. | Rewrite validation claims. |
| `imf-statistics.md` | ✗ page says `test_imf_statistics.py`; actual is `test_imf_physics.py` | Actual tests check slopes, breakpoints, Chabrier parameters, massive-star fractions, differentiability. No HMC recovered-slope table. | ✗ | Rename file references and demote inferred-slope table. |
| `binary-imf.md` | ✗ no `tests/validation/test_binary_imf_recovery.py` | Related scripts exist under `validation/imf/validate_binary_aware_recovery.py`; not pytest validation. | ❓ JSON exists, not rerun. | Mark as offline validation, not pytest suite. |
| `fractal-substructure.md` | ✗ no `tests/validation/test_fractal_substructure.py` | Unit tests exist (`tests/unit/cluster/test_fdf.py`, `test_fractal.py`, `tests/unit/substructure/*`). | ❓ Calibration table not rerun. | Create validation test or mark aspirational/offline. |
| `mass-segregation.md` | ✗ no `tests/validation/test_mass_segregation.py` | Unit tests cover monotonicity/Λ_MSR in `tests/unit/cluster/test_cluster_ic.py` and helper behavior in `test_mass_segregation.py`; no 10-realization spot table. | ❓ `Λ_MSR = 1.92 ± 0.13` is not anchored in validation file. | Demote spot table or add reproducible validation. |
| `gravoturbulent-pp20.md` | ⚠ unit tests present; no `tests/validation/test_bm19_forward.py` | PP20 unit tests are real (`tests/unit/physics/test_pp20_zeta_canonical.py`, `test_parmentier.py`). BM19 validation page overstates pytest coverage. | ✓ for PP20 constants by test source; ❓ for BM19 forward claims. | Split PP20 unit validation from BM19/offline validation. |
| `two-component.md` | ✗ no `tests/validation/test_two_component.py` | Unit `tests/unit/test_populations.py` exists but API differs from page. | ✗ | Rewrite around live API. |
| `tidal-truncation.md` | ✗ no `tests/validation/test_tidal_truncation.py` | Unit `tests/unit/test_tidal.py` likely covers formulas; not validation suite. | ❓ `51.31 pc` is formula-consistent, not rerun. | Rename to unit test or add validation file. |
| `analytical-test-cases.md` | ✗ no `tests/validation/test_analytical_physics.py` | Unit `tests/unit/analytical/test_analytical.py` exists. | ❓ | Rename/demote. |
| `physics-tests.md` | ✗ no `tests/validation/test_cross_cutting.py` or `test_physics_validation.py` | There is `tests/integration/test_jax_compatibility.py`; no named cross-cutting validation files. | ✗ | Remove or create tests. |

## Part 3 - bibliography accuracy audit

### DOI / volume / page metadata

Crossref verified title, volume, and pages for: `Plummer1911`, `King1966`, `ElsonFallFreeman1987`, `Gieles2015`, `Salpeter1955`, `Chabrier2003`, `Kroupa2001`, `Maschberger2013`, `Marks2012`, `Sana2012`, `Moe2017`, `Moe2019`, `Cartwright2004`, `Goodwin2004`, `Allison2009`, `Baumgardt2008`, `Kuepper2011`, `FederrathKlessen2012`, `Burkhart2018`, `TanKrumholzMcKee2006`, `Kainulainen2014`, `Kritsuk2011`, `Hurley2000`, `Kuepper2011_McLuster`, `Riley2022_COMPAS`, `Szecsi2022_BoOST`.

| Entry | Status | Evidence | Action needed |
|---|---:|---|---|
| `Jerabkova2018` | ⚠ | DOI resolves, but Crossref title is "Impact of metallicity and star formation rate..." while bib title says UCD/GC formation. | Verify intended Jerabkova paper and correct title/detail page. |
| `Subr2008` | ✗ | DOI resolves to an unrelated primordial non-Gaussianity paper; Crossref pages `1613-1620` vs bib `1673-1680`. | Replace DOI/pages with correct Šubr et al. paper metadata. |
| `Burkhart2021` | ✗ | DOI `10.3847/1538-4357/abc484` resolves to "CATS", ApJ 905, 14; the self-gravitating gas fraction paper is Burkhart & Mocz 2019, ApJ 879, 129, DOI `10.3847/1538-4357/ab25ed`. | Correct year/volume/page/DOI and detail page. |
| `ParmentierPasquali2020` | ✗ | DOI `10.3847/1538-4357/abb6f8` resolves to a reverberation-mapped Mg ii paper, ApJ 903, 86. Target PP20 paper is ApJ 903, 56, DOI `10.3847/1538-4357/abb8d3`. | Correct DOI in bib and detail page. |
| `Hurley2002` | ⚠ | Crossref API returned 404, but web lookup confirms DOI `10.1046/j.1365-8711.2002.05038.x`, MNRAS 329, 897-928. | Keep DOI; maybe Crossref endpoint quirk. |
| `Equinox` | ⚠ | Crossref API returned 404 for Zenodo DOI. | Verify via Zenodo if this entry is intended for MyST citation. |
| `Aarseth1974`, `JAX` | ⚠ | No DOI in bib. | Accept if intentional; add URL/ADS bibcode if desired. |

### Per-paper detail pages

There are **31 bibliography entries** in `references.bib`, not 27. There are **20 per-paper detail pages** plus the index. Missing per-paper pages for cited/nontrivial entries: `Aarseth1974`, `Gieles2015`, `Marks2012`, `Jerabkova2018`, `Subr2008`, `Kuepper2011`, `TanKrumholzMcKee2006`, `Kritsuk2011`, `Hurley2002`, `Riley2022_COMPAS`, `Szecsi2022_BoOST`, `JAX`, `Equinox`.

All per-paper "Use in progenax" markdown links resolve to real pages. Some linked pages do not actually cite the paper yet, and several detail pages point at aspirational implementation symbols:

| Detail page | Status | Evidence | Action needed |
|---|---:|---|---|
| `baumgardt-2008.md` | ✗ | "Use" points at `progenax.profiles.mass_segregation.apply_mass_segregation_baumgardt`, which does not exist. | Replace with `progenax.cluster.mass_segregation.energy_sorted_segregation` / `MassSegregationLayer`. |
| `parmentier-pasquali-2020.md` | ✗ | DOI wrong as above; abstract is faithful to PP20 but attached to wrong DOI. | Correct DOI. |
| `burkhart-2021.md` | ✗ | Metadata describes 2019 Burkhart & Mocz paper, not the DOI/year in bib/page. | Correct entry/page. |
| `chabrier-2003.md` and `salpeter-1955.md` | ✓ | Detail-page use links resolve and cited chapters cite the keys. | Keep. |
| Other 16 detail pages | ⚠ | Links resolve and abstracts are broadly faithful at the one-paragraph level, but ADS abstract fidelity was not fully checked. | Full ADS abstract audit remains. |

## Part 4 - gap analysis

### Code covered but under-documented

Recommended action: add a "cluster internals / advanced substructure" status page before claiming full site coverage. The most substantial undocumented code is:

- `src/progenax/cluster/fdf_density.py`: `DensityField3D`, turbulent/BM19 density-field initializers, PMF sampling, density-tail IC generation.
- `src/progenax/cluster/turbulence.py`: Mach-to-density-width, Larson scaling, virial Mach, environment forcing parameter `b`.
- `src/progenax/cluster/fdf_tail.py`: BM19/PN11 tail PMFs and sampling.
- `src/progenax/cluster/fdf_config.py` and `fdf_calibration.py`: cluster-type to FDF layer mappings and calibration loader.
- `src/progenax/cluster/validation.py`: λ_seg sweeps, fractal-dimension sweeps, toy inverse problem.
- `src/progenax/diagnostics/*`: `compute_lambda_msr`, Q approximations, azimuthal variation.
- `src/progenax/imf/differentiable_binary.py`: differentiable binary-fraction/model helpers.
- `src/progenax/dynamics/virial.py`: virial helpers not surfaced in API docs.

### Docs covered but implementation unclear

Recommended classification:

- **Implemented:** Plummer/King/EFF profiles, Plummer/King/EFF velocity DFs, canonical IMF classes, binary orbital mechanics and simple population distributions, PP20 ζ, BM19/Pn11 core functions, tidal utilities, current simple two-component generator.
- **Partially implemented:** FDF and mass segregation exist under `progenax.cluster`, but public docs use wrong paths/names and overstate validation.
- **Aspirational/planned:** `ComponentSpec`, `Q_target_global`, `ParticleSystem`, `SystemParams`, `EnvironmentIMF`, `CMDOperator`, `LIMEPYProfile`, `Moe17Period`, `Moe17BinaryPopulation`, `generate_fdf_positions`, `apply_mass_segregation_baumgardt`, `bm19_inference.posterior_chain`.

### Per-paper bibliography gaps

Code docstrings cite papers without matching per-paper detail pages: Aarseth 1974, Gieles & Zocchi 2015, Marks 2012, Jerabkova 2018, Subr 2008, Kuepper 2011/McLuster, Tan/Krumholz/McKee 2006, Kritsuk 2011, Hurley 2002, and the JAX/Equinox software references.

### Cross-link density

The markdown link graph has 12 pages with no inbound markdown links besides TOC membership:

- `40-howto/gradient-based-r_h-fit.md`
- `40-howto/mix-plummer-positions-king-velocities.md`
- `40-howto/set-up-virial-cluster.md`
- `90-development-log/2025-12-07-progenax-review.md`
- `90-development-log/2026-02-12-imf-hmc-recovery.md`
- `90-development-log/2026-02-13-binary-aware-imf-recovery-impl.md`
- `90-development-log/2026-02-13-binary-aware-imf-recovery.md`
- `90-development-log/2026-02-13-precision-scaling-panel.md`
- `90-development-log/by-topic.md`
- `90-development-log/code-reviews.md`
- `90-development-log/whats-changed.md`
- `99-bibliography/ecosystem-papers.md`

### Phase E tutorial gap

Not safe to promote as-is. `first-plummer-sphere.md`, `differentiable-ic.md`, and `imf-sampling.md` all need executable API correction first (`sample(key, n)`, `ChabrierIMF`, `PowerLawIMF(exponents=..., breakpoints=...)`, and realistic NumPyro setup labels).

### Validation regenerations

Confirmed PP20 stale-artifact list from `docs/notes/2026-04-28-pp20-fix.md`: `validation/turbulence/bm19_fdf_suite/VALIDATION_SUMMARY.md`, `b5_zeta_comparison.png`, `b6_pp20_diagram.png`, and `e5_pp20_diagram.png`. Additional refresh candidates:

- `docs/website/50-validation/plot-gallery.md` references `validation/plots/*`, but no `validation/plots` directory appears in this checkout.
- Any committed BM19 outputs that pin ζ values need audit; the PP20 note says none were found in `tests/`, but this audit did not rerun the turbulence scripts.
- API reference anchors are generated but not valid MyST targets; regenerate/fix `docs/website/scripts/build_api_reference.py` before relying on full symbol index links.

## Build and verification commands

Commands attempted:

```bash
pytest tests/validation -q
uv run pytest tests/validation -q
cd docs/website && myst build --html
```

Results:

- `pytest`: unavailable (`command not found`).
- `uv run pytest tests/validation -q`: failed dependency resolution because `jaxstro` was not found in the package registry.
- `myst build --html`: built pages but exited nonzero after `listen EPERM: operation not permitted 0.0.0.0:3100`; also emitted content warnings/errors listed above.


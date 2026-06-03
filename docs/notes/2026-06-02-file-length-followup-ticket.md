# Ticket: remaining files over the 500-LOC limit

**Opened:** 2026-06-02 · **Status:** 📋 **OPEN — follow-up** · **Source:** audit "oversize units" (file-length) + 2026-06 hardening Batch 6

## Context

The 2026-06 hardening split the audit-named oversize **functions**
(`generate_cluster_ic`, `generate_fractal_ic_density`) and the >1000-LOC **files**
chosen for that PR (`binary.py`, `fdf_density.py`, `analytical/core.py`,
`imf/environment.py`) into subpackages, each module <500 LOC, public APIs
unchanged. (Note: `cluster/core.py` is 485 LOC — already compliant; the "core.py"
the audit listed at 1061 LOC was `analytical/core.py`.)

The ecosystem limit is **file ≤500 LOC (300 preferred)**. The files below still
exceed it and were deferred to keep the hardening PR's regression surface bounded.

## Remaining files > 500 LOC (measured 2026-06-02, post-split)

| File | LOC | Suggested split seam |
| --- | --- | --- |
| `cluster/fdf.py` | 767 | displacement-layer FDF vs the `generate_fractal_ic` pipeline |
| `cluster/fdf_tail.py` | 562 | BM19 tail PMFs vs the `pn11_legacy` comparison path |
| `binaries/kepler.py` | 558 | Kepler elements vs orbital-state propagation |
| `cluster/validation.py` | 543 | per-diagnostic validators |
| `cluster/fractal_gw_legacy.py` | 535 | generation vs rescale/virialize (legacy; may instead be retired when the FDF path fully supersedes it) |
| `binaries/population.py` | 531 | period dists vs eccentricity dists vs orientation/mass-dependent |
| `profiles/king.py` | 508 | profile/ODE vs the lowered-Maxwellian density helpers |
| `gravoturb/bm19_model.py` | 507 | PDF model vs the σ_s²/s_t closure relations |

## Approach (when scheduled)

Same pattern as Batch 6: map the internal dependency graph (confirm acyclic),
slice along cohesive seams with `sed` (byte-exact), re-export from each
subpackage `__init__.py` so callsites are unchanged, and verify each split is
behavior-preserving with the relevant test subset (and an RNG byte-identical
check for any stochastic sampler, as was done for `generate_cluster_ic`).

These are maintainability-only refactors (the code is correct), so they are
lower priority than open science items.

## Function-length follow-up (>100-LOC functions in the split files)

The 2026-06 hardening split the audit-named oversize *files*, but several
*functions* (inherited from the old monoliths) still exceed the project's
**100-LOC function limit (50 preferred)**. These should be addressed alongside
the file-length splits above. Measured 2026-06-03 (def line → end):

| Function | LOC | File |
| --- | --- | --- |
| `sample_positions_tail` | 231 | `cluster/fdf_density/sampling.py` |
| `init_bm19_density_field` | 160 | `cluster/fdf_density/field_init.py` |
| `two_body_kepler` | 145 | `analytical/two_body.py` |
| `generate_fractal_ic_density` | 140 | `cluster/fdf_density/pipeline.py` |
| `env_to_imf_params` | 137 | `imf/mapping.py` |
| `init_turbulent_density_field` | 131 | `cluster/fdf_density/field_init.py` |
| `solar_system_inner_4` | 125 | `analytical/solar_system.py` |
| `three_body_figure_eight` | 111 | `analytical/few_body.py` |
| `solar_system_full` | 103 | `analytical/solar_system.py` |

(`init_bm19_density_field` is 160 LOC post-Batch-C — the 2026-06-03 concreteness
guard added ~3 lines over the audit's 157.) As with the file splits, these are
maintainability-only — the functions are correct and tested.

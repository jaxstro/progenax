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

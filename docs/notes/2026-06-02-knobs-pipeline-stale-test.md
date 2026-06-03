# Stale test: `tests/integration/test_knobs_pipeline.py`

**Opened:** 2026-06-02 · **Status:** ✅ **RESOLVED — deleted 2026-06-02 (Batch 2)** · **Source:** Batch 0 (audit **M4**)

> **Resolution (Anna: "remove or refactor stale tests"):** the file was **deleted**. It
> uniquely tested only the two *removed* functions (`apply_mass_segregation_baumgardt`,
> `apply_fractal_overlay_radial`); every still-existing API it touched is covered elsewhere
> (`energy_sorted_segregation` → `test_mass_segregation.py`/`test_cluster_ic.py`,
> `generate_two_component_cluster` → `test_populations.py`, binaries → `test_population.py`,
> anisotropy/rotation/tidal/fractal each in their own unit tests). So deletion lost **no**
> coverage of existing functionality. The diagnosis below is retained as a record.

## Summary

`tests/integration/test_knobs_pipeline.py` is **refactor-orphaned**. It exercises a
pre-refactor "knobs pipeline" API that no longer exists in the package. A single dead
top-level import (`tests/integration/test_knobs_pipeline.py:13`) raised `ModuleNotFoundError`
during pytest **collection**, which aborts the *entire* `pytest tests/` session (0 tests run).
The 2026-06-01 audit's "812 passed" baseline was obtained only by *excluding* this file.

The audit's M4 remedy ("correct the import path `profiles` → `cluster`") **under-diagnosed**
the problem: the symbol was not merely moved, it was redesigned with an incompatible
signature, and two other imports in the file were removed entirely. A path-only fix still
`ImportError`s.

**Batch 0 action (Anna-approved):** quarantine via a module-level
`pytest.skip(..., allow_module_level=True)` placed *above* the dead imports, so the suite
collects and the ~812 healthy tests run. No physics was invented; no real test was weakened.
**This ticket tracks the real resolution for Batch 4.**

## Evidence — the three dead imports

| Import (test line) | Reality (verified 2026-06-02) |
|---|---|
| `from progenax.profiles.mass_segregation import apply_mass_segregation_baumgardt` (L13, 199, 431; **7 call sites**: L47, 115, 150, 179, 218, 466) | The module moved `profiles → cluster`, **and** the function was redesigned. The symbol `apply_mass_segregation_baumgardt` is **defined nowhere** in `src/`. The current function is `progenax.cluster.mass_segregation.energy_sorted_segregation`. |
| `from progenax.substructure.fractal import generate_fractal_positions, apply_fractal_overlay_radial` (L239–240) and `... import apply_fractal_overlay_radial` (L432) | `progenax.substructure.fractal` **does not exist**. `generate_fractal_positions` now lives in `progenax.cluster.fractal_gw_legacy` (itself legacy, also slated for Batch 4). `apply_fractal_overlay_radial` is **defined nowhere**. |
| `from progenax.binaries.population import (LogNormalPeriod, ThermalEccentricity, sample_isotropic_orientations, ...)` and the L433–441 binary imports | ✅ All still exist — these are fine. |

### Signature incompatibility (the core blocker)

```text
# What the test calls (pre-refactor API, removed):
apply_mass_segregation_baumgardt(positions, velocities, masses, s=..., key=..., G=...)
    -> (positions, velocities)                                   # 2-tuple, in-place reposition + strength s

# What exists now (src/progenax/cluster/mass_segregation.py:33):
energy_sorted_segregation(key, masses, positions_pool, velocities_pool, potential_fn)
    -> (masses, positions, velocities)                           # 3-tuple, pool + potential_fn assignment
```

The new function performs energy-ordered assignment from an **orbit pool** (`N_pool > N`)
given an analytic `potential_fn`; it has no `s` (segregation-strength) argument — partial
segregation is now handled by `lambda_seg` blending in the IC generator, not in this function.
You cannot wrap one onto the other without writing new physics, so no thin adapter is
appropriate here.

## Git provenance

- `54295c9` feat(profiles): add Baumgardt-style energy-ranked mass segregation *(original `apply_mass_segregation_baumgardt`)*
- `c8107d7` refactor(profiles): Rewrite mass_segregation with Baumgardt algorithm *(redesign → `energy_sorted_segregation`)*
- `1159769` refactor(cluster): rename fractal.py to fractal_gw_legacy.py *(removes `substructure.fractal`)*
- `13df53f` test: Clean up progenax tests to physics-only (566 → 213 tests) *(big test cull that left this file behind)*

## Batch 4 decision (choose one, per-item approval)

1. **Rewrite** `test_knobs_pipeline.py` against the current APIs: build an orbit pool +
   analytic `potential_fn` and call `energy_sorted_segregation`; drop the removed
   `apply_fractal_overlay_radial` knob (or repoint fractal coverage at the supported path).
   *Pro:* restores real end-to-end integration coverage of the knobs pipeline.
   *Con:* non-trivial; depends on the Batch-2 DF work and the Batch-4 `fractal_gw_legacy`
   disposition.
2. **Delete** the file. *Pro:* fastest; the API it tests is gone. *Con:* loses the
   integration-pipeline coverage; triggers the deletion gate.

**Recommendation:** decide alongside the `fractal_gw_legacy` disposition in Batch 4, since the
fractal-overlay knob is entangled with it. Until then, the quarantine keeps CI honest.

## How to lift the quarantine

Remove the `pytest.skip(..., allow_module_level=True)` block at the top of
`tests/integration/test_knobs_pipeline.py` once the file is rewritten (option 1) — or delete
the file (option 2). Re-run `uv run --no-sync pytest tests/integration/test_knobs_pipeline.py -q`
to confirm.

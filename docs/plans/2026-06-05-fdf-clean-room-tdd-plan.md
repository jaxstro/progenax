# Gravoturbulent-FDF Clean-Room Rewrite — TDD Implementation Plan (canonical)

> Repo-tracked canonical copy of the approved TDD plan. Authoritative spec:
> [`2026-06-05-fdf-clean-room-spec.md`](2026-06-05-fdf-clean-room-spec.md) §8.
> Approved by Anna 2026-06-05. Sub-skills: `superpowers:test-driven-development`,
> `superpowers:executing-plans`. Every physics task opens the cited PDF first.

**Goal:** Rebuild the gravoturbulent-1D + FDF-3D IC subsystem clean-room from PDF-grounded
theory into a standalone `gravoturb_fdf` package, validated by a committed AC1–AC10
acceptance suite that prints real numbers, with released progenax severed from it and the
old modules deleted.

**Architecture:** `src/experimental/gravoturb_fdf/` (importable as `gravoturb_fdf`, **not**
in the progenax wheel), one-way dep on `progenax.cluster.turbulence`. Layers: `theory/`
(BM19/PP20/PN11/PDF — JAX, differentiable), `field/` (GRF + rank copula, tail, sampling,
pipeline), `diagnostics/` (CW04 Q — numpy/scipy, non-diff), `validation/` (AC scripts).
Phases P0→P6, HITL-gated at phase boundaries.

**Tech:** JAX/Equinox/jaxtyping in core paths; numpy/scipy only in `diagnostics/` + test
fixtures. CGS microphysics, solar units stellar. pytest marker `experimental`; uv (jax
0.10.1) + conda astro (jax 0.7.0).

---

## Context

The FDF/gravoturbulent subsystem (~15.3k LOC) was built experimentally by older models with
a documented tendency to fabricate (caught: PP20 fake p=1.3 pole; fabricated BM19 s_t; PN11
prefactor 2.3× off; a "validated" suite running white-noise through a √N-less Q estimator →
nonphysical Q≈0.13). Decision (spec §8): clean-room rebuild into `src/experimental/` as a
follow-up-paper feature excluded from the initial release. Existing code = untrusted
reference. "Validated" appears only next to fresh printed output.

## Operating rules

- **TDD RED→GREEN**; never weaken a test, fix the physics. **Trust-nothing Step 0:** open
  the cited PDF, confirm the equation, before writing the test.
- **HITL cadence (Anna, 2026-06-05):** commit each task autonomously (TDD + verified);
  check in + give next-steps **at phase boundaries**; surface consequential decisions
  (API shape, physics) before acting.
- **Verify** at phase boundaries with **uv (jax 0.10.1)**: `env -u VIRTUAL_ENV uv run
  --no-sync pytest tests/ -q`. (Conda/jax-0.7.0 was dropped from the loop on 2026-06-05;
  P0 was both-env-verified before that.)
- **Released-core invariant (corrected):** the ~814 released-core tests stay green at every
  gate. ~404 of the original 1217 are subsystem tests retired in P5 and replaced by
  `tests/experimental/`. (See P0 appendix.)
- Atomic commits; trailer `Co-Authored-By: Claude Opus 4.8 (1M context)
  <noreply@anthropic.com>`; push to origin/main only on Anna's explicit go (one final PR).

## Acceptance criteria (each = a committed script that PRINTS the number)

| # | Criterion | Threshold | Phase |
|---|-----------|-----------|-------|
| AC1 | BM19 scalars vs analytic (σ_s², s_t, f_dense, lognormal limit) | <1e-6 | P1 |
| AC2 | Mass conservation ∫eˢ p_LN ds = 1 | ±1e-3 | P1 |
| AC3 | ζ anchors ζ(0)=1, ζ(1)=1.089, ζ(1.5)=√2, ζ(1.67)=1.79 | <0.1% | P1 |
| AC4 | ζ_FDF vs analytic ζ(p), power law, p<1.7 | few % | P1 |
| AC5 | CW04 Q sanity: uniform 0.79±0.04; fractal D=1.5→0.47, 2.0→0.58, 2.5→0.70 | CW04 σ | P1 |
| AC6 | **CORNERSTONE** f_tail≈f_dense (rank copula, 128³) | <5% single, <1% ens | P2 |
| AC7 | Q(f_sub) monotone↓, near CW04 D-targets, real range | Q∈[0.4,0.8]+bands | P3 |
| AC8 | grad signs ∂f_dense/∂ℳ<0, ∂f_dense/∂α<0, ∂ζ/∂α<0 | sign + FD agree | P1 |
| AC9 | FD-vs-autodiff on public diff entry points | rel err <1e-4 | P1 |
| AC10 | Full suite both envs (released-core + experimental) | 100% | every gate |

## Phases (bite-sized RED→GREEN tasks live in the session plan; summary here)

- **P0 — sever + scaffold (DONE; see appendix).** Stand up `gravoturb_fdf`; sever every
  core→subsystem coupling; importer-free guard; no module deletion yet.
- **P1 — acceptance harness.** Author `theory/{bm19,pp20,pn11,pdf}.py` + `diagnostics/q.py`
  clean-room; AC1–AC5, AC8, AC9 as printing scripts. PDF Step-0 per module.
- **P2 — IC pipeline + cornerstone.** `field/{field,tail,sampling,pipeline}.py`; GRF + rank
  copula; AC6 (make-or-break; report honestly, do not tune).
- **P3 — f_sub→Q calibration (headline).** Tiered 64³ smoke → 128³ production; AC7 + bands;
  smooth differentiable Q(f_sub;σ_s,β) surrogate.
- **P4 — turbulence grounding.** Verify σ_s²/β(ℳ)/α_vir(Σ)/Larson vs FK10/Heyer/Kim&Ryu;
  correct citations; label heuristics. (turbulence.py stays in released core.)
- **P5 — quarantine→delete + consolidation.** Delete the subsystem modules + their ~390
  retained tests (importer-free guard already green); enforce LOC limits.
- **P6 — docs.** Regenerate VALIDATION_SUMMARY + website/per-paper notes from fresh AC runs.

---

## Appendix — P0 inventory result (recorded 2026-06-05)

**Baseline:** 1217 tests collected (both envs at `ae1a65d`).

**Subsystem footprint inside released `tests/` (~404 tests, retired in P5):**
`tests/unit/physics/` = 180 (BM19/PP20/PN11); `tests/unit/cluster/` test_fdf 49, test_fdf_tail
34, test_fractal 34, test_gravoturbulent 25, test_fdf_density 25, test_tail_sampling 24,
test_fdf_calibration 3, test_density_normalization 2, test_pn11_routing 1; `tests/unit/
substructure/` 45; `tests/validation/test_bm19_field_tail.py` 4.

**Couplings severed in P0 (core→subsystem):**
1. `progenax/__init__.py` re-export of `fractal_gw_legacy` symbols — removed (commit `8005651`).
2. `cluster/core.py` legacy-FDF branch via `FractalLayer` — **Option A**: `FractalLayer` +
   `fractal=` removed from the released `SpatialStructureParams` API entirely; substructure
   moves to experimental; `cluster/validation.py` fractal helpers dropped (commit `870a1f3`).
3. `imf/environment/birth_environment.py` reached turbulence via `cluster.fdf_config`
   re-exports — repointed directly to `progenax.cluster.turbulence` (commit `4010661`).
4. **(found during P0)** `cluster/__init__.py` re-exported the entire FDF/gravoturbulent API
   (7 module-level imports) — removed; `progenax.cluster` now surfaces only core + mass-seg +
   constants + turbulence (commit `868fa15`).

**Importer-free invariant:** enforced by `tests/experimental/unit/test_core_severance.py::
test_no_core_module_imports_subsystem_at_module_level` (AST scan; GREEN). Released core has
zero module-level subsystem imports → P5 deletion is pure file removal. Subsystem modules
still import standalone for reference during P1–P4.

**Delete map for P5 (modules):** `cluster/fdf.py`, `cluster/fractal_gw_legacy.py`,
`cluster/fdf_calibration.py`, `cluster/fdf_config.py`, `cluster/gravoturbulent.py`,
`cluster/fdf_tail.py`, `cluster/fdf_density/*`, `cluster/fdf_hyperparams.py`, `gravoturb/*`;
suite `validation/.../bm19_fdf_suite/` (√N-less Q + white-noise); + the ~390 retained
subsystem tests. (`cluster/constants.py` stays — turbulence depends on it.)

**P0 commits:** `1c96978` scaffold · `8005651` sever 1 · `870a1f3` Option A (sever 2) ·
`4010661` sever 3 · `868fa15` cluster re-export removal (4) + importer-free guard.

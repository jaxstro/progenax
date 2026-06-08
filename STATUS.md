# progenax — status

next: segregation module validated in progenax (Λ_MSR analytic Tier-A, 8 tests + plots; released-core 814→822) + stale validate scripts cleaned (3 deleted, 2 refactored); gravax dynamical-segregation experiment DEFERRED (docs/notes/2026-06-08-...). Next: methods-paper Figs 1-3 (buildable now) + Tier-C density-correlated mass placement + mass-weighted metric
blocker: none
due:

## Current focus
_Seeded 2026-06-07 by the brain STATUS.md convention (`~/brain/work/meta/status-convention.md`). Update in your progenax session; the brain pulls `next:`/`blocker:`/`due:` via `federate.py`._

Differentiable ICs / population generation (IMF, binaries, cluster profiles). Feeds the Cottrell census (Aim 1) + the jaxstro methods paper. Mature, actively committed.

## Open
- [x] FDF ICs: spherical envelope (envelope × turbulent fluctuation; reuse Plummer/EFF profiles)
- [x] FDF ICs: end-to-end `build_cluster_ic` (envelope + coherent velocities + chosen virial Q)
- [x] FDF ICs: validation + curated figure gallery (`validation/cluster_acceptance.py`, 5/5 AC, 4 figs)
- [x] substructure diagnostic to parameterize: CW04 (m̄,s̄) plane — m̄ ≈ concentration axis, Q/s̄ resolves β
- [ ] (optional) m̄↔β quantitative calibration table for inference-grade substructure recovery
- [ ] methods-paper-ready API + validation figures

## Nice-to-do / revisit later
- Filamentary (non-Gaussian-phase) morphology for the FDF field — deferred (second-order for the
  N-body/binary purpose at matched P(k)+clumpiness+virial). Why-wait + pedagogical implementation
  (turbulent-shock / Zel'dovich displacement + copula; equations) in
  docs/plans/2026-06-07-gravoturb-fdf-spherical-ic-design.md (§ NICE-TO-DO). Revisit only if a
  dynamical observable distinguishes filament-vs-blob ICs, or for direct morphology/gas→star studies.

## Decided (2026-06-07)
- 2D-projected β inference from star catalogues: **banked as a methods result, not optimizing
  further** (cosmic-variance + dynamics limited; gas measures β more directly). Analytic high-N
  SBC p=0.82, flowjax NPE low-N p=0.20. Pivot to FDF ICs as a forward generative tool + (future)
  3D Gaia inference where the projection wall vanishes. Captured to brain inbox 2026-06-07.

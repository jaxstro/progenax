# progenax — status

next: NEW SESSION — systematic validation audit of all progenax modules + pub-quality methods figures. Entry point: docs/plans/2026-06-08-validation-audit-and-methods-figures-spec.md (per-module: brainstorm plot set → build pub-quality figures → embed on 50-validation page → BUILD-VERIFY render → update index dashboard). Refactor/remove stale validate scripts as found. Done so far: mass-segregation/plummer/two-component pages current+figure-bearing+build-verified; Λ_MSR validated; Tier C built.
blocker: none
due:

## Current focus
_Seeded 2026-06-07 by the brain STATUS.md convention (`~/brain/work/meta/status-convention.md`). Update in your progenax session; the brain pulls `next:`/`blocker:`/`due:` via `federate.py`._

Differentiable ICs / population generation (IMF, binaries, cluster profiles). Feeds the Cottrell census (Aim 1) + the jaxstro methods paper. Mature, actively committed.

## Recent progress (2026-06-08)
- **Segregation module validated in progenax (no gravax).** Λ_MSR diagnostic
  (`progenax.diagnostics.compute_lambda_msr`, Allison 2009) validated against analytic ground
  truth — `tests/validation/test_mass_segregation_physics.py` (8 tests: unsegregated→1, exact
  N=2, maximal→≫1, inverse→<1, convergence, binary caveat). Released-core **814→822**.
- **Paper grounding.** `docs/website/99-bibliography/per-paper/allison-2009.md` rewritten + verified
  vs the held ApJ 700 L99 PDF; found+fixed a docstring mis-citation (L99 Eq.1 = Spitzer t_seg, not
  Λ_MSR; formal def is MNRAS 395,1449 — PDF not held).
- **Validate-script cleanup (round 1).** 5 of 7 `scripts/validate_*.py` were broken (FDF clean-room +
  feature removal). Deleted 3 obsolete (`validate_fdf`, `validate_tail_sampling`,
  `validate_imf_extensions`); refactored `validate_mass_segregation` + `validate_cluster_ic` to the
  live API (all PASS). Commits 0745731, d327774.
- **Gravax dynamical mass-segregation experiment DEFERRED** (Anna's call) — full spec in
  `docs/notes/2026-06-08-gravax-segregation-validation-followup.md`.

## Open
- [x] FDF ICs: spherical envelope, `build_cluster_ic`, validation gallery, CW04 (m̄,s̄) plane
- [x] Λ_MSR segregation diagnostic validated (analytic Tier-A) + stale validate-script cleanup r1
- [ ] **validation/plots + experimental scratch-driver cleanup (round 2)** — retire banked-2D-β
      scratch (`_d0*`,`_v2*`,`_v3/4/5` + their plots); classify verified-vs-stale (in progress)
- [ ] integrate verified validation plots into the website docs pages
- [ ] Tier C: density-correlated mass placement (`correlated_mass_assignment`) + grounded
      mass-weighted substructure metric
- [ ] methods-paper Figs 1–3 (buildable now) + (deferred) gravax Figs 4–5
- [ ] (optional) m̄↔β quantitative calibration table

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

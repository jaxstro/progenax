# progenax — status

next: FDF IC pipeline — validate+test+plot, add a spherical cluster-shape envelope, design a parameterizable substructure diagnostic
blocker: none
due:

## Current focus
_Seeded 2026-06-07 by the brain STATUS.md convention (`~/brain/work/meta/status-convention.md`). Update in your progenax session; the brain pulls `next:`/`blocker:`/`due:` via `federate.py`._

Differentiable ICs / population generation (IMF, binaries, cluster profiles). Feeds the Cottrell census (Aim 1) + the jaxstro methods paper. Mature, actively committed.

## Open
- [ ] FDF ICs: spherical envelope (envelope × turbulent fluctuation; reuse Plummer/EFF profiles)
- [ ] FDF ICs: validation + curated figure gallery
- [ ] substructure diagnostic to parameterize (decouple concentration from substructure)
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

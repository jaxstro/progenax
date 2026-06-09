# progenax — status

next: Validation audit — RESUME AT kinematics/rotation-anisotropy (solid-body + differential rotation; check for validation tier/page) then substructure (CW04 Q diagnostic) + diagnostics (energy-sorted segregation); defer gravoturb_fdf. **Michie DONE (2026-06-08)**: new tests/validation/test_michie_physics.py (12 tests, released-core 830→842) + scripts/validate_michie.py (5 figs) + 50-validation/michie-anisotropy.md. KEY FINDING: Michie-King β(r) is SUPPRESSED below the pure Osipkov-Merritt ceiling r²/(r²+r_a²) because the lowering term −exp(−J²/2r_a²σ²) breaks the f(Q) form — validated the sampler against the model's OWN DF β (2nd-moment oracle), not OM. γ/W0/r_c/M all differentiable. Per-module: status check → brainstorm plot set (HITL) → build pub figures → embed → BUILD-VERIFY → dashboard. **Plummer + King + EFF (2026-06-08) all done**: dedicated scripts/validate_{plummer,king,eff}.py on shared scripts/_plotstyle.py, each 5 pub figures printing expected-vs-measured; validation tables carry Measured columns; validate_profiles.py deleted (superseded). EFF: fixed false page claims (no γ>3 ValueError, no from_rh r_h-mapping, no KS test, "γ differentiable unlike King's W0" was wrong — both differentiable); γ=5→Plummer exact; γ AD-grad 2e-11. Entry point: docs/plans/2026-06-08-validation-audit-and-methods-figures-spec.md (per-module: brainstorm plot set → build pub-quality figures → embed on 50-validation page → BUILD-VERIFY render → update index dashboard). Done so far: **Plummer (2026-06-08, 5 pub figures)** + **King (32 tests, 5 figs)** + two-component + mass-segregation pages current+figure-bearing+build-verified; Λ_MSR validated; Tier C built. NEW shared scripts/_plotstyle.py (Okabe-Ito + ApJ rcParams); King refactored onto it. Plummer: dedicated scripts/validate_plummer.py (density+CDF, velocity-equilibrium, Beta(3/2,9/2), gradient AD-vs-FD, isotropy), each printing expected-vs-measured; Plummer+King validation tables now carry a Measured column (real regenerated numbers, fixed old overclaims). LIMEPY reframed across 13 pages (cite Gieles2015, own differentiable lowered-model family) + new roadmap 10-theory/spatial-profiles/lowered-model-family.md. EFF figure rewrite in validate_profiles.py still pending the EFF audit (validate_profiles.py is now Plummer-stale — its validate_plummer() superseded by scripts/validate_plummer.py; clean it during EFF).
blocker: none
due:

## Current focus
_Seeded 2026-06-07 by the brain STATUS.md convention (`~/brain/work/meta/status-convention.md`). Update in your progenax session; the brain pulls `next:`/`blocker:`/`due:` via `federate.py`._

Differentiable ICs / population generation (IMF, binaries, cluster profiles). Feeds the Cottrell census (Aim 1) + the jaxstro methods paper. Mature, actively committed.

## Recent progress (2026-06-08)
- **King auto-domain + differentiability.** `from_W0_rc`/`KingVelocityDF` now
  auto-size the ODE domain from W0 (tracer-safe: falls back to fixed domain under
  jit/grad-over-W0, so differentiability is preserved) — high-W0 (to 15) models
  "just work"; backward-compatible for W0≤9. Released core 822→830. Added Fig 5
  gradient-validation (AD vs FD: r_c 2e-10, W0 2e-7, M 1.6e-6) and CORRECTED the
  docs: W0 IS differentiable (diffrax dψ/dW0); only the scalar r_t crossing is
  blocked (argmax). Joint (W0, r_c, M) gradient/HMC inference is supported.
- **profiles/King audit + release-hardening.** king-profile.md rewritten to the
  exact tested tolerances; 4 build-verified figures via scripts/validate_king.py
  (concentration vs Table II, density-vs-oracle [Option A: K-function overlay
  removed], velocity equilibrium, W₀ sweep); 50-validation index dashboard added.
  Release-grade stale-text sweep across all King files (king.py, king_df.py,
  test_king_physics.py, king.md + king-dfs.md theory, validate_profiles.py):
  removed internal audit labels (B2.0/B2.1/M5/M6/C2), bug-history comments, the
  uncited "W0>12 unstable" note, and the **fabricated** "validation cross-checks
  against LIMEPY at g=1" claim; fixed king-dfs.md sampling description to match the
  code (per-particle 1-D inverse-CDF via vmap, not a 2-D table). 24 King tests +
  822 released-core green; all 4 figures build-verified in _build + page JSON.
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
- [ ] **Differentiable King r_t (Approach B, IFT)** — DEFERRED until a scalar-r_t
      use case is concrete (headline: tidal-field/Jacobi coupling r_t≈r_J → infer
      Galactic potential/orbits from cluster limiting radii, wiring in
      progenax.tidal). Plan + science cases:
      docs/plans/2026-06-08-king-differentiable-tidal-radius-deferred.md
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

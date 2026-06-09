# progenax — status

next: **VALIDATION-AUDIT CAMPAIGN COMPLETE** — every released module now at publication standard (5 build-verified figures + Measured-column table each). 2026-06-09 closed the last rows: analytical-test-cases (R2, push 9b96bc5) + tidal-truncation (R3 harden-then-validate, push 2db8405) + a full scientific-correctness pass on IMF/binary/environment (push a9bf54a). Only gravoturb_fdf/pp20 remain ⚠️ (experimental, repo-only, deferred by instruction). Released-core 866→893. Optional future: gravoturb_fdf release-hardening; the deferred items in §8 (differentiable King r_t, unified lowered-model family, self-consistent rotating equilibria).

SCIENTIFIC-CORRECTNESS PASS (2026-06-09, a9bf54a): line-by-line review of the full IMF/binary/environment code vs HELD PDFs (3 parallel audits + independent verification). KEY BUG: `alpha3_marks_plane` used the Marks+2012 PUBLISHED TYPO threshold +0.87 (a missing minus sign); the 2014 erratum (now held docs/core-papers/marks-2014-erratum.pdf) + Marks Fig.6 prove it's -0.87, and the authors used -0.87 (Table 1 unaffected). Fixed → corrected Marks ≡ Jeřábková IGIMF (gap 0.008, SAME relation; the earlier "two divergent models" Fig.4 was wrong). Also: differentiable.py 4-segment power_integral α=1 divide-by-zero (NaN likelihood+grad → added safe-denominator guard + TDD regression). Provenance lesson REPEATED: a review agent falsely said Salpeter/Kroupa/Maschberger PDFs not held → I propagated it into imf-statistics.md; they ARE held → corrected (all 4 IMF forms now PDF-verified). Chabrier mean 0.607 over [0.08,100] CONFIRMED correct (independent integral 0.604); the "~0.35" expectation is the mean integrated to m_min=0.01 (substellar included). Everything else PASSED (IMF forms+Jacobians, Kepler/orbit conservation, binary-aware 1/m1 Jacobian via integral=E[f_b], Moe Table 13, Marks Table 3/4 + 8π density).

CAMPAIGN R2/R3 (2026-06-09, pushed): **analytical-test-cases** (9b96bc5): scripts/validate_analytical.py — two-body Kepler (E=−Gm₁m₂/2a, closes 1.1e-7), Chenciner–Montgomery figure-eight (L=0 exact, closes 4.3e-8), Kepler III across 8 planets vs observed periods (max rel 6.9e-3) + IAU 2009 mass ratios, harmonic oscillator, adversarial spun-figure-8 (L=0.30 fails closure); fixed stale page (false "no test_analytical_physics.py", wrong Aarseth1974→Chenciner-Montgomery cite). **tidal-truncation** (2db8405, harden-then-validate): NEW tests/validation/test_tidal_physics.py (9 tests) validates r_J vs the restricted-3-body L1 point (Hill correction ∝(m/M_g)^(1/3), slope 0.34), the 3Ω² force balance, the (3/2)^(1/3) Keplerian-vs-isothermal factor, and apply_tidal_truncation vs analytic Plummer M(<r); scripts/validate_tidal.py 5 figs; differentiable in r_t via straight-through surrogate (vs analytic shell, median 0.02).

CAMPAIGN R2 EXTENSION (2026-06-08, pushed): 3 more modules to pub standard, each 5 build-verified figures + Measured-column tables. **imf-statistics** (c0b8cfe): scripts/validate_imfs.py rewrite onto _plotstyle — recovered slopes via sample-MLE (Salpeter 2.347±0.004, Kroupa 0.294/1.297/2.301), PDF/CDF coverage+KS, mean_mass() vs 200k log-grid ref (≤1.2e-4) with an adversarial linear-grid failure-mode panel, inference-layer gradient (rel 1e-5); fixed classic.md stale Chabrier params (0.22/0.57 were the *system* IMF → code defaults single-star 0.08/0.69), wrong class name, non-existent API kwargs. **binary-imf** (d43e0c5): scripts/validate_binaries.py — Kepler/orbit oracles (machine-precision), Moe+17 q-dist+KS (mass-dependent twin excess 15.2%→6.3%), HEADLINE regenerated "confidently wrong" via fast differentiable MLE (no MCMC): naive single-star fit 17.8σ-wrong at N=1e5 (α̂=2.21 vs 2.30) while binary-aware recovers 2.298 (0.4σ); bias∝f_b →0 at f_b=0; grad 8e-6; fixed stale binary_aware_validation.py script name. **environment-IMF** (9c358d9, harden-then-validate): NEW tests/validation/test_environment_physics.py (12 tests vs published Marks+2012 Table 1 GC α3 + Table 4 slopes) → released-core 866→878; scripts/validate_environment.py — GC anchors (NGC 7078 most top-heavy 0.844 vs pub 0.76), Fundamental Plane (density beats metallicity 7:1), low-mass slopes exact, HONEST Marks-vs-Jeřábková cross-model (both top-heavy mechanism but ~0.7 zero-point divergence reported not asserted away), grad 9e-8. All myst-build-verified (figs in _build/site/public + page JSON); new 50-validation/environment-imf.md wired into nav+dashboard+audit-report.

CAMPAIGN SUMMARY (2026-06-08): 6 modules at publication standard, each 5 build-verified figures + Measured-column tables: Plummer, King(retrofit), EFF, Michie, rotation/OM-anisotropy, substructure/CW04 Q; diagnostics (Λ_MSR + energy-sorted segregation) already done. Shared scripts/_plotstyle.py; LIMEPY reframed across 13 pages + roadmap (10-theory/spatial-profiles/lowered-model-family.md); validate_profiles.py deleted. Released-core 830→861 (+31 validation tests, all collect clean, build clean). Commits f964ebf, 4db6dbb, ee5f299, fcbfa32, 66cedaf on local main. Findings handled honestly: King W0 differentiable (corrected docs); EFF γ=5≡Plummer exact; Michie β suppressed below Osipkov-Merritt (validated vs DF moment oracle); OM stretch realizes β exactly (the Michie contrast); q_approx kNN-surrogate gradient (median 0.9%, cell-boundary worst-case). **rotation/anisotropy DONE (2026-06-08)**: new tests/validation/test_rotation_anisotropy_physics.py (10 tests, released-core 842→852) + scripts/validate_rotation_anisotropy.py (5 figs) + 50-validation/rotation-om-anisotropy.md. Solid-body v_φ=ΩR slope exact (resid 3e-14), L_z budget exact, differential curve <1e-6; OM anisotropy β(r)=r²/(r²+r_a²) realized EXACTLY for Plummer+EFF (dev 0.024/0.016) via Merritt-1985 direction-stretch — the clean CONTRAST to Michie's suppressed β. Ω/v_peak/r_a all differentiable (3e-9/7e-9/8e-6). **Michie DONE (2026-06-08)**: new tests/validation/test_michie_physics.py (12 tests, released-core 830→842) + scripts/validate_michie.py (5 figs) + 50-validation/michie-anisotropy.md. KEY FINDING: Michie-King β(r) is SUPPRESSED below the pure Osipkov-Merritt ceiling r²/(r²+r_a²) because the lowering term −exp(−J²/2r_a²σ²) breaks the f(Q) form — validated the sampler against the model's OWN DF β (2nd-moment oracle), not OM. γ/W0/r_c/M all differentiable. Per-module: status check → brainstorm plot set (HITL) → build pub figures → embed → BUILD-VERIFY → dashboard. **Plummer + King + EFF (2026-06-08) all done**: dedicated scripts/validate_{plummer,king,eff}.py on shared scripts/_plotstyle.py, each 5 pub figures printing expected-vs-measured; validation tables carry Measured columns; validate_profiles.py deleted (superseded). EFF: fixed false page claims (no γ>3 ValueError, no from_rh r_h-mapping, no KS test, "γ differentiable unlike King's W0" was wrong — both differentiable); γ=5→Plummer exact; γ AD-grad 2e-11. Entry point: docs/plans/2026-06-08-validation-audit-and-methods-figures-spec.md (per-module: brainstorm plot set → build pub-quality figures → embed on 50-validation page → BUILD-VERIFY render → update index dashboard). Done so far: **Plummer (2026-06-08, 5 pub figures)** + **King (32 tests, 5 figs)** + two-component + mass-segregation pages current+figure-bearing+build-verified; Λ_MSR validated; Tier C built. NEW shared scripts/_plotstyle.py (Okabe-Ito + ApJ rcParams); King refactored onto it. Plummer: dedicated scripts/validate_plummer.py (density+CDF, velocity-equilibrium, Beta(3/2,9/2), gradient AD-vs-FD, isotropy), each printing expected-vs-measured; Plummer+King validation tables now carry a Measured column (real regenerated numbers, fixed old overclaims). LIMEPY reframed across 13 pages (cite Gieles2015, own differentiable lowered-model family) + new roadmap 10-theory/spatial-profiles/lowered-model-family.md. EFF figure rewrite in validate_profiles.py still pending the EFF audit (validate_profiles.py is now Plummer-stale — its validate_plummer() superseded by scripts/validate_plummer.py; clean it during EFF).
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

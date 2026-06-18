# Changelog

All notable changes to progenax are documented here. This project adheres to
[Semantic Versioning](https://semver.org/).

## 0.1.0 (unreleased)

First public-ready release candidate. This cycle resolves the release blockers from
the 2026-06-11 pre-release adversarial audit
(`docs/reviews/2026-06-11-prerelease-adversarial-audit.md`), moving progenax from Beta
toward Release Candidate. Finding IDs (R*, J*, S*, A*, T*, D*, M*) refer to that report.

### Breaking changes

- **Explicit `G` now required on every velocity-sampling surface** (A2): the
  `G=None → DEFAULT_UNITS.G` convenience default was removed, per the MANDATORY
  explicit-units policy. Affected surfaces: `PlummerVelocityDF`, `KingVelocityDF`,
  `EFFVelocityDF`, `MichieVelocityDF`, and `LIMEPYVelocityDF` `.sample_velocities`;
  `MultiComponentCluster.sample_cluster`; and `sample_velocities_pipeline`. Callers
  must now pass an explicit `G` (e.g. `STELLAR.G`) — omitting it raises `TypeError`.
  Forward values are bit-identical to the previous explicit-`G` behavior; only the
  silent fallback is gone.

### Science fixes

- **Moe & Di Stefano F_twin normalization** (R3): the twin excess was mixed against the
  whole q ∈ [0.1, 1] population instead of the paper's q > 0.3 convention (realized
  F_twin 0.367 vs Table-13 0.300 at solar logP=1). `MoeDiStefano2017Full` rebuilt as a
  jointly-normalized PL + ft/(1−ft)·I_B twin block; pinned at four Table-13 nodes (`0adc74e`).
- **King/Michie/EFF core resolution at high concentration** (R4): the linear 1000-pt
  position-CDF grid under-resolved the core (measured +18% at W0=9, +270% at W0=12, 0.3 r_c).
  Replaced with a sqrt-stretched `r = r_t·u²` grid + non-uniform trapezoid; EFF's
  `compute_profile_potential` made consistent (`d35164e`, `f57e34d`).
- **`sample_fixed_n` silent mass shortfall** (R5): asking 500 M☉ for n=1000 silently
  returned 349 M☉. Now raises on an unreachable target; first tests for all four
  mass-target sampling modes (`9085e88`).
- **`compute_stellar_radii` mass–radius relation** (R6): exponents were inverted vs MS
  homology (10 M☉ → 6.3 R☉). Adopted Demircan & Kahraman (1991) (`20c62ec`).
- **PowerLawIMF α=1 NaN gradient** (R10): ported the `exp_safe` double-where so gradients
  are finite at exactly α=1 (`b98ff0f`).
- **EFF γ=3 sub-virial offset** pinned (S2); honest equilibrium caveats for rotation
  overlays, tidal truncation, mass segregation, the binary energy budget, and subgroup
  virial (S3/S4/S6/S10/S16) (`240ac81`, `7946fdf`).

### JAX / API hardening

- King `r_t` boundary-pinning is now flagged (traced) or refused (eager) instead of
  silently wrong (J4) (`43d5481`).
- `virial_scale` refuses T=0 input and is deduped with `rescale_velocities_to_virial` (J5)
  (`6d3d7b6`).
- `q_approx` uses Python-`if` dispatch on static N (no double-branch compile) (J6) (`cb1b94b`).
- Inconsistent `KingProfile` r_t now warns; the dead `KingVelocityDF.r_t` field removed
  (S1/A3) (`3a86878`). Zero-rotation-axis now raises (S15) (`7946fdf`).
- Single source of truth for the `VelocityDF` protocol (A1) (`83948ba`).

### Documentation

- Exported the 15 documented IMF/binary symbols from `progenax.__init__`; removed the
  phantom `IGIMF`/`EnvironmentIMF` class claims (the environment-dependent IMF is the
  functional `BirthEnvironment` + `env_to_imf_params` API) (R7) (`a09e5a8`).
- README rewritten against the current API, with an executable-examples smoke test that
  runs every code block; installation.md fixed (R7/D3/D4/H2) (`b825b2a`).
- Removed the fabricated `pip install progenax-legacy` claim (D5/H4) (`7ac2547`).
- Reconciled the three conflicting test-count claims (T5/L3/D6) (`e0943be`).

### Packaging & CI

- **CI un-broken** (R1/D2): the lock was decoupled from the private `jaxstroviz` sibling
  (CI had been red since 2026-06-10); the unit tier is sharded and the JAX compilation
  cache is bounded in CI to fix OOM kills (`7fe4cec`, `2bb8f44`, `bc98083`).
- Nightly + release-tag lane running the slow-marked headline validations, with
  `PROGENAX_STRICT_REFS=1` turning a missing reference cache into a failure (R8/T4)
  (`02689fb`, `095e3b2`).
- **LICENSE** added (Apache-2.0, © 2026 Anna Rosen) + author/classifiers/URL in the wheel
  metadata (D1) (`a688e53`).
- `[diagnostics]` extra + lazy numpy/scipy imports with actionable errors — `import
  progenax.diagnostics` no longer crashes in a clean install (R9) (`b140bac`).
- wheel-smoke (clean-venv import) + nightly Python version matrix (3.10/3.13) (M6/T3)
  (`bc98083`).

### Known limitations / deferred (require a separate decision or arc)

- **Not yet pip-installable for outsiders** (R2): the runtime depends on the unpublished
  sibling `jaxstro`. Resolution (publish `jaxstro` to PyPI, vendor its `units`+`jaxconfig`,
  or a git-URL dependency) is a strategy decision tracked separately.
- A two-sided quantile stretch for `sample_fixed_n` (R5 follow-up) is deferred.

  (Units-policy A2 — DF `G=None` defaults vs protocol-wide explicit G — is now resolved;
  see "Breaking changes" above.)

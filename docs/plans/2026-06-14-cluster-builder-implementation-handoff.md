# `build_cluster` implementation — session handoff (kickoff prompt)

**Date:** 2026-06-14 · **Companion to:** the ratified design
[`2026-06-14-cluster-builder-api-design.md`](2026-06-14-cluster-builder-api-design.md) (commit `0a7a424`)

Paste the block below to kick off the implementation session. It is self-contained: it carries the
ratified decisions, the open questions to brainstorm first, the verified codebase facts, the
Definition-of-Complete, the demo/versatility scope, the strict protocol, and the local gate commands.

---

```
KICKOFF — implement the `build_cluster` convenience IC-builder API (progenax / jaxstro)

ROLE & GOAL
You are continuing progenax (~/projects/jaxstro-dev/progenax, part of the jaxstro ecosystem).
This session: (1) brainstorm the remaining OPEN design questions, then (2) implement the
`build_cluster` convenience IC-builder API to full "Definition of Complete" — unit + integration
tests, a validation script with plots, grad-gate registration, a versatile demo set, and a
completion doc — under strict human-in-the-loop (HITL) with Anna.

START STATE
- `main` is clean at the merged provenance-audit arc (commit 3baa556, pushed). Working tree clean.
- FIRST: `git switch -c feat/cluster-builders` off main.
- The API is RATIFIED. READ THIS FIRST and treat its decisions as settled (do not relitigate):
  docs/plans/2026-06-14-cluster-builder-api-design.md  (commit 0a7a424).

RATIFIED DESIGN (summary — full detail in the doc)
- Hybrid shape: one generic engine + thin named aliases.
  build_cluster(profile, *, key, masses=None, n=None, imf=None, units=None, Q=0.5,
                anisotropy_radius=None, tidal_radius=None, rotation=None,
                revirialize=False, softening=0.0) -> ICResult
- Parameterize by a PROFILE OBJECT (not a model= string).
- New primitive: matched_velocity_df(profile, anisotropy_radius=None) -> VelocityDF — auto-pairs the
  equilibrium DF (kills the r_h-desync footgun). Mapping: Plummer→PlummerVelocityDF,
  King→KingVelocityDF, EFF→EFFVelocityDF, Michie→MichieVelocityDF, LIMEPY→LIMEPYVelocityDF.
  anisotropy_radius (OM β=r²/(r²+r_a²)) is valid ONLY for Plummer/EFF DFs.
- Aliases: build_plummer_cluster(n|masses, r_h, …), build_king_cluster(W0, r_c, …),
  build_eff_cluster(r_h, γ, …) — construct the profile and delegate.
- Mass spec: `masses=array` (inference path) OR `n[+imf]` (generative; `n` alone → equal 1 Msun).
- Modifiers REUSE existing code: anisotropy via matched DF; tidal via apply_tidal_truncation
  (shape-preserving, differentiable in r_t, masses→0 "ghosts"; document the S4 super-virial caveat +
  `revirialize` opt-in; King/LIMEPY profiles are the stationary tidal route; ERROR on double-truncation);
  rotation via apply_solid_body_rotation (float ω, or a RotationSpec for differential).
- Differentiability: profile scalar params, anisotropy_radius, tidal_radius, rotation.omega, Q are
  TRACED leaves; n, key, units, profile *type* are static. Headline:
  jax.grad(lambda r_h: loss(build_cluster(PlummerProfile(r_h), masses=m, key=k))).
- Units: convenience wrapper, units=None → DEFAULT_UNITS (STELLAR); core gets explicit G=units.G.

OPEN QUESTIONS — BRAINSTORM THESE FIRST (use the brainstorming skill; ONE question at a time;
lead with a recommendation; get Anna's explicit ratification BEFORE writing builder code):
  1. Aliases — ship build_michie_cluster / build_limepy_cluster too, or generic-only for those?
  2. Multi-component — should build_cluster accept a MultiComponentCluster, or stay strictly
     single-population (MultiComponentCluster.from_* remains the multi path)?
  3. Binary path — fold a binaries option into build_cluster, or keep build_binary_cluster
     separate (ratified default: separate)?
  4. matched_velocity_df — first-class public primitive (own docs/tests) or internal helper?
  5. Inference ergonomics — add a thin params→ICResult wrapper for HMC, or is the signature enough?
  6. revirialize default for tidal — confirm False (+ document S4) is right.

REAL CODEBASE FACTS (verified 2026-06-14; re-verify before relying)
- Core in src/progenax/builders.py:
    build_spatial_ic(profile, masses, velocity_df, key, G, Q=0.5, softening=0.0, id_offset=0) -> ICResult
    build_binary_cluster(profile, velocity_df, primary_imf, companion_model, target, key, *, units, Q=0.5, …)
  builders.py is 523 LOC (OVER the 500-LOC limit) → put new code in a NEW module
  src/progenax/builders_cluster.py; re-export from progenax/__init__.py.
- ICResult (eqx.Module): positions, velocities, masses, stellar_radii, ids, primordial_system_id,
  is_primordial_secondary, component_id.
- Modifiers: src/progenax/tidal.py
    apply_tidal_truncation(positions, velocities, masses, r_t, grad_width=0.05)
      -> (pos, vel, masses_truncated[r>r_t→0], keep_mask)   # shape-preserving, differentiable in r_t
    jacobi_radius(M_cluster, M_galaxy, R_galactic)
  src/progenax/kinematics/rotation.py:
    apply_solid_body_rotation(velocities, positions, omega, axis); apply_differential_rotation(…)
- Profiles/DFs exported from `progenax` (Plummer/King/EFF/Michie/LIMEPY + their *VelocityDF`s).
  King's r_t is a profile param (stationary tidal route).
- GRAD-AUDIT GATE (do not skip): tests/validation/grad_audit/manifest.py holds SYMBOL_CATEGORY over
  every progenax.__all__ symbol + a coverage ratchet + an __all__ cross-check. EVERY new public symbol
  (build_cluster, the aliases, matched_velocity_df, RotationSpec) MUST be added to SYMBOL_CATEGORY
  (AUDITED) and get grad-audit Cases in tests/validation/grad_audit/registry.py — otherwise the ratchet
  REDS the gate. The AD-vs-FD assertions for r_h, r_a, r_t, ω belong there.

DEFINITION OF COMPLETE (CLAUDE.md — all five; Anna explicitly wants integration tests + demos + versatility):
  1. Unit (tests/unit/builders/): matched_velocity_df pairs all 5 profiles scale-matched; mass-spec
     resolution (4 paths + 2 error cases); build_cluster ≡ manual build_spatial_ic BIT-IDENTICAL in the
     base case (proves pure sugar — no physics drift); anisotropy threading + unsupported-model errors;
     tidal masses→0 + double-truncate error; rotation L_z>0; units=None→STELLAR; aliases ≡ build_cluster.
  2. Integration (tests/integration/): build_cluster end-to-end across ALL profiles → equilibrium ICs
     (Q≈0.5 unscaled), each modifier applied, and jit+grad through the whole call.
  3. Validation (validation/validate_cluster_builders.py + validation/plots/): Q≈0.5 per alias,
     density-profile recovery, tidal-cut correctness, rotation L_z; expected-vs-measured tables;
     publication-quality figures (reuse scripts/_plotstyle.py).
  4. Differentiability: grad-audit registry Cases for r_h, r_a, r_t, ω (AD-vs-FD, jit-safe), manifest updated.
  5. Completion doc: .claude-work/TASK_build_cluster_COMPLETE.md.

VERSATILITY / DEMOS (Anna: "make it versatile"):
  Build a demo set showing — (a) the one-call onboarding one-liner build_plummer_cluster(n=1000, r_h=1.0);
  (b) differentiable θ→ICResult inference (gradient/Fisher on r_h, r_a, r_t, ω — the B-series pattern,
  now one-call); (c) all 5 profiles via the generic engine; (d) each modifier with a physical readout
  (β(r), tidal cut, L_z); (e) generative vs inference paths. Reuse the existing _demo_inference harness +
  scripts/_plotstyle.py where possible. Gate each demo (ALL PASS, capture run records).

STRICT PROTOCOL (NON-NEGOTIABLE):
  - brainstorming skill BEFORE any builder/solver code; present trade-offs; ONE question at a time;
    Anna ratifies each design choice.
  - HITL: Anna approves at every checkpoint; talk to her often; no silent decisions; she approves every
    deletion before it lands.
  - TDD RED→GREEN→REFACTOR. NEVER weaken a test/tolerance to pass — fix the physics.
  - JAX-native only (jax.numpy, equinox, lax.scan; NO numpy/scipy in core; everything differentiable;
    no while_loop in hot paths).
  - Limits: file ≤500 LOC (300 preferred), function ≤100 LOC (50 preferred), ≤15 fields.
  - Units: explicit G in core; units=None→DEFAULT_UNITS ONLY in the convenience wrapper.
  - CI MINUTES ARE EXHAUSTED → verify LOCALLY. GitHub workflows are disabled_manually and have no
    push-to-main trigger — keep it that way. Do NOT push/merge without Anna's explicit go.
  - Commit per verified batch; end commit messages with:
    Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

LOCAL GATES (run these; do not rely on CI):
  FAST (inner loop):
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
      env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -m "not slow" -n auto
  FULL (commit/phase gate): same command WITHOUT `-m "not slow"`.
  GRAD-GATE: env -u VIRTUAL_ENV uv run --no-sync pytest tests/validation/grad_audit -q
             && env -u VIRTUAL_ENV uv run --no-sync python scripts/audit_gradients.py   # exit 0
  DOCS: cd docs/website && myst build --html   # must be 0 warnings

FIRST STEPS:
  1. Read docs/plans/2026-06-14-cluster-builder-api-design.md fully.
  2. git switch -c feat/cluster-builders   (off main @ 3baa556).
  3. Invoke the brainstorming skill; resolve the 6 open questions with Anna.
  4. TDD-implement in order: matched_velocity_df → build_cluster → aliases → modifiers →
     grad-gate registration → integration tests → validation+plots → demos → completion doc.
     Commit per verified batch; HITL throughout; verify LOCALLY; do not push without Anna's go.
```
